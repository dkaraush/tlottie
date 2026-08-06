//! Shape evaluation shared by the frame walker and CPU test oracle.

use super::*;
use alloc::vec::Vec;

pub(crate) struct ShapeWalker<'a> {
  pub(crate) scratch: &'a mut RenderScratch,
  pub(crate) frame: f32,
  pub(crate) clip: &'a ClipQuad,
  pub(crate) curve_tolerance: f32,
  pub(crate) width: usize,
  pub(crate) height: usize,
  pub(crate) antialias: bool,
  pub(crate) color_override: Option<Color>,
  /// Retains complete canonical contours for translation-parametric replay.
  /// Only used with an empty authored clip stack; the rasterizer still clips
  /// translated edges to the target bounds while depositing coverage.
  pub(crate) unbounded: bool,
}

/// A paint recorded during the walk. `range` indexes the geometry arena;
/// contours are snapshotted only AFTER the whole walk, so path modifiers
/// (trim) that appear later in the tree still affect earlier paints —
/// rlottie mutates paths first and paints afterwards.
enum PendingPaint {
  Solid {
    rule: FillRule,
    color: Color,
    opacity: f32,
  },
  Gradient {
    rule: FillRule,
    lut: alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
    lut_id: u64,
    map: GradientMap,
  },
  Stroke {
    color: Option<Color>,
    lut: Option<(alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>, u64, GradientMap)>,
    opacity: f32,
    hw: f32,
    cap: crate::stroke::Cap,
    join: crate::stroke::Join,
    miter_limit: f32,
    pattern: Vec<f32>,
    dash_offset: f32,
  },
}

pub(crate) struct PendingJob {
  paint: PendingPaint,
  start: usize,
  end: usize,
}

/// Materialized draw operation, executed in reverse walk order
/// (first-listed item paints on top).
pub(crate) enum DrawJob {
  Solid {
    key: u128,
    contours: Vec<Contour>,
    borrowed: Option<usize>,
    rule: FillRule,
    color: Color,
    opacity: f32,
  },
  Gradient {
    key: u128,
    src_key: u128,
    contours: Vec<Contour>,
    borrowed: Option<usize>,
    rule: FillRule,
    lut: alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
    map: GradientMap,
  },
}

impl ShapeWalker<'_> {
  pub(crate) fn walk_shapes(&mut self, shapes: &[Shape], m: Mat2x3, opacity: f32, depth: usize) -> Result<(Vec<(Contour, bool)>, Vec<PendingJob>)> {
    let mut arena: Vec<(Contour, bool)> = Vec::new();
    let mut pending: Vec<PendingJob> = Vec::new();
    self.walk(shapes, m, opacity, depth, &mut arena, &mut pending)?;
    Ok((arena, pending))
  }

  #[cfg(test)]
  pub(super) fn render_shape_jobs_cpu(&mut self, canvas: &mut Canvas<'_>, arena: &[(Contour, bool)], pending: &[PendingJob], mut record: Option<&mut Vec<ReplayJob>>) {
    // Materialize AFTER all modifiers ran, execute in reverse. Fused:
    // materialize is pure per-job (the arena is immutable once the walk
    // finished), so each job's geometry is built, drawn, and freed before
    // the next — nothing forces all jobs' contours to coexist.
    for pj in pending.iter().rev() {
      let contours = match self.materialize(pj, arena, &|_| false) {
        DrawJob::Solid {
          key,
          contours,
          borrowed,
          rule,
          color,
          opacity,
        } => {
          let geometry = borrowed.and_then(|index| arena.get(index).map(|(contour, _)| core::slice::from_ref(contour))).unwrap_or(&contours);
          canvas.fill::<false>(&mut self.scratch.cov_cache, key, geometry, rule, color, opacity);
          if let Some(rec) = record.as_deref_mut() {
            rec.push(ReplayJob::Solid { key, rule, color, opacity });
          }
          contours
        }
        DrawJob::Gradient {
          key,
          src_key,
          contours,
          borrowed,
          rule,
          lut,
          map,
        } => {
          let geometry = borrowed.and_then(|index| arena.get(index).map(|(contour, _)| core::slice::from_ref(contour))).unwrap_or(&contours);
          canvas.fill_gradient::<false>(&mut self.scratch.cov_cache, key, src_key, geometry, rule, &lut, &map);
          if let Some(rec) = record.as_deref_mut() {
            rec.push(ReplayJob::Gradient {
              key,
              src_key,
              rule,
              lut: lut.clone(),
              map: map.clone(),
            });
          }
          contours
        }
      };
      for c in contours {
        self.scratch.put_pts(c.points);
      }
    }
  }

  /// Replays a static layer's recorded paints straight from the coverage
  /// cache. Returns false (touching nothing) if any needed cache entry
  /// was evicted — the caller then takes the normal path.
  #[cfg(test)]
  pub(super) fn replay_jobs(&mut self, canvas: &mut Canvas<'_>, jobs: &[ReplayJob]) -> bool {
    let all_present = jobs.iter().all(|j| match j {
      ReplayJob::Solid { key, .. } => self.scratch.cov_cache.contains(*key),
      ReplayJob::Gradient { key, src_key, .. } => self.scratch.cov_cache.contains(*src_key) || self.scratch.cov_cache.contains(*key),
    });
    if !all_present {
      return false;
    }
    self.scratch.cov_cache.pinned = true;
    for j in jobs {
      match j {
        ReplayJob::Solid { key, rule, color, opacity } => {
          canvas.fill::<false>(&mut self.scratch.cov_cache, *key, &[], *rule, *color, *opacity);
        }
        ReplayJob::Gradient { key, src_key, rule, lut, map } => {
          canvas.fill_gradient::<false>(&mut self.scratch.cov_cache, *key, *src_key, &[], *rule, lut, map);
        }
      }
    }
    self.scratch.cov_cache.pinned = false;
    self.scratch.cov_cache.rotate_if_needed();
    true
  }

  pub(crate) fn materialize(&mut self, pj: &PendingJob, arena: &[(Contour, bool)], retained: &impl Fn(u128) -> bool) -> DrawJob {
    let slice = arena.get(pj.start..pj.end.min(arena.len())).unwrap_or(&[]);
    match &pj.paint {
      PendingPaint::Solid { rule, color, opacity } => {
        let key = self.fill_key(slice, *rule);
        let hit = retained(key);
        let borrowed = (!hit && slice.len() == 1 && self.clip_is_noop(&slice[0].0)).then_some(pj.start);
        DrawJob::Solid {
          key,
          contours: if hit || borrowed.is_some() {
            Vec::new() // hit: coverage replays, geometry unneeded
          } else {
            let mut v = Vec::with_capacity(slice.len());
            for (c, _) in slice {
              let copy = self.pooled_copy(c);
              v.push(self.clip_all_owned(copy));
            }
            v
          },
          borrowed,
          rule: *rule,
          color: *color,
          opacity: *opacity,
        }
      }
      PendingPaint::Gradient { rule, lut, lut_id, map } => {
        let key = self.fill_key(slice, *rule);
        let src_key = Self::src_key_of(key, *lut_id, map);
        let hit = retained(key);
        let borrowed = (!hit && slice.len() == 1 && self.clip_is_noop(&slice[0].0)).then_some(pj.start);
        DrawJob::Gradient {
          key,
          src_key,
          contours: if hit || borrowed.is_some() {
            Vec::new()
          } else {
            let mut v = Vec::with_capacity(slice.len());
            for (c, _) in slice {
              let copy = self.pooled_copy(c);
              v.push(self.clip_all_owned(copy));
            }
            v
          },
          borrowed,
          rule: *rule,
          lut: lut.clone(),
          map: map.clone(),
        }
      }
      PendingPaint::Stroke {
        color,
        lut,
        opacity,
        hw,
        cap,
        join,
        miter_limit,
        pattern,
        dash_offset,
      } => {
        let stroke_key = self.stroke_key(slice, *hw, *cap, *join, *miter_limit, pattern, *dash_offset);
        let gradient_keys = lut.as_ref().map(|(_, lut_id, map)| Self::src_key_of(stroke_key, *lut_id, map));
        let hit = retained(stroke_key);
        if hit {
          // Hit: coverage replays from the cache — skip dashing,
          // stroking and clipping wholesale (measured 12-13% of
          // 64px frames spent stroking geometry whose coverage
          // was already cached).
          return match lut {
            Some((lut, _lut_id, map)) => DrawJob::Gradient {
              key: stroke_key,
              src_key: gradient_keys.unwrap_or(stroke_key),
              contours: Vec::new(),
              borrowed: None,
              rule: FillRule::NonZero,
              lut: lut.clone(),
              map: map.clone(),
            },
            None => DrawJob::Solid {
              key: stroke_key,
              contours: Vec::new(),
              borrowed: None,
              rule: FillRule::NonZero,
              color: color.unwrap_or(Color::BLACK),
              opacity: *opacity,
            },
          };
        }
        let solo = slice.len() == 1 && pattern.is_empty();
        let mut contours: Vec<Contour> = Vec::new();
        let mut pieces: Vec<Contour> = Vec::new();
        for (contour, closed) in slice {
          // Every stroke emission (segment rect, join wedge or
          // miter tip, cap) stays within hw·max(miter_limit, √2)
          // of the source polyline, so ONE inflated-bbox test on
          // the source proves every piece clip-free — replacing
          // thousands of per-piece bbox scans (measured 12% of
          // stroke-heavy 64px frames).
          let margin = hw * miter_limit.max(1.5);
          let skip_clip = self.contour_clip_is_noop_inflated(contour, margin);
          pieces.clear();
          {
            let target = if skip_clip { &mut contours } else { &mut pieces };
            if pattern.is_empty() {
              stroke_polyline(
                &contour.points,
                &contour.anchors,
                *closed,
                *hw,
                *cap,
                *join,
                *miter_limit,
                &mut self.scratch.pts_pool,
                &mut self.scratch.stroke_segments,
                target,
                solo,
              );
            } else {
              for (piece, piece_anchors) in dash_polyline(&contour.points, &contour.anchors, *closed, pattern, *dash_offset) {
                stroke_polyline(
                  &piece,
                  &piece_anchors,
                  false,
                  *hw,
                  *cap,
                  *join,
                  *miter_limit,
                  &mut self.scratch.pts_pool,
                  &mut self.scratch.stroke_segments,
                  target,
                  false,
                );
              }
            }
          }
          if !skip_clip {
            for p in pieces.drain(..) {
              let clipped = self.clip_all_owned(p);
              contours.push(clipped);
            }
          }
        }
        match lut {
          Some((lut, _lut_id, map)) => DrawJob::Gradient {
            key: stroke_key,
            src_key: gradient_keys.unwrap_or(stroke_key),
            contours,
            borrowed: None,
            rule: FillRule::NonZero,
            lut: lut.clone(),
            map: map.clone(),
          },
          None => DrawJob::Solid {
            key: stroke_key,
            contours,
            borrowed: None,
            rule: FillRule::NonZero,
            color: color.unwrap_or(Color::BLACK),
            opacity: *opacity,
          },
        }
      }
    }
  }

  /// Forward pass. Geometry goes into `arena` (device space, unclipped);
  /// paints record their scope's arena range; trims mutate the arena
  /// range of their scope eagerly.
  #[allow(clippy::too_many_arguments)]
  fn walk(&mut self, shapes: &[Shape], m: Mat2x3, opacity: f32, depth: usize, arena: &mut Vec<(Contour, bool)>, pending: &mut Vec<PendingJob>) -> Result<()> {
    if depth > MAX_RENDER_DEPTH {
      return Ok(());
    }
    let scope_start = arena.len();
    let jobs_start = pending.len();
    for shape in shapes {
      match shape {
        Shape::Group(g) => {
          let (gm, gop) = transform_at(&g.transform, self.frame);
          let child_m = m.concat(gm);
          self.walk(&g.shapes, child_m, opacity * gop, depth + 1, arena, pending)?;
        }
        Shape::Path(p) => {
          let data = p.path.eval(self.frame);
          let closed = data.closed;
          let contour = self.scratch.take_contour();
          arena.push((flatten_path_reusing(&data, &m, self.curve_tolerance, contour), closed));
        }
        Shape::Rect(r) => {
          let pos = r.position.eval(self.frame);
          let size = r.size.eval(self.frame);
          let radius = r.radius.eval(self.frame);
          arena.push((rect_contour(pos, size, radius, r.reversed, &m, self.curve_tolerance), true));
        }
        Shape::Ellipse(e) => {
          let pos = e.position.eval(self.frame);
          let size = e.size.eval(self.frame);
          arena.push((ellipse_contour(pos, size, e.reversed, &m, self.curve_tolerance), true));
        }
        Shape::Polystar(ps) => {
          let data = polystar_path(
            ps.star,
            ps.reversed,
            ps.points.eval(self.frame),
            ps.position.eval(self.frame),
            ps.rotation.eval(self.frame),
            ps.inner_radius.eval(self.frame),
            ps.outer_radius.eval(self.frame),
            ps.inner_roundness.eval(self.frame),
            ps.outer_roundness.eval(self.frame),
          );
          let contour = self.scratch.take_contour();
          arena.push((flatten_path_reusing(&data, &m, self.curve_tolerance, contour), true));
        }
        Shape::RoundCorners(rc) => {
          let radius = rc.radius.eval(self.frame);
          let det = (m.a * m.d - m.b * m.c).abs();
          let radius = radius * det.sqrt();
          if radius > 0.0 {
            if let Some(range) = arena.get_mut(scope_start..) {
              for (contour, closed) in range {
                *contour = round_polyline_corners(contour, *closed, radius);
              }
            }
          }
        }
        Shape::Trim(tr) => {
          self.apply_trim(tr, arena, pending, scope_start);
        }
        Shape::Repeater(rp) => {
          self.apply_repeater(rp, m, arena, pending, scope_start, jobs_start);
        }
        Shape::Fill(f) => {
          let color = self.color_override.unwrap_or_else(|| f.color.eval(self.frame));
          let fill_opacity = (f.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
          let paint_opacity = opacity * fill_opacity;
          if opacity_byte(color.a * paint_opacity) == 0 {
            continue;
          }
          pending.push(PendingJob {
            paint: PendingPaint::Solid {
              rule: f.rule,
              color,
              opacity: paint_opacity,
            },
            start: scope_start,
            // rlottie: a paint covers only geometry that
            // precedes it in the walk (verified: a fill/stroke
            // listed before a path or nested group does NOT
            // paint that later geometry).
            end: arena.len(), // resolved at scope end
          });
        }
        Shape::GradientFill(gf) => {
          // Gradient geometry stays in local space; pixels are
          // pulled back through the inverse CTM at fill time.
          let fill_opacity = (gf.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
          let paint_opacity = opacity * fill_opacity;
          if opacity_byte(paint_opacity) == 0 {
            continue;
          }
          let start_p = gf.start.eval(self.frame);
          let end_p = gf.end.eval(self.frame);
          let inv = m.inverse();
          let stops = gf.stops.eval(self.frame);
          let (lut, lut_id) = self.scratch.lut_for(&stops, gf.color_count, paint_opacity);
          let map = match gf.kind {
            GradientKind::Linear => linear_map(start_p, end_p, inv),
            GradientKind::Radial => radial_map(start_p, end_p, inv, gf.highlight_len.eval(self.frame), gf.highlight_angle.eval(self.frame)),
          };
          pending.push(PendingJob {
            paint: PendingPaint::Gradient { rule: gf.rule, lut, lut_id, map },
            start: scope_start,
            // rlottie: a paint covers only geometry that
            // precedes it in the walk (verified: a fill/stroke
            // listed before a path or nested group does NOT
            // paint that later geometry).
            end: arena.len(),
          });
        }
        Shape::Stroke(st) => {
          let stroke_opacity = (st.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
          let paint_opacity = opacity * stroke_opacity;
          let color = self.color_override.unwrap_or_else(|| st.color.eval(self.frame));
          if opacity_byte(color.a * paint_opacity) == 0 {
            continue;
          }
          let width = st.width.eval(self.frame);
          let scale = stroke_scale(&m);
          let hw = 0.5 * width * scale;
          if hw <= 0.0 || !hw.is_finite() {
            continue;
          }
          let (pattern, dash_offset) = self.dash_pattern(&st.dashes, scale);
          pending.push(PendingJob {
            paint: PendingPaint::Stroke {
              color: Some(color),
              lut: None,
              opacity: paint_opacity,
              hw,
              cap: st.cap,
              join: st.join,
              miter_limit: st.miter_limit,
              pattern,
              dash_offset,
            },
            start: scope_start,
            // rlottie: a paint covers only geometry that
            // precedes it in the walk (verified: a fill/stroke
            // listed before a path or nested group does NOT
            // paint that later geometry).
            end: arena.len(),
          });
        }
        Shape::GradientStroke(gs) => {
          let stroke_opacity = (gs.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
          let paint_opacity = opacity * stroke_opacity;
          if opacity_byte(paint_opacity) == 0 {
            continue;
          }
          let width = gs.width.eval(self.frame);
          let scale = stroke_scale(&m);
          let hw = 0.5 * width * scale;
          if hw <= 0.0 || !hw.is_finite() {
            continue;
          }
          let start_p = gs.start.eval(self.frame);
          let end_p = gs.end.eval(self.frame);
          let inv = m.inverse();
          let stops = gs.stops.eval(self.frame);
          let (lut, lut_id) = self.scratch.lut_for(&stops, gs.color_count, paint_opacity);
          let map = match gs.kind {
            GradientKind::Linear => linear_map(start_p, end_p, inv),
            GradientKind::Radial => radial_map(start_p, end_p, inv, gs.highlight_len.eval(self.frame), gs.highlight_angle.eval(self.frame)),
          };
          let (pattern, dash_offset) = self.dash_pattern(&gs.dashes, scale);
          pending.push(PendingJob {
            paint: PendingPaint::Stroke {
              color: None,
              lut: Some((lut, lut_id, map)),
              opacity: opacity * stroke_opacity,
              hw,
              cap: gs.cap,
              join: gs.join,
              miter_limit: gs.miter_limit,
              pattern,
              dash_offset,
            },
            start: scope_start,
            // rlottie: a paint covers only geometry that
            // precedes it in the walk (verified: a fill/stroke
            // listed before a path or nested group does NOT
            // paint that later geometry).
            end: arena.len(),
          });
        }
      }
    }
    let _ = jobs_start;
    Ok(())
  }

  /// Dash array, rlottie semantics (model::Dash::getDashInfo): values are
  /// consumed POSITIONALLY in file order — the `n` role tags are ignored —
  /// and the LAST value is the offset. An even-length list gets rlottie's
  /// fixup first: the last value moves to the end as the offset and the
  /// second-to-last is duplicated into its place as a synthesized gap
  /// ([d,g] → dash=d, gap=d, offset=g — NOT period d+g).
  fn dash_pattern(&self, dashes: &[DashElement], scale: f32) -> (Vec<f32>, f32) {
    let mut raw: Vec<f32> = dashes.iter().map(|d| d.value.eval(self.frame) * scale).collect();
    if raw.len() <= 1 {
      return (Vec::new(), 0.0);
    }
    if raw.len() % 2 == 0 {
      let last = raw.last().copied().unwrap_or(0.0);
      let prev = raw.get(raw.len() - 2).copied().unwrap_or(0.0);
      if let Some(slot) = raw.last_mut() {
        *slot = prev; // duplicate previous dash as the missing gap
      }
      raw.push(last); // original last value becomes the offset
    }
    let offset = raw.pop().unwrap_or(0.0);
    for v in raw.iter_mut() {
      *v = v.max(0.0);
    }
    // AE "draw the dash even if dash value is 0" quirk: rlottie
    // (lottieitem.cpp LOTStrokeItem::updateRenderNode) forces the first
    // dash length to 0.1 when it is zero, AFTER scaling. This flips
    // VDasher's mNoLength false so a zero-dash/zero-gap array (`[d:0,o:0]`,
    // e.g. ShibaInu's heart outlines) reaches the `mNoGap → return solid
    // path` branch and renders as a SOLID stroke rather than nothing.
    if let Some(first) = raw.first_mut() {
      if first.abs() < 1e-6 {
        *first = 0.1;
      }
    }
    (raw, offset)
  }

  /// Repeater: replaces this scope's geometry with transformed copies and
  /// duplicates this scope's earlier paints per copy.
  fn apply_repeater(&self, rp: &crate::model::Repeater, m: Mat2x3, arena: &mut Vec<(Contour, bool)>, pending: &mut Vec<PendingJob>, scope_start: usize, jobs_start: usize) {
    let copies = rp.copies.eval(self.frame).clamp(0.0, 64.0) as usize;
    if copies <= 1 {
      return;
    }
    let offset = rp.offset.eval(self.frame);
    let so = (rp.start_opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
    let eo = (rp.end_opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
    // rlottie builds each copy's matrix PARAMETRICALLY with multiplier
    // mult = offset + i (LOTRepeaterTransform::matrix): position and
    // rotation scale linearly, scale is raised to the mult power, the
    // anchor applies once. A matrix power step^k is NOT the same once
    // rotation is nonzero (it spirals; the reference draws a row).
    let rp_anchor = rp.transform.anchor.eval(self.frame);
    let rp_pos = rp.transform.position.eval(self.frame);
    let rp_scale = rp.transform.scale.eval(self.frame);
    let rp_rot = rp.transform.rotation.eval(self.frame);
    let m_inv = m.inverse();
    let base_end = arena.len();
    let base_len = base_end - scope_start;
    let prior_jobs: Vec<(usize, usize, usize)> = pending.iter().enumerate().skip(jobs_start).map(|(i, pj)| (i, pj.start, pj.end)).collect();
    for i in 0..copies {
      let mult = offset + i as f32;
      let t_local = Mat2x3::translate(rp_pos.x * mult, rp_pos.y * mult)
        .concat(Mat2x3::translate(rp_anchor.x, rp_anchor.y))
        .concat(Mat2x3::scale((rp_scale.x / 100.0).powf(mult), (rp_scale.y / 100.0).powf(mult)))
        .concat(Mat2x3::rotate(rp_rot * mult))
        .concat(Mat2x3::translate(-rp_anchor.x, -rp_anchor.y));
      let t_dev = m.concat(t_local).concat(m_inv);
      let alpha = if copies > 1 { so + (eo - so) * (i as f32 / (copies - 1) as f32) } else { so };
      let block_start = arena.len();
      for idx in scope_start..base_end {
        let (contour, closed) = match arena.get(idx) {
          Some((c, cl)) => (
            Contour {
              points: c.points.iter().map(|p| t_dev.apply(*p)).collect(),
              anchors: c.anchors.clone(),
              // Repeater copies carry an extra device-space
              // conjugated transform; the flatten matrix no
              // longer describes them (device measure).
              inv_lin: None,
            },
            *cl,
          ),
          None => continue,
        };
        arena.push((contour, closed));
      }
      for &(ji, js, je) in &prior_jobs {
        if let Some(orig) = pending.get(ji) {
          let shift = block_start - scope_start;
          let np = clone_paint(&orig.paint, alpha);
          pending.push(PendingJob {
            paint: np,
            start: js + shift,
            end: if je == usize::MAX { usize::MAX } else { je + shift },
          });
        }
      }
    }
    // Originals are replaced by the copies: blank the base geometry and
    // disarm the original paint jobs.
    if let Some(range) = arena.get_mut(scope_start..base_end) {
      for (c, _) in range {
        c.points.clear();
      }
    }
    let _ = base_len;
    for pj in pending.iter_mut().skip(jobs_start).take(prior_jobs.len()) {
      pj.end = pj.start; // empty range
    }
  }

  fn apply_trim(&self, tr: &crate::model::Trim, arena: &mut Vec<(Contour, bool)>, pending: &mut Vec<PendingJob>, scope_start: usize) {
    let start_pct = tr.start.eval(self.frame) / 100.0;
    let end_pct = tr.end.eval(self.frame) / 100.0;
    // rlottie LOTTrimData::segment() (lottiemodel.h): offset is
    // fmod(deg,360)/360 and a window pushed past the path end WRAPS,
    // yielding TWO ranges — a "loop" segment with ss > ee.
    let offset = (tr.offset.eval(self.frame) % 360.0) / 360.0;
    let diff = (start_pct - end_pct).abs();
    if diff <= 1e-6 {
      if let Some(geoms) = arena.get_mut(scope_start..) {
        for (c, _) in geoms {
          c.points.clear();
        }
      }
      return;
    }
    if diff >= 1.0 - 1e-6 {
      return; // full path
    }
    let s = start_pct + offset;
    let e = end_pct + offset;
    let noloop = |a: f32, b: f32| (a.min(b), a.max(b));
    let loopf = |a: f32, b: f32| (a.max(b), a.min(b)); // ss > ee marks wrap
    let (ss, ee) = if offset >= 0.0 {
      if s <= 1.0 && e <= 1.0 {
        noloop(s, e)
      } else if s > 1.0 && e > 1.0 {
        noloop(s - 1.0, e - 1.0)
      } else if s > 1.0 {
        loopf(s - 1.0, e)
      } else {
        loopf(s, e - 1.0)
      }
    } else if s >= 0.0 && e >= 0.0 {
      noloop(s, e)
    } else if s < 0.0 && e < 0.0 {
      noloop(1.0 + s, 1.0 + e)
    } else if s < 0.0 {
      loopf(1.0 + s, e)
    } else {
      loopf(s, 1.0 + e)
    };
    // Fractional ranges along the path; two when the window wraps.
    let ranges: [(f32, f32); 2] = if ss <= ee { [(ss, ee), (0.0, 0.0)] } else { [(0.0, ee), (ss, 1.0)] };
    let range_count = if ss <= ee { 1 } else { 2 };

    match tr.mode {
      TrimMode::Simultaneous => {
        let mut i = scope_start;
        while i < arena.len() {
          let Some((contour, closed)) = arena.get(i) else {
            break;
          };
          let closed = *closed;
          let total = polyline_length(&contour.points, closed, contour.inv_lin);
          if closed && range_count == 2 {
            // Closed path: the wrapped window is one continuous
            // piece across the seam; extract_by_length wraps.
            let Some((contour, cl)) = arena.get_mut(i) else {
              break;
            };
            let (pts, anc) = extract_by_length(&contour.points, &contour.anchors, true, ss * total, (1.0 + ee) * total, contour.inv_lin);
            contour.points = pts;
            contour.anchors = anc;
            *cl = false;
            i += 1;
            continue;
          }
          let mut pieces: Vec<(Vec<Vec2>, Vec<bool>)> = Vec::with_capacity(range_count);
          for &(lo, hi) in ranges.iter().take(range_count) {
            if hi > lo + 1e-6 {
              let Some((contour, _)) = arena.get(i) else {
                break;
              };
              let piece = extract_by_length(&contour.points, &contour.anchors, closed, lo * total, hi * total, contour.inv_lin);
              if piece.0.len() >= 2 {
                pieces.push(piece);
              }
            }
          }
          i = splice_trimmed(arena, pending, i, pieces);
        }
      }
      TrimMode::Individual => {
        let totals: Vec<f32> = arena.get(scope_start..).unwrap_or(&[]).iter().map(|(c, cl)| polyline_length(&c.points, *cl, c.inv_lin)).collect();
        let grand: f32 = totals.iter().sum();
        if grand <= 1e-6 {
          return;
        }
        let mut i = scope_start;
        let mut acc = 0.0f32;
        let mut ti = 0usize;
        while i < arena.len() {
          let total = totals.get(ti).copied().unwrap_or(0.0);
          let Some((_, closed)) = arena.get(i) else {
            break;
          };
          let closed = *closed;
          let mut pieces: Vec<(Vec<Vec2>, Vec<bool>)> = Vec::new();
          for &(lo, hi) in ranges.iter().take(range_count) {
            let c0 = (lo * grand - acc).clamp(0.0, total);
            let c1 = (hi * grand - acc).clamp(0.0, total);
            if c1 > c0 + 1e-6 {
              let Some((contour, _)) = arena.get(i) else {
                break;
              };
              let piece = extract_by_length(&contour.points, &contour.anchors, closed, c0, c1, contour.inv_lin);
              if piece.0.len() >= 2 {
                pieces.push(piece);
              }
            }
          }
          acc += total;
          ti += 1;
          i = splice_trimmed(arena, pending, i, pieces);
        }
      }
    }
  }

  /// True when clipping is a bit-exact no-op for this contour: its bbox
  /// is inside the viewport AND inside every precomp clip quad (all under
  /// the clippers' own non-strict inside tests, so the Sutherland–Hodgman
  /// passes would return the polygon unchanged). Measured: on precomp-
  /// heavy files 100% of clip calls were fully inside — this one bbox
  /// pass replaces `1 + |clip|` full S-H passes and their allocations.
  fn clip_is_noop(&self, c: &Contour) -> bool {
    if self.unbounded {
      return true;
    }
    let wf = self.width as f32;
    let hf = self.height as f32;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &c.points {
      // NaN/inf must take the slow path: min/max ignore NaN, but the
      // clippers' `>= 0.0` comparisons treat NaN as OUTSIDE and cut.
      if !(p.x.is_finite() && p.y.is_finite()) {
        return false;
      }
      x0 = x0.min(p.x);
      y0 = y0.min(p.y);
      x1 = x1.max(p.x);
      y1 = y1.max(p.y);
    }
    c.points.is_empty() || (x0 >= 0.0 && y0 >= 0.0 && x1 <= wf && y1 <= hf && self.clip.iter().all(|q| quad_contains_box(q, x0, y0, x1, y1)))
  }

  /// Hash of everything besides the paint's own geometry that determines
  /// coverage: canvas dims + the precomp clip-quad stack.
  pub(super) fn clip_sig(&self) -> Hasher128 {
    let mut h = Hasher128::new();
    h.mix(self.width as u32);
    h.mix(self.height as u32);
    h.mix(u32::from(self.antialias));
    for q in self.clip.iter() {
      for p in q {
        h.mix(p.x.to_bits());
        h.mix(p.y.to_bits());
      }
    }
    h
  }

  /// Source-level coverage key for a fill: exact bits of the UNclipped
  /// arena geometry + rule + clip signature. Coverage is a deterministic
  /// function of these, so a hit can skip snapshot+clip+raster wholesale.
  pub(super) fn fill_key(&self, slice: &[(Contour, bool)], rule: FillRule) -> u128 {
    let mut h = self.clip_sig();
    h.mix(1); // paint kind tag: fill
    h.mix(match rule {
      FillRule::NonZero => 1,
      FillRule::EvenOdd => 2,
    });
    for (c, _) in slice {
      for p in &c.points {
        h.mix(p.x.to_bits());
        h.mix(p.y.to_bits());
      }
      h.mix(c.points.len() as u32);
    }
    h.finish()
  }

  /// Source-plane key: coverage key + LUT id + gradient map bits.
  fn src_key_of(key: u128, lut_id: u64, map: &GradientMap) -> u128 {
    let mut h = Hasher128::new();
    h.mix(3); // plane kind tag: gradient source
    h.mix(key as u32);
    h.mix((key >> 32) as u32);
    h.mix((key >> 64) as u32);
    h.mix((key >> 96) as u32);
    h.mix(lut_id as u32);
    h.mix((lut_id >> 32) as u32);
    let mh = map.content_hash();
    h.mix(mh as u32);
    h.mix((mh >> 32) as u32);
    h.finish()
  }

  /// Source-level coverage key for a stroke: geometry (points + anchor
  /// flags + closed — all three feed the stroker) + every stroke
  /// parameter + clip signature.
  #[allow(clippy::too_many_arguments)]
  fn stroke_key(&self, slice: &[(Contour, bool)], hw: f32, cap: crate::stroke::Cap, join: crate::stroke::Join, miter_limit: f32, pattern: &[f32], dash_offset: f32) -> u128 {
    let mut h = self.clip_sig();
    h.mix(2); // paint kind tag: stroke
    h.mix(hw.to_bits());
    h.mix(miter_limit.to_bits());
    h.mix(match cap {
      crate::stroke::Cap::Butt => 1,
      crate::stroke::Cap::Round => 2,
      crate::stroke::Cap::Square => 3,
    });
    h.mix(match join {
      crate::stroke::Join::Miter => 1,
      crate::stroke::Join::Round => 2,
      crate::stroke::Join::Bevel => 3,
    });
    for v in pattern {
      h.mix(v.to_bits());
    }
    h.mix(pattern.len() as u32);
    h.mix(dash_offset.to_bits());
    for (c, closed) in slice {
      for p in &c.points {
        h.mix(p.x.to_bits());
        h.mix(p.y.to_bits());
      }
      let mut bits: u32 = 0;
      for (i, &a) in c.anchors.iter().enumerate() {
        bits = (bits << 1) | u32::from(a);
        if i % 32 == 31 {
          h.mix(bits);
          bits = 0;
        }
      }
      h.mix(bits);
      h.mix(c.anchors.len() as u32);
      h.mix(c.points.len() as u32);
      h.mix(u32::from(*closed));
    }
    h.finish()
  }

  /// Copies a borrowed arena contour into a pooled buffer (anchors are
  /// not carried: nothing after materialization reads them).
  fn pooled_copy(&mut self, c: &Contour) -> Contour {
    let mut v = self.scratch.pts_pool.pop().unwrap_or_default();
    v.clear();
    v.extend_from_slice(&c.points);
    Contour {
      points: v,
      anchors: Vec::new(),
      inv_lin: None,
    }
  }

  /// clip_is_noop for a source polyline whose derived geometry may
  /// extend up to `margin` beyond it (stroke pieces): tests the inflated
  /// bbox against the same non-strict viewport + quad conditions.
  fn contour_clip_is_noop_inflated(&self, c: &Contour, margin: f32) -> bool {
    if self.unbounded {
      return true;
    }
    if !margin.is_finite() {
      return false;
    }
    let wf = self.width as f32;
    let hf = self.height as f32;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &c.points {
      if !(p.x.is_finite() && p.y.is_finite()) {
        return false;
      }
      x0 = x0.min(p.x);
      y0 = y0.min(p.y);
      x1 = x1.max(p.x);
      y1 = y1.max(p.y);
    }
    if c.points.is_empty() {
      return true;
    }
    let (x0, y0, x1, y1) = (x0 - margin, y0 - margin, x1 + margin, y1 + margin);
    x0 >= 0.0 && y0 >= 0.0 && x1 <= wf && y1 <= hf && self.clip.iter().all(|q| quad_contains_box(q, x0, y0, x1, y1))
  }

  #[cfg(test)]
  pub(super) fn clip_all(&self, c: &Contour) -> Contour {
    if self.clip_is_noop(c) {
      return c.clone();
    }
    let wf = self.width as f32;
    let hf = self.height as f32;
    let mut c = c.clone();
    for quad in self.clip.iter() {
      c = clip_to_quad(&c, quad);
    }
    clip_contour(&c, wf, hf)
  }

  /// clip_all for OWNED temporaries (stroke pieces): moves the contour
  /// through unchanged when nothing clips — the borrowing variant clones
  /// every piece, which the profiler showed as pure allocator churn.
  fn clip_all_owned(&self, c: Contour) -> Contour {
    if self.unbounded || self.clip_is_noop(&c) {
      return c;
    }
    let wf = self.width as f32;
    let hf = self.height as f32;
    let mut c = c;
    for quad in self.clip.iter() {
      c = clip_to_quad(&c, quad);
    }
    clip_contour(&c, wf, hf)
  }
}

/// Replaces `arena[idx]` with the trimmed `pieces` (0, 1, or 2 open
/// polylines). Extra pieces are INSERTED right after `idx` so they stay
/// inside every paint range that covered the original contour; recorded
/// job indices past the insertion point are shifted to compensate.
/// Returns the index of the next original entry.
fn splice_trimmed(arena: &mut Vec<(Contour, bool)>, pending: &mut Vec<PendingJob>, idx: usize, mut pieces: Vec<(Vec<Vec2>, Vec<bool>)>) -> usize {
  let (first, first_anchors) = if pieces.is_empty() { (Vec::new(), Vec::new()) } else { pieces.remove(0) };
  if let Some((contour, closed)) = arena.get_mut(idx) {
    contour.points = first;
    contour.anchors = first_anchors;
    *closed = false;
  }
  let extra = pieces.len();
  for (k, (pts, anc)) in pieces.into_iter().enumerate() {
    let at = (idx + 1 + k).min(arena.len());
    arena.insert(
      at,
      (
        Contour {
          points: pts,
          anchors: anc,
          inv_lin: None,
        },
        false,
      ),
    );
  }
  if extra > 0 {
    for pj in pending.iter_mut() {
      if pj.start > idx {
        pj.start += extra;
      }
      if pj.end != usize::MAX && pj.end > idx {
        pj.end += extra;
      }
    }
  }
  idx + 1 + extra
}

/// Clones a pending paint with opacity scaled (repeater copies).
fn clone_paint(p: &PendingPaint, alpha: f32) -> PendingPaint {
  match p {
    PendingPaint::Solid { rule, color, opacity } => PendingPaint::Solid {
      rule: *rule,
      color: *color,
      opacity: opacity * alpha,
    },
    PendingPaint::Gradient { rule, lut, lut_id, map } => PendingPaint::Gradient {
      rule: *rule,
      lut: lut.clone(),
      lut_id: *lut_id,
      map: map.clone(),
    },
    PendingPaint::Stroke {
      color,
      lut,
      opacity,
      hw,
      cap,
      join,
      miter_limit,
      pattern,
      dash_offset,
    } => PendingPaint::Stroke {
      color: *color,
      lut: lut.clone(),
      opacity: opacity * alpha,
      hw: *hw,
      cap: *cap,
      join: *join,
      miter_limit: *miter_limit,
      pattern: pattern.clone(),
      dash_offset: *dash_offset,
    },
  }
}

/// Scalar stroke-width scale for a (possibly non-uniform) CTM, matching
/// rlottie's getScale() (lottieitem.cpp): |M·(√2,√2)|/2, i.e. the RMS of the
/// mapped axis lengths — NOT the geometric mean det.sqrt(). Equal for
/// uniform scale; diverges exactly where non-uniform-scale strokes did.
fn stroke_scale(m: &Mat2x3) -> f32 {
  (((m.a + m.c) * (m.a + m.c) + (m.b + m.d) * (m.b + m.d)) * 0.5).sqrt()
}

fn linear_map(start: Vec2, end: Vec2, inv: Mat2x3) -> GradientMap {
  let dx = end.x - start.x;
  let dy = end.y - start.y;
  let len_sq = dx * dx + dy * dy;
  GradientMap {
    inv,
    kind: GradientMapKind::Linear {
      sx: start.x,
      sy: start.y,
      dx,
      dy,
      inv_len_sq: if len_sq > 1e-9 { 1.0 / len_sq } else { 0.0 },
    },
  }
}

/// Polyline length in rlottie's metric (see geometry::seg_len_rlottie) so
/// trim fractions land where rlottie puts them along curves.
fn polyline_length(points: &[crate::math::Vec2], closed: bool, inv: Option<[f32; 4]>) -> f32 {
  if points.len() < 2 {
    return 0.0;
  }
  let seg_count = if closed { points.len() } else { points.len() - 1 };
  let mut total = 0.0f32;
  for i in 0..seg_count {
    let Some(a) = points.get(i) else { break };
    let Some(b) = points.get(if i + 1 == points.len() { 0 } else { i + 1 }) else {
      break;
    };
    total += crate::geometry::seg_len_measured(b.x - a.x, b.y - a.y, inv);
  }
  total
}
