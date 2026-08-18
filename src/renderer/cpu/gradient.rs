//! Gradient lookup tables, coordinate maps, and compositing kernels.

use super::*;
use alloc::vec::Vec;

/// Builds a premultiplied RGBA8 lookup table from independently interpolated
/// Lottie color and opacity stops.
pub(crate) fn build_gradient_lut(stops: &FloatList, color_count: usize, opacity: f32) -> [u32; GRADIENT_LUT_SIZE] {
  let data = &stops.0;
  let n = color_count.min(data.len() / 4);
  let mut lut = [0u32; GRADIENT_LUT_SIZE];
  let opac = data.get(n * 4..).unwrap_or(&[]);
  let opacity_stop_count = if opac.len() >= 4 && opac.len() % 2 == 0 { opac.len() / 2 } else { 0 };

  for (i, slot) in lut.iter_mut().enumerate() {
    let t = i as f32 / (GRADIENT_LUT_SIZE - 1) as f32;
    let color = |index: usize| {
      let base = index.saturating_mul(4);
      (
        data.get(base).copied().unwrap_or(0.0),
        data.get(base + 1).copied().unwrap_or(0.0),
        data.get(base + 2).copied().unwrap_or(0.0),
        data.get(base + 3).copied().unwrap_or(0.0),
      )
    };
    let first = color(0);
    let last = color(n.saturating_sub(1));
    let (r, g, b) = if n == 0 {
      (1.0, 1.0, 1.0)
    } else if t <= first.0 {
      (first.1, first.2, first.3)
    } else if t >= last.0 {
      (last.1, last.2, last.3)
    } else {
      let mut out = (last.1, last.2, last.3);
      for index in 1..n {
        let p0 = color(index - 1);
        let p1 = color(index);
        if t <= p1.0 {
          let f = if p1.0 > p0.0 { ((t - p0.0) / (p1.0 - p0.0)).clamp(0.0, 1.0) } else { 1.0 };
          out = (p0.1 + (p1.1 - p0.1) * f, p0.2 + (p1.2 - p0.2) * f, p0.3 + (p1.3 - p0.3) * f);
          break;
        }
      }
      out
    };
    let stop_alpha = if opacity_stop_count == 0 {
      1.0
    } else {
      let opacity_stop = |index: usize| {
        let base = index.saturating_mul(2);
        (opac.get(base).copied().unwrap_or(0.0), opac.get(base + 1).copied().unwrap_or(1.0))
      };
      let first = opacity_stop(0);
      let last = opacity_stop(opacity_stop_count - 1);
      if t <= first.0 {
        first.1
      } else if t >= last.0 {
        last.1
      } else {
        let mut out = last.1;
        for index in 1..opacity_stop_count {
          let p0 = opacity_stop(index - 1);
          let p1 = opacity_stop(index);
          if t <= p1.0 {
            let f = if p1.0 > p0.0 { ((t - p0.0) / (p1.0 - p0.0)).clamp(0.0, 1.0) } else { 1.0 };
            out = p0.1 + (p1.1 - p0.1) * f;
            break;
          }
        }
        out
      }
    };
    let a = (stop_alpha * opacity).clamp(0.0, 1.0);
    let pa = (a * 255.0 + 0.5) as u32;
    let pr = (r.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
    let pg = (g.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
    let pb = (b.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
    *slot = crate::pixel::pack_premultiplied_rgba(pr, pg, pb, pa);
  }
  lut
}

/// Gradient parametrization. The shape geometry (`sx/sy/…`) lives in LOCAL
/// (shape) space and each device pixel is pulled back through `inv`, the
/// inverse CTM — this is what makes a radial gradient an ellipse under
/// non-uniform scale/shear, matching rlottie (vdrawhelper.cpp setupMatrix:
/// spans are inverse-transformed before the radial distance is measured).
#[derive(Clone)]
pub(crate) struct GradientMap {
  pub(crate) inv: Mat2x3,
  pub(crate) kind: GradientMapKind,
}

#[derive(Clone)]
pub(crate) enum GradientMapKind {
  Linear {
    sx: f32,
    sy: f32,
    dx: f32,
    dy: f32,
    inv_len_sq: f32,
  },
  Radial {
    sx: f32,
    sy: f32,
    inv_r: f32,
  },
  /// Two-point (focal) radial, rlottie/Qt semantics: focal circle
  /// (fx,fy,r=0) → outer circle (C,r). `dx/dy = C−F`, `a = r² − |C−F|²`.
  Focal {
    fx: f32,
    fy: f32,
    dx: f32,
    dy: f32,
    a: f32,
    r: f32,
  },
}

impl GradientMap {
  /// Exact-bits content hash (for the source-plane cache key).
  pub(crate) fn content_hash(&self) -> u64 {
    let mut h = Hasher128::new();
    for v in [self.inv.a, self.inv.b, self.inv.c, self.inv.d, self.inv.tx, self.inv.ty] {
      h.mix(v.to_bits());
    }
    match &self.kind {
      GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
        for v in [*sx, *sy, *dx, *dy, *inv_len_sq] {
          h.mix(v.to_bits());
        }
        h.mix(1);
      }
      GradientMapKind::Radial { sx, sy, inv_r } => {
        for v in [*sx, *sy, *inv_r] {
          h.mix(v.to_bits());
        }
        h.mix(2);
      }
      GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
        for v in [*fx, *fy, *dx, *dy, *a, *r] {
          h.mix(v.to_bits());
        }
        h.mix(3);
      }
    }
    h.finish() as u64
  }
}

/// Builds the radial/focal gradient map from LOCAL-space points; `inv` is
/// the inverse CTM used to pull device pixels back into that space.
pub(crate) fn radial_map(start: Vec2, end: Vec2, inv: Mat2x3, hl_len_pct: f32, hl_angle_deg: f32) -> GradientMap {
  let dx = end.x - start.x;
  let dy = end.y - start.y;
  // rlottie measures the radius with its approximate metric
  // (radial.cradius = VLine::length(start, end)).
  let r = crate::geometry::seg_len_rlottie(dx, dy);
  if hl_len_pct.abs() < 0.01 || r < 1e-6 {
    return GradientMap {
      inv,
      kind: GradientMapKind::Radial {
        sx: start.x,
        sy: start.y,
        inv_r: if r > 1e-6 { 1.0 / r } else { 0.0 },
      },
    };
  }
  // Focal point (rlottie lottiemodel.cpp): at highlight-length % of the
  // radius from the center, along the start→end direction rotated by the
  // highlight angle. Only exactly-100% is nudged to 99%.
  let mut progress = hl_len_pct / 100.0;
  if (progress - 1.0).abs() < 1e-4 {
    progress = 0.99;
  }
  let base_angle = dy.atan2(dx) + hl_angle_deg.to_radians();
  let fx = start.x + r * progress * base_angle.cos();
  let fy = start.y + r * progress * base_angle.sin();
  let cdx = start.x - fx;
  let cdy = start.y - fy;
  let a = r * r - (cdx * cdx + cdy * cdy);
  GradientMap {
    inv,
    kind: GradientMapKind::Focal { fx, fy, dx: cdx, dy: cdy, a, r },
  }
}

impl Canvas<'_> {
  #[cfg(test)]
  pub(crate) fn fill_gradient<const TRACK_ROWS: bool>(
    &mut self,
    cache: &mut CovCache,
    key: u128,
    src_key: u128,
    contours: &[Contour],
    rule: FillRule,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
  ) {
    self.fill_gradient_translated::<TRACK_ROWS>(cache, key, src_key, contours, crate::renderer::frame::Point::default(), rule, lut, map);
  }

  pub(crate) fn fill_gradient_translated<const TRACK_ROWS: bool>(
    &mut self,
    cache: &mut CovCache,
    key: u128,
    src_key: u128,
    contours: &[Contour],
    translation: crate::renderer::frame::Point,
    rule: FillRule,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
  ) {
    let w = self.w;
    let antialias = self.antialias;
    let dst_clear = self.dirty.is_empty();
    // Fastest path: the paint's premultiplied coverage-scaled SOURCE
    // pixels are cached (geometry + LUT + map all repeat) — replay as a
    // pure composite; bit-exact vs blend_gradient_px (same formula).
    if let Some(e) = cache.get(src_key) {
      let PlaneData::Src(data) = &e.data else {
        return;
      };
      let mut off = 0usize;
      for &(y, x0, len) in &e.rows {
        let (y, x0, len) = (y as usize, x0 as usize, len as usize);
        let lo = y.saturating_mul(w).saturating_add(x0);
        let hi = lo.saturating_add(len);
        let (Some(dst_row), Some(src_row)) = (self.pixels.get_mut(lo..hi), data.get(off..off + len)) else {
          break;
        };
        self.dirty.mark_row(y, x0, x0 + len);
        if TRACK_ROWS {
          mark_row_bounds(&mut self.dirty_rows, y, x0, x0 + len);
        }
        if dst_clear {
          dst_row.copy_from_slice(src_row);
        } else {
          crate::simd::composite_over_span(dst_row, src_row, 255);
        }
        off += len;
      }
      return;
    }
    let radial_opaque = matches!(&map.kind, GradientMapKind::Radial { .. }) && lut.iter().all(|&pixel| pixel >> 24 == 255);
    let mut src_entry = CovEntry {
      rows: Vec::new(),
      data: PlaneData::Src(Vec::new()),
    };
    let mut had_cov_hit = false;
    // Hoisted out of the `cache.get` borrow — `capture_enabled` must be
    // read before the mutable borrow extends through the cov entry.
    let capture_enabled = cache.capture_enabled();
    if let Some(e) = cache.get(key) {
      if let PlaneData::Spans(spans) = &e.data {
        // Mode-S coverage hit: replay spans through the identical
        // per-pixel gradient math via a synthesized uniform cov
        // row, and (size permitting) capture the source plane so
        // later frames replay as a pure composite — same
        // proven-to-repeat rule as the Cov arm (profiled:
        // gradient_srcs was the top cost at 720 without this).
        let px_total: usize = spans.iter().map(|&s| unpack_span(s).2).sum();
        // Only capture the source plane when it will actually be
        // admitted AND the cache is still learning — when frozen
        // `insert` is a no-op, so paying the two-pass
        // gradient_row_capture (src plane materialized to DRAM) for
        // every cov hit throws the bytes away. The fused gradient_row
        // is bit-identical and round-trip-free.
        let capture = capture_enabled && px_total * 4 + spans.len() * 12 + 64 <= SRC_ENTRY_MAX;
        for &s in spans {
          let (y, x0, len, cov) = unpack_span(s);
          let lo = y.saturating_mul(w).saturating_add(x0);
          let Some(dst_row) = self.pixels.get_mut(lo..lo.saturating_add(len)) else {
            break;
          };
          self.dirty.mark_row(y, x0, x0 + len);
          if TRACK_ROWS {
            mark_row_bounds(&mut self.dirty_rows, y, x0, x0 + len);
          }
          if capture {
            // Source-plane capture needs actual per-pixel bytes.
            if cov == 255 && self.row_ones.len() < len {
              self.row_ones.resize(len, 255);
            }
            if cov != 255 {
              self.row_cov.clear();
              self.row_cov.resize(len, cov);
            }
            let cr: &[u8] = if cov == 255 { self.row_ones.get(..len).unwrap_or(&[]) } else { &self.row_cov };
            src_entry.rows.push((y as u32, x0 as u32, len as u32));
            if let PlaneData::Src(sd) = &mut src_entry.data {
              if dst_clear {
                gradient_row_capture_clear(dst_row, cr, y, x0, lut, map, sd);
              } else {
                gradient_row_capture(dst_row, cr, y, x0, lut, map, sd);
              }
            }
          } else if dst_clear {
            gradient_span_uniform_clear(dst_row, cov, y, x0, lut, map);
          } else {
            gradient_span_uniform(dst_row, cov, y, x0, lut, map, radial_opaque);
          }
        }
        if !capture {
          return;
        }
        had_cov_hit = true;
      } else {
        let PlaneData::Cov(data) = &e.data else {
          return;
        };
        had_cov_hit = true;
        // Only capture the source plane if it will actually be admitted:
        // 4 bytes/pixel, size known exactly from the coverage entry.
        // Oversized planes (720px gradients) previously re-captured every
        // frame just to be rejected by insert — measured at 49% of the
        // worst effects file.
        let capture = capture_enabled && data.len() * 4 + e.rows.len() * 12 + 64 <= SRC_ENTRY_MAX;
        let mut off = 0usize;
        for &(y, x0, len) in &e.rows {
          let (y, x0, len) = (y as usize, x0 as usize, len as usize);
          let lo = y.saturating_mul(w).saturating_add(x0);
          let hi = lo.saturating_add(len);
          let (Some(dst_row), Some(cov_row)) = (self.pixels.get_mut(lo..hi), data.get(off..off + len)) else {
            break;
          };
          self.dirty.mark_row(y, x0, x0 + len);
          if TRACK_ROWS {
            mark_row_bounds(&mut self.dirty_rows, y, x0, x0 + len);
          }
          if capture {
            src_entry.rows.push((y as u32, x0 as u32, len as u32));
            if let PlaneData::Src(sd) = &mut src_entry.data {
              if dst_clear {
                gradient_row_capture_clear(dst_row, cov_row, y, x0, lut, map, sd);
              } else {
                gradient_row_capture(dst_row, cov_row, y, x0, lut, map, sd);
              }
            }
          } else if dst_clear {
            gradient_srcs(dst_row, cov_row, y, x0, lut, map);
          } else {
            gradient_row(dst_row, cov_row, y, x0, lut, map);
          }
          off += len;
        }
        if !capture {
          return;
        }
      }
    }
    if had_cov_hit {
      cache.insert(src_key, src_entry);
      return;
    }
    if contours.is_empty() {
      return;
    }
    let pixels = &mut *self.pixels;
    let dirty = &mut self.dirty;
    let dirty_rows = &mut self.dirty_rows;
    if mode_s_wins(contours, w * self.h) {
      // Mode S: spans feed the same gradient_row math through a
      // synthesized uniform cov row.
      drop(src_entry);
      let cells = self.cells.as_mut().expect("fresh gradient requires cell rasterizer");
      cells.reset();
      cells.fill_contours_translated(contours, translation.x, translation.y);
      let capture = cache.capture_enabled();
      let mut spans: Vec<u64> = core::mem::take(&mut self.span_buf);
      spans.clear();
      let mut px_total = 0usize;
      let mut overflow = false;
      cells.sweep_spans(rule, antialias, |y, x0, len, cov| {
        let lo = y.saturating_mul(w).saturating_add(x0);
        let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
          return;
        };
        dirty.mark_row(y, x0, x0 + len);
        if TRACK_ROWS {
          mark_row_bounds(dirty_rows, y, x0, x0 + len);
        }
        if !capture {
        } else if spans.len() < SPAN_CAPTURE_MAX {
          spans.push(pack_span(y, x0, len, cov));
          px_total += len;
        } else {
          overflow = true;
        }
        if dst_clear {
          gradient_span_uniform_clear(dst_row, cov, y, x0, lut, map);
        } else {
          gradient_span_uniform(dst_row, cov, y, x0, lut, map, radial_opaque);
        }
      });
      if overflow || !capture {
        self.span_buf = spans; // recycle
        return; // uncacheable either way; skip entry construction
      }
      let entry = if spans_fragmented(&spans, px_total) {
        let e = spans_to_cov_entry(&spans);
        self.span_buf = spans; // recycle
        e
      } else {
        CovEntry {
          rows: Vec::new(),
          data: PlaneData::Spans(spans),
        }
      };
      cache.insert(key, entry);
      return;
    }
    let raster = self.raster.as_mut().expect("fresh gradient requires dense rasterizer");
    raster.reset();
    raster.fill_contours_translated(contours, translation.x, translation.y);
    let capture = cache.capture_enabled();
    let mut entry = if capture && w.saturating_mul(self.h) <= 160 * 160 {
      let (row_cap, data_cap) = raster.capture_capacities();
      CovEntry {
        rows: Vec::with_capacity(row_cap),
        data: PlaneData::Cov(Vec::with_capacity(data_cap)),
      }
    } else {
      CovEntry::default()
    };
    // Fresh rasterization: do NOT capture the source plane here — most
    // fresh geometry is animated and never repeats, and the capture
    // (resize + per-pixel src writes) measured 13.6% of gradient-heavy
    // 320px frames. The src plane is captured on the first COV hit
    // instead, i.e. only for geometry proven to repeat.
    drop(src_entry);
    raster.sweep(rule, antialias, |y, x0, cov_row| {
      let lo = y.saturating_mul(w).saturating_add(x0);
      let hi = lo.saturating_add(cov_row.len());
      let Some(dst_row) = pixels.get_mut(lo..hi) else {
        return;
      };
      dirty.mark_row(y, x0, x0 + cov_row.len());
      if TRACK_ROWS {
        mark_row_bounds(dirty_rows, y, x0, x0 + cov_row.len());
      }
      if capture {
        entry.rows.push((y as u32, x0 as u32, cov_row.len() as u32));
        if let PlaneData::Cov(d) = &mut entry.data {
          d.extend_from_slice(cov_row);
        }
      }
      if dst_clear {
        gradient_srcs(dst_row, cov_row, y, x0, lut, map);
      } else {
        gradient_row(dst_row, cov_row, y, x0, lut, map);
      }
    });
    if capture {
      cache.insert(key, entry);
    }
  }

  pub(crate) fn fill_gradient_translated_alpha<const TRACK_ROWS: bool>(
    &mut self,
    cache: &mut CovCache,
    key: u128,
    src_key: u128,
    contours: &[Contour],
    translation: crate::renderer::frame::Point,
    rule: FillRule,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
    alpha: u8,
  ) {
    let k = u32::from(alpha);
    if k == 255 {
      self.fill_gradient_translated::<TRACK_ROWS>(cache, key, src_key, contours, translation, rule, lut, map);
      return;
    }
    let w = self.w;
    let antialias = self.antialias;
    // Fastest: the frozen source plane is already cached - replay as a pure
    // composite over the covered pixels.
    if let Some(e) = cache.get(src_key) {
      let PlaneData::Src(data) = &e.data else {
        return;
      };
      let mut off = 0usize;
      for &(y, x0, len) in &e.rows {
        let (y, x0, len) = (y as usize, x0 as usize, len as usize);
        let lo = y.saturating_mul(w).saturating_add(x0);
        let hi = lo.saturating_add(len);
        let (Some(dst_row), Some(src_row)) = (self.pixels.get_mut(lo..hi), data.get(off..off + len)) else {
          break;
        };
        self.dirty.mark_row(y, x0, x0 + len);
        if TRACK_ROWS {
          mark_row_bounds(&mut self.dirty_rows, y, x0, x0 + len);
        }
        crate::simd::composite_over_span(dst_row, src_row, k);
        off += len;
      }
      return;
    }
    let capture_enabled = cache.capture_enabled();
    let mut row_scratch: Vec<u32> = Vec::new();
    let mut src_rows: Vec<(u32, u32, u32)> = Vec::new();
    let mut src_data: Vec<u32> = Vec::new();
    let pixels = &mut *self.pixels;
    let dirty = &mut self.dirty;
    let dirty_rows = &mut self.dirty_rows;
    // Synthesize the source for coverage row `cr` and composite with `k`. When
    // `capture`, append the source to the source plane for later pure-composite
    // replay; otherwise reuse `row_scratch`.
    let mut emit_row = |dst_row: &mut [u32], cr: &[u8], y: usize, x0: usize, capture: bool| {
      if capture {
        let base = src_data.len();
        src_data.resize(base + cr.len(), 0);
        gradient_srcs(&mut src_data[base..base + cr.len()], cr, y, x0, lut, map);
        dirty.mark_row(y, x0, x0 + cr.len());
        if TRACK_ROWS {
          mark_row_bounds(dirty_rows, y, x0, x0 + cr.len());
        }
        crate::simd::composite_over_span(dst_row, &src_data[base..base + cr.len()], k);
      } else {
        row_scratch.resize(cr.len(), 0);
        gradient_srcs(&mut row_scratch[..cr.len()], cr, y, x0, lut, map);
        dirty.mark_row(y, x0, x0 + cr.len());
        if TRACK_ROWS {
          mark_row_bounds(dirty_rows, y, x0, x0 + cr.len());
        }
        crate::simd::composite_over_span(dst_row, &row_scratch[..cr.len()], k);
      }
    };
    // Coverage hit: reuse cached coverage, capture the source plane when
    // admissible so later frames replay as a pure composite.
    if let Some(e) = cache.get(key) {
      let mut had = false;
      match &e.data {
        PlaneData::Cov(data) => {
          let caps = capture_enabled && data.len() * 4 + e.rows.len() * 12 + 64 <= SRC_ENTRY_MAX;
          let mut off = 0usize;
          for &(y, x0, len) in &e.rows {
            let (y, x0, len) = (y as usize, x0 as usize, len as usize);
            let lo = y.saturating_mul(w).saturating_add(x0);
            let hi = lo.saturating_add(len);
            let (Some(dst_row), Some(cov_row)) = (pixels.get_mut(lo..hi), data.get(off..off + len)) else {
              break;
            };
            had = true;
            if caps {
              src_rows.push((y as u32, x0 as u32, len as u32));
              emit_row(dst_row, cov_row, y, x0, true);
            } else {
              emit_row(dst_row, cov_row, y, x0, false);
            }
            off += len;
          }
        }
        PlaneData::Spans(spans) => {
          let px_total: usize = spans.iter().map(|&s| unpack_span(s).2).sum();
          let caps = capture_enabled && px_total * 4 + spans.len() * 12 + 64 <= SRC_ENTRY_MAX;
          for &s in spans {
            let (y, x0, len, cov) = unpack_span(s);
            let lo = y.saturating_mul(w).saturating_add(x0);
            let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
              break;
            };
            had = true;
            let cr: &[u8] = if cov == 255 && self.row_ones.len() < len {
              self.row_ones.resize(len, 255);
              self.row_ones.get(..len).unwrap_or(&[])
            } else if cov != 255 {
              self.row_cov.clear();
              self.row_cov.resize(len, cov);
              &self.row_cov
            } else {
              &self.row_ones[..len.min(self.row_ones.len())]
            };
            if caps {
              src_rows.push((y as u32, x0 as u32, len as u32));
              emit_row(dst_row, cr, y, x0, true);
            } else {
              emit_row(dst_row, cr, y, x0, false);
            }
          }
        }
        PlaneData::Src(_) => {}
      }
      if had && !src_rows.is_empty() {
        cache.insert(src_key, CovEntry {
          rows: src_rows,
          data: PlaneData::Src(src_data),
        });
      }
      return;
    }
    if contours.is_empty() {
      return;
    }
    let capture = capture_enabled;
    if mode_s_wins(contours, w * self.h) {
      // Mode S: spans feed the same source math through a synthesized row.
      let cells = self.cells.as_mut().expect("fresh gradient requires cell rasterizer");
      cells.reset();
      cells.fill_contours_translated(contours, translation.x, translation.y);
      let mut spans: Vec<u64> = core::mem::take(&mut self.span_buf);
      spans.clear();
      let mut px_total = 0usize;
      let mut overflow = false;
      cells.sweep_spans(rule, antialias, |y, x0, len, cov| {
        let lo = y.saturating_mul(w).saturating_add(x0);
        let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
          return;
        };
        if capture {
          if spans.len() < SPAN_CAPTURE_MAX {
            spans.push(pack_span(y, x0, len, cov));
            px_total += len;
          } else {
            overflow = true;
          }
        }
        let cr: &[u8] = if cov == 255 {
          if self.row_ones.len() < len {
            self.row_ones.resize(len, 255);
          }
          &self.row_ones[..len.min(self.row_ones.len())]
        } else {
          self.row_cov.clear();
          self.row_cov.resize(len, cov);
          &self.row_cov
        };
        emit_row(dst_row, cr, y, x0, false);
      });
      if capture && !overflow && px_total * 4 + spans.len() * 12 + 64 <= COV_ENTRY_MAX {
        let entry = if spans_fragmented(&spans, px_total) {
          let e = spans_to_cov_entry(&spans);
          self.span_buf = spans; // recycle
          e
        } else {
          CovEntry {
            rows: Vec::new(),
            data: PlaneData::Spans(spans),
          }
        };
        cache.insert(key, entry);
      } else {
        self.span_buf = spans; // recycle
      }
      return;
    }
    let raster = self.raster.as_mut().expect("fresh gradient requires dense rasterizer");
    raster.reset();
    raster.fill_contours_translated(contours, translation.x, translation.y);
    let mut entry = if capture && w.saturating_mul(self.h) <= 160 * 160 {
      let (row_cap, data_cap) = raster.capture_capacities();
      CovEntry {
        rows: Vec::with_capacity(row_cap),
        data: PlaneData::Cov(Vec::with_capacity(data_cap)),
      }
    } else {
      CovEntry::default()
    };
    // Fresh rasterization captures only the COVERAGE; the source plane is
    // captured on the first cov hit later (same policy as the legacy path).
    raster.sweep(rule, antialias, |y, x0, cov_row| {
      let lo = y.saturating_mul(w).saturating_add(x0);
      let hi = lo.saturating_add(cov_row.len());
      let Some(dst_row) = pixels.get_mut(lo..hi) else {
        return;
      };
      if capture {
        entry.rows.push((y as u32, x0 as u32, cov_row.len() as u32));
        if let PlaneData::Cov(d) = &mut entry.data {
          d.extend_from_slice(cov_row);
        }
      }
      emit_row(dst_row, cov_row, y, x0, false);
    });
    if capture {
      cache.insert(key, entry);
    }
  }

}

/// gradient_row variant that also CAPTURES the premultiplied,
/// coverage-scaled source pixel per position (0 where the paint leaves the
/// destination untouched). Blending src over dst here uses the identical
/// integer formula as blend_gradient_px, so captured-replay via
/// composite_over_span(k=255) is bit-exact.
fn gradient_row_capture(dst_row: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap, out: &mut Vec<u32>) {
  let base = out.len();
  out.resize(base + dst_row.len(), 0);
  let Some(srcs) = out.get_mut(base..) else {
    return;
  };
  gradient_srcs(srcs, cov_row, y, x0, lut, map);
  crate::simd::composite_over_span(dst_row, srcs, 255);
}

/// Clear-destination form of [`gradient_row_capture`]: source-over onto
/// transparent pixels is exactly the coverage-scaled premultiplied source.
fn gradient_row_capture_clear(dst_row: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap, out: &mut Vec<u32>) {
  let base = out.len();
  out.resize(base + dst_row.len(), 0);
  let Some(srcs) = out.get_mut(base..) else {
    return;
  };
  gradient_srcs(srcs, cov_row, y, x0, lut, map);
  dst_row.copy_from_slice(srcs);
}

/// Computes the premultiplied coverage-scaled source pixels of one gradient
/// row into `srcs` (same per-pixel math as gradient_row's t/LUT sampling).
fn gradient_srcs(srcs: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap) {
  let inv = map.inv;
  let yf = y as f32 + 0.5;
  // Row origin: local coords at device column 0 (pixel-center x = 0.5).
  // Every position below is anchored to the ABSOLUTE device column X and
  // computed as one rounded `base + X·step`, so a pixel's bits do not
  // depend on the span/row/sub-run x0 it is reached through
  // (segmentation-invariant — the strong byte-exact-cache invariant).
  let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
  let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
  match &map.kind {
    GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
      let n = srcs.len().min(cov_row.len());
      let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
      let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
      // Run-batched like the radial/focal arms; `row_base + X·dt` form
      // at absolute column X (corpus-gated association change).
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = srcs.get_mut(i..i + run) {
            crate::simd::linear_lut_fill(out, lut, row_base, dt, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        let t = row_base + (x0 + i) as f32 * dt;
        if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
          if cov == 255 {
            *s = lut_sample(lut, t); // src_px(255, c) == c exactly
          } else if cov != 0 {
            *s = src_px(cov, lut_sample(lut, t));
          } else {
            // Explicit zero: lets gradient_row keep its scratch
            // buffer dirty across rows (no per-row memset).
            *s = 0;
          }
        }
        i += 1;
      }
    }
    GradientMapKind::Radial { sx, sy, inv_r } => {
      // Full-coverage runs (interiors; every mode-S span) go through
      // the 4-lane kernel — the post-span profile had this loop as
      // the single largest 720px cost (gradient_srcs 74 samples).
      // `dd` positions are computed as `dd0 + X·step` at absolute
      // column X in BOTH paths (not sequential adds): sub-ULP
      // different from the historical loop, corpus-gated like round 2.
      let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
      let n = srcs.len().min(cov_row.len());
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = srcs.get_mut(i..i + run) {
            crate::simd::radial_lut_fill(out, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        let xf = (x0 + i) as f32;
        let ddx = dd0x + xf * inv.a;
        let ddy = dd0y + xf * inv.b;
        if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
          if cov == 255 {
            let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
            *s = lut_sample(lut, t);
          } else if cov != 0 {
            let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
            *s = src_px(cov, lut_sample(lut, t));
          } else {
            *s = 0; // see Linear arm: scratch stays dirty
          }
        }
        i += 1;
      }
    }
    GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
      if a.abs() < 1e-9 {
        srcs.fill(0); // callers assume every pixel was stored
        return;
      }
      let inv2a = 1.0 / (2.0 * a);
      // Same run-batched structure as the Radial arm; positions in
      // `g0 + X·step` form at absolute column X (corpus-gated).
      let (g0x, g0y) = (lx0 - fx, ly0 - fy);
      let n = srcs.len().min(cov_row.len());
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = srcs.get_mut(i..i + run) {
            crate::simd::focal_lut_fill(out, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        let xf = (x0 + i) as f32;
        let gx = g0x + xf * inv.a;
        let gy = g0y + xf * inv.b;
        if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
          // Every pixel gets a store (transparent cases write 0)
          // so gradient_row's scratch stays dirty across rows.
          let mut v = 0u32;
          if cov != 0 {
            let b = 2.0 * (gx * dx + gy * dy);
            let gg = gx * gx + gy * gy;
            let det = b * b + 4.0 * a * gg;
            if det >= 0.0 {
              let sq = det.sqrt();
              let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
              if r * sroot >= 0.0 {
                let c = lut_sample(lut, sroot);
                v = if cov == 255 { c } else { src_px(cov, c) };
              }
            }
          }
          *s = v;
        }
        i += 1;
      }
    }
  }
}

/// Coverage-scales a premultiplied LUT color: the s_* terms of
/// blend_gradient_px, packed.
#[inline(always)]
fn src_px(cov: u8, src: u32) -> u32 {
  let s_a0 = (src >> 24) & 0xff;
  if s_a0 == 0 {
    return 0;
  }
  let covu = u32::from(cov);
  let s_a = (s_a0 * covu + 127) / 255;
  let s_r = (((src >> 16) & 0xff) * covu + 127) / 255;
  let s_g = (((src >> 8) & 0xff) * covu + 127) / 255;
  let s_b = ((src & 0xff) * covu + 127) / 255;
  (s_a << 24) | (s_r << 16) | (s_g << 8) | s_b
}

/// Blends one coverage row of a gradient paint into `dst_row` (row `y`,
/// starting column `x0`) in a SINGLE fused pass: each source pixel's
/// coverage-scaled premultiplied color is computed (t/LUT stepping identical
/// to gradient_srcs) and immediately source-overed into `dst_row`, with NO
/// materialized src plane. Bit-for-bit identical to the historical
/// gradient_srcs + composite_over_span(k=255) two-pass — the same per-pixel
/// operations, only the intermediate DRAM buffer is elided (the measured
/// A76 win: gradient_srcs writes the plane, composite reads it back, and on
/// the stacked-gradient packs that round-trip is DRAM-bandwidth-bound).
fn gradient_row(dst_row: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap) {
  let n = dst_row.len().min(cov_row.len());
  // Short rows: the tight per-pixel fused loop (blend_gradient_px ==
  // src_px + blend_premult_px) — no run scan, no runs long enough to reach
  // the batched kernels anyway.
  if n < 32 {
    let (Some(d), Some(c)) = (dst_row.get_mut(..n), cov_row.get(..n)) else {
      return;
    };
    gradient_row_scalar(d, c, y, x0, lut, map);
    return;
  }
  let (Some(d), Some(c)) = (dst_row.get_mut(..n), cov_row.get(..n)) else {
    return;
  };
  gradient_over(d, c, y, x0, lut, map);
}

/// Blends a mode-S uniform-coverage gradient span directly. This is the
/// same math as `gradient_row` over a synthetic constant coverage row, but
/// full-coverage spans can jump straight to the fused LUT-over kernels and
/// partial spans avoid allocating/filling a temporary coverage slice.
fn gradient_span_uniform(dst_row: &mut [u32], cov: u8, y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap, radial_opaque: bool) {
  if cov == 0 || dst_row.is_empty() {
    return;
  }
  let inv = map.inv;
  let yf = y as f32 + 0.5;
  let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
  let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
  match &map.kind {
    GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
      let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
      let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
      if cov == 255 {
        crate::simd::linear_lut_over(dst_row, lut, row_base, dt, x0 as f32);
        return;
      }
      for (i, dst) in dst_row.iter_mut().enumerate() {
        let t = row_base + (x0 + i) as f32 * dt;
        blend_gradient_px(dst, cov, lut_sample(lut, t));
      }
    }
    GradientMapKind::Radial { sx, sy, inv_r } => {
      let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
      if cov == 255 {
        if radial_opaque {
          crate::simd::radial_lut_fill(dst_row, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, x0 as f32);
        } else {
          crate::simd::radial_lut_over(dst_row, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, x0 as f32);
        }
        return;
      }
      for (i, dst) in dst_row.iter_mut().enumerate() {
        let xf = (x0 + i) as f32;
        let ddx = dd0x + xf * inv.a;
        let ddy = dd0y + xf * inv.b;
        let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
        blend_gradient_px(dst, cov, lut_sample(lut, t));
      }
    }
    GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
      if a.abs() < 1e-9 {
        return;
      }
      let inv2a = 1.0 / (2.0 * a);
      let (g0x, g0y) = (lx0 - fx, ly0 - fy);
      if cov == 255 {
        crate::simd::focal_lut_over(dst_row, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, x0 as f32);
        return;
      }
      for (i, dst) in dst_row.iter_mut().enumerate() {
        let xf = (x0 + i) as f32;
        let gx = g0x + xf * inv.a;
        let gy = g0y + xf * inv.b;
        let b = 2.0 * (gx * dx + gy * dy);
        let gg = gx * gx + gy * gy;
        let det = b * b + 4.0 * a * gg;
        if det >= 0.0 {
          let sq = det.sqrt();
          let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
          if r * sroot >= 0.0 {
            blend_gradient_px(dst, cov, lut_sample(lut, sroot));
          }
        }
      }
    }
  }
}

/// Clear-destination form of [`gradient_span_uniform`].
fn gradient_span_uniform_clear(dst_row: &mut [u32], cov: u8, y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap) {
  if cov == 0 || dst_row.is_empty() {
    return;
  }
  if cov == 255 {
    match &map.kind {
      GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
        let inv = map.inv;
        let yf = y as f32 + 0.5;
        let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
        let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
        let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
        let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
        crate::simd::linear_lut_fill(dst_row, lut, row_base, dt, x0 as f32);
      }
      GradientMapKind::Radial { sx, sy, inv_r } => {
        let inv = map.inv;
        let yf = y as f32 + 0.5;
        let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
        let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
        let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
        crate::simd::radial_lut_fill(dst_row, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, x0 as f32);
      }
      GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
        if a.abs() < 1e-9 {
          return;
        }
        let inv = map.inv;
        let yf = y as f32 + 0.5;
        let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
        let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
        let inv2a = 1.0 / (2.0 * a);
        let (g0x, g0y) = (lx0 - fx, ly0 - fy);
        crate::simd::focal_lut_fill(dst_row, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, x0 as f32);
      }
    }
    return;
  }

  if dst_row.len() > 1024 {
    gradient_span_uniform(dst_row, cov, y, x0, lut, map, false);
    return;
  }
  let mut cov_row = [0u8; 1024];
  if let Some(row) = cov_row.get_mut(..dst_row.len()) {
    row.fill(cov);
    gradient_srcs(dst_row, row, y, x0, lut, map);
  }
}

/// Fused generate+blend of one gradient coverage row — the single-pass form
/// of `gradient_srcs` + `composite_over_span(k=255)`. Full-coverage runs
/// (>=16px, the bulk of gradient pixels) go through the fused `*_lut_over`
/// NEON kernels; partial-coverage and short-run pixels blend scalar via
/// `blend_gradient_px`. The per-pixel LUT math is byte-identical to
/// gradient_srcs (same segmentation-invariant `base + X·step` form), and the
/// blend is the identical integer source-over composite_over_span uses, so
/// the output matches the two-pass path bit-for-bit.
fn gradient_over(dst_row: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap) {
  let inv = map.inv;
  let yf = y as f32 + 0.5;
  // Row origin: local coords at device column 0 (pixel-center x = 0.5) —
  // identical anchoring to gradient_srcs/gradient_row_scalar.
  let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
  let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
  match &map.kind {
    GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
      let n = dst_row.len().min(cov_row.len());
      let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
      let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = dst_row.get_mut(i..i + run) {
            crate::simd::linear_lut_over(out, lut, row_base, dt, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
          if cov != 0 {
            let t = row_base + (x0 + i) as f32 * dt;
            blend_gradient_px(d, cov, lut_sample(lut, t));
          }
        }
        i += 1;
      }
    }
    GradientMapKind::Radial { sx, sy, inv_r } => {
      let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
      let n = dst_row.len().min(cov_row.len());
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = dst_row.get_mut(i..i + run) {
            crate::simd::radial_lut_over(out, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
          if cov != 0 {
            let xf = (x0 + i) as f32;
            let ddx = dd0x + xf * inv.a;
            let ddy = dd0y + xf * inv.b;
            let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
            blend_gradient_px(d, cov, lut_sample(lut, t));
          }
        }
        i += 1;
      }
    }
    GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
      if a.abs() < 1e-9 {
        return; // gradient_srcs fills 0 (all transparent) → dst untouched
      }
      let inv2a = 1.0 / (2.0 * a);
      let (g0x, g0y) = (lx0 - fx, ly0 - fy);
      let n = dst_row.len().min(cov_row.len());
      let mut i = 0usize;
      while i < n {
        let run = cov_row.get(i..).map(|c| c.iter().take_while(|&&v| v == 255).count()).unwrap_or(0);
        if run >= 16 {
          if let Some(out) = dst_row.get_mut(i..i + run) {
            crate::simd::focal_lut_over(out, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, (x0 + i) as f32);
          }
          i += run;
          continue;
        }
        if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
          if cov != 0 {
            let xf = (x0 + i) as f32;
            let gx = g0x + xf * inv.a;
            let gy = g0y + xf * inv.b;
            let b = 2.0 * (gx * dx + gy * dy);
            let gg = gx * gx + gy * gy;
            let det = b * b + 4.0 * a * gg;
            if det >= 0.0 {
              let sq = det.sqrt();
              let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
              if r * sroot >= 0.0 {
                blend_gradient_px(d, cov, lut_sample(lut, sroot));
              }
            }
          }
        }
        i += 1;
      }
    }
  }
}

/// Fused scalar row blend (short rows; also the reference form for
/// gradient_srcs' batched kernels).
fn gradient_row_scalar(dst_row: &mut [u32], cov_row: &[u8], y: usize, x0: usize, lut: &[u32; GRADIENT_LUT_SIZE], map: &GradientMap) {
  let inv = map.inv;
  let yf = y as f32 + 0.5;
  // Row origin at device column 0 (pixel-center x = 0.5). Positions are
  // anchored to the ABSOLUTE device column X = x0 + i and computed as one
  // rounded `base + X·step` per pixel — identical to gradient_srcs, so a
  // pixel's bits are the same via this short-row path, the batched
  // kernels, or any span/row segmentation (segmentation-invariant).
  let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
  let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
  match &map.kind {
    GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
      let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
      let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
      for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
        if cov != 0 {
          let t = row_base + (x0 + i) as f32 * dt;
          blend_gradient_px(dst, cov, lut_sample(lut, t));
        }
      }
    }
    GradientMapKind::Radial { sx, sy, inv_r } => {
      let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
      for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
        if cov != 0 {
          let xf = (x0 + i) as f32;
          let ddx = dd0x + xf * inv.a;
          let ddy = dd0y + xf * inv.b;
          let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
          blend_gradient_px(dst, cov, lut_sample(lut, t));
        }
      }
    }
    GradientMapKind::Focal { fx, fy, dx, dy, a, r } => {
      if a.abs() < 1e-9 {
        return; // rlottie: vIsZero(a) → transparent
      }
      let inv2a = 1.0 / (2.0 * a);
      let (g0x, g0y) = (lx0 - fx, ly0 - fy);
      for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
        if cov != 0 {
          // rlottie fetch_radial_gradient: solve
          // a·s² + b·s − |g|² = 0, take the LARGER root; no
          // real solution / behind the focal cone → skip
          // (transparent).
          let xf = (x0 + i) as f32;
          let gx = g0x + xf * inv.a;
          let gy = g0y + xf * inv.b;
          let b = 2.0 * (gx * dx + gy * dy);
          let gg = gx * gx + gy * gy;
          let det = b * b + 4.0 * a * gg;
          if det >= 0.0 {
            let sq = det.sqrt();
            let s = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
            if r * s >= 0.0 {
              blend_gradient_px(dst, cov, lut_sample(lut, s));
            }
          }
        }
      }
    }
  }
}

/// Clamped LUT sample at gradient position `t`. Non-finite `t` (degenerate
/// transform, focal edge cases) yields transparent, like the old NaN skip.
#[inline(always)]
fn lut_sample(lut: &[u32; GRADIENT_LUT_SIZE], t: f32) -> u32 {
  if !t.is_finite() {
    return 0;
  }
  let t = t.clamp(0.0, 1.0);
  lut.get((t * (GRADIENT_LUT_SIZE - 1) as f32 + 0.5) as usize).copied().unwrap_or(0)
}

/// Coverage-modulated premultiplied source-over of one gradient pixel.
#[inline(always)]
fn blend_gradient_px(dst: &mut u32, cov: u8, src: u32) {
  let s_a0 = (src >> 24) & 0xff;
  if s_a0 == 0 {
    return;
  }
  let covu = u32::from(cov);
  let s_a = (s_a0 * covu + 127) / 255;
  let s_r = (((src >> 16) & 0xff) * covu + 127) / 255;
  let s_g = (((src >> 8) & 0xff) * covu + 127) / 255;
  let s_b = ((src & 0xff) * covu + 127) / 255;
  let d = *dst;
  let inv = 256 - s_a;
  let o_a = s_a + (((d >> 24) & 0xff) * inv >> 8);
  let o_r = s_r + (((d >> 16) & 0xff) * inv >> 8);
  let o_g = s_g + (((d >> 8) & 0xff) * inv >> 8);
  let o_b = s_b + ((d & 0xff) * inv >> 8);
  *dst = (o_a.min(255) << 24) | (o_r.min(255) << 16) | (o_g.min(255) << 8) | o_b.min(255);
}

#[cfg(test)]
#[path = "tests/gradient.rs"]
mod tests;
