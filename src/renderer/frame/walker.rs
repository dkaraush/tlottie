//! Unstable renderer-neutral evaluated-frame pipeline.
//!
//! Renderer calls borrow the walker's temporary geometry and are consumed
//! synchronously. Renderers decide independently what, if anything, to
//! retain or cache.

#![allow(missing_docs)]

use crate::error::{Error, Result};
use crate::geometry::{clip_contour, clip_to_quad, flatten_path, rect_contour, Contour};
use crate::limits::Limits;
use crate::math::{Color, Mat2x3, Vec2};
use crate::model::{shapes_have_multiple_visible_paints, Composition, FillRule, Layer, LayerKind};
use crate::renderer::cpu::executor::{layer_transform_at, opacity_byte, parent_chain_matrix, ClipQuad, DrawJob, GradientMapKind, PendingJob, RenderCtx, RenderScratch, ShapeWalker, MAX_PRECOMP_DEPTH};

use super::renderer::*;

/// Stateful evaluator that streams one frame into a renderer.
#[derive(Default)]
pub struct FrameWalker {
  scratch: RenderScratch,
}

impl FrameWalker {
  /// Evaluates `composition` at `frame_index` and synchronously streams its
  /// renderer-neutral drawing operations into `renderer`.
  pub fn render(&mut self, composition: &Composition, frame_index: f32, width: u32, height: u32, options: crate::RenderOptions, renderer: &mut impl FrameRenderer) -> Result<()> {
    walk_frame(composition, frame_index, width, height, options, self, renderer)
  }
}

/// Evaluates one frame and synchronously invokes `renderer` without
/// allocating an owned operation or contour list.
pub fn walk_frame_into(comp: &Composition, frame_index: f32, width: u32, height: u32, options: crate::RenderOptions, renderer: &mut impl FrameRenderer) -> Result<()> {
  FrameWalker::default().render(comp, frame_index, width, height, options, renderer)
}

fn walk_frame(comp: &Composition, frame_index: f32, width: u32, height: u32, options: crate::RenderOptions, walker: &mut FrameWalker, renderer: &mut impl FrameRenderer) -> Result<()> {
  let limits = Limits::default();
  if width == 0 || height == 0 || width > limits.max_dimension || height > limits.max_dimension {
    return Err(Error::InvalidLottie {
      offset: 0,
      what: "render size out of range",
    });
  }

  let max_frame = comp.frame_count().saturating_sub(1) as f32;
  let frame_in_range = if frame_index.is_finite() { frame_index.clamp(0.0, max_frame) } else { 0.0 };
  let frame = comp.in_point + frame_in_range;
  let base = Mat2x3::scale(width as f32 / comp.width.max(1) as f32, height as f32 / comp.height.max(1) as f32);

  let scratch = &mut walker.scratch;
  let ctx = RenderCtx {
    comp,
    continuous: frame_in_range.fract() != 0.0,
    #[cfg(test)]
    antialias: options.antialias,
    curve_tolerance: options.curve_tolerance,
  };
  ctx.collect_layers(scratch, width as usize, height as usize, options.antialias, &comp.layers, base, frame, 1.0, &Vec::new(), 0, renderer)
}

fn rule_of(rule: FillRule) -> Rule {
  match rule {
    FillRule::NonZero => Rule::NonZero,
    FillRule::EvenOdd => Rule::EvenOdd,
  }
}

fn premul_argb(color: Color, opacity: f32) -> u32 {
  let a = (color.a * opacity).clamp(0.0, 1.0);
  // Match Canvas::fill exactly: straight channels and paint alpha truncate
  // independently, then premultiplication uses rounded byte division.
  let ai = (a * 255.0) as u32;
  let premul = |channel: f32| {
    let straight = (channel.clamp(0.0, 1.0) * 255.0) as u32;
    (straight * ai + 127) / 255
  };
  let ri = premul(color.r);
  let gi = premul(color.g);
  let bi = premul(color.b);
  (ai.min(255) << 24) | (ri.min(255) << 16) | (gi.min(255) << 8) | bi.min(255)
}

fn geometry_key(contours: &[Contour], rule: Rule, width: usize, height: usize, antialias: bool) -> u128 {
  let mut lo = 0xcbf2_9ce4_8422_2325u64;
  let mut hi = 0x9e37_79b9_7f4a_7c15u64;
  for contour in contours {
    for point in &contour.points {
      for word in [point.x.to_bits(), point.y.to_bits()] {
        lo = (lo ^ u64::from(word)).wrapping_mul(0x100_0000_01b3);
        hi ^= lo.rotate_left(17).wrapping_add(u64::from(word));
        hi = hi.wrapping_mul(0x9e37_79b1_85eb_ca87);
      }
    }
  }
  lo ^= match rule {
    Rule::NonZero => 0,
    Rule::EvenOdd => 1,
  };
  for word in [width as u64, height as u64, u64::from(antialias)] {
    lo = (lo ^ word).wrapping_mul(0x100_0000_01b3);
    hi ^= lo.rotate_left(17).wrapping_add(word);
    hi = hi.wrapping_mul(0x9e37_79b1_85eb_ca87);
  }
  (u128::from(hi) << 64) | u128::from(lo)
}

impl RenderCtx<'_> {
  #[allow(clippy::too_many_arguments)]
  fn collect_layers(
    &self,
    scratch: &mut RenderScratch,
    width: usize,
    height: usize,
    antialias: bool,
    layers: &[Layer],
    base: Mat2x3,
    frame: f32,
    opacity: f32,
    clip: &ClipQuad,
    precomp_depth: usize,
    renderer: &mut impl FrameRenderer,
  ) -> Result<()> {
    if precomp_depth > MAX_PRECOMP_DEPTH {
      return Ok(());
    }
    let mut consumed_as_matte = vec![false; layers.len()];
    for (i, l) in layers.iter().enumerate() {
      if l.matte.is_some() {
        if let Some(slot) = i.checked_sub(1).and_then(|j| consumed_as_matte.get_mut(j)) {
          *slot = true;
        }
      }
    }
    for (idx, layer) in layers.iter().enumerate().rev() {
      if consumed_as_matte.get(idx).copied().unwrap_or(false) || layer.matte_src || !self.layer_visible(layer, frame) {
        continue;
      }
      let (layer_m, layer_opacity) = layer_transform_at(layer, frame);
      let m = base.concat(parent_chain_matrix(layers, layer, frame)).concat(layer_m);
      let combined_opacity = opacity * layer_opacity;
      let group_opacity = opacity_byte(combined_opacity);
      if group_opacity == 0 {
        continue;
      }
      if let Some(kind) = layer.matte {
        let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) else {
          continue;
        };
        // A track-matte consumer is transparent while its source layer is
        // outside [ip, op). Evaluating an inactive source leaks stale matte
        // artwork after its authored lifetime (real files use consecutive
        // matte sources to hand content off between frame ranges).
        if !self.layer_visible(src, frame) {
          continue;
        }
        renderer.save_layer();
        let (src_m, src_opacity) = layer_transform_at(src, frame);
        let source_matrix = base.concat(parent_chain_matrix(layers, src, frame)).concat(src_m);
        // A matte source's layer opacity applies to its flattened result, not
        // independently to every child of a precomp. Carry it into the fused
        // matte composite instead of distributing it through the source tree.
        self.collect_layer_content(scratch, width, height, antialias, src, source_matrix, frame, 1.0, clip, precomp_depth, renderer)?;
        self.collect_masks(width, height, src, source_matrix, frame, clip, renderer);
        renderer.save_layer();
        self.collect_layer_content(scratch, width, height, antialias, layer, m, frame, 1.0, clip, precomp_depth, renderer)?;
        self.collect_masks(width, height, layer, m, frame, clip, renderer);
        renderer.end_layer(Composite::Matte {
          kind,
          opacity: group_opacity as u8,
          source_opacity: opacity_byte(src_opacity) as u8,
        });
        continue;
      }
      let complex_precomp = if layer.kind == LayerKind::Precomp {
        layer
          .ref_id
          .as_deref()
          .and_then(|ref_id| self.comp.assets.iter().find(|asset| asset.id == ref_id))
          .is_some_and(|asset| asset.layers.len() > 1)
      } else {
        false
      };
      // Layer opacity applies to the flattened result of a shape layer.
      // Folding it into every fill/stroke makes overlapping paints more
      // opaque than authored (notably cloud shading made from several
      // overlapping white shapes).
      let translucent_shape = group_opacity < 255 && layer.kind == LayerKind::Shape && shapes_have_multiple_visible_paints(&layer.shapes, frame);
      let isolate = !layer.masks.is_empty() || translucent_shape || (group_opacity < 255 && complex_precomp);
      if isolate {
        renderer.save_layer();
      }
      self.collect_layer_content(
        scratch,
        width,
        height,
        antialias,
        layer,
        m,
        frame,
        if isolate { 1.0 } else { combined_opacity },
        clip,
        precomp_depth,
        renderer,
      )?;
      if !layer.masks.is_empty() {
        self.collect_masks(width, height, layer, m, frame, clip, renderer);
      }
      if isolate {
        renderer.end_layer(Composite::Over { opacity: group_opacity as u8 });
      }
    }
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn collect_layer_content(
    &self,
    scratch: &mut RenderScratch,
    width: usize,
    height: usize,
    antialias: bool,
    layer: &Layer,
    m: Mat2x3,
    frame: f32,
    content_opacity: f32,
    clip: &ClipQuad,
    precomp_depth: usize,
    renderer: &mut impl FrameRenderer,
  ) -> Result<()> {
    if opacity_byte(content_opacity) == 0 {
      return Ok(());
    }
    match layer.kind {
      LayerKind::Shape => {
        let mut walker = ShapeWalker {
          scratch,
          frame,
          clip,
          curve_tolerance: self.curve_tolerance,
          width,
          height,
          antialias,
        };
        let (arena, pending) = walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
        walker.collect_shape_jobs(&arena, &pending, renderer);
        for (contour, _) in arena {
          walker.scratch.put_contour(contour);
        }
      }
      LayerKind::Solid => {
        if let Some((sw, sh, color)) = layer.solid {
          let contour = rect_contour(Vec2::new(sw * 0.5, sh * 0.5), Vec2::new(sw, sh), 0.0, false, &m, self.curve_tolerance);
          let contours = core::slice::from_ref(&contour);
          let key = geometry_key(contours, Rule::NonZero, width, height, antialias);
          renderer.draw(
            Geometry::new(contours, key),
            Paint::Solid(SolidPaint {
              rule: Rule::NonZero,
              argb: premul_argb(color, content_opacity),
              color,
              opacity: content_opacity,
            }),
          );
        }
      }
      LayerKind::Precomp => {
        let Some(ref_id) = layer.ref_id.as_deref() else {
          return Ok(());
        };
        let Some(asset) = self.comp.assets.iter().find(|a| a.id == ref_id) else {
          return Ok(());
        };
        let mut child_clip: ClipQuad = clip.clone();
        if let Some((w, h)) = layer.precomp_size {
          child_clip.push([m.apply(Vec2::new(0.0, 0.0)), m.apply(Vec2::new(w, 0.0)), m.apply(Vec2::new(w, h)), m.apply(Vec2::new(0.0, h))]);
        }
        let sr = if layer.time_stretch.abs() > 1e-6 { layer.time_stretch } else { 1.0 };
        let quant = |v: f32| if self.continuous { v } else { v.trunc() };
        let child_frame = match &layer.time_remap {
          Some(tm) => {
            let dur = (self.comp.out_point - self.comp.in_point - 1.0).max(0.0);
            let fr = self.comp.frame_rate.max(1e-6);
            let pos = if dur > 0.0 { (tm.eval(frame) * fr / dur).clamp(0.0, 1.0) } else { 0.0 };
            quant(pos * dur / sr)
          }
          None => quant((frame - layer.start_time) / sr),
        };
        self.collect_layers(
          scratch,
          width,
          height,
          antialias,
          &asset.layers,
          m,
          child_frame,
          content_opacity,
          &child_clip,
          precomp_depth + 1,
          renderer,
        )?;
      }
      LayerKind::Null | LayerKind::Other(_) => {}
    }
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn collect_masks(&self, width: usize, height: usize, layer: &Layer, matrix: Mat2x3, frame: f32, clip: &ClipQuad, renderer: &mut impl FrameRenderer) {
    let masks = layer.masks.iter().filter(|mask| matches!(mask.mode, b'a' | b's' | b'i' | b'f')).collect::<Vec<_>>();
    let count = masks.len();
    for (index, mask) in masks.into_iter().enumerate() {
      let data = mask.path.eval(frame);
      let mut contour = flatten_path(&data, &matrix, self.curve_tolerance);
      for quad in clip {
        contour = clip_to_quad(&contour, quad);
      }
      contour = clip_contour(&contour, width as f32, height as f32);
      let contours = core::slice::from_ref(&contour);
      renderer.apply_mask(
        Geometry::new(contours, geometry_key(contours, Rule::NonZero, width, height, false)),
        mask.mode,
        mask.invert,
        ((mask.opacity.eval(frame) / 100.0).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        index == 0,
        index + 1 == count,
      );
    }
  }
}

impl ShapeWalker<'_> {
  fn collect_shape_jobs(&mut self, arena: &[(Contour, bool)], pending: &[PendingJob], renderer: &mut impl FrameRenderer) {
    for pj in pending.iter().rev() {
      match self.materialize(pj, arena, &|key| renderer.retains_geometry(key)) {
        DrawJob::Solid {
          key,
          contours,
          borrowed,
          rule,
          color,
          opacity,
          ..
        } => {
          let geometry = borrowed.and_then(|index| arena.get(index).map(|(contour, _)| core::slice::from_ref(contour))).unwrap_or(&contours);
          renderer.draw(
            Geometry::new(geometry, key),
            Paint::Solid(SolidPaint {
              rule: rule_of(rule),
              argb: premul_argb(color, opacity),
              color,
              opacity,
            }),
          );
          for c in contours {
            self.scratch.put_pts(c.points);
          }
        }
        DrawJob::Gradient {
          key,
          src_key,
          contours,
          borrowed,
          rule,
          lut,
          map,
          ..
        } => {
          let geometry = borrowed.and_then(|index| arena.get(index).map(|(contour, _)| core::slice::from_ref(contour))).unwrap_or(&contours);
          let gradient = GradientPaint {
            rule: rule_of(rule),
            lut,
            transform: GradientTransform {
              a: map.inv.a,
              b: map.inv.b,
              c: map.inv.c,
              d: map.inv.d,
              tx: map.inv.tx,
              ty: map.inv.ty,
            },
            kind: match map.kind {
              GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq } => GradientKind::Linear { sx, sy, dx, dy, inv_len_sq },
              GradientMapKind::Radial { sx, sy, inv_r } => GradientKind::Radial { sx, sy, inv_r },
              GradientMapKind::Focal { fx, fy, dx, dy, a, r } => GradientKind::Focal { fx, fy, dx, dy, a, r },
            },
            source_key: src_key,
          };
          renderer.draw(Geometry::new(geometry, key), Paint::Gradient(&gradient));
          for c in contours {
            self.scratch.put_pts(c.points);
          }
        }
      }
    }
  }
}

#[cfg(test)]
#[path = "tests/walker.rs"]
mod tests;
