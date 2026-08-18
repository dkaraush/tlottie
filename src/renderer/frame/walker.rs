//! Unstable renderer-neutral evaluated-frame pipeline.
//!
//! Renderer calls borrow the walker's temporary geometry and are consumed
//! synchronously. Renderers decide independently what, if anything, to
//! retain or cache.

#![allow(missing_docs)]

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::error::{Error, Result};
use crate::geometry::{clip_contour, clip_to_quad, flatten_path, rect_contour, Contour};
use crate::limits::Limits;
use crate::math::{Color, Mat2x3, Vec2};
use crate::model::{shapes_have_multiple_visible_paints, shapes_static, Composition, FillRule, Layer, LayerKind};
use crate::renderer::cpu::executor::{layer_transform_at, opacity_byte, parent_chain_matrix, ClipQuad, DrawJob, GradientMapKind, PendingJob, RenderCtx, RenderScratch, ShapeWalker, MAX_PRECOMP_DEPTH};
use alloc::vec;
use alloc::vec::Vec;

use super::renderer::*;

/// Stateful evaluator that streams one frame into a renderer.
#[derive(Default)]
pub struct FrameWalker {
  scratch: RenderScratch,
  static_jobs: StaticJobCache,
}

const STATIC_JOB_CACHE_BYTES: usize = 1024 * 1024;
const STATIC_JOB_ENTRY_BYTES: usize = 256 * 1024;
const STATIC_JOB_CACHE_ENTRIES: usize = 256;

#[derive(Clone)]
enum CachedPaint {
  Solid(SolidPaint),
  Gradient(GradientPaint),
}

#[derive(Clone)]
struct CachedJob {
  key: u128,
  contours: Vec<Contour>,
  paint: CachedPaint,
}

impl CachedJob {
  fn replay(&self, tx: f32, ty: f32, layer_alpha: f32, renderer: &mut impl FrameRenderer) {
    let key = translated_key(self.key, tx, ty);
    let geometry = Geometry::translated(&self.contours, key, tx, ty);
    match &self.paint {
      CachedPaint::Solid(paint) => {
        let mut paint = *paint;
        // An animated layer opacity recorded a 1.0 paint; re-apply the
        // current fraction (the coverage is geometry-only, so this stays a cheap
        // per-frame solid blend over the covered pixels — byte-exact fold).
        if layer_alpha < 1.0 {
          paint.opacity = layer_alpha;
        }
        renderer.draw(geometry, Paint::Solid(paint));
      }
      CachedPaint::Gradient(paint) => {
        let mut paint = paint.clone();
        paint.transform.tx -= paint.transform.a * tx + paint.transform.c * ty;
        paint.transform.ty -= paint.transform.b * tx + paint.transform.d * ty;
        paint.source_key = translated_key(paint.source_key, tx, ty);
        paint.alpha = opacity_byte(layer_alpha) as u8;
        renderer.draw(geometry, Paint::Gradient(&paint));
      }
    }
  }
}

fn translated_key(key: u128, tx: f32, ty: f32) -> u128 {
  if tx == 0.0 && ty == 0.0 {
    return key;
  }
  let mut lo = key as u64;
  let mut hi = (key >> 64) as u64;
  for word in [tx.to_bits(), ty.to_bits()] {
    lo = (lo ^ u64::from(word)).wrapping_mul(0x100_0000_01b3);
    hi ^= lo.rotate_left(17).wrapping_add(u64::from(word));
    hi = hi.wrapping_mul(0x9e37_79b1_85eb_ca87);
  }
  (u128::from(hi) << 64) | u128::from(lo)
}

#[derive(Clone)]
struct StaticContext {
  composition: usize,
  layer: usize,
  matrix: [u32; 6],
  opacity: u32,
  color: [u32; 5],
  width: usize,
  height: usize,
  antialias: bool,
  curve_tolerance: u32,
  clip: Vec<[u32; 8]>,
  /// Layer opacity is animated: the paint is frozen at 100% and the current
  /// frame's opacity is re-applied at replay.
  dynamic: bool,
}

impl StaticContext {
  #[allow(clippy::too_many_arguments)]
  fn new(comp: &Composition, layer: &Layer, matrix: Mat2x3, opacity: f32, color: Option<Color>, width: usize, height: usize, antialias: bool, curve_tolerance: f32, clip: &ClipQuad, dynamic: bool) -> Self {
    let color = match color {
      Some(color) => [1, color.r.to_bits(), color.g.to_bits(), color.b.to_bits(), color.a.to_bits()],
      None => [0; 5],
    };
    Self {
      composition: comp as *const Composition as usize,
      layer: layer as *const Layer as usize,
      matrix: [matrix.a.to_bits(), matrix.b.to_bits(), matrix.c.to_bits(), matrix.d.to_bits(), matrix.tx.to_bits(), matrix.ty.to_bits()],
      opacity: opacity.to_bits(),
      color,
      width,
      height,
      antialias,
      curve_tolerance: curve_tolerance.to_bits(),
      clip: clip
        .iter()
        .map(|quad| {
          [
            quad[0].x.to_bits(),
            quad[0].y.to_bits(),
            quad[1].x.to_bits(),
            quad[1].y.to_bits(),
            quad[2].x.to_bits(),
            quad[2].y.to_bits(),
            quad[3].x.to_bits(),
            quad[3].y.to_bits(),
          ]
        })
        .collect(),
      dynamic,
    }
  }

  fn signature(&self) -> u128 {
    let mut lo = 0xcbf2_9ce4_8422_2325u64;
    let mut hi = 0x9e37_79b9_7f4a_7c15u64;
    let mut mix = |word: u64| {
      lo = (lo ^ word).wrapping_mul(0x100_0000_01b3);
      hi ^= lo.rotate_left(17).wrapping_add(word);
      hi = hi.wrapping_mul(0x9e37_79b1_85eb_ca87);
    };
    for word in [
      self.composition as u64,
      self.layer as u64,
      self.width as u64,
      self.height as u64,
      self.antialias as u64,
      self.curve_tolerance as u64,
      self.opacity as u64,
    ] {
      mix(word);
    }
    for &word in &self.matrix {
      mix(u64::from(word));
    }
    for &word in &self.color {
      mix(u64::from(word));
    }
    for quad in &self.clip {
      for &word in quad {
        mix(u64::from(word));
      }
    }
    (u128::from(hi) << 64) | u128::from(lo)
  }
}

impl PartialEq for StaticContext {
  fn eq(&self, other: &Self) -> bool {
    self.composition == other.composition
      && self.layer == other.layer
      && self.matrix == other.matrix
      && self.opacity == other.opacity
      && self.color == other.color
      && self.width == other.width
      && self.height == other.height
      && self.antialias == other.antialias
      && self.curve_tolerance == other.curve_tolerance
      && self.clip == other.clip
      && self.dynamic == other.dynamic
  }
}

struct CachedLayerJobs {
  context: StaticContext,
  jobs: Vec<CachedJob>,
  bytes: usize,
  /// Jobs were painted frozen at 100% because the layer opacity animates; every
  /// replay must re-apply the current frame's layer opacity on top.
  apply_layer_alpha: bool,
}

#[derive(Default)]
struct StaticJobCache {
  entries: crate::compat::HashMap<u128, CachedLayerJobs>,
  seen: crate::compat::HashSet<u128>,
  rejected: crate::compat::HashSet<u128>,
  static_flags: crate::compat::HashMap<(usize, usize), bool>,
  bytes: usize,
  #[cfg(test)]
  hits: usize,
}

impl StaticJobCache {
  fn layer_is_static(&mut self, composition: &Composition, layer: &Layer) -> bool {
    let key = (composition as *const Composition as usize, layer as *const Layer as usize);
    if let Some(&is_static) = self.static_flags.get(&key) {
      return is_static;
    }
    let is_static = shapes_static(&layer.shapes);
    if self.static_flags.len() >= STATIC_JOB_CACHE_ENTRIES {
      self.static_flags.clear();
    }
    self.static_flags.insert(key, is_static);
    is_static
  }

  fn replay(&mut self, signature: u128, context: &StaticContext, tx: f32, ty: f32, layer_alpha: f32, renderer: &mut impl FrameRenderer) -> bool {
    let Some(entry) = self.entries.get(&signature) else {
      return false;
    };
    // A hash collision is a miss, never an incorrect replay.
    if entry.context != *context {
      return false;
    }
    // Frozen-at-100% layers (animated layer opacity) pick up the current frame's
    // opacity here; baked layers replay at 1.0 and keep their paint opacity.
    let alpha = if entry.apply_layer_alpha { layer_alpha } else { 1.0 };
    for job in &entry.jobs {
      job.replay(tx, ty, alpha, renderer);
    }
    #[cfg(test)]
    {
      self.hits += 1;
    }
    true
  }

  fn should_record(&mut self, signature: u128) -> bool {
    if self.rejected.contains(&signature) {
      return false;
    }
    if self.seen.remove(&signature) {
      return true;
    }
    if self.seen.len() >= STATIC_JOB_CACHE_ENTRIES {
      self.seen.clear();
    }
    self.seen.insert(signature);
    false
  }

  fn insert(&mut self, signature: u128, context: StaticContext, jobs: Vec<CachedJob>) {
    let bytes = static_jobs_bytes(&context, &jobs);
    if bytes > STATIC_JOB_ENTRY_BYTES {
      self.reject(signature);
      return;
    }
    if let Some(previous) = self.entries.remove(&signature) {
      self.bytes = self.bytes.saturating_sub(previous.bytes);
    }
    if self.entries.len() >= STATIC_JOB_CACHE_ENTRIES || self.bytes.saturating_add(bytes) > STATIC_JOB_CACHE_BYTES {
      self.reject(signature);
      return;
    }
    self.bytes = self.bytes.saturating_add(bytes);
    let apply_layer_alpha = context.dynamic;
    self.entries.insert(signature, CachedLayerJobs { context, jobs, bytes, apply_layer_alpha });
  }

  fn reject(&mut self, signature: u128) {
    if self.rejected.len() >= STATIC_JOB_CACHE_ENTRIES {
      self.rejected.clear();
    }
    self.rejected.insert(signature);
  }
}

fn static_jobs_bytes(context: &StaticContext, jobs: &Vec<CachedJob>) -> usize {
  let mut bytes = core::mem::size_of::<CachedLayerJobs>()
    .saturating_add(context.clip.capacity().saturating_mul(core::mem::size_of::<[u32; 8]>()))
    .saturating_add(jobs.capacity().saturating_mul(core::mem::size_of::<CachedJob>()));
  let mut gradient_luts: Vec<*const [u32; GRADIENT_LUT_SIZE]> = Vec::new();
  for job in jobs {
    bytes = bytes.saturating_add(job.contours.capacity().saturating_mul(core::mem::size_of::<Contour>()));
    for contour in &job.contours {
      bytes = bytes
        .saturating_add(contour.points.capacity().saturating_mul(core::mem::size_of::<Vec2>()))
        .saturating_add(contour.anchors.capacity().saturating_mul(core::mem::size_of::<bool>()));
    }
    if let CachedPaint::Gradient(paint) = &job.paint {
      let ptr = alloc::sync::Arc::as_ptr(&paint.lut);
      if !gradient_luts.contains(&ptr) {
        gradient_luts.push(ptr);
        bytes = bytes.saturating_add(core::mem::size_of::<[u32; GRADIENT_LUT_SIZE]>());
      }
    }
  }
  bytes.saturating_add(gradient_luts.capacity().saturating_mul(core::mem::size_of::<*const [u32; GRADIENT_LUT_SIZE]>()))
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

  let FrameWalker { scratch, static_jobs } = walker;
  let ctx = RenderCtx {
    comp,
    continuous: frame_in_range.fract() != 0.0,
    #[cfg(test)]
    antialias: options.antialias,
    curve_tolerance: options.curve_tolerance,
  };
  ctx.collect_layers(
    scratch,
    static_jobs,
    width as usize,
    height as usize,
    options.antialias,
    &comp.layers,
    base,
    frame,
    1.0,
    None,
    &Vec::new(),
    0,
    renderer,
  )
}

fn rule_of(rule: FillRule) -> Rule {
  match rule {
    FillRule::NonZero => Rule::NonZero,
    FillRule::EvenOdd => Rule::EvenOdd,
  }
}

fn premul_rgba(color: Color, opacity: f32) -> u32 {
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
  crate::pixel::pack_premultiplied_rgba(ri.min(255), gi.min(255), bi.min(255), ai.min(255))
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
    static_jobs: &mut StaticJobCache,
    width: usize,
    height: usize,
    antialias: bool,
    layers: &[Layer],
    base: Mat2x3,
    frame: f32,
    opacity: f32,
    inherited_color: Option<Color>,
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
        self.collect_layer_content(
          scratch,
          static_jobs,
          width,
          height,
          antialias,
          src,
          source_matrix,
          frame,
          1.0,
          inherited_color,
          clip,
          precomp_depth,
          1.0,
          false,
          renderer,
        )?;
        self.collect_masks(width, height, src, source_matrix, frame, clip, renderer);
        renderer.save_layer();
        self.collect_layer_content(scratch, static_jobs, width, height, antialias, layer, m, frame, 1.0, inherited_color, clip, precomp_depth, 1.0, false, renderer)?;
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
      // A single paint can absorb a *constant* layer opacity into its fill and
      // still replay a static flattened job (byte-exact, baked LUT). When the
      // layer opacity animates, the folded paint changes every frame and the
      // static-job cache never hits. Freeze the fill at 100% (constant →
      // geometry + gradient source stay cached) and re-apply the current alpha on
      // *every* emission: cache hits, record frames and the deferred loop all
      // thread `combined_opacity` straight into the paint (SolidPaint.opacity /
      // GradientPaint.alpha) — no offscreen surface, byte-exact fold.
      // Small-icon regime: the frozen canonical replay wins there; above 224 the
      // per-pixel gradient re-composite outweighs the saved walk (measured on
      // NeonEmoji), so bake the opacity and stay byte-close to baseline.
      let dynamic_translucent = !isolate
        && group_opacity < 255
        && layer.kind == LayerKind::Shape
        && !layer.transform.opacity.is_static()
        && layer.transform.geometry_static()
        && shapes_static(&layer.shapes)
        && clip.is_empty()
        && width.max(height) <= 224;
      let decorrelated_alpha = if dynamic_translucent { combined_opacity } else { 1.0 };
      self.collect_layer_content(
        scratch,
        static_jobs,
        width,
        height,
        antialias,
        layer,
        m,
        frame,
        if isolate { 1.0 } else if dynamic_translucent { 1.0 } else { combined_opacity },
        inherited_color,
        clip,
        precomp_depth,
        decorrelated_alpha,
        dynamic_translucent,
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
    static_jobs: &mut StaticJobCache,
    width: usize,
    height: usize,
    antialias: bool,
    layer: &Layer,
    m: Mat2x3,
    frame: f32,
    content_opacity: f32,
    inherited_color: Option<Color>,
    clip: &ClipQuad,
    precomp_depth: usize,
    layer_alpha: f32,
    dynamic_layer: bool,
    renderer: &mut impl FrameRenderer,
  ) -> Result<()> {
    if opacity_byte(content_opacity) == 0 {
      return Ok(());
    }
    let color_override = layer.color_override.or(inherited_color);
    match layer.kind {
      LayerKind::Shape => {
        // A device translation does not change static local contours or paint.
        // Retain one canonical (tx=ty=0) job list and supply the exact current
        // translation to the renderer. Clipped layers remain exact-context
        // cached because their clipped contour topology can change as they move.
        let layer_static = static_jobs.layer_is_static(self.comp, layer);
        // At large output sizes, processing complete off-canvas canonical
        // contours costs more than the saved walk. Keep the exact-context
        // path there; this cache targets the small-icon regime where the
        // retained-state gap was measured.
        // Capture always runs at the real device matrix, bounded.
        //
        // Capturing at a canonical origin instead, so that a pure
        // translation became a replay parameter, forced the recording pass
        // to run `unbounded`: clipping cannot be baked in when the position
        // is meant to vary. That handed the rasteriser unclipped geometry,
        // which no other frame ever produces, and dropped scanlines on the
        // recording frame — a visible hole in 31 of 260 sampled Telegram
        // fixtures. It also measured flat, so the reuse was not worth it.
        //
        // Animated-opacity layers keep the frozen-at-100% paint: their
        // geometry transform is static, so the real-matrix capture is still
        // cacheable and every replay re-applies the current layer opacity.
        let static_context = layer_static.then(|| StaticContext::new(self.comp, layer, m, content_opacity, color_override, width, height, antialias, self.curve_tolerance, clip, dynamic_layer));
        let signature = static_context.as_ref().map(StaticContext::signature);
        let replay_hit = if let (Some(context), Some(signature)) = (static_context.as_ref(), signature) {
          static_jobs.replay(signature, context, 0.0, 0.0, layer_alpha, renderer)
        } else {
          false
        };
        if replay_hit {
          return Ok(());
        }
        // Dynamic layers rebuild and store the frozen canonical job on every miss so
        // they always emit with this frame's opacity; baked layers defer to the cap.
        let record = signature.is_some_and(|signature| dynamic_layer || static_jobs.should_record(signature));
        let mut walker = ShapeWalker {
          scratch,
          frame,
          clip,
          curve_tolerance: self.curve_tolerance,
          width,
          height,
          antialias,
          color_override,
          unbounded: false,
        };
        let (arena, pending) = walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
        let mut recorded = record.then(|| Vec::with_capacity(pending.len()));
        walker.collect_shape_jobs(&arena, &pending, renderer, recorded.as_mut(), !record);
        for (contour, _) in arena {
          walker.scratch.put_contour(contour);
        }
        if let (Some(context), Some(signature), Some(jobs)) = (static_context, signature, recorded) {
          for job in &jobs {
            job.replay(0.0, 0.0, if dynamic_layer { layer_alpha } else { 1.0 }, renderer);
          }
          static_jobs.insert(signature, context, jobs);
        }
      }
      LayerKind::Solid => {
        if let Some((sw, sh, color)) = layer.solid {
          let color = color_override.unwrap_or(color);
          let contour = rect_contour(Vec2::new(sw * 0.5, sh * 0.5), Vec2::new(sw, sh), 0.0, false, &m, self.curve_tolerance);
          let contours = core::slice::from_ref(&contour);
          let key = geometry_key(contours, Rule::NonZero, width, height, antialias);
          renderer.draw(
            Geometry::new(contours, key),
            Paint::Solid(SolidPaint {
              rule: Rule::NonZero,
              rgba: premul_rgba(color, content_opacity),
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
          static_jobs,
          width,
          height,
          antialias,
          &asset.layers,
          m,
          child_frame,
          content_opacity,
          color_override,
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
  fn collect_shape_jobs(&mut self, arena: &[(Contour, bool)], pending: &[PendingJob], renderer: &mut impl FrameRenderer, mut record: Option<&mut Vec<CachedJob>>, emit: bool) {
    for pj in pending.iter().rev() {
      // Recording must materialize complete geometry even when this renderer
      // already retains coverage. The recorded jobs are backend-independent
      // and must still render correctly after a later coverage eviction.
      let recording = record.is_some();
      match self.materialize(pj, arena, &|key| !recording && renderer.retains_geometry(key)) {
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
          let paint = SolidPaint {
            rule: rule_of(rule),
            rgba: premul_rgba(color, opacity),
            color,
            opacity,
          };
          if let Some(record) = record.as_deref_mut() {
            record.push(CachedJob {
              key,
              contours: geometry.to_vec(),
              paint: CachedPaint::Solid(paint),
            });
          }
          if emit {
            renderer.draw(Geometry::new(geometry, key), Paint::Solid(paint));
          }
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
            alpha: 255,
          };
          if let Some(record) = record.as_deref_mut() {
            record.push(CachedJob {
              key,
              contours: geometry.to_vec(),
              paint: CachedPaint::Gradient(gradient.clone()),
            });
          }
          if emit {
            renderer.draw(Geometry::new(geometry, key), Paint::Gradient(&gradient));
          }
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
