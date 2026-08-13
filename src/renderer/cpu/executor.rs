//! Frame rendering: evaluate the model at a frame, flatten geometry,
//! rasterize, and composite into a premultiplied RGBA8 buffer.

use super::mapped_surface::Surface;
use crate::cells::CellRaster;
#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
#[cfg(test)]
use crate::error::Error;
use crate::error::Result;
#[cfg(test)]
use crate::geometry::flatten_path;
use crate::geometry::{
  clip_contour, clip_to_quad, dash_polyline, ellipse_contour, extract_by_length, flatten_path_reusing, polystar_path, quad_contains_box, rect_contour, round_polyline_corners, Contour,
};
#[cfg(test)]
use crate::limits::Limits;
use crate::math::{Color, Mat2x3, Vec2};
#[cfg(test)]
use crate::model::shapes_have_multiple_paints;
#[cfg(test)]
use crate::model::LayerKind;
use crate::model::{Composition, DashElement, FillRule, FloatList, GradientKind, Layer, Shape, Transform, TrimMode};
use crate::raster::Rasterizer;
use crate::renderer::frame::renderer::GRADIENT_LUT_SIZE;
use crate::stroke::{stroke_polyline, StrokeSegment};
use alloc::vec::Vec;

/// Maximum group recursion while rendering (matches parse-side bound).
const MAX_RENDER_DEPTH: usize = 40;
/// Gradient LUT resolution; rlottie uses a 1024-entry table
/// (VGradient::colorTableSize) — 256 visibly quantizes steep ramps.
/// Maximum precomp nesting during render.
pub(crate) const MAX_PRECOMP_DEPTH: usize = 16;

/// Reusable render-time buffers: rasterizer accumulators, offscreen pixel
/// planes, mask planes, gradient LUT memos. Owned by [`crate::CPURenderer`]
/// (one per playing instance) so nothing is reallocated between frames —
/// at 720px a single frame otherwise allocates and zeroes multiple MB.
/// Dropping it frees everything; it carries no per-composition state.
#[derive(Default)]
pub(crate) struct RenderScratch {
  rasters: Vec<Rasterizer>,
  cells_pool: Vec<CellRaster>,
  #[cfg(test)]
  bufs_u32: Vec<Vec<u32>>,
  /// CPU frame-renderer surfaces are returned already transparent. Unlike
  /// general scratch planes, acquiring one therefore does not need to clear
  /// the full canvas again.
  surface_u32: Vec<Surface>,
  bufs_u8: Vec<Vec<u8>>,
  /// Gradient LUT memoization: building a 1024-entry premultiplied table
  /// from the stop list is pure, and stop values repeat across frames
  /// (static gradients: every frame; animated: on hold segments and loop
  /// repeats). Keyed by the exact input bits — no collision risk.
  lut_cache: crate::compat::HashMap<Vec<u32>, alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>>,
  lut_key: Vec<u32>,
  /// Recycled contour point buffers: stroke pieces + fill snapshots draw
  /// from here and return after their paint executes (measured: 2,683
  /// piece allocations per 64px frame on stroke-heavy files).
  pts_pool: Vec<Vec<Vec2>>,
  /// Recycled source contours (points + anchor provenance) produced while
  /// evaluating authored paths. Unlike paint snapshots, these need their
  /// anchor flags for trim/dash/stroke processing, so retain the complete
  /// contour between frames instead of allocating two vectors per path.
  contour_pool: Vec<Contour>,
  /// Reused normalized stroke segments; stroke calls execute serially.
  stroke_segments: Vec<StrokeSegment>,
  /// Memoized per-layer staticness (keyed by the Layer's stable address
  /// inside the Arc'd Composition).
  #[cfg(test)]
  static_flags: crate::compat::HashMap<usize, bool>,
  /// Static-layer job lists: replay the exact fill calls (by coverage
  /// key) without walking/evaluating/flattening the shape tree at all —
  /// the per-frame cost rlottie avoids via its own static detection.
  #[cfg(test)]
  jobs_cache: crate::compat::HashMap<u128, Vec<ReplayJob>>,
  /// Two-touch admission for jobs_cache: a replay key must be seen twice
  /// before recording (a static layer under an ANIMATED parent produces a
  /// fresh key every frame and must not flood the cache).
  #[cfg(test)]
  jobs_seen: std::collections::HashSet<u128>,
  /// Content-addressed coverage cache (the GOALS design lever): keyed by
  /// the exact bits of a paint's clipped device-space geometry, valued by
  /// the rasterizer's coverage rows. Measured repeat rates on the heavy
  /// set: 74-96% within one loop, ~100% across loop replays. Coverage is
  /// deterministic per geometry, so replaying rows is bit-exact.
  pub(crate) cov_cache: CovCache,
}

#[path = "coverage.rs"]
mod coverage;
use coverage::*;
pub(crate) use coverage::{mode_s_wins, pack_span, unpack_span, CovCache, CovEntry, PlaneData, SPAN_CAPTURE_MAX};

/// Bound on recycled point buffers (each typically 4-26 points).
const PTS_POOL_CAP: usize = 4096;
/// Do not pin exceptionally large authored paths in a renderer for its whole
/// lifetime. Ordinary sticker contours are far below this bound.
const CONTOUR_REUSE_POINT_CAP: usize = 4096;

/// One recorded paint of a static layer: everything the fused execute loop
/// needs except geometry (which replays from the coverage cache by key).
#[cfg(test)]
enum ReplayJob {
  Solid {
    key: u128,
    rule: FillRule,
    color: Color,
    opacity: f32,
  },
  Gradient {
    key: u128,
    src_key: u128,
    rule: FillRule,
    lut: alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
    map: GradientMap,
  },
}

/// Bound on cached static-layer job lists.
#[cfg(test)]
const JOBS_CACHE_CAP: usize = 1024;

/// Bound on memoized gradient LUTs (4 KB each). A composition rarely has
/// more than a few dozen distinct gradients; continuously-animated stops
/// would otherwise grow the map without bound, so overflow clears it.
const LUT_CACHE_CAP: usize = 512;

/// Bound on pooled objects per kind — offscreen depth beyond this is rare
/// (matte pairs + a couple of nested composites); excess simply frees.
/// (Halved from 8 in the RSS audit: 8 pooled 320px rasterizers+planes are
/// ~6 MB of mostly-idle buffers.)
const SCRATCH_POOL_CAP: usize = 4;

impl RenderScratch {
  pub(crate) fn take_raster(&mut self, w: usize, h: usize) -> Rasterizer {
    match self.rasters.pop() {
      Some(mut r) => {
        r.reshape(w, h);
        r
      }
      None => Rasterizer::new(w, h),
    }
  }

  pub(crate) fn put_raster(&mut self, mut r: Rasterizer) {
    if self.rasters.len() < SCRATCH_POOL_CAP {
      r.reset();
      self.rasters.push(r);
    }
  }

  pub(crate) fn take_cells(&mut self, w: usize, h: usize) -> CellRaster {
    match self.cells_pool.pop() {
      Some(mut c) => {
        c.reshape(w, h);
        c
      }
      None => CellRaster::new(w, h),
    }
  }

  pub(crate) fn put_cells(&mut self, mut c: CellRaster) {
    if self.cells_pool.len() < SCRATCH_POOL_CAP {
      c.reset();
      self.cells_pool.push(c);
    }
  }

  #[cfg(test)]
  pub(crate) fn take_u32(&mut self, n: usize) -> Vec<u32> {
    let mut b = self.bufs_u32.pop().unwrap_or_default();
    b.clear();
    b.resize(n, 0);
    b
  }

  #[cfg(test)]
  pub(crate) fn put_u32(&mut self, b: Vec<u32>) {
    if self.bufs_u32.len() < SCRATCH_POOL_CAP {
      self.bufs_u32.push(b);
    }
  }

  pub(crate) fn take_surface_u32(&mut self, n: usize) -> Surface {
    let mut b = self.surface_u32.pop().unwrap_or_default();
    b.resize_zeroed(n);
    b
  }

  pub(crate) fn put_surface_u32(&mut self, mut b: Surface, width: usize, rows: &[RowBounds]) {
    if width != 0 {
      let height = b.len() / width;
      for (y, &row) in rows.iter().take(height).enumerate() {
        if row.is_empty() {
          continue;
        }
        let x0 = row.x0.min(width);
        let x1 = row.x1.saturating_add(1).min(width);
        b[y * width + x0..y * width + x1].fill(0);
      }
    }
    if self.surface_u32.len() < SCRATCH_POOL_CAP {
      self.surface_u32.push(b);
    }
  }

  /// Returns a length-`n` u8 buffer WITHOUT a full-length fill: contents are
  /// unspecified (stale pool bytes) except that a freshly grown tail is
  /// zeroed. The only caller ([`RenderCtx::build_mask`]) reads just a bounded
  /// sub-rectangle (the offscreen dirty box) and seeds exactly that region
  /// itself via [`fill_rows_u8`], so skipping the O(n) fill is the whole
  /// point. Single-mask rendering likewise seeds only the active layer's
  /// tracked row spans before reading them. In steady state the pooled
  /// buffer is already `n` long and no fill happens at all.
  pub(crate) fn take_u8_uninit(&mut self, n: usize) -> Vec<u8> {
    let mut b = self.bufs_u8.pop().unwrap_or_default();
    if b.len() != n {
      b.clear();
      b.resize(n, 0);
    }
    b
  }

  pub(crate) fn take_u8(&mut self, n: usize, value: u8) -> Vec<u8> {
    let mut b = self.bufs_u8.pop().unwrap_or_default();
    b.resize(n, value);
    b.fill(value);
    b
  }

  pub(crate) fn put_u8(&mut self, b: Vec<u8>) {
    if self.bufs_u8.len() < SCRATCH_POOL_CAP {
      self.bufs_u8.push(b);
    }
  }

  pub(crate) fn put_pts(&mut self, mut v: Vec<Vec2>) {
    if self.pts_pool.len() < PTS_POOL_CAP {
      v.clear();
      self.pts_pool.push(v);
    }
  }

  pub(crate) fn take_contour(&mut self) -> Contour {
    self.contour_pool.pop().unwrap_or_default()
  }

  pub(crate) fn put_contour(&mut self, mut contour: Contour) {
    if self.contour_pool.len() < PTS_POOL_CAP && contour.points.capacity() <= CONTOUR_REUSE_POINT_CAP && contour.anchors.capacity() <= CONTOUR_REUSE_POINT_CAP {
      contour.points.clear();
      contour.anchors.clear();
      contour.inv_lin = None;
      self.contour_pool.push(contour);
    }
  }

  /// Returns the LUT plus a 64-bit id of its exact inputs (used in the
  /// gradient source-plane cache key).
  fn lut_for(&mut self, stops: &crate::model::FloatList, color_count: usize, opacity: f32) -> (alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>, u64) {
    self.lut_key.clear();
    self.lut_key.reserve(stops.0.len() + 2);
    self.lut_key.push(color_count as u32);
    self.lut_key.push(opacity.to_bits());
    for v in &stops.0 {
      self.lut_key.push(v.to_bits());
    }
    let mut h = Hasher128::new();
    for &w in &self.lut_key {
      h.mix(w);
    }
    let id = h.finish() as u64;
    if let Some(lut) = self.lut_cache.get(self.lut_key.as_slice()) {
      return (lut.clone(), id);
    }
    let lut: alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]> = alloc::sync::Arc::new(build_gradient_lut(stops, color_count, opacity));
    if self.lut_cache.len() >= LUT_CACHE_CAP {
      self.lut_cache.clear();
    }
    self.lut_cache.insert(self.lut_key.clone(), lut.clone());
    (lut, id)
  }
}

/// Test oracle for the pre-command CPU execution path.
#[cfg(test)]
pub(crate) fn render_pooled(composition: &Composition, scratch: &mut RenderScratch, frame_index: f32, pixels: &mut [u32], width: u32, height: u32, options: crate::RenderOptions) -> Result<()> {
  let limits = Limits::default();
  if width == 0 || height == 0 || width > limits.max_dimension || height > limits.max_dimension {
    return Err(Error::InvalidLottie {
      offset: 0,
      what: "render size out of range",
    });
  }
  let need = (width as usize).saturating_mul(height as usize);
  let Some(buf) = pixels.get_mut(..need) else {
    return Err(Error::InvalidLottie {
      offset: 0,
      what: "pixel buffer too small",
    });
  };
  if options.clear {
    buf.fill(0);
  }

  let max_frame = composition.frame_count().saturating_sub(1) as f32;
  let frame_in_range = if frame_index.is_finite() { frame_index.clamp(0.0, max_frame) } else { 0.0 };
  let frame = composition.in_point + frame_in_range;

  let base = Mat2x3::scale(width as f32 / composition.width.max(1) as f32, height as f32 / composition.height.max(1) as f32);

  scratch.cov_cache.set_budget_for_canvas(width as usize, height as usize);
  scratch.cov_cache.frame_tick();
  let raster = scratch.take_raster(width as usize, height as usize);
  let cells = scratch.take_cells(width as usize, height as usize);
  let mut canvas = Canvas::with_raster(buf, width as usize, height as usize, raster, cells, options.antialias);
  if !options.clear {
    canvas.dirty.mark_row(0, 0, 1);
  }
  let ctx = RenderCtx {
    comp: composition,
    continuous: frame_in_range.fract() != 0.0,
    antialias: options.antialias,
    curve_tolerance: options.curve_tolerance,
  };
  let res = ctx.render_layers(scratch, &mut canvas, &composition.layers, base, frame, 1.0, &Vec::new(), 0);
  scratch.put_raster(canvas.raster.take().expect("direct canvas has rasterizer"));
  scratch.put_cells(canvas.cells.take().expect("direct canvas has cell rasterizer"));
  res
}

#[doc(hidden)]
/// Stack of convex clip quads (device-space precomp viewports). Nested
/// precomps INTERSECT their viewports (rlottie: `mask = clipper.rle() &
/// mask`, lottieitem.cpp renderHelper) — applying convex clips in sequence
/// is exactly their intersection. Empty = no precomp clip.
pub(crate) type ClipQuad = Vec<[Vec2; 4]>;

pub(crate) struct RenderCtx<'a> {
  pub(crate) comp: &'a Composition,
  /// True when the requested root frame is fractional. Integer frames keep
  /// rlottie's integer-truncated precomp child time (bit-exact parity);
  /// fractional frames evaluate the whole tree continuously — quantizing
  /// precomp interiors would defeat in-between rendering.
  pub(crate) continuous: bool,
  /// Whether edge coverage remains fractional or is thresholded to binary.
  #[cfg(test)]
  pub(crate) antialias: bool,
  /// Maximum device-space error used while flattening cubic curves.
  pub(crate) curve_tolerance: f32,
}

impl RenderCtx<'_> {
  /// Renders a layer list (root composition or precomp asset) bottom-up.
  /// `frame` is in this list's local time; `base` maps this list's
  /// coordinate space to the device.
  #[allow(clippy::too_many_arguments)]
  #[cfg(test)]
  fn render_layers(&self, scratch: &mut RenderScratch, canvas: &mut Canvas<'_>, layers: &[Layer], base: Mat2x3, frame: f32, opacity: f32, clip: &ClipQuad, precomp_depth: usize) -> Result<()> {
    if precomp_depth > MAX_PRECOMP_DEPTH {
      return Ok(()); // over-deep nesting is dropped, not fatal
    }
    // A layer with `tt` is matted by the layer directly above it in file
    // order; that source layer is not drawn on its own.
    let mut consumed_as_matte = vec![false; layers.len()];
    for (i, l) in layers.iter().enumerate() {
      if l.matte.is_some() {
        if let Some(slot) = i.checked_sub(1).and_then(|j| consumed_as_matte.get_mut(j)) {
          *slot = true;
        }
      }
    }
    for (idx, layer) in layers.iter().enumerate().rev() {
      if consumed_as_matte.get(idx).copied().unwrap_or(false) {
        continue;
      }
      if layer.matte_src {
        continue; // matte-only layer without a consumer right below
      }
      if !self.layer_visible(layer, frame) {
        continue;
      }
      let (layer_m, layer_opacity) = layer_transform_at(layer, frame);
      let m = base.concat(parent_chain_matrix(layers, layer, frame)).concat(layer_m);
      let combined_opacity = opacity * layer_opacity;
      let k = opacity_byte(combined_opacity);
      if k == 0 {
        continue;
      }

      // A translucent shape layer is flattened before its layer opacity is
      // applied. Folding opacity into every paint would compound it where
      // fills and strokes overlap.
      let translucent_shape = k < 255 && layer.kind == LayerKind::Shape && shapes_have_multiple_paints(&layer.shapes);
      let complex_precomp = layer_opacity < 0.999
        && matches!(layer.kind, LayerKind::Precomp)
        && layer
          .ref_id
          .as_deref()
          .and_then(|id| self.comp.assets.iter().find(|a| a.id == id))
          .map(|a| a.layers.len() > 1)
          .unwrap_or(false);
      let needs_offscreen = !layer.masks.is_empty() || layer.matte.is_some() || translucent_shape || complex_precomp;
      if !needs_offscreen {
        self.draw_layer_content(scratch, canvas, layer, m, frame, combined_opacity, clip, precomp_depth)?;
        continue;
      }

      // Offscreen path: render content at full opacity, modulate by
      // masks and matte, then composite with the layer opacity.
      // All buffers come from (and return to) the scratch pools.
      let (w, h) = (canvas.w, canvas.h);
      let mut buf_a = scratch.take_u32(w * h);
      let da; // content bounds of buf_a: outside it everything is 0
      {
        let raster = scratch.take_raster(w, h);
        let cells = scratch.take_cells(w, h);
        let mut off = Canvas::with_raster(&mut buf_a, w, h, raster, cells, self.antialias);
        let res = self.draw_layer_content(scratch, &mut off, layer, m, frame, 1.0, clip, precomp_depth);
        da = off.dirty;
        scratch.put_raster(off.raster.take().expect("offscreen has rasterizer"));
        scratch.put_cells(off.cells.take().expect("offscreen has cell rasterizer"));
        res?;
      }
      // Every following pass is a per-pixel function that maps 0 → 0
      // on the destination (mask multiply, matte multiply, source-
      // over of 0), so bounding all of them to buf_a's dirty box is
      // exact — and measured offscreens are often EMPTY (skip all).
      if !da.is_empty() {
        if !layer.masks.is_empty() {
          // `da` is exactly the region the modulate below reads and
          // the only region where buf_a is nonzero — bound the mask
          // build to it (byte-exact; see build_mask).
          let maskbuf = self.build_mask(scratch, layer, m, frame, w, h, da);
          for_rows_boxed(&mut buf_a, w, da, |y, row| {
            let lo = y * w + da.x0;
            if let Some(mask_row) = maskbuf.get(lo..lo + row.len()) {
              modulate(row, mask_row);
            }
          });
          scratch.put_u8(maskbuf);
        }
        if layer.matte.is_some() {
          if let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) {
            let mut buf_b = scratch.take_u32(w * h);
            let (src_m, src_op) = layer_transform_at(src, frame);
            // A track-matte source is sampled for the consumer's lifetime;
            // its own in/out range controls standalone drawing only. Some
            // exports end the source one frame before its consumer.
            let sm = base.concat(parent_chain_matrix(layers, src, frame)).concat(src_m);
            let raster = scratch.take_raster(w, h);
            let cells = scratch.take_cells(w, h);
            let mut off = Canvas::with_raster(&mut buf_b, w, h, raster, cells, self.antialias);
            let res = self.draw_layer_content(scratch, &mut off, src, sm, frame, 1.0, clip, precomp_depth);
            // Source content bounds: buf_b is 0 outside `db`,
            // where the mask modulate below is a no-op.
            let db = off.dirty;
            scratch.put_raster(off.raster.take().expect("matte has rasterizer"));
            scratch.put_cells(off.cells.take().expect("matte has cell rasterizer"));
            res?;
            if !src.masks.is_empty() && !db.is_empty() {
              let maskbuf = self.build_mask(scratch, src, sm, frame, w, h, db);
              // Bound the modulate to `db` too — outside it
              // buf_b is 0 (modulate maps 0 → 0), so this is
              // byte-exact vs the former full-plane modulate.
              for_rows_boxed(&mut buf_b, w, db, |y, row| {
                let lo = y * w + db.x0;
                if let Some(mask_row) = maskbuf.get(lo..lo + row.len()) {
                  modulate(row, mask_row);
                }
              });
              scratch.put_u8(maskbuf);
            }
            let kind = layer.matte.unwrap_or(1);
            for_rows_boxed(&mut buf_a, w, da, |y, row| {
              let lo = y * w + da.x0;
              if let Some(src_row) = buf_b.get(lo..lo + row.len()) {
                apply_matte(row, src_row, kind, opacity_byte(src_op) as u8, self.comp.channel_order);
              }
            });
            scratch.put_u32(buf_b);
          }
        }
        canvas.dirty.union(da);
        for_rows_boxed(canvas.pixels, w, da, |y, row| {
          let lo = y * w + da.x0;
          if let Some(src_row) = buf_a.get(lo..lo + row.len()) {
            crate::simd::composite_over_span(row, src_row, k);
          }
        });
      }
      scratch.put_u32(buf_a);
    }
    Ok(())
  }

  pub(crate) fn layer_visible(&self, layer: &Layer, frame: f32) -> bool {
    // Half-open lifetime [ip, op): the out-point frame is NOT drawn
    // (patched rlottie lottieitem.cpp LOTLayerItem::visible uses
    // `frameNo() < outFrame()`). Five independent BROKEN clusters were
    // single-frame spikes at exactly frame == op.
    !layer.hidden && frame >= layer.in_point && frame < layer.out_point
  }

  /// Draws one layer's content (shape tree / precomp / solid) into `canvas`.
  #[allow(clippy::too_many_arguments)]
  #[cfg(test)]
  fn draw_layer_content(&self, scratch: &mut RenderScratch, canvas: &mut Canvas<'_>, layer: &Layer, m: Mat2x3, frame: f32, content_opacity: f32, clip: &ClipQuad, precomp_depth: usize) -> Result<()> {
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
          width: canvas.w,
          height: canvas.h,
          antialias: canvas.antialias,
          color_override: layer.color_override,
          unbounded: false,
        };
        let lp = layer as *const Layer as usize;
        let is_static = *walker.scratch.static_flags.entry(lp).or_insert_with(|| crate::model::shapes_static(&layer.shapes));
        if is_static {
          // Static shape tree: the job list is a deterministic
          // function of (device matrix, folded opacity, clip) —
          // replay it without evaluating/flattening anything.
          let mut h = walker.clip_sig();
          h.mix(4); // replay-key tag
          for v in [m.a, m.b, m.c, m.d, m.tx, m.ty] {
            h.mix(v.to_bits());
          }
          h.mix(content_opacity.to_bits());
          // Via u64: usize is 32-bit on wasm32, where `>> 32`
          // on the pointer itself would overflow.
          h.mix(lp as u64 as u32);
          h.mix(((lp as u64) >> 32) as u32);
          let rkey = h.finish();
          if let Some(jobs) = walker.scratch.jobs_cache.remove(&rkey) {
            let ok = walker.replay_jobs(canvas, &jobs);
            walker.scratch.jobs_cache.insert(rkey, jobs);
            if ok {
              return Ok(());
            }
          }
          let admit_now = !self.continuous && self.comp.frame_count() <= 1;
          if admit_now || walker.scratch.jobs_seen.contains(&rkey) {
            let mut rec = Vec::new();
            let (arena, pending) = walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
            walker.render_shape_jobs_cpu(canvas, &arena, &pending, Some(&mut rec));
            if walker.scratch.jobs_cache.len() >= JOBS_CACHE_CAP {
              walker.scratch.jobs_cache.clear();
            }
            walker.scratch.jobs_cache.insert(rkey, rec);
          } else {
            if walker.scratch.jobs_seen.len() >= 4 * JOBS_CACHE_CAP {
              walker.scratch.jobs_seen.clear();
            }
            walker.scratch.jobs_seen.insert(rkey);
            let (arena, pending) = walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
            walker.render_shape_jobs_cpu(canvas, &arena, &pending, None);
          }
        } else {
          let (arena, pending) = walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
          walker.render_shape_jobs_cpu(canvas, &arena, &pending, None);
        }
      }
      LayerKind::Solid => {
        if let Some((sw, sh, color)) = layer.solid {
          let color = layer.color_override.unwrap_or(color);
          let contour = rect_contour(Vec2::new(sw * 0.5, sh * 0.5), Vec2::new(sw, sh), 0.0, false, &m, self.curve_tolerance);
          let walker = ShapeWalker {
            scratch,
            frame,
            clip,
            curve_tolerance: self.curve_tolerance,
            width: canvas.w,
            height: canvas.h,
            antialias: canvas.antialias,
            color_override: layer.color_override,
            unbounded: false,
          };
          let key = walker.fill_key(core::slice::from_ref(&(contour.clone(), true)), FillRule::NonZero);
          let contours: Vec<Contour> = if walker.scratch.cov_cache.contains(key) { Vec::new() } else { vec![walker.clip_all(&contour)] };
          canvas.fill::<false>(&mut walker.scratch.cov_cache, key, &contours, FillRule::NonZero, color, content_opacity);
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
        // rlottie evaluates precomp children at INTEGER frames:
        // LOTLayerData::timeRemap returns int (lottiemodel.h), so
        // both branches truncate toward zero. The remap branch maps
        // tm's seconds through frameAtPos — pos clamped to [0,1]
        // over frameDuration = op − ip − 1 — and divides by the
        // time stretch (which tlottie previously omitted there).
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
        self.render_layers(scratch, canvas, &asset.layers, m, child_frame, content_opacity, &child_clip, precomp_depth + 1)?;
      }
      LayerKind::Null | LayerKind::Other(_) => {}
    }
    Ok(())
  }

  /// Rasterizes a layer's mask stack into a full-canvas u8 buffer, but only
  /// the `bound` sub-rectangle is computed exactly — every pixel OUTSIDE
  /// `bound` is left with unspecified (stale) contents. The sole callers read
  /// the mask only inside their offscreen's dirty box (`da` for a layer's own
  /// masks, `db` for a matte source's), which they pass as `bound`; the
  /// offscreen is transparent (0) everywhere outside that box, and both
  /// consumers (mask `modulate`, matte `modulate`) map dst 0 → 0, so a mask
  /// value there can never change a pixel. Bounding the three former
  /// full-plane passes (acc init-fill, per-mask `tmp` clear, per-mask
  /// accumulate) to `bound` is therefore byte-exact.
  ///
  /// Mode mapping follows the patched rlottie parser (lottieparser.cpp):
  /// 'a' Add, 's' Subtract, 'i' Intersect, 'f' Difference; 'd', 'l', 'n' and
  /// anything else are None — the mask contributes nothing, and a layer whose
  /// masks are all None draws nothing (empty maskRle).
  ///
  /// Per-mode "outside" value — the mask at a pixel inside `bound` but
  /// OUTSIDE every mask geometry (coverage t=0 for all masks). This is why we
  /// bound to `bound` (the dirty box) and NOT to the mask geometry bbox: the
  /// outside value is generally nonzero and mode-dependent, and the bounded
  /// accumulate reproduces it exactly because t=0 flows through the unchanged
  /// per-pixel body. With `c = round((invert? 255−t : t)·op/255)`, at t=0:
  /// non-inverted → c=0; inverted → c=op (= round(255·op/255)). Folding one
  /// mask over the running `cur`:
  ///   'a' Add   → c + round((255−c)·cur/255)   t0 non-inv: cur   inv: op+…·cur
  ///   's' Sub   → round(cur·(255−c)/255)        t0 non-inv: cur   inv: cur·(255−op)
  ///   'i' Isect → round(cur·c/255)              t0 non-inv: 0     inv: cur·op
  ///   'f' Diff  → |cur − c|                     t0 non-inv: cur   inv: |cur−op|
  /// `first_additive` seeds `cur`: 0 for a leading 'a'/'f' mask, 255 for a
  /// leading 's'/'i'. So a non-inverted intersect (or an all-add stack seeded
  /// at 0) collapses the outside toward 0, while any inverted mask leaves a
  /// nonzero outside — the outside region inside `bound` must be walked, not
  /// assumed. The bounded accumulate over `bound` does exactly that, pixel by
  /// pixel, identically to the former full-plane loop.
  #[allow(clippy::too_many_arguments)]
  #[cfg(test)]
  fn build_mask(&self, scratch: &mut RenderScratch, layer: &Layer, m: Mat2x3, frame: f32, w: usize, h: usize, bound: DirtyBox) -> Vec<u8> {
    let effective = |mode: u8| matches!(mode, b'a' | b's' | b'i' | b'f');
    let first_additive = layer.masks.iter().find(|mask| effective(mask.mode)).map(|mask| matches!(mask.mode, b'a' | b'f')).unwrap_or(true); // all-None: acc stays 0 → layer hidden
                                                                                                                                            // acc/tmp stay full-canvas length (indexed by y*w+x) but only `bound`
                                                                                                                                            // is initialized/cleared/walked — see the note above. Outside `bound`
                                                                                                                                            // they hold stale pool bytes that no consumer ever reads.
    let init = if first_additive { 0u8 } else { 255u8 };
    let mut acc: Vec<u8> = scratch.take_u8_uninit(w * h);
    fill_rows_u8(&mut acc, w, bound, init);
    let mut raster = scratch.take_raster(w, h);
    let mut cells = scratch.take_cells(w, h);
    let mut tmp: Vec<u8> = scratch.take_u8_uninit(w * h);
    for mask in &layer.masks {
      if !effective(mask.mode) {
        continue;
      }
      let opacity = (mask.opacity.eval(frame) / 100.0).clamp(0.0, 1.0);
      let data = mask.path.eval(frame);
      let contour = flatten_path(&data, &m, self.curve_tolerance);
      let clipped = clip_contour(&contour, w as f32, h as f32);
      // Clear only `bound`; the rasterizer sweep below overwrites the
      // mask's covered pixels, leaving `bound`-but-uncovered pixels at 0
      // (coverage t=0), exactly as the former `tmp.fill(0)` did there.
      fill_rows_u8(&mut tmp, w, bound, 0);
      if mode_s_wins(core::slice::from_ref(&clipped), w * h) {
        cells.reset();
        cells.fill_contours(core::slice::from_ref(&clipped));
        cells.sweep_spans(FillRule::NonZero, self.antialias, |y, x0, len, cov| {
          let lo = y * w + x0;
          if let Some(dst) = tmp.get_mut(lo..lo + len) {
            dst.fill(cov);
          }
        });
      } else {
        raster.reset();
        raster.fill_contours(core::slice::from_ref(&clipped));
        raster.sweep(FillRule::NonZero, self.antialias, |y, x0, cov_row| {
          let lo = y * w + x0;
          if let Some(dst) = tmp.get_mut(lo..lo + cov_row.len()) {
            dst.copy_from_slice(cov_row);
          }
        });
      }
      let op = (opacity * 255.0 + 0.5) as u32;
      // Full-plane accumulate, bounded to `bound`: the per-pixel body is
      // byte-identical to the former loop — only its iteration range
      // shrinks from w*h to the dirty box.
      for y in bound.y0..=bound.y1 {
        let lo = y * w + bound.x0;
        let hi = y * w + bound.x1 + 1;
        let (Some(acc_row), Some(tmp_row)) = (acc.get_mut(lo..hi), tmp.get(lo..hi)) else {
          continue;
        };
        for (a, &t) in acc_row.iter_mut().zip(tmp_row.iter()) {
          let mut c = u32::from(t);
          if mask.invert {
            c = 255 - c;
          }
          c = (c * op + 127) / 255;
          let cur = u32::from(*a);
          let next = match mask.mode {
            b's' => (cur * (255 - c) + 127) / 255,
            b'i' => (cur * c + 127) / 255,
            b'f' => cur.abs_diff(c), // Difference (XOR-like)
            // 'a' Add combines with SrcOver (rlottie blitSrcOver:
            // b + (255−b)·a/255), NOT a saturating sum — overlapping
            // partial-coverage masks differ (190 vs 255).
            _ => c + ((255 - c) * cur + 127) / 255,
          };
          *a = next as u8;
        }
      }
    }
    scratch.put_u8(tmp);
    scratch.put_raster(raster);
    scratch.put_cells(cells);
    acc
  }
}

/// Fills the `bound` sub-rectangle of a `w`-stride u8 plane with `v`. Seeds
/// only the region build_mask actually walks (the offscreen dirty box),
/// replacing the former full-plane `take_u8` fill and per-mask `tmp.fill(0)`.
#[cfg(test)]
fn fill_rows_u8(buf: &mut [u8], w: usize, b: DirtyBox, v: u8) {
  if b.is_empty() {
    return;
  }
  for y in b.y0..=b.y1 {
    let lo = y * w + b.x0;
    let hi = (y * w + b.x1 + 1).min(buf.len());
    if lo >= hi {
      continue;
    }
    if let Some(row) = buf.get_mut(lo..hi) {
      row.fill(v);
    }
  }
}

/// Multiplies premultiplied RGBA pixels by a u8 coverage buffer.
/// Calls `f(y, row)` for each row of the dirty box, with `row` being the
/// box's column span `[b.x0, b.x1]` of that row.
#[cfg(test)]
fn for_rows_boxed(pixels: &mut [u32], w: usize, b: DirtyBox, mut f: impl FnMut(usize, &mut [u32])) {
  if b.is_empty() {
    return;
  }
  for y in b.y0..=b.y1 {
    let lo = y * w + b.x0;
    let hi = (y * w + b.x1 + 1).min(pixels.len());
    if lo >= hi {
      continue;
    }
    if let Some(row) = pixels.get_mut(lo..hi) {
      f(y, row);
    }
  }
}

pub(crate) fn modulate(pixels: &mut [u32], mask: &[u8]) {
  for (px, &mk) in pixels.iter_mut().zip(mask.iter()) {
    if mk == 255 {
      continue;
    }
    if mk == 0 {
      *px = 0;
      continue;
    }
    let m = u32::from(mk);
    let p = *px;
    let a = (((p >> 24) & 0xff) * m + 127) / 255;
    let r = (((p >> 16) & 0xff) * m + 127) / 255;
    let g = (((p >> 8) & 0xff) * m + 127) / 255;
    let b = ((p & 0xff) * m + 127) / 255;
    *px = (a << 24) | (r << 16) | (g << 8) | b;
  }
}

/// Applies a matte source (`src`) onto `dst` premultiplied pixels.
/// kind: 1 alpha, 2 inverted alpha, 3 luma, 4 inverted luma.
pub(crate) fn apply_matte(dst: &mut [u32], src: &[u32], kind: u8, source_opacity: u8, order: crate::ChannelOrder) {
  if kind == 1 || kind == 2 {
    crate::simd::apply_matte_alpha(dst, src, source_opacity, kind == 2);
    return;
  }
  for (d, &s) in dst.iter_mut().zip(src.iter()) {
    let scaled_alpha = luma_premult(s, order);
    let factor = if kind == 3 { scaled_alpha } else { 255 - scaled_alpha };
    if factor == 255 {
      continue;
    }
    if factor == 0 {
      *d = 0;
      continue;
    }
    let p = *d;
    let a = (((p >> 24) & 0xff) * factor + 127) / 255;
    let r = (((p >> 16) & 0xff) * factor + 127) / 255;
    let g = (((p >> 8) & 0xff) * factor + 127) / 255;
    let b = ((p & 0xff) * factor + 127) / 255;
    *d = (a << 24) | (r << 16) | (g << 8) | b;
  }
}

/// Luma of a premultiplied pixel, rlottie semantics: unpremultiply, then
/// Rec.601 weights on the straight color (the matte's own alpha does not
/// scale the luma).
///
/// The ONLY place in the pipeline where a channel's identity matters, since
/// the Rec.601 weights differ per channel: under [`ChannelOrder::Bgra`] the
/// low byte holds blue and bits 16..24 hold red, so the two weights trade
/// places. Everything else blends channels independently and needs no such
/// knowledge.
fn luma_premult(p: u32, order: crate::ChannelOrder) -> u32 {
  let a = (p >> 24) & 0xff;
  if a == 0 {
    return 0;
  }
  let mut low = p & 0xff;
  let mut g = (p >> 8) & 0xff;
  let mut high = (p >> 16) & 0xff;
  if a != 255 {
    low = (low * 255) / a;
    g = (g * 255) / a;
    high = (high * 255) / a;
  }
  let (low_weight, high_weight) = match order {
    crate::ChannelOrder::Rgba => (299, 114),
    crate::ChannelOrder::Bgra => (114, 299),
  };
  ((low * low_weight + g * 587 + high * high_weight) / 1000).min(255)
}

/// Composites premultiplied `src` over `dst` with a global opacity factor.
/// Combined matrix of all ancestors of `layer` within `layers` (not
/// including the layer itself). Cycle-safe: walks at most `layers.len()`.
pub(crate) fn parent_chain_matrix(layers: &[Layer], layer: &Layer, frame: f32) -> Mat2x3 {
  let mut chain: Vec<Mat2x3> = Vec::new();
  let mut current = layer.parent;
  let mut steps = 0usize;
  while let Some(parent_ind) = current {
    steps += 1;
    if steps > layers.len() {
      break; // cycle; refuse to loop forever
    }
    let Some(parent) = layers.iter().find(|l| l.index == parent_ind) else {
      break;
    };
    let (m, _) = transform_at(&parent.transform, frame);
    chain.push(m);
    current = parent.parent;
  }
  let mut result = Mat2x3::IDENTITY;
  for m in chain.iter().rev() {
    result = result.concat(*m);
  }
  result
}

/// Layer-level transform: adds auto-orient rotation from the position path
/// derivative when the layer requests it.
pub(crate) fn layer_transform_at(layer: &Layer, frame: f32) -> (Mat2x3, f32) {
  let (m, opacity) = transform_at(&layer.transform, frame);
  if !layer.auto_orient {
    return (m, opacity);
  }
  let before = layer.transform.position.eval(frame - 0.5);
  let after = layer.transform.position.eval(frame + 0.5);
  let dx = after.x - before.x;
  let dy = after.y - before.y;
  if dx * dx + dy * dy < 1e-9 {
    return (m, opacity);
  }
  let angle = dy.atan2(dx).to_degrees();
  // Auto-orient rotates around the anchor, i.e. composes like rotation:
  // re-apply an extra rotation between position and the rest.
  layer.transform.anchor.eval(frame);
  let pos = layer.transform.position.eval(frame);
  let extra = Mat2x3::translate(pos.x, pos.y).concat(Mat2x3::rotate(angle)).concat(Mat2x3::translate(-pos.x, -pos.y));
  (extra.concat(m), opacity)
}

/// Evaluates a Transform at a frame into (matrix, opacity 0..=1).
fn transform_at(tf: &Transform, frame: f32) -> (Mat2x3, f32) {
  let anchor = tf.anchor.eval(frame);
  let position = tf.position.eval(frame);
  let scale = tf.scale.eval(frame);
  let rotation = tf.rotation.eval(frame);
  let opacity = (tf.opacity.eval(frame) / 100.0).clamp(0.0, 1.0);
  // NOTE: skew (sk/sa) is parsed but NOT applied — rlottie (patched and
  // upstream) never reads those fields; its matrix is translate·rotate·
  // scale·translate(−anchor) only. Applying AE-correct shear diverged on
  // every skewed file (contract review: transforms).
  let m = Mat2x3::translate(position.x, position.y)
    .concat(Mat2x3::rotate(rotation))
    .concat(Mat2x3::scale(scale.x / 100.0, scale.y / 100.0))
    .concat(Mat2x3::translate(-anchor.x, -anchor.y));
  (m, opacity)
}

pub(crate) fn opacity_byte(opacity: f32) -> u32 {
  (opacity.clamp(0.0, 1.0) * 255.0) as u32
}

#[path = "canvas.rs"]
mod canvas;
pub(crate) use canvas::{mark_row_bounds, Canvas, DirtyBox, RowBounds};

#[path = "gradient.rs"]
mod gradient;
use gradient::*;
pub(crate) use gradient::{GradientMap, GradientMapKind};

#[path = "../frame/evaluator.rs"]
mod evaluator;
pub(crate) use evaluator::{DrawJob, PendingJob, ShapeWalker};

#[cfg(test)]
#[path = "tests/cov_cache.rs"]
mod cov_freeze_tests;

#[cfg(test)]
#[path = "tests/mask_bounds.rs"]
mod mask_bound_tests;
