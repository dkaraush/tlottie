//! CPU implementation of renderer-neutral frame operations.

#![allow(unsafe_code)]

use crate::model::FillRule;
use crate::renderer::frame::{Composite, FrameRenderer, Geometry, GradientKind, GradientPaint, Paint, Rule};
use alloc::vec::Vec;

use super::executor::{apply_matte, mode_s_wins, modulate, Canvas, CovCache, DirtyBox, GradientMap, GradientMapKind, RowBounds};
use super::CPURenderer;

struct BitmapReset<'a>(&'a mut CPURenderer);

impl Drop for BitmapReset<'_> {
  fn drop(&mut self) {
    self.0.bitmap = None;
    while let Some(surface) = self.0.surfaces.pop() {
      let _dirty = self.0.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
      let rows = self.0.surface_rows.pop().unwrap_or_default();
      self.0.state.put_surface_u32(surface, self.0.width, &rows);
      self.0.row_bounds_pool.push(rows);
    }
    self.0.bitmap_dirty = false;
    if let Some(mask) = self.0.mask_accumulator.take() {
      self.0.state.put_u8(mask);
    }
  }
}

impl CPURenderer {
  /// Temporarily binds a caller-owned bitmap while `render` streams drawing
  /// operations into this renderer.
  pub(super) fn with_bitmap<T>(&mut self, pixels: &mut [u32], width: u32, height: u32, options: crate::RenderOptions, render: impl FnOnce(&mut Self) -> crate::Result<T>) -> crate::Result<T> {
    if self.bitmap.is_some() {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "CPU renderer already has a bound bitmap",
      });
    }
    let Some(expected) = (width as usize).checked_mul(height as usize) else {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "render size overflow",
      });
    };
    let Some(target) = pixels.get_mut(..expected) else {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "pixel buffer too small",
      });
    };
    if options.clear {
      target.fill(0);
    }
    self.width = width as usize;
    self.height = height as usize;
    self.antialias = options.antialias;
    self.alpha_only = options.alpha_only;
    // Existing pixels are destination content. Marking them dirty makes
    // empty-target fast paths use the same source-over compositors as
    // layers, masks, and mattes.
    self.bitmap_dirty = !options.clear;
    self.state.cov_cache.set_budget_for_canvas(self.width, self.height);
    self.state.cov_cache.frame_tick();
    self.bitmap = Some(core::ptr::NonNull::from(target));
    let reset = BitmapReset(self);
    render(&mut *reset.0)
  }

  fn active(&mut self) -> &mut [u32] {
    match self.surfaces.last_mut() {
      Some(surface) => surface,
      None => match self.bitmap {
        Some(mut bitmap) => {
          // SAFETY: with_bitmap holds the source slice mutably for the entire
          // callback and BitmapReset clears the pointer before that borrow ends.
          unsafe { bitmap.as_mut() }
        }
        None => &mut [],
      },
    }
  }

  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>) {
    let destination_dirty = self.surface_dirty.last().is_some_and(|bounds| !bounds.is_empty()) || (self.surface_dirty.is_empty() && self.bitmap_dirty);
    let key = geometry.cache_key;
    let contours = geometry.raw_contours();
    let translation = geometry.raw_translation();
    let retained = paint_is_retained(&self.state.cov_cache, key, paint, self.alpha_only);
    let work = (!retained).then(|| (self.state.take_raster(self.width, self.height), self.state.take_cells(self.width, self.height)));
    let scratch = &mut self.state;
    let pixels = match self.surfaces.last_mut() {
      Some(surface) => surface.as_mut_slice(),
      None => match self.bitmap {
        Some(mut bitmap) => {
          // SAFETY: the bitmap is bound for the duration of with_bitmap.
          unsafe { bitmap.as_mut() }
        }
        None => return,
      },
    };
    let dirty_rows = self.surface_rows.last_mut().map(Vec::as_mut_slice);
    let track_rows = dirty_rows.is_some();
    let mut canvas = match work {
      Some((raster, cells)) => Canvas::with_raster_and_rows(pixels, self.width, self.height, raster, cells, self.antialias, dirty_rows),
      None => Canvas::with_retained_rows(pixels, self.width, self.height, self.antialias, dirty_rows),
    };
    if destination_dirty {
      // `Canvas` uses an empty dirty box to select a gradient copy fast
      // path. It is recreated for each streamed command, so carry the
      // destination's content state across commands explicitly.
      canvas.dirty.mark_row(0, 0, 1);
    }
    match (paint, track_rows) {
      (Paint::Solid(solid), true) => canvas.fill_translated::<true>(
        &mut scratch.cov_cache,
        key,
        contours,
        translation,
        fill_rule(solid.rule),
        alpha_color(solid.color, self.alpha_only),
        solid.opacity,
      ),
      (Paint::Solid(solid), false) => canvas.fill_translated::<false>(
        &mut scratch.cov_cache,
        key,
        contours,
        translation,
        fill_rule(solid.rule),
        alpha_color(solid.color, self.alpha_only),
        solid.opacity,
      ),
      (Paint::Gradient(gradient), true) => {
        let map = gradient_map(gradient);
        fill_gradient::<true>(&mut canvas, &mut scratch.cov_cache, key, contours, translation, gradient, &map, self.alpha_only);
      }
      (Paint::Gradient(gradient), false) => {
        let map = gradient_map(gradient);
        fill_gradient::<false>(&mut canvas, &mut scratch.cov_cache, key, contours, translation, gradient, &map, self.alpha_only);
      }
    }
    let draw_dirty = canvas.dirty;
    if let Some(raster) = canvas.raster.take() {
      scratch.put_raster(raster);
    }
    if let Some(cells) = canvas.cells.take() {
      scratch.put_cells(cells);
    }
    if let Some(dirty) = self.surface_dirty.last_mut() {
      dirty.union(draw_dirty);
    } else {
      self.bitmap_dirty |= !draw_dirty.is_empty();
    }
  }

  fn end_layer(&mut self, composite: Composite) {
    match composite {
      Composite::Over { opacity } => {
        let Some(source) = self.surfaces.pop() else {
          return;
        };
        let source_dirty = self.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
        let source_rows = self.surface_rows.pop().unwrap_or_default();
        let width = self.width;
        composite_over_rows(self.active(), &source, width, &source_rows, source_dirty, opacity);
        if !source_dirty.is_empty() {
          self.mark_active_dirty(source_dirty, &source_rows);
        }
        self.state.put_surface_u32(source, self.width, &source_rows);
        self.row_bounds_pool.push(source_rows);
      }
      Composite::Matte { kind, opacity, source_opacity } => {
        let Some(mut target) = self.surfaces.pop() else {
          return;
        };
        let target_dirty = self.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
        let target_rows = self.surface_rows.pop().unwrap_or_default();
        let Some(source) = self.surfaces.pop() else {
          return;
        };
        let _source_dirty = self.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
        let source_rows = self.surface_rows.pop().unwrap_or_default();
        apply_matte(&mut target, &source, kind, source_opacity, self.comp.channel_order);
        let width = self.width;
        composite_over_rows(self.active(), &target, width, &target_rows, target_dirty, opacity);
        if !target_dirty.is_empty() {
          self.mark_active_dirty(target_dirty, &target_rows);
        }
        self.state.put_surface_u32(target, self.width, &target_rows);
        self.state.put_surface_u32(source, self.width, &source_rows);
        self.row_bounds_pool.push(target_rows);
        self.row_bounds_pool.push(source_rows);
      }
    }
  }

  fn mark_active_dirty(&mut self, bounds: DirtyBox, rows: &[RowBounds]) {
    if let Some(dirty) = self.surface_dirty.last_mut() {
      dirty.union(bounds);
      if let Some(active_rows) = self.surface_rows.last_mut() {
        for (active, &source) in active_rows.iter_mut().zip(rows) {
          if !source.is_empty() {
            active.mark(source.x0, source.x1.saturating_add(1));
          }
        }
      }
    } else {
      self.bitmap_dirty = true;
    }
  }

  fn apply_mask(&mut self, geometry: crate::renderer::frame::Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    let len = self.width.saturating_mul(self.height);
    // A single mask needs no accumulator: its final factor is just its
    // coverage (add/intersect/difference) or inverse coverage (subtract),
    // after inversion and opacity. Restrict modulation to the active
    // layer's tracked row spans instead of scanning the full canvas. Small
    // animated masks on a 720px surface otherwise spend most of the frame
    // initializing, combining, and applying three full-size mask planes.
    if first && last {
      let mut active_rows = self.surface_rows.pop();
      if let Some(rows) = active_rows.as_deref_mut() {
        if mode_s_wins(geometry.raw_contours(), len) {
          let mut cells = self.state.take_cells(self.width, self.height);
          cells.reset();
          cells.fill_contours(geometry.raw_contours());
          let width = self.width;
          let antialias = self.antialias;
          modulate_single_mask_spans(self.active(), &mut cells, width, rows, mode, inverted, opacity, antialias);
          self.state.put_cells(cells);
          self.surface_rows.push(active_rows.expect("checked above"));
          return;
        }
      }
      let mut coverage = if let Some(rows) = active_rows.as_deref() {
        let mut coverage = self.state.take_u8_uninit(len);
        clear_mask_rows(&mut coverage, self.width, rows);
        coverage
      } else {
        self.state.take_u8(len, 0)
      };
      let width = self.width;
      if mode_s_wins(geometry.raw_contours(), len) {
        let mut cells = self.state.take_cells(self.width, self.height);
        cells.reset();
        cells.fill_contours(geometry.raw_contours());
        cells.sweep_spans(FillRule::NonZero, self.antialias, |y, x0, span_len, value| {
          let lo = y.saturating_mul(width).saturating_add(x0);
          if let Some(dst) = coverage.get_mut(lo..lo.saturating_add(span_len)) {
            dst.fill(value);
          }
        });
        self.state.put_cells(cells);
      } else {
        let mut raster = self.state.take_raster(self.width, self.height);
        raster.reset();
        raster.fill_contours(geometry.raw_contours());
        raster.sweep(FillRule::NonZero, self.antialias, |y, x0, row| {
          let lo = y.saturating_mul(width).saturating_add(x0);
          if let Some(dst) = coverage.get_mut(lo..lo.saturating_add(row.len())) {
            dst.copy_from_slice(row);
          }
        });
        self.state.put_raster(raster);
      }
      if let Some(rows) = active_rows {
        modulate_single_mask_rows(self.active(), &mut coverage, width, &rows, mode, inverted, opacity);
        self.surface_rows.push(rows);
      } else {
        prepare_single_mask(&mut coverage, mode, inverted, opacity);
        modulate(self.active(), &coverage);
      }
      self.state.put_u8(coverage);
      return;
    }
    if first || self.mask_accumulator.is_none() {
      let initial = if matches!(mode, b'a' | b'f') { 0 } else { 255 };
      if let Some(previous) = self.mask_accumulator.take() {
        self.state.put_u8(previous);
      }
      self.mask_accumulator = Some(self.state.take_u8(len, initial));
    }
    let mut coverage = self.state.take_u8(len, 0);
    let width = self.width;
    if mode_s_wins(geometry.raw_contours(), len) {
      let mut cells = self.state.take_cells(self.width, self.height);
      cells.reset();
      cells.fill_contours(geometry.raw_contours());
      cells.sweep_spans(FillRule::NonZero, self.antialias, |y, x0, span_len, value| {
        let lo = y.saturating_mul(width).saturating_add(x0);
        if let Some(dst) = coverage.get_mut(lo..lo.saturating_add(span_len)) {
          dst.fill(value);
        }
      });
      self.state.put_cells(cells);
    } else {
      let mut raster = self.state.take_raster(self.width, self.height);
      raster.reset();
      raster.fill_contours(geometry.raw_contours());
      raster.sweep(FillRule::NonZero, self.antialias, |y, x0, row| {
        let lo = y.saturating_mul(width).saturating_add(x0);
        if let Some(dst) = coverage.get_mut(lo..lo.saturating_add(row.len())) {
          dst.copy_from_slice(row);
        }
      });
      self.state.put_raster(raster);
    }
    if let Some(accumulator) = self.mask_accumulator.as_mut() {
      for (current, &sample) in accumulator.iter_mut().zip(&coverage) {
        let mut contribution = u32::from(sample);
        if inverted {
          contribution = 255 - contribution;
        }
        contribution = (contribution * u32::from(opacity) + 127) / 255;
        let old = u32::from(*current);
        *current = match mode {
          b's' => ((old * (255 - contribution) + 127) / 255) as u8,
          b'i' => ((old * contribution + 127) / 255) as u8,
          b'f' => old.abs_diff(contribution) as u8,
          _ => (contribution + ((255 - contribution) * old + 127) / 255) as u8,
        };
      }
    }
    self.state.put_u8(coverage);
    if last {
      if let Some(mask) = self.mask_accumulator.take() {
        modulate(self.active(), &mask);
        self.state.put_u8(mask);
      }
    }
  }
}

fn clear_mask_rows(mask: &mut [u8], width: usize, rows: &[RowBounds]) {
  if width == 0 {
    return;
  }
  let height = mask.len() / width;
  for (y, &bounds) in rows.iter().take(height).enumerate() {
    if bounds.is_empty() {
      continue;
    }
    let x0 = bounds.x0.min(width);
    let x1 = bounds.x1.saturating_add(1).min(width);
    if x0 < x1 {
      if let Some(row) = mask.get_mut(y * width + x0..y * width + x1) {
        row.fill(0);
      }
    }
  }
}

fn prepare_single_mask(mask: &mut [u8], mode: u8, inverted: bool, opacity: u8) {
  for sample in mask {
    *sample = single_mask_factor(*sample, mode, inverted, opacity);
  }
}

#[inline]
fn single_mask_factor(sample: u8, mode: u8, inverted: bool, opacity: u8) -> u8 {
  let contribution = if inverted { 255 - u32::from(sample) } else { u32::from(sample) };
  let contribution = (contribution * u32::from(opacity) + 127) / 255;
  if mode == b's' {
    (255 - contribution) as u8
  } else {
    contribution as u8
  }
}

fn modulate_single_mask_spans(pixels: &mut [u32], cells: &mut super::cells::CellRaster, width: usize, rows: &mut [RowBounds], mode: u8, inverted: bool, opacity: u8, antialias: bool) {
  if width == 0 {
    return;
  }
  let outside = single_mask_factor(0, mode, inverted, opacity);
  let mut cursor_y = usize::MAX;
  let mut cursor = 0usize;
  let mut kept_x0 = usize::MAX;
  let mut kept_x1 = 0usize;
  cells.sweep_span_rows(FillRule::NonZero, antialias, |y, span| {
    let Some(&bounds) = rows.get(y) else {
      return;
    };
    if bounds.is_empty() {
      return;
    }
    let row_x0 = bounds.x0.min(width);
    let row_x1 = bounds.x1.saturating_add(1).min(width);
    if row_x0 >= row_x1 {
      return;
    }
    if cursor_y != y {
      cursor_y = y;
      cursor = row_x0;
      kept_x0 = usize::MAX;
      kept_x1 = 0;
    }
    if let Some((span_x0, span_len, coverage)) = span {
      let span_x1 = span_x0.saturating_add(span_len).min(row_x1);
      let span_x0 = span_x0.max(row_x0).min(span_x1);
      if cursor < span_x0 {
        modulate_uniform_range(pixels, width, y, cursor, span_x0, outside);
      }
      let paint_x0 = cursor.max(span_x0);
      if paint_x0 < span_x1 {
        let factor = single_mask_factor(coverage, mode, inverted, opacity);
        modulate_uniform_range(pixels, width, y, paint_x0, span_x1, factor);
        if factor != 0 {
          kept_x0 = kept_x0.min(paint_x0);
          kept_x1 = kept_x1.max(span_x1);
        }
        cursor = span_x1;
      }
    } else {
      if cursor < row_x1 {
        modulate_uniform_range(pixels, width, y, cursor, row_x1, outside);
      }
      if outside == 0 {
        if let Some(bounds) = rows.get_mut(y) {
          *bounds = if kept_x0 < kept_x1 { RowBounds { x0: kept_x0, x1: kept_x1 - 1 } } else { RowBounds::empty() };
        }
      }
      cursor_y = usize::MAX;
    }
  });
}

#[inline]
fn modulate_uniform_range(pixels: &mut [u32], width: usize, y: usize, x0: usize, x1: usize, factor: u8) {
  if x0 >= x1 || factor == 255 {
    return;
  }
  let lo = y.saturating_mul(width).saturating_add(x0);
  let hi = y.saturating_mul(width).saturating_add(x1);
  let Some(span) = pixels.get_mut(lo..hi) else {
    return;
  };
  if factor == 0 {
    span.fill(0);
    return;
  }
  let mask = [factor];
  for pixel in span {
    modulate(core::slice::from_mut(pixel), &mask);
  }
}

fn modulate_single_mask_rows(pixels: &mut [u32], mask: &mut [u8], width: usize, rows: &[RowBounds], mode: u8, inverted: bool, opacity: u8) {
  if width == 0 {
    return;
  }
  let height = pixels.len().min(mask.len()) / width;
  for (y, &bounds) in rows.iter().take(height).enumerate() {
    if bounds.is_empty() {
      continue;
    }
    let x0 = bounds.x0.min(width);
    let x1 = bounds.x1.saturating_add(1).min(width);
    if x0 >= x1 {
      continue;
    }
    let start = y * width + x0;
    let end = y * width + x1;
    if let Some(mask_row) = mask.get_mut(start..end) {
      prepare_single_mask(mask_row, mode, inverted, opacity);
    }
    if let (Some(pixel_row), Some(mask_row)) = (pixels.get_mut(start..end), mask.get(start..end)) {
      modulate(pixel_row, mask_row);
    }
  }
}

fn alpha_color(mut color: crate::math::Color, alpha_only: bool) -> crate::math::Color {
  if alpha_only {
    color.r = 0.0;
    color.g = 0.0;
    color.b = 0.0;
  }
  color
}

fn paint_is_retained(cache: &CovCache, key: u128, paint: Paint<'_>, alpha_only: bool) -> bool {
  if cache.contains_coverage(key) {
    return true;
  }
  let Paint::Gradient(gradient) = paint else {
    return false;
  };
  if !alpha_only {
    return cache.contains(gradient.source_key);
  }
  let first_alpha = gradient.lut.first().copied().unwrap_or(0) >> 24;
  if gradient.lut.iter().all(|pixel| pixel >> 24 == first_alpha) {
    false
  } else {
    cache.contains(gradient.source_key ^ (1u128 << 127))
  }
}

fn fill_gradient<const TRACK_ROWS: bool>(
  canvas: &mut Canvas<'_>,
  cache: &mut CovCache,
  key: u128,
  contours: &[crate::geometry::Contour],
  translation: crate::renderer::frame::Point,
  gradient: &GradientPaint,
  map: &GradientMap,
  alpha_only: bool,
) {
  if !alpha_only {
    if gradient.alpha < 255 {
      canvas.fill_gradient_translated_alpha::<TRACK_ROWS>(cache, key, gradient.source_key, contours, translation, fill_rule(gradient.rule), &gradient.lut, map, gradient.alpha);
      return;
    }
    canvas.fill_gradient_translated::<TRACK_ROWS>(cache, key, gradient.source_key, contours, translation, fill_rule(gradient.rule), &gradient.lut, map);
    return;
  }

  let first_alpha = gradient.lut.first().copied().unwrap_or(0) >> 24;
  if gradient.lut.iter().all(|pixel| pixel >> 24 == first_alpha) {
    let color = crate::math::Color {
      r: 0.0,
      g: 0.0,
      b: 0.0,
      a: first_alpha as f32 / 255.0 * f32::from(gradient.alpha) / 255.0,
    };
    canvas.fill_translated::<TRACK_ROWS>(cache, key, contours, translation, fill_rule(gradient.rule), color, 1.0);
    return;
  }

  let mut alpha_lut = [0u32; crate::renderer::frame::GRADIENT_LUT_SIZE];
  if gradient.alpha < 255 {
    let k = u32::from(gradient.alpha);
    for (out, &pixel) in alpha_lut.iter_mut().zip(gradient.lut.iter()) {
      *out = ((pixel >> 24) * k / 255) << 24;
    }
    let plane_key = gradient.source_key ^ (1u128 << 127) ^ (u128::from(gradient.alpha) << 32);
    canvas.fill_gradient_translated::<TRACK_ROWS>(cache, key, plane_key, contours, translation, fill_rule(gradient.rule), &alpha_lut, map);
    return;
  }
  for (out, &pixel) in alpha_lut.iter_mut().zip(gradient.lut.iter()) {
    *out = pixel & 0xff00_0000;
  }
  canvas.fill_gradient_translated::<TRACK_ROWS>(cache, key, gradient.source_key ^ (1u128 << 127), contours, translation, fill_rule(gradient.rule), &alpha_lut, map);
}

impl FrameRenderer for CPURenderer {
  fn save_layer(&mut self) {
    let layer = self.state.take_surface_u32(self.width.saturating_mul(self.height));
    let mut rows = self.row_bounds_pool.pop().unwrap_or_default();
    rows.clear();
    rows.resize(self.height, RowBounds::empty());
    self.surfaces.push(layer);
    self.surface_dirty.push(DirtyBox::empty());
    self.surface_rows.push(rows);
  }

  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>) {
    self.draw(geometry, paint);
  }

  fn apply_mask(&mut self, geometry: Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    self.apply_mask(geometry, mode, inverted, opacity, first, last);
  }

  fn end_layer(&mut self, composite: Composite) {
    self.end_layer(composite);
  }

  fn retains_geometry(&self, cache_key: u128) -> bool {
    self.state.cov_cache.contains(cache_key)
  }
}

fn composite_over_rows(destination: &mut [u32], source: &[u32], width: usize, rows: &[RowBounds], bounds: DirtyBox, opacity: u8) {
  if bounds.is_empty() || width == 0 {
    return;
  }
  let height = destination.len().min(source.len()) / width;
  let y0 = bounds.y0.min(height).min(rows.len());
  let y1 = bounds.y1.saturating_add(1).min(height).min(rows.len());
  for y in y0..y1 {
    let row_bounds = rows[y];
    if row_bounds.is_empty() {
      continue;
    }
    let x0 = row_bounds.x0.min(width);
    let x1 = row_bounds.x1.saturating_add(1).min(width);
    if x0 >= x1 {
      continue;
    }
    let row = y * width;
    crate::simd::composite_over_span(&mut destination[row + x0..row + x1], &source[row + x0..row + x1], u32::from(opacity));
  }
}

#[cfg(test)]
fn composite_over_box(destination: &mut [u32], source: &[u32], width: usize, bounds: DirtyBox, opacity: u8) {
  if bounds.is_empty() || width == 0 {
    return;
  }
  let height = destination.len().min(source.len()) / width;
  let x0 = bounds.x0.min(width);
  let x1 = bounds.x1.saturating_add(1).min(width);
  let y0 = bounds.y0.min(height);
  let y1 = bounds.y1.saturating_add(1).min(height);
  if x0 >= x1 || y0 >= y1 {
    return;
  }
  for y in y0..y1 {
    let row = y * width;
    let src_row = &source[row + x0..row + x1];
    // A DirtyBox has one shared x range, but isolated vector layers can be
    // much narrower on individual rows. Keep the dense SIMD compositor while
    // excluding each row's transparent margins.
    let Some(first) = src_row.iter().position(|&pixel| pixel != 0) else {
      continue;
    };
    let last = src_row.iter().rposition(|&pixel| pixel != 0).unwrap_or(first) + 1;
    crate::simd::composite_over_span(&mut destination[row + x0 + first..row + x0 + last], &src_row[first..last], u32::from(opacity));
  }
}

#[cfg(test)]
mod composite_tests {
  use super::*;

  fn composite_over_box_reference(destination: &mut [u32], source: &[u32], width: usize, bounds: DirtyBox, opacity: u8) {
    if bounds.is_empty() || width == 0 {
      return;
    }
    let height = destination.len().min(source.len()) / width;
    let x0 = bounds.x0.min(width);
    let x1 = bounds.x1.saturating_add(1).min(width);
    let y0 = bounds.y0.min(height);
    let y1 = bounds.y1.saturating_add(1).min(height);
    if x0 >= x1 || y0 >= y1 {
      return;
    }
    for y in y0..y1 {
      let row = y * width;
      crate::simd::composite_over_span(&mut destination[row + x0..row + x1], &source[row + x0..row + x1], u32::from(opacity));
    }
  }

  #[test]
  fn composite_over_box_row_trimming_matches_full_rows() {
    let width = 24;
    let height = 5;
    let bounds = DirtyBox { x0: 2, y0: 1, x1: 21, y1: 4 };
    let mut source = vec![0; width * height];
    source[width + 8] = 0x8040_2010;
    source[width + 16] = 0xff10_2030;
    source[2 * width + 2..2 * width + 22].fill(0x4020_1008);
    source[4 * width + 20] = 0x0101_0000;
    let original: Vec<u32> = (0..width * height).map(|i| 0xff00_0000 | ((i as u32 * 0x0101_01) & 0x00ff_ffff)).collect();
    for opacity in [1, 17, 128, 254, 255] {
      let mut expected = original.clone();
      let mut actual = original.clone();
      composite_over_box_reference(&mut expected, &source, width, bounds, opacity);
      composite_over_box(&mut actual, &source, width, bounds, opacity);
      assert_eq!(actual, expected, "opacity={opacity}");
    }
  }

  #[test]
  fn composite_over_recorded_rows_matches_full_rows() {
    let width = 24;
    let height = 5;
    let bounds = DirtyBox { x0: 2, y0: 1, x1: 21, y1: 4 };
    let mut source = vec![0; width * height];
    source[width + 8] = 0x8040_2010;
    source[width + 16] = 0xff10_2030;
    source[2 * width + 2..2 * width + 22].fill(0x4020_1008);
    source[4 * width + 20] = 0x0101_0000;
    let mut rows = vec![RowBounds::empty(); height];
    for y in 0..height {
      let row = &source[y * width..(y + 1) * width];
      if let Some(first) = row.iter().position(|&pixel| pixel != 0) {
        let last = row.iter().rposition(|&pixel| pixel != 0).unwrap_or(first) + 1;
        rows[y].mark(first, last);
      }
    }
    let original = vec![0xff20_4060; width * height];
    for opacity in [1, 17, 128, 254, 255] {
      let mut expected = original.clone();
      let mut actual = original.clone();
      composite_over_box_reference(&mut expected, &source, width, bounds, opacity);
      composite_over_rows(&mut actual, &source, width, &rows, bounds, opacity);
      assert_eq!(actual, expected, "opacity={opacity}");
    }
  }
}

fn fill_rule(rule: Rule) -> FillRule {
  match rule {
    Rule::NonZero => FillRule::NonZero,
    Rule::EvenOdd => FillRule::EvenOdd,
  }
}

fn gradient_map(paint: &GradientPaint) -> GradientMap {
  let transform = paint.transform;
  GradientMap {
    inv: crate::math::Mat2x3 {
      a: transform.a,
      b: transform.b,
      c: transform.c,
      d: transform.d,
      tx: transform.tx,
      ty: transform.ty,
    },
    kind: match paint.kind {
      GradientKind::Linear { sx, sy, dx, dy, inv_len_sq } => GradientMapKind::Linear { sx, sy, dx, dy, inv_len_sq },
      GradientKind::Radial { sx, sy, inv_r } => GradientMapKind::Radial { sx, sy, inv_r },
      GradientKind::Focal { fx, fy, dx, dy, a, r } => GradientMapKind::Focal { fx, fy, dx, dy, a, r },
    },
  }
}

#[cfg(test)]
#[path = "tests/pipeline_equivalence.rs"]
mod tests;
