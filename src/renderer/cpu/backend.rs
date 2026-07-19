//! CPU implementation of renderer-neutral frame operations.

#![allow(unsafe_code)]

use crate::model::FillRule;
use crate::renderer::frame::{Composite, FrameRenderer, Geometry, GradientKind, GradientPaint, Paint, Rule};

use super::executor::{apply_matte, modulate, Canvas, DirtyBox, GradientMap, GradientMapKind};
use super::CPURenderer;

struct BitmapReset<'a>(&'a mut CPURenderer);

impl Drop for BitmapReset<'_> {
  fn drop(&mut self) {
    self.0.bitmap = None;
    while let Some(surface) = self.0.surfaces.pop() {
      let dirty = self.0.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
      self.0.state.put_surface_u32(surface, self.0.width, dirty);
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
    target.fill(0);
    self.width = width as usize;
    self.height = height as usize;
    self.antialias = options.antialias;
    self.bitmap_dirty = false;
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
    let raster = self.state.take_raster(self.width, self.height);
    let cells = self.state.take_cells(self.width, self.height);
    let key = geometry.cache_key;
    let contours = geometry.raw_contours();
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
    let mut canvas = Canvas::with_raster(pixels, self.width, self.height, raster, cells, self.antialias);
    if destination_dirty {
      // `Canvas` uses an empty dirty box to select a gradient copy fast
      // path. It is recreated for each streamed command, so carry the
      // destination's content state across commands explicitly.
      canvas.dirty.mark_row(0, 0, 1);
    }
    match paint {
      Paint::Solid(solid) => canvas.fill(&mut scratch.cov_cache, key, contours, fill_rule(solid.rule), solid.color, solid.opacity),
      Paint::Gradient(gradient) => {
        let map = gradient_map(gradient);
        canvas.fill_gradient(&mut scratch.cov_cache, key, gradient.source_key, contours, fill_rule(gradient.rule), &gradient.lut, &map);
      }
    }
    let draw_dirty = canvas.dirty;
    scratch.put_raster(canvas.raster);
    scratch.put_cells(canvas.cells);
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
        let width = self.width;
        composite_over_box(self.active(), &source, width, source_dirty, opacity);
        if !source_dirty.is_empty() {
          self.mark_active_dirty(source_dirty);
        }
        self.state.put_surface_u32(source, self.width, source_dirty);
      }
      Composite::Matte { kind, opacity } => {
        let Some(mut target) = self.surfaces.pop() else {
          return;
        };
        let target_dirty = self.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
        let Some(source) = self.surfaces.pop() else {
          return;
        };
        let _source_dirty = self.surface_dirty.pop().unwrap_or_else(DirtyBox::empty);
        apply_matte(&mut target, &source, kind);
        let width = self.width;
        composite_over_box(self.active(), &target, width, target_dirty, opacity);
        if !target_dirty.is_empty() {
          self.mark_active_dirty(target_dirty);
        }
        self.state.put_surface_u32(target, self.width, target_dirty);
        self.state.put_surface_u32(source, self.width, _source_dirty);
      }
    }
  }

  fn mark_active_dirty(&mut self, bounds: DirtyBox) {
    if let Some(dirty) = self.surface_dirty.last_mut() {
      dirty.union(bounds);
    } else {
      self.bitmap_dirty = true;
    }
  }

  fn apply_mask(&mut self, geometry: crate::renderer::frame::Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    let len = self.width.saturating_mul(self.height);
    if first || self.mask_accumulator.is_none() {
      let initial = if matches!(mode, b'a' | b'f') { 0 } else { 255 };
      if let Some(previous) = self.mask_accumulator.take() {
        self.state.put_u8(previous);
      }
      self.mask_accumulator = Some(self.state.take_u8(len, initial));
    }
    let mut coverage = self.state.take_u8(len, 0);
    let mut raster = self.state.take_raster(self.width, self.height);
    raster.reset();
    raster.fill_contours(geometry.raw_contours());
    let width = self.width;
    raster.sweep(FillRule::NonZero, self.antialias, |y, x0, row| {
      let lo = y.saturating_mul(width).saturating_add(x0);
      if let Some(dst) = coverage.get_mut(lo..lo.saturating_add(row.len())) {
        dst.copy_from_slice(row);
      }
    });
    self.state.put_raster(raster);
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

impl FrameRenderer for CPURenderer {
  fn save_layer(&mut self) {
    let layer = self.state.take_surface_u32(self.width.saturating_mul(self.height));
    self.surfaces.push(layer);
    self.surface_dirty.push(DirtyBox::empty());
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
