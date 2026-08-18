//! Direct Alpha8 rendering backend.

use crate::model::FillRule;
use crate::renderer::frame::{Composite, FrameRenderer, Geometry, GradientKind, GradientPaint, Paint, Rule};
use alloc::vec::Vec;

use super::executor::{mode_s_wins, pack_span, unpack_span, CovEntry, PlaneData, RenderScratch, SPAN_CAPTURE_MAX};

pub(super) struct Alpha8Renderer<'a> {
  pub(super) pixels: &'a mut [u8],
  pub(super) width: usize,
  pub(super) height: usize,
  pub(super) antialias: bool,
  pub(super) state: &'a mut RenderScratch,
  surfaces: Vec<Vec<u8>>,
  mask: Option<Vec<u8>>,
  gradient_row: Vec<u32>,
  gradient_alpha: Vec<u8>,
}

impl<'a> Alpha8Renderer<'a> {
  pub(super) fn new(pixels: &'a mut [u8], width: usize, height: usize, antialias: bool, clear: bool, state: &'a mut RenderScratch) -> Self {
    if clear {
      pixels.fill(0);
    }
    Self {
      pixels,
      width,
      height,
      antialias,
      state,
      surfaces: Vec::new(),
      mask: None,
      gradient_row: Vec::new(),
      gradient_alpha: Vec::new(),
    }
  }

  fn active(&mut self) -> &mut [u8] {
    self.surfaces.last_mut().map_or(self.pixels, Vec::as_mut_slice)
  }

  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>) {
    let width = self.width;
    let antialias = self.antialias;
    let key = geometry.cache_key;
    let translation = geometry.raw_translation();
    let uniform_gradient_alpha = match paint {
      Paint::Gradient(gradient) => {
        let alpha = (gradient.lut.first().copied().unwrap_or(0) >> 24) as u8;
        gradient.lut.iter().all(|pixel| (*pixel >> 24) as u8 == alpha).then_some(alpha)
      }
      Paint::Solid(_) => None,
    };

    if let Some(entry) = self.state.cov_cache.get(key) {
      let (gradient_row, gradient_alpha) = (&mut self.gradient_row, &mut self.gradient_alpha);
      let target = match self.surfaces.last_mut() {
        Some(surface) => surface.as_mut_slice(),
        None => &mut *self.pixels,
      };
      match &entry.data {
        PlaneData::Cov(data) => {
          let mut offset = 0usize;
          for &(y, x0, len) in &entry.rows {
            let (y, x0, len) = (y as usize, x0 as usize, len as usize);
            let start = y.saturating_mul(width).saturating_add(x0);
            if let (Some(row), Some(coverage)) = (target.get_mut(start..start.saturating_add(len)), data.get(offset..offset.saturating_add(len))) {
              blend_paint(row, coverage, y, x0, paint, uniform_gradient_alpha, gradient_row, gradient_alpha);
            }
            offset = offset.saturating_add(len);
          }
          return;
        }
        PlaneData::Spans(spans) => {
          for &span in spans {
            let (y, x0, len, coverage) = unpack_span(span);
            let start = y.saturating_mul(width).saturating_add(x0);
            if let Some(row) = target.get_mut(start..start.saturating_add(len)) {
              blend_uniform_paint(row, coverage, y, x0, paint, uniform_gradient_alpha, gradient_row, gradient_alpha);
            }
          }
          return;
        }
        PlaneData::Src(_) => {}
      }
    }

    if mode_s_wins(geometry.raw_contours(), width.saturating_mul(self.height)) {
      let mut cells = self.state.take_cells(width, self.height);
      cells.fill_contours_translated(geometry.raw_contours(), translation.x, translation.y);
      let capture = self.state.cov_cache.capture_enabled();
      let mut spans = Vec::new();
      let mut capture_overflow = false;
      let (gradient_row, gradient_alpha) = (&mut self.gradient_row, &mut self.gradient_alpha);
      let target = match self.surfaces.last_mut() {
        Some(surface) => surface.as_mut_slice(),
        None => &mut *self.pixels,
      };
      cells.sweep_spans(
        fill_rule(match paint {
          Paint::Solid(solid) => solid.rule,
          Paint::Gradient(gradient) => gradient.rule,
        }),
        antialias,
        |y, x0, len, coverage| {
          let start = y.saturating_mul(width).saturating_add(x0);
          if let Some(row) = target.get_mut(start..start.saturating_add(len)) {
            blend_uniform_paint(row, coverage, y, x0, paint, uniform_gradient_alpha, gradient_row, gradient_alpha);
          }
          if capture {
            if spans.len() < SPAN_CAPTURE_MAX {
              spans.push(pack_span(y, x0, len, coverage));
            } else {
              capture_overflow = true;
            }
          }
        },
      );
      self.state.put_cells(cells);
      if capture && !capture_overflow {
        self.state.cov_cache.insert(
          key,
          CovEntry {
            rows: Vec::new(),
            data: PlaneData::Spans(spans),
          },
        );
      }
      return;
    }

    let mut raster = self.state.take_raster(width, self.height);
    raster.fill_contours_translated(geometry.raw_contours(), translation.x, translation.y);
    let capture = self.state.cov_cache.capture_enabled();
    let mut entry = CovEntry::default();
    let (gradient_row, gradient_alpha) = (&mut self.gradient_row, &mut self.gradient_alpha);
    let target = match self.surfaces.last_mut() {
      Some(surface) => surface.as_mut_slice(),
      None => &mut *self.pixels,
    };
    raster.sweep(
      fill_rule(match paint {
        Paint::Solid(solid) => solid.rule,
        Paint::Gradient(gradient) => gradient.rule,
      }),
      antialias,
      |y, x0, coverage| {
        let start = y.saturating_mul(width).saturating_add(x0);
        let Some(row) = target.get_mut(start..start.saturating_add(coverage.len())) else {
          return;
        };
        blend_paint(row, coverage, y, x0, paint, uniform_gradient_alpha, gradient_row, gradient_alpha);
        if capture {
          entry.rows.push((y as u32, x0 as u32, coverage.len() as u32));
          if let PlaneData::Cov(data) = &mut entry.data {
            data.extend_from_slice(coverage);
          }
        }
      },
    );
    self.state.put_raster(raster);
    if capture {
      self.state.cov_cache.insert(key, entry);
    }
  }

  fn apply_mask(&mut self, geometry: Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    let len = self.width.saturating_mul(self.height);
    if first || self.mask.is_none() {
      if let Some(old) = self.mask.take() {
        self.state.put_u8(old);
      }
      self.mask = Some(self.state.take_u8(len, if matches!(mode, b'a' | b'f') { 0 } else { 255 }));
    }
    let mut coverage = self.state.take_u8(len, 0);
    let mut raster = self.state.take_raster(self.width, self.height);
    raster.fill_contours(geometry.raw_contours());
    let width = self.width;
    raster.sweep(FillRule::NonZero, self.antialias, |y, x0, row| {
      let start = y.saturating_mul(width).saturating_add(x0);
      if let Some(out) = coverage.get_mut(start..start.saturating_add(row.len())) {
        out.copy_from_slice(row);
      }
    });
    self.state.put_raster(raster);
    if let Some(mask) = self.mask.as_mut() {
      crate::simd::alpha_mask_combine(mask, &coverage, mode, inverted, opacity);
    }
    self.state.put_u8(coverage);
    if last {
      if let Some(mask) = self.mask.take() {
        crate::simd::alpha_multiply(self.active(), &mask);
        self.state.put_u8(mask);
      }
    }
  }

  pub(super) fn finish(mut self) {
    if let Some(mask) = self.mask.take() {
      self.state.put_u8(mask);
    }
    for surface in self.surfaces.drain(..) {
      self.state.put_u8(surface);
    }
  }
}

impl FrameRenderer for Alpha8Renderer<'_> {
  fn save_layer(&mut self) {
    self.surfaces.push(self.state.take_u8(self.width.saturating_mul(self.height), 0));
  }

  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>) {
    self.draw(geometry, paint);
  }

  fn apply_mask(&mut self, geometry: Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    self.apply_mask(geometry, mode, inverted, opacity, first, last);
  }

  fn end_layer(&mut self, composite: Composite) {
    match composite {
      Composite::Over { opacity } => {
        if let Some(source) = self.surfaces.pop() {
          composite_over(self.active(), &source, opacity);
          self.state.put_u8(source);
        }
      }
      Composite::Matte { kind, opacity, source_opacity } => {
        let Some(mut target) = self.surfaces.pop() else { return };
        let Some(source) = self.surfaces.pop() else {
          self.state.put_u8(target);
          return;
        };
        crate::simd::alpha_matte(&mut target, &source, source_opacity, kind != 1);
        composite_over(self.active(), &target, opacity);
        self.state.put_u8(target);
        self.state.put_u8(source);
      }
    }
  }

  fn retains_geometry(&self, cache_key: u128) -> bool {
    self.state.cov_cache.contains_coverage(cache_key)
  }
}

fn fill_rule(rule: Rule) -> FillRule {
  match rule {
    Rule::NonZero => FillRule::NonZero,
    Rule::EvenOdd => FillRule::EvenOdd,
  }
}

fn blend_solid(destination: &mut [u8], coverage: &[u8], alpha: u8) {
  crate::simd::alpha_blend_solid(destination, coverage, alpha);
}

fn blend_paint(destination: &mut [u8], coverage: &[u8], y: usize, x0: usize, paint: Paint<'_>, uniform_gradient_alpha: Option<u8>, gradient_row: &mut Vec<u32>, gradient_alpha: &mut Vec<u8>) {
  match paint {
    Paint::Solid(solid) => {
      let alpha = (solid.color.a * solid.opacity).clamp(0.0, 1.0);
      blend_solid(destination, coverage, (alpha * 255.0) as u8);
    }
    Paint::Gradient(gradient) => match uniform_gradient_alpha {
      Some(255) => blend_coverage(destination, coverage),
      Some(alpha) => blend_solid(destination, coverage, alpha),
      None => blend_gradient(destination, coverage, y, x0, gradient, gradient_row, gradient_alpha),
    },
  }
}

fn blend_uniform_paint(destination: &mut [u8], coverage: u8, y: usize, x0: usize, paint: Paint<'_>, uniform_gradient_alpha: Option<u8>, gradient_row: &mut Vec<u32>, gradient_alpha: &mut Vec<u8>) {
  match paint {
    Paint::Solid(solid) => {
      let alpha = (solid.color.a * solid.opacity).clamp(0.0, 1.0);
      blend_uniform(destination, coverage, (alpha * 255.0) as u8);
    }
    Paint::Gradient(gradient) => match uniform_gradient_alpha {
      Some(255) => blend_uniform(destination, coverage, 255),
      Some(alpha) => blend_uniform(destination, coverage, alpha),
      None => {
        sample_gradient_row(gradient_row, destination.len(), y, x0, gradient);
        extract_gradient_alpha(gradient_alpha, gradient_row);
        crate::simd::alpha_blend_solid(destination, gradient_alpha, coverage);
      }
    },
  }
}

fn blend_uniform(destination: &mut [u8], coverage: u8, alpha: u8) {
  crate::simd::alpha_blend_uniform(destination, coverage, alpha);
}

fn blend_coverage(destination: &mut [u8], coverage: &[u8]) {
  crate::simd::alpha_blend_solid(destination, coverage, 255);
}

fn composite_over(destination: &mut [u8], source: &[u8], opacity: u8) {
  crate::simd::alpha_composite_over(destination, source, opacity);
}

fn blend_gradient(destination: &mut [u8], coverage: &[u8], y: usize, x0: usize, paint: &GradientPaint, gradient_row: &mut Vec<u32>, gradient_alpha: &mut Vec<u8>) {
  sample_gradient_row(gradient_row, destination.len().min(coverage.len()), y, x0, paint);
  extract_gradient_alpha(gradient_alpha, gradient_row);
  crate::simd::alpha_blend_product(destination, coverage, gradient_alpha);
}

fn extract_gradient_alpha(out: &mut Vec<u8>, samples: &[u32]) {
  out.clear();
  out.extend(samples.iter().map(|sample| (sample >> 24) as u8));
}

fn sample_gradient_row(out: &mut Vec<u32>, len: usize, y: usize, x0: usize, paint: &GradientPaint) {
  out.resize(len, 0);
  let out = out.as_mut_slice();
  let inv = paint.transform;
  let yf = y as f32 + 0.5;
  let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
  let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
  match paint.kind {
    GradientKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
      let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
      let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
      crate::simd::linear_lut_fill(out, paint.lut.as_ref(), row_base, dt, x0 as f32);
    }
    GradientKind::Radial { sx, sy, inv_r } => {
      crate::simd::radial_lut_fill(out, paint.lut.as_ref(), lx0 - sx, ly0 - sy, inv.a, inv.b, inv_r, x0 as f32);
    }
    GradientKind::Focal { fx, fy, dx, dy, a, r } => {
      if a.abs() < 1e-9 {
        out.fill(0);
      } else {
        crate::simd::focal_lut_fill(out, paint.lut.as_ref(), lx0 - fx, ly0 - fy, inv.a, inv.b, dx, dy, a, 1.0 / (2.0 * a), r, x0 as f32);
      }
    }
  }
}
