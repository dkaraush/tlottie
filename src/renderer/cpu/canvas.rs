use super::*;
use alloc::vec::Vec;

/// Inclusive rectangle containing every pixel written to a canvas.
/// Pixels outside it remain transparent.
#[derive(Clone, Copy)]
pub(crate) struct DirtyBox {
  pub(crate) x0: usize,
  pub(crate) y0: usize,
  pub(crate) x1: usize,
  pub(crate) y1: usize,
}

impl DirtyBox {
  pub(crate) fn empty() -> DirtyBox {
    DirtyBox {
      x0: usize::MAX,
      y0: usize::MAX,
      x1: 0,
      y1: 0,
    }
  }

  pub(crate) fn is_empty(&self) -> bool {
    self.x0 > self.x1 || self.y0 > self.y1
  }

  /// Marks the half-open column range `[x0, x1)` of row `y`.
  pub(crate) fn mark_row(&mut self, y: usize, x0: usize, x1: usize) {
    if x1 <= x0 {
      return;
    }
    self.x0 = self.x0.min(x0);
    self.x1 = self.x1.max(x1 - 1);
    self.y0 = self.y0.min(y);
    self.y1 = self.y1.max(y);
  }

  pub(crate) fn union(&mut self, other: DirtyBox) {
    if other.is_empty() {
      return;
    }
    self.x0 = self.x0.min(other.x0);
    self.y0 = self.y0.min(other.y0);
    self.x1 = self.x1.max(other.x1);
    self.y1 = self.y1.max(other.y1);
  }
}

#[derive(Clone, Copy)]
pub(crate) struct RowBounds {
  pub(crate) x0: usize,
  pub(crate) x1: usize,
}

impl RowBounds {
  pub(crate) fn empty() -> Self {
    Self { x0: usize::MAX, x1: 0 }
  }

  pub(crate) fn is_empty(self) -> bool {
    self.x0 > self.x1
  }

  pub(crate) fn mark(&mut self, x0: usize, x1: usize) {
    if x1 <= x0 {
      return;
    }
    self.x0 = self.x0.min(x0);
    self.x1 = self.x1.max(x1 - 1);
  }
}

pub(crate) fn mark_row_bounds(rows: &mut Option<&mut [RowBounds]>, y: usize, x0: usize, x1: usize) {
  if let Some(row) = rows.as_deref_mut().and_then(|rows| rows.get_mut(y)) {
    row.mark(x0, x1);
  }
}

pub(crate) struct Canvas<'a> {
  pub(crate) pixels: &'a mut [u32],
  pub(crate) w: usize,
  pub(crate) h: usize,
  pub(crate) antialias: bool,
  pub(crate) raster: Option<Rasterizer>,
  /// Mode-S sparse cell engine, selected per paint when the contours'
  /// bbox extent exceeds [`MODE_S_MIN_EXTENT`].
  pub(crate) cells: Option<CellRaster>,
  /// Union of all rows/columns written since creation (fills, gradient
  /// fills, and nested offscreen composites all mark it).
  pub(crate) dirty: DirtyBox,
  /// Optional per-row bounds owned by the active offscreen layer. Recording
  /// them while drawing avoids rediscovering sparse bounds from pixels when
  /// the layer is composited.
  pub(crate) dirty_rows: Option<&'a mut [RowBounds]>,
  /// Uniform-coverage row scratch: mode-S gradient spans synthesize a
  /// constant cov row here so gradient_row's per-pixel math (and its
  /// bit-exactness contract with the cache replay) is reused verbatim.
  pub(crate) row_cov: Vec<u8>,
  /// All-255 coverage row, grown on demand: opaque uniform gradient
  /// spans (the common mode-S case) borrow it instead of re-filling
  /// row_cov per span.
  pub(crate) row_ones: Vec<u8>,
  /// Recycled mode-S span-capture buffer: taken at the top of each
  /// fresh fill, returned unless the buffer moved into a cache entry
  /// (chunky captures become `PlaneData::Spans` verbatim). Avoids a
  /// realloc-growth chain per fresh mode-S paint.
  pub(crate) span_buf: Vec<u64>,
}

impl<'a> Canvas<'a> {
  #[cfg(test)]
  pub(crate) fn with_raster(pixels: &'a mut [u32], w: usize, h: usize, raster: Rasterizer, cells: CellRaster, antialias: bool) -> Self {
    Self::with_raster_and_rows(pixels, w, h, raster, cells, antialias, None)
  }

  pub(crate) fn with_raster_and_rows(pixels: &'a mut [u32], w: usize, h: usize, raster: Rasterizer, cells: CellRaster, antialias: bool, dirty_rows: Option<&'a mut [RowBounds]>) -> Self {
    Canvas {
      pixels,
      w,
      h,
      antialias,
      raster: Some(raster),
      cells: Some(cells),
      dirty: DirtyBox::empty(),
      dirty_rows,
      row_cov: Vec::new(),
      row_ones: Vec::new(),
      span_buf: Vec::new(),
    }
  }

  pub(crate) fn with_retained_rows(pixels: &'a mut [u32], w: usize, h: usize, antialias: bool, dirty_rows: Option<&'a mut [RowBounds]>) -> Self {
    Canvas {
      pixels,
      w,
      h,
      antialias,
      raster: None,
      cells: None,
      dirty: DirtyBox::empty(),
      dirty_rows,
      row_cov: Vec::new(),
      row_ones: Vec::new(),
      span_buf: Vec::new(),
    }
  }
}

impl Canvas<'_> {
  /// Rasterizes `contours` and blends `color` (straight alpha, 0..=1
  /// components) with `opacity`, premultiplied source-over.
  #[cfg(test)]
  pub(crate) fn fill<const TRACK_ROWS: bool>(&mut self, cache: &mut CovCache, key: u128, contours: &[Contour], rule: crate::model::FillRule, color: Color, opacity: f32) {
    self.fill_translated::<TRACK_ROWS>(cache, key, contours, crate::renderer::frame::Point::default(), rule, color, opacity);
  }

  pub(crate) fn fill_translated<const TRACK_ROWS: bool>(
    &mut self,
    cache: &mut CovCache,
    key: u128,
    contours: &[Contour],
    translation: crate::renderer::frame::Point,
    rule: crate::model::FillRule,
    color: Color,
    opacity: f32,
  ) {
    let alpha = (color.a * opacity).clamp(0.0, 1.0);
    if alpha <= 0.0 {
      return;
    }
    let (sr, sg, sb) = (
      (color.r.clamp(0.0, 1.0) * 255.0) as u32,
      (color.g.clamp(0.0, 1.0) * 255.0) as u32,
      (color.b.clamp(0.0, 1.0) * 255.0) as u32,
    );
    // The reference rasterizers quantize paint opacity by truncation.
    // Rounding is visibly biased for animations made from many very faint
    // overlapping shapes (for example 0.68% must become alpha 1, not 2).
    let sa = (alpha * 255.0) as u32;
    let w = self.w;
    let antialias = self.antialias;
    if let Some(e) = cache.get(key) {
      // Cache hit: replay coverage (identical bytes to a fresh
      // rasterization of the same geometry, whichever mode made it).
      match &e.data {
        PlaneData::Cov(data) => {
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
            crate::simd::fill_span_solid(dst_row, cov_row, sr, sg, sb, sa, w > 128);
            off += len;
          }
        }
        PlaneData::Spans(spans) => {
          if sa == 255 {
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
              crate::simd::fill_span_uniform_opaque(dst_row, cov, sr, sg, sb);
            }
          } else {
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
              crate::simd::fill_span_uniform(dst_row, cov, sr, sg, sb, sa);
            }
          }
        }
        PlaneData::Src(_) => {}
      }
      return;
    }
    if contours.is_empty() {
      return;
    }
    let pixels = &mut *self.pixels;
    let dirty = &mut self.dirty;
    let dirty_rows = &mut self.dirty_rows;
    if mode_s_wins(contours, w * self.h) {
      // Mode S: sparse cells — no w×h plane, cost ∝ edge crossings.
      let cells = self.cells.as_mut().expect("fresh fill requires cell rasterizer");
      cells.reset();
      cells.fill_contours_translated(contours, translation.x, translation.y);
      let capture = cache.capture_enabled();
      let mut spans: Vec<u64> = core::mem::take(&mut self.span_buf);
      spans.clear();
      let mut px_total = 0usize;
      let mut overflow = false;
      let mut render_span = |y: usize, x0: usize, len: usize, cov: u8, opaque: bool| {
        let lo = y.saturating_mul(w).saturating_add(x0);
        let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
          return;
        };
        dirty.mark_row(y, x0, x0 + len);
        if TRACK_ROWS {
          mark_row_bounds(dirty_rows, y, x0, x0 + len);
        }
        if capture {
          if spans.len() < SPAN_CAPTURE_MAX {
            spans.push(pack_span(y, x0, len, cov));
            px_total += len;
          } else {
            overflow = true;
          }
        }
        if opaque {
          crate::simd::fill_span_uniform_opaque(dst_row, cov, sr, sg, sb);
        } else {
          crate::simd::fill_span_uniform(dst_row, cov, sr, sg, sb, sa);
        }
      };
      if sa == 255 {
        cells.sweep_spans(rule, antialias, |y, x0, len, cov| render_span(y, x0, len, cov, true));
      } else {
        cells.sweep_spans(rule, antialias, |y, x0, len, cov| render_span(y, x0, len, cov, false));
      }
      // Chunky span lists cache as spans (denser, uniform replay);
      // fragmented ones as rows (fast row replay).
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
    let raster = self.raster.as_mut().expect("fresh fill requires dense rasterizer");
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
      crate::simd::fill_span_solid(dst_row, cov_row, sr, sg, sb, sa, w > 128);
    });
    if capture {
      cache.insert(key, entry);
    }
  }
}
