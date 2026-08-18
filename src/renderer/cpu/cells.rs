//! Sparse cell/span rasterizer. FreeType-style edges deposit (cover, area)
//! records into per-scanline cell buckets; the sweep sorts each row's
//! cells by x and walks them left→right, emitting horizontal spans of
//! UNIFORM coverage. Cost is proportional to edge crossings, not covered
//! area — interior pixels are free until the blend.
//!
//! Numerics: input points snap to 24.8 fixed point; every deposit is an
//! exact i32. Each segment is normalized top→bottom before splitting, so
//! a segment and its reversal produce identical magnitudes with opposite
//! sign and cancel EXACTLY after same-cell merge (the seam-exactness
//! contract the f32 accumulator provides in mode D, but integral).
//!
//! It is not bit-identical to the dense rasterizer because it uses different
//! summation and 1/256-coordinate snapping. Selection is based on paint size.

use crate::geometry::Contour;
use crate::model::FillRule;
use alloc::vec::Vec;

/// Subpixel bits (24.8 fixed point, FreeType's PIXEL_BITS = 8).
const PIX_B: i32 = 8;
/// One pixel in fixed-point units.
const ONE: i32 = 1 << PIX_B;
#[derive(Clone, Copy)]
struct Cell {
  x: i32,
  cover: i32,
  area: i32,
}

/// Sparse cell rasterizer; pooled and reused like `raster::Rasterizer`.
pub(crate) struct CellRaster {
  w: usize,
  h: usize,
  /// Per-scanline unsorted cell buckets. Row Vecs keep their capacity
  /// across reset (pool semantics); only touched rows are cleared.
  rows: Vec<Vec<Cell>>,
  min_y: usize,
  max_y: usize,
}

impl CellRaster {
  pub fn new(w: usize, h: usize) -> Self {
    CellRaster {
      w,
      h,
      rows: (0..h).map(|_| Vec::new()).collect(),
      min_y: usize::MAX,
      max_y: 0,
    }
  }

  /// Re-targets a pooled instance; keeps row-bucket allocations.
  pub fn reshape(&mut self, w: usize, h: usize) {
    if self.rows.len() < h {
      self.rows.resize_with(h, Vec::new);
    }
    self.w = w;
    self.h = h;
    self.min_y = usize::MAX;
    self.max_y = 0;
  }

  pub fn reset(&mut self) {
    if self.min_y <= self.max_y {
      for row in self.rows.get_mut(self.min_y..=self.max_y.min(self.h.saturating_sub(1))).unwrap_or_default() {
        row.clear();
      }
    }
    self.min_y = usize::MAX;
    self.max_y = 0;
  }

  /// Deposits into an already-resolved row. The row is fetched ONCE per
  /// `scanline` call (`ey` is constant across it) — the per-column
  /// `rows.get_mut(ey)` re-resolution was measured at ~1% of 720px
  /// frames (host, IceMan/DeathNote class).
  #[inline]
  fn bump_row(row: &mut Vec<Cell>, ex: i32, cover: i32, area: i32) {
    if cover == 0 && area == 0 {
      return;
    }
    // Consecutive deposits usually hit the same cell; merge in place.
    if let Some(last) = row.last_mut() {
      if last.x == ex {
        last.cover += cover;
        last.area += area;
        return;
      }
    }
    row.push(Cell { x: ex, cover, area });
  }

  /// Deposits one row-segment (both endpoints within scanline `ey`),
  /// walking the pixel columns it crosses. `x1/x2` are fixed-point
  /// horizontal positions, `fy1/fy2` subpixel y offsets (0..=ONE) within
  /// the row, already direction-normalized: the winding sign is `dir`.
  ///
  /// Column crossings use an exact floor-div remainder DDA (FreeType's
  /// FT_DIV_MOD pattern): ONE division per segment, adds per column —
  /// the per-boundary i64 division was the hottest line at 720px. The
  /// walk is normalized left→right, so a segment and its reversal still
  /// split identically (seam exactness).
  fn scanline(&mut self, ey: usize, x1: i32, fy1: i32, x2: i32, fy2: i32, dir: i32) {
    let dy = fy2 - fy1;
    if dy == 0 {
      return;
    }
    let Some(row) = self.rows.get_mut(ey) else {
      return;
    };
    let ex1 = x1 >> PIX_B;
    let ex2 = x2 >> PIX_B;
    if ex1 == ex2 {
      // Whole segment inside one column.
      let fx1 = x1 - (ex1 << PIX_B);
      let fx2 = x2 - (ex1 << PIX_B);
      Self::bump_row(row, ex1, dir * dy, dir * dy * (fx1 + fx2));
      return;
    }
    // Normalize left→right; the swapped parameterization traverses y
    // backwards, so the deposit sign flips with it.
    let (lx, ly, rx, ry, dirw) = if x1 <= x2 { (x1, fy1, x2, fy2, dir) } else { (x2, fy2, x1, fy1, -dir) };
    let exl = lx >> PIX_B;
    let exr = rx >> PIX_B;
    let dx_t = i64::from(rx - lx); // > 0 (distinct columns)
    let dy_t = i64::from(ry - ly);
    // y at the first right boundary, then exact incremental steps. Euclidean
    // floor division by the positive divisor `dx_t`, computed as trunc `/`+`%`
    // (one idiv pair per dividend) plus a negative-remainder fixup — bit-exact
    // with `div_euclid`/`rem_euclid` (profiled: div_euclid is the sweep
    // walk hotspot at 720px).
    let bx0 = (exl + 1) << PIX_B;
    let num0 = i64::from(bx0 - lx) * dy_t;
    let mut tq = num0 / dx_t;
    let mut tr = num0 % dx_t;
    if tr < 0 {
      tq -= 1;
      tr += dx_t;
    }
    let mut ycur = ly + tq as i32;
    let mut rem = tr;
    let stepnum = i64::from(ONE) * dy_t;
    let mut q = stepnum / dx_t;
    let mut r = stepnum % dx_t;
    if r < 0 {
      q -= 1;
      r += dx_t;
    }
    let q = q as i32;
    // First column: enters at lx, exits at the boundary (fx_b = ONE).
    let fx_a = lx - (exl << PIX_B);
    let d0 = ycur - ly;
    Self::bump_row(row, exl, dirw * d0, dirw * d0 * (fx_a + ONE));
    // Middle columns: enter fx 0, exit fx ONE → area = d·ONE.
    let mut yprev = ycur;
    let mut ex = exl + 1;
    while ex < exr {
      ycur += q;
      rem += r;
      if rem >= dx_t {
        rem -= dx_t;
        ycur += 1;
      }
      let d = ycur - yprev;
      Self::bump_row(row, ex, dirw * d, dirw * d * ONE);
      yprev = ycur;
      ex += 1;
    }
    // Last column: enters at fx 0, exits at rx.
    let fx_b = rx - (exr << PIX_B);
    let dl = ry - yprev;
    Self::bump_row(row, exr, dirw * dl, dirw * dl * fx_b);
  }

  /// Accumulates one edge segment (float device coordinates, y-down).
  /// Callers pre-clip to the viewport; excursions are clamped.
  fn draw_line(&mut self, x0f: f32, y0f: f32, x1f: f32, y1f: f32) {
    if !(x0f.is_finite() && y0f.is_finite() && x1f.is_finite() && y1f.is_finite()) {
      return;
    }
    // Snap to 24.8. Clamp generously before the cast (i32 range).
    let wf = (self.w as f32) * ONE as f32;
    let hf = (self.h as f32) * ONE as f32;
    let fx = |v: f32| ((v * ONE as f32).clamp(0.0, wf)) as i32;
    let fy = |v: f32| ((v * ONE as f32).clamp(0.0, hf)) as i32;
    let (mut x0, mut y0, mut x1, mut y1) = (fx(x0f), fy(y0f), fx(x1f), fy(y1f));
    if y0 == y1 {
      return;
    }
    // Normalize top→bottom; dir carries the winding sign. A segment and
    // its reversal normalize to IDENTICAL coordinates, so their deposits
    // are exact negations (seam cancellation).
    let dir = if y0 < y1 {
      1i32
    } else {
      core::mem::swap(&mut x0, &mut x1);
      core::mem::swap(&mut y0, &mut y1);
      -1i32
    };
    let ey0 = y0 >> PIX_B;
    let ey1 = y1 >> PIX_B;
    let lo = (ey0.max(0) as usize).min(self.h.saturating_sub(1));
    let hi = ((y1 - 1) >> PIX_B).max(0) as usize;
    self.min_y = self.min_y.min(lo);
    self.max_y = self.max_y.max(hi.min(self.h.saturating_sub(1)));

    if ey0 == ey1 {
      let base = ey0 << PIX_B;
      if ey0 >= 0 && (ey0 as usize) < self.h {
        self.scanline(ey0 as usize, x0, y0 - base, x1, y1 - base, dir);
      }
      return;
    }
    // Split at each integer scanline from the ORIGINAL endpoints (exact
    // i64 interpolation ⇒ reversal-stable split points). The split x at row
    // boundary `by` is `x0 + trun((by − y0)·dx/dy)`. `by` grows by
    // exactly ONE per scanline, so the numerator grows by B = ONE·|dx| each
    // step and the quotient can be maintained incrementally (one div+mod per
    // segment instead of one per scanline — the per-boundary i64 division
    // was a documented 720px hotspot). Truncation is reproduced exactly via a
    // magnitude DDA with a sign factor, so split points are bit-identical.
    let dx_total = i64::from(x1 - x0);
    let dy_total = i64::from(y1 - y0); // > 0 after normalization
    let (s, adx) = if dx_total < 0 { (-1i64, -dx_total) } else { (1i64, dx_total) };
    debug_assert!(dy_total > 0);
    let by0 = i64::from((ey0 + 1) << PIX_B);
    let a0 = (by0 - i64::from(y0)) * adx;
    let b = i64::from(ONE) * adx;
    let d = dy_total;
    let q = b / d;
    let r = b % d;
    let mut qmag = a0 / d;
    let mut rem = a0 % d;
    let mut xa = x0;
    let mut ya = y0;
    let mut ey = ey0;
    while ey < ey1 {
      let by = (ey + 1) << PIX_B; // bottom boundary of row ey
      let xb_at = x0 + (s * qmag) as i32;
      if ey >= 0 && (ey as usize) < self.h {
        let base = ey << PIX_B;
        self.scanline(ey as usize, xa, ya - base, xb_at, by - base, dir);
      }
      xa = xb_at;
      ya = by;
      // Advance the magnitude DDA: |num| grows by B = q·d + r per row.
      rem += r;
      if rem >= d {
        rem -= d;
        qmag += q + 1;
      } else {
        qmag += q;
      }
      ey += 1;
    }
    if ey >= 0 && (ey as usize) < self.h && ya < y1 {
      let base = ey << PIX_B;
      self.scanline(ey as usize, xa, ya - base, x1, y1 - base, dir);
    }
  }

  pub fn fill_contours(&mut self, contours: &[Contour]) {
    self.fill_contours_translated(contours, 0.0, 0.0);
  }

  pub fn fill_contours_translated(&mut self, contours: &[Contour], tx: f32, ty: f32) {
    for contour in contours {
      let pts = &contour.points;
      if pts.len() < 3 {
        continue;
      }
      for (i, cur) in pts.iter().enumerate() {
        let next = pts.get(i + 1).or_else(|| pts.first());
        if let Some(next) = next {
          self.draw_line(cur.x + tx, cur.y + ty, next.x + tx, next.y + ty);
        }
      }
    }
  }

  /// Converts an accumulated area value to a coverage byte under `rule`,
  /// with the SAME quantization as mode D's sweep (`c*255 + 0.5` on the
  /// clamped/mirrored fraction) so the two engines differ only by the
  /// 24.8 coordinate snap.
  #[inline]
  fn coverage(area: i32, rule: FillRule, antialias: bool) -> u8 {
    /// area units of one fully covered pixel (cover ONE × width 2·ONE).
    const FULL: i64 = (ONE as i64) * (ONE as i64) * 2;
    let a = i64::from(area);
    let v = match rule {
      FillRule::NonZero => a.abs().min(FULL),
      FillRule::EvenOdd => {
        let m = a.rem_euclid(2 * FULL);
        if m > FULL {
          2 * FULL - m
        } else {
          m
        }
      }
    };
    if antialias {
      ((v * 255 + FULL / 2) / FULL) as u8
    } else if v * 2 >= FULL {
      255
    } else {
      0
    }
  }

  /// Sorts each touched row's cells and emits horizontal spans of uniform
  /// coverage: `f(y, x0, len, cov)`. Adjacent emissions with equal
  /// coverage are MERGED into one span before the callback: piece-union
  /// interiors are criss-crossed by seam edges whose cells keep coverage
  /// at 255 on both sides (winding 1→2→1), and without merging every
  /// such cell split the interior into avg-7px fragments (measured on
  /// DeathNote@720: 14M spans/120 frames), defeating the ≥16px NEON
  /// blit gate and tripling cache entry sizes. Zero-coverage gaps are
  /// not emitted and naturally break merge runs.
  pub fn sweep_spans(&mut self, rule: FillRule, antialias: bool, mut f: impl FnMut(usize, usize, usize, u8)) {
    self.sweep_spans_impl::<false>(rule, antialias, &mut |y, span| {
      if let Some((x0, len, cov)) = span {
        f(y, x0, len, cov);
      }
    });
  }

  /// Like [`Self::sweep_spans`], but also calls `row_end` for every canvas
  /// row. This lets sparse consumers process zero-coverage gaps without
  /// materializing a full width×height coverage plane.
  pub fn sweep_span_rows(&mut self, rule: FillRule, antialias: bool, mut f: impl FnMut(usize, Option<(usize, usize, u8)>)) {
    self.sweep_spans_impl::<true>(rule, antialias, &mut f);
  }

  fn sweep_spans_impl<const ALL_ROWS: bool>(&mut self, rule: FillRule, antialias: bool, f: &mut impl FnMut(usize, Option<(usize, usize, u8)>)) {
    if self.min_y > self.max_y {
      if ALL_ROWS {
        for y in 0..self.h {
          f(y, None);
        }
      }
      return;
    }
    let w = self.w;
    let y0 = if ALL_ROWS { 0 } else { self.min_y };
    let y1 = if ALL_ROWS { self.h.saturating_sub(1) } else { self.max_y.min(self.h.saturating_sub(1)) };
    for y in y0..=y1 {
      let Some(row) = self.rows.get_mut(y) else {
        if ALL_ROWS {
          f(y, None);
        }
        continue;
      };
      if row.is_empty() {
        if ALL_ROWS {
          f(y, None);
        }
        continue;
      }
      row.sort_unstable_by_key(|c| c.x);
      // Pending merged span: (x0, len, cov).
      let mut pend: Option<(usize, usize, u8)> = None;
      macro_rules! emit {
        ($x0:expr, $len:expr, $cov:expr) => {{
          let (x0, len, cov) = ($x0, $len, $cov);
          match &mut pend {
            Some((px0, plen, pcov)) if *pcov == cov && *px0 + *plen == x0 => {
              *plen += len;
            }
            Some((px0, plen, pcov)) => {
              f(y, Some((*px0, *plen, *pcov)));
              pend = Some((x0, len, cov));
            }
            None => pend = Some((x0, len, cov)),
          }
        }};
      }
      let mut cover: i32 = 0;
      let mut i = 0usize;
      let mut prev_end: i32 = i32::MIN;
      while let Some(&Cell { x, .. }) = row.get(i) {
        // Gap span between previous cell and this one.
        if prev_end != i32::MIN && x > prev_end && cover != 0 {
          let cov = Self::coverage(cover * (ONE * 2), rule, antialias);
          if cov != 0 {
            let x0 = prev_end.max(0) as usize;
            let x1 = (x.max(0) as usize).min(w);
            if x1 > x0 {
              emit!(x0, x1 - x0, cov);
            }
          }
        }
        // Merge all cells sharing this x.
        let mut cell_cover = 0i32;
        let mut cell_area = 0i32;
        while let Some(&Cell { x: cx, cover: cc, area: ca }) = row.get(i) {
          if cx != x {
            break;
          }
          cell_cover += cc;
          cell_area += ca;
          i += 1;
        }
        cover += cell_cover;
        let area = cover * (ONE * 2) - cell_area;
        if area != 0 && x >= 0 && (x as usize) < w {
          let cov = Self::coverage(area, rule, antialias);
          if cov != 0 {
            emit!(x as usize, 1, cov);
          }
        }
        prev_end = x + 1;
      }
      if let Some((px0, plen, pcov)) = pend.take() {
        f(y, Some((px0, plen, pcov)));
      }
      if ALL_ROWS {
        f(y, None);
      }
    }
  }
}

#[cfg(test)]
#[path = "tests/cells.rs"]
mod tests;
