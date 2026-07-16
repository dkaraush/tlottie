//! Scanline coverage rasterizer, signed-area accumulation style (font-rs):
//! each polygon edge deposits signed per-pixel area deltas; a prefix sum
//! per row yields winding-number coverage with analytic anti-aliasing.
//!
//! v0 keeps a full w×h f32 accumulation buffer. The cell/RLE engine with
//! small-mask output replaces this in the rasterizer phase.

use crate::geometry::Contour;
use crate::model::FillRule;

pub(crate) struct Rasterizer {
    w: usize,
    h: usize,
    acc: Vec<f32>,
    /// Rows/columns touched: sweep and clear cost scale with the shape's
    /// bounding box, not the canvas (critical: dozens of small paints per
    /// frame at 512px would otherwise each pay full-width row walks).
    min_y: usize,
    max_y: usize,
    /// Per-row touched x-range (inclusive min, inclusive max of spill cell).
    /// Thin diagonal shapes (strokes!) touch few columns per row; sweeping
    /// each row only within its own range beats a global bounding box.
    rows: Vec<(u32, u32)>,
    /// Scratch coverage row, reused across sweeps.
    cov: Vec<u8>,
}

const ROW_EMPTY: (u32, u32) = (u32::MAX, 0);

impl Rasterizer {
    pub fn new(w: usize, h: usize) -> Self {
        // Row stride is w+1: edges that land exactly on x == w deposit their
        // delta into the row's own slack slot instead of leaking into the
        // next row's first pixel (which would corrupt its winding).
        Rasterizer {
            w,
            h,
            acc: vec![0.0; (w + 1).saturating_mul(h).saturating_add(1)],
            min_y: usize::MAX,
            max_y: 0,
            rows: vec![ROW_EMPTY; h],
            cov: vec![0; w],
        }
    }

    fn stride(&self) -> usize {
        self.w + 1
    }

    /// Re-targets a pooled rasterizer to new dimensions, reusing the
    /// allocations. Callers must hand back rasterizers in the reset state
    /// (acc all-zero, rows all ROW_EMPTY) — then only growth needs filling.
    pub fn reshape(&mut self, w: usize, h: usize) {
        let need = (w + 1).saturating_mul(h).saturating_add(1);
        if self.acc.len() < need {
            self.acc.resize(need, 0.0);
        }
        if self.rows.len() < h {
            self.rows.resize(h, ROW_EMPTY);
        }
        if self.cov.len() < w {
            self.cov.resize(w, 0);
        }
        self.w = w;
        self.h = h;
        self.min_y = usize::MAX;
        self.max_y = 0;
    }

    pub fn reset(&mut self) {
        if self.min_y <= self.max_y {
            let stride = self.stride();
            for y in self.min_y..=self.max_y.min(self.h.saturating_sub(1)) {
                let Some(&(rx0, rx1)) = self.rows.get(y) else {
                    continue;
                };
                if rx0 == ROW_EMPTY.0 {
                    continue;
                }
                let x0 = rx0 as usize;
                let x1 = (rx1 as usize + 2).min(stride); // +1 spill, +1 exclusive
                let lo = y * stride + x0;
                let hi = (y * stride + x1).min(self.acc.len());
                if let Some(slice) = self.acc.get_mut(lo..hi) {
                    slice.fill(0.0);
                }
                if let Some(slot) = self.rows.get_mut(y) {
                    *slot = ROW_EMPTY;
                }
            }
        }
        self.min_y = usize::MAX;
        self.max_y = 0;
    }

    /// Accumulates one edge. Coordinates must already be clipped to the
    /// viewport (clip_contour); small excursions are tolerated via clamping.
    fn draw_line(&mut self, x0f: f32, y0f: f32, x1f: f32, y1f: f32) {
        if !(y0f.is_finite() && y1f.is_finite() && x0f.is_finite() && x1f.is_finite()) {
            return;
        }
        if (y0f - y1f).abs() <= 1e-9 {
            return;
        }
        let (dir, p0x, p0y, p1x, p1y) = if y0f < y1f {
            (1.0f32, x0f, y0f, x1f, y1f)
        } else {
            (-1.0f32, x1f, y1f, x0f, y0f)
        };
        let dxdy = (p1x - p0x) / (p1y - p0y);
        let mut x = p0x;
        let y_start = (p0y.max(0.0)) as usize;
        let y_end = (p1y.min(self.h as f32).ceil().max(0.0)) as usize;
        if p0y < 0.0 {
            x -= p0y * dxdy;
        }
        self.min_y = self.min_y.min(y_start);
        self.max_y = self.max_y.max(y_end.min(self.h.saturating_sub(1)));

        let wmax = self.w as f32;
        let stride = self.stride();
        for y in y_start..y_end.min(self.h) {
            let linestart = y * stride;
            let yf = y as f32;
            let dy = (yf + 1.0).min(p1y) - yf.max(p0y);
            if dy <= 0.0 {
                x += dxdy * 0.0;
                continue;
            }
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (mut x0, mut x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
            x0 = x0.clamp(0.0, wmax);
            x1 = x1.clamp(0.0, wmax);
            let x0floor = x0.floor();
            let x0i = x0floor as usize;
            let x1ceil = x1.ceil();
            let x1i = x1ceil as usize;
            if let Some(slot) = self.rows.get_mut(y) {
                slot.0 = slot.0.min(x0i as u32);
                slot.1 = slot.1.max(x1i as u32);
            }
            // One slice per row instead of a checked lookup per cell. The
            // slice is stride+1 long: cell indices reach x1i <= w and the
            // single-column branch touches x0i+1 <= w+1 — that last slot is
            // the row's spill / the global slack cell (acc has stride*h+1
            // entries precisely so this slice always exists).
            let Some(row) = self.acc.get_mut(linestart..linestart + stride + 1) else {
                x = xnext;
                continue;
            };
            let mut cell = |i: usize, v: f32| {
                if let Some(slot) = row.get_mut(i) {
                    *slot += v;
                }
            };
            if x1i <= x0i + 1 {
                // Whole trapezoid within one pixel column.
                let xmf = 0.5 * (x + xnext) - x0floor;
                let xmf = xmf.clamp(0.0, 1.0);
                cell(x0i, d - d * xmf);
                cell(x0i + 1, d * xmf);
            } else {
                let s = 1.0 / (x1 - x0);
                let x0f = x0 - x0floor;
                let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
                let x1f = x1 - x1ceil + 1.0;
                let am = 0.5 * s * x1f * x1f;
                cell(x0i, d * a0);
                if x1i == x0i + 2 {
                    cell(x0i + 1, d * (1.0 - a0 - am));
                } else {
                    let a1 = s * (1.5 - x0f);
                    cell(x0i + 1, d * (a1 - a0));
                    let ds = d * s;
                    if let Some(mid) = row.get_mut((x0i + 2)..x1i.saturating_sub(1)) {
                        for slot in mid {
                            *slot += ds;
                        }
                    }
                    let a2 = a1 + (x1i.saturating_sub(x0i).saturating_sub(3)) as f32 * s;
                    if let Some(slot) = row.get_mut(x1i.saturating_sub(1)) {
                        *slot += d * (1.0 - a2 - am);
                    }
                }
                if let Some(slot) = row.get_mut(x1i) {
                    *slot += d * am;
                }
            }
            x = xnext;
        }
    }

    pub fn fill_contours(&mut self, contours: &[Contour]) {
        for contour in contours {
            let pts = &contour.points;
            if pts.len() < 3 {
                continue;
            }
            for (i, cur) in pts.iter().enumerate() {
                let next = pts.get(i + 1).or_else(|| pts.first());
                if let Some(next) = next {
                    self.draw_line(cur.x, cur.y, next.x, next.y);
                }
            }
        }
    }

    /// Runs the per-row prefix sum over the touched bounding box and hands
    /// (row, first_column, coverage slice) to `f`. Coverage values 0..=255;
    /// columns outside the slice have zero coverage.
    pub fn sweep(
        &mut self,
        rule: FillRule,
        antialias: bool,
        mut f: impl FnMut(usize, usize, &[u8]),
    ) {
        if self.min_y > self.max_y {
            return;
        }
        let stride = self.stride();
        // Split borrow: coverage scratch vs accumulation buffer.
        let mut cov = core::mem::take(&mut self.cov);
        for y in self.min_y..=self.max_y.min(self.h.saturating_sub(1)) {
            let Some(&(rx0, rx1)) = self.rows.get(y) else {
                continue;
            };
            if rx0 == ROW_EMPTY.0 {
                continue;
            }
            let x0 = (rx0 as usize).min(self.w.saturating_sub(1));
            let x1 = (rx1 as usize + 1).min(self.w); // exclusive
            if x1 <= x0 {
                continue;
            }
            let lo = y.saturating_mul(stride).saturating_add(x0);
            let hi = lo.saturating_add(x1 - x0).min(self.acc.len());
            let Some(row) = self.acc.get(lo..hi) else {
                continue;
            };
            let Some(cov_slice) = cov.get_mut(..row.len()) else {
                continue;
            };
            let mut sum = 0.0f32;
            for (dst, src) in cov_slice.iter_mut().zip(row.iter()) {
                sum += *src;
                let c = match rule {
                    FillRule::NonZero => sum.abs().min(1.0),
                    FillRule::EvenOdd => {
                        let m = sum.rem_euclid(2.0);
                        if m > 1.0 {
                            2.0 - m
                        } else {
                            m
                        }
                    }
                };
                *dst = if antialias {
                    (c * 255.0 + 0.5) as u8
                } else if c >= 0.5 {
                    255
                } else {
                    0
                };
            }
            let filled = cov_slice.len();
            f(y, x0, cov.get(..filled).unwrap_or(&[]));
        }
        self.cov = cov;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Contour;
    use crate::math::Vec2;

    fn cov_sum(r: &mut Rasterizer, rule: FillRule) -> u64 {
        let mut total = 0u64;
        r.sweep(rule, true, |_, _, row| {
            total += row.iter().map(|&c| u64::from(c)).sum::<u64>();
        });
        total
    }

    #[test]
    fn fills_a_square() {
        let mut r = Rasterizer::new(20, 20);
        let square = Contour {
            points: vec![
                Vec2::new(5.0, 5.0),
                Vec2::new(15.0, 5.0),
                Vec2::new(15.0, 15.0),
                Vec2::new(5.0, 15.0),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        r.fill_contours(&[square]);
        let total = cov_sum(&mut r, FillRule::NonZero);
        // 10x10 px fully covered = 100 * 255.
        assert!((total as i64 - 25_500).abs() < 300, "total={total}");
    }

    #[test]
    fn disabled_antialiasing_emits_binary_coverage() {
        let mut r = Rasterizer::new(20, 20);
        let square = Contour {
            points: vec![
                Vec2::new(5.25, 5.25),
                Vec2::new(14.75, 5.25),
                Vec2::new(14.75, 14.75),
                Vec2::new(5.25, 14.75),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        r.fill_contours(&[square]);
        let mut saw_covered = false;
        r.sweep(FillRule::NonZero, false, |_, _, row| {
            saw_covered |= row.contains(&255);
            assert!(row.iter().all(|&coverage| coverage == 0 || coverage == 255));
        });
        assert!(saw_covered);
    }

    #[test]
    fn winding_direction_independent() {
        let mut r = Rasterizer::new(20, 20);
        let ccw = Contour {
            points: vec![
                Vec2::new(5.0, 5.0),
                Vec2::new(5.0, 15.0),
                Vec2::new(15.0, 15.0),
                Vec2::new(15.0, 5.0),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        r.fill_contours(&[ccw]);
        let total = cov_sum(&mut r, FillRule::NonZero);
        assert!((total as i64 - 25_500).abs() < 300, "total={total}");
    }

    #[test]
    fn evenodd_hole() {
        let mut r = Rasterizer::new(30, 30);
        let outer = Contour {
            points: vec![
                Vec2::new(5.0, 5.0),
                Vec2::new(25.0, 5.0),
                Vec2::new(25.0, 25.0),
                Vec2::new(5.0, 25.0),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        let inner = Contour {
            points: vec![
                Vec2::new(10.0, 10.0),
                Vec2::new(20.0, 10.0),
                Vec2::new(20.0, 20.0),
                Vec2::new(10.0, 20.0),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        r.fill_contours(&[outer, inner]);
        let total = cov_sum(&mut r, FillRule::EvenOdd);
        // 20x20 minus 10x10 hole = 300 px.
        assert!((total as i64 - 300 * 255).abs() < 600, "total={total}");
    }

    #[test]
    fn half_covered_pixel_antialiases() {
        let mut r = Rasterizer::new(4, 4);
        // Triangle covering half of one pixel row region.
        let tri = Contour {
            points: vec![
                Vec2::new(1.0, 1.0),
                Vec2::new(3.0, 1.0),
                Vec2::new(1.0, 3.0),
            ],
            anchors: Vec::new(),
            ..Default::default()
        };
        r.fill_contours(&[tri]);
        let total = cov_sum(&mut r, FillRule::NonZero);
        // Triangle area = 2 px → ~510.
        assert!((total as i64 - 510).abs() < 60, "total={total}");
    }
}
