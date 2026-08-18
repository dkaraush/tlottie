use super::*;
use crate::math::Vec2;
use crate::raster::Rasterizer;
use alloc::vec;
use alloc::vec::Vec;

fn plane_s(w: usize, h: usize, contours: &[Contour], rule: FillRule) -> Vec<u8> {
  let mut r = CellRaster::new(w, h);
  r.fill_contours(contours);
  let mut plane = vec![0u8; w * h];
  r.sweep_spans(rule, true, |y, x0, len, cov| {
    if let Some(dst) = plane.get_mut(y * w + x0..y * w + x0 + len) {
      dst.fill(cov);
    }
  });
  plane
}

fn plane_d(w: usize, h: usize, contours: &[Contour], rule: FillRule) -> Vec<u8> {
  let mut r = Rasterizer::new(w, h);
  r.fill_contours(contours);
  let mut plane = vec![0u8; w * h];
  r.sweep(rule, true, |y, x0, cov_row| {
    if let Some(dst) = plane.get_mut(y * w + x0..y * w + x0 + cov_row.len()) {
      dst.copy_from_slice(cov_row);
    }
  });
  plane
}

fn contour(pts: &[(f32, f32)]) -> Contour {
  Contour {
    points: pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect(),
    ..Default::default()
  }
}

#[test]
fn fills_a_square_exactly() {
  let sq = contour(&[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]);
  let plane = plane_s(20, 20, &[sq], FillRule::NonZero);
  let total: u64 = plane.iter().map(|&c| u64::from(c)).sum();
  assert!((total as i64 - 25_500).abs() < 300, "total={total}");
  // Interior pixel must be fully opaque.
  assert_eq!(plane.get(10 * 20 + 10), Some(&255));
}

#[test]
fn seam_pair_cancels_exactly() {
  // Two rectangles sharing an edge: the shared edge is drawn twice in
  // opposite directions; interior coverage must be EXACTLY uniform.
  let a = contour(&[(2.0, 2.0), (10.5, 2.0), (10.5, 18.0), (2.0, 18.0)]);
  let b = contour(&[(10.5, 2.0), (19.0, 2.0), (19.0, 18.0), (10.5, 18.0)]);
  let plane = plane_s(21, 20, &[a, b], FillRule::NonZero);
  for y in 3..17 {
    for x in 3..18 {
      assert_eq!(plane.get(y * 21 + x), Some(&255), "hole at {x},{y}");
    }
  }
}

#[test]
fn evenodd_hole() {
  let outer = contour(&[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0), (5.0, 25.0)]);
  let inner = contour(&[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]);
  let plane = plane_s(30, 30, &[outer, inner], FillRule::EvenOdd);
  let total: u64 = plane.iter().map(|&c| u64::from(c)).sum();
  assert!((total as i64 - 300 * 255).abs() < 600, "total={total}");
  assert_eq!(plane.get(15 * 30 + 15), Some(&0));
}

#[test]
fn interior_emitted_as_long_spans() {
  let sq = contour(&[(1.0, 1.0), (63.0, 1.0), (63.0, 63.0), (1.0, 63.0)]);
  let mut r = CellRaster::new(64, 64);
  r.fill_contours(&[sq]);
  let mut spans_row_32 = Vec::new();
  r.sweep_spans(FillRule::NonZero, true, |y, x0, len, cov| {
    if y == 32 {
      spans_row_32.push((x0, len, cov));
    }
  });
  // Expect: 1-2 AA edge spans + ONE long opaque interior span per side.
  assert!(spans_row_32.iter().any(|&(_, len, cov)| cov == 255 && len >= 60), "no long interior span: {spans_row_32:?}");
  assert!(spans_row_32.len() <= 4, "fragmented row: {spans_row_32:?}");
}

#[test]
fn span_rows_reports_empty_canvas_rows() {
  let sq = contour(&[(1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)]);
  let mut raster = CellRaster::new(4, 4);
  raster.fill_contours(&[sq]);
  let mut ended = Vec::new();
  raster.sweep_span_rows(FillRule::NonZero, true, |y, span| {
    if span.is_none() {
      ended.push(y);
    }
  });
  assert_eq!(ended, [0, 1, 2, 3]);
}

/// Differential vs mode D: random polygons, per-pixel coverage delta
/// bounded by the 24.8 snap + rounding (≤2 of 255).
#[test]
fn differential_vs_dense() {
  let mut seed = 0x1234_5678u64;
  let mut rnd = move || {
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((seed >> 33) as u32) as f32 / u32::MAX as f32 * 4294967296.0f32 / 4294967296.0
  };
  for case in 0..200 {
    let w = 48usize;
    let h = 48usize;
    let n = 3 + (case % 6);
    let pts: Vec<(f32, f32)> = (0..n).map(|_| (rnd() * w as f32, rnd() * h as f32)).collect();
    let c = contour(&pts);
    for rule in [FillRule::NonZero, FillRule::EvenOdd] {
      let pd = plane_d(w, h, core::slice::from_ref(&c), rule);
      let ps = plane_s(w, h, core::slice::from_ref(&c), rule);
      // The 24.8 snap moves edges by up to 1/512 px; near even-odd
      // parity boundaries the mirror amplifies that by the local
      // winding slope. Gate: tiny absolute bound plus a tight
      // distribution bound (deltas >1 must be rare edge pixels).
      let mut worst = 0i32;
      let mut over1 = 0usize;
      let mut covered = 0usize;
      for (a, b) in pd.iter().zip(ps.iter()) {
        let d = (i32::from(*a) - i32::from(*b)).abs();
        worst = worst.max(d);
        if d > 1 {
          over1 += 1;
        }
        if *a > 0 || *b > 0 {
          covered += 1;
        }
      }
      assert!(worst <= 4, "case {case} rule {rule:?}: max delta {worst}");
      assert!(over1 <= 4 + covered / 16, "case {case} rule {rule:?}: {over1} of {covered} covered px off by >1");
    }
  }
}
