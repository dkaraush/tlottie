use super::*;
use alloc::vec;
use alloc::vec::Vec;

fn stroke(pts: &[(f32, f32)], closed: bool, hw: f32, cap: Cap, join: Join, ml: f32) -> Vec<Contour> {
  let v: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
  let anchors = vec![true; v.len()];
  let mut pool = Vec::new();
  let mut segments = Vec::new();
  let mut out = Vec::new();
  stroke_outline(&v, &anchors, closed, hw, cap, join, ml, &mut pool, &mut segments, &mut out);
  out
}

fn shoelace(pts: &[Vec2]) -> f32 {
  let mut area = 0.0f32;
  for (i, p) in pts.iter().enumerate() {
    let q = pts.get(i + 1).or_else(|| pts.first()).copied().unwrap_or(*p);
    area += p.x * q.y - q.x * p.y;
  }
  area * 0.5
}

#[test]
fn one_step_arc_uses_known_outgoing_normal() {
  let start = Vec2::new(1.0, 0.0);
  let end = Vec2::new(0.96, 0.28);
  let mut border = Border::new(vec![start]);
  border.arc_to(Vec2::new(0.0, 0.0), 1.0, start, end, 0.283_794_1);
  assert_eq!(border.pts.len(), 2);
  assert_eq!(border.pts[1], end);
}

#[test]
fn straight_band_area_and_sign() {
  let out = stroke(&[(0.0, 0.0), (10.0, 0.0)], false, 2.0, Cap::Butt, Join::Miter, 4.0);
  assert_eq!(out.len(), 1);
  let a = shoelace(&out.first().map(|c| c.points.clone()).unwrap_or_default());
  // Band = 10 x 4, canonical sign POSITIVE (legacy piece sign).
  assert!((a - 40.0).abs() < 0.5, "area {a}");
}

#[test]
fn closed_square_rings_sign_invariant() {
  // The design's THE regression test: both input directions must
  // yield outer ring sign == open-band sign (+), inner ring (-).
  let ccw = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
  let cw = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
  for pts in [&ccw[..], &cw[..]] {
    let out = stroke(pts, true, 1.0, Cap::Butt, Join::Miter, 4.0);
    assert_eq!(out.len(), 2, "rings for {pts:?}");
    let mut areas: Vec<f32> = out.iter().map(|c| shoelace(&c.points)).collect();
    areas.sort_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap_or(core::cmp::Ordering::Equal));
    let inner = areas.first().copied().unwrap_or(0.0);
    let outer = areas.last().copied().unwrap_or(0.0);
    assert!(outer > 0.0, "outer ring must be + ({outer} for {pts:?})");
    assert!(inner < 0.0, "inner ring must be - ({inner} for {pts:?})");
    // 12x12 outer minus 8x8 inner.
    assert!((outer - 144.0).abs() < 1.0, "outer {outer}");
    assert!((inner + 64.0).abs() < 1.0, "inner {inner}");
  }
}

#[test]
fn reversal_miter_is_bevel_stub() {
  // A -> B -> A reversal with miter join must NOT emit a
  // miter_limit*hw needle (VFT:926-942).
  let out = stroke(&[(0.0, 0.0), (10.0, 0.0), (0.0, 0.001)], false, 2.0, Cap::Butt, Join::Miter, 10.0);
  let max_x = out.iter().flat_map(|c| c.points.iter()).fold(f32::MIN, |m, p| m.max(p.x));
  assert!(max_x <= 12.1, "reversal produced a needle: max_x {max_x}");
}

#[test]
fn over_limit_miter_bevels() {
  // 170 degree turn, limit 4: sec(85deg) ~ 11.5 > 4 -> bevel.
  let out = stroke(&[(0.0, 0.0), (10.0, 0.0), (0.17, 0.87)], false, 2.0, Cap::Butt, Join::Miter, 4.0);
  let max_x = out.iter().flat_map(|c| c.points.iter()).fold(f32::MIN, |m, p| m.max(p.x));
  assert!(max_x <= 12.1, "over-limit miter not beveled: max_x {max_x}");
}

#[test]
fn no_panic_garbage() {
  for pts in [
    vec![(0.0f32, 0.0f32)],
    vec![(0.0, 0.0), (0.0, 0.0)],
    vec![(f32::NAN, 0.0), (1.0, 1.0), (2.0, 0.0)],
    vec![(0.0, 0.0), (1e30, 1e30), (-1e30, 1e30)],
  ] {
    let v: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let anchors = vec![true; v.len()];
    let mut pool = Vec::new();
    let mut segments = Vec::new();
    let mut out = Vec::new();
    for closed in [false, true] {
      stroke_outline(&v, &anchors, closed, 2.0, Cap::Round, Join::Round, 4.0, &mut pool, &mut segments, &mut out);
    }
  }
}
