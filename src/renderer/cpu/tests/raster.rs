use super::*;
use crate::geometry::Contour;
use crate::math::Vec2;
use alloc::vec;
use alloc::vec::Vec;

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
    points: vec![Vec2::new(5.0, 5.0), Vec2::new(15.0, 5.0), Vec2::new(15.0, 15.0), Vec2::new(5.0, 15.0)],
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
    points: vec![Vec2::new(5.25, 5.25), Vec2::new(14.75, 5.25), Vec2::new(14.75, 14.75), Vec2::new(5.25, 14.75)],
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
    points: vec![Vec2::new(5.0, 5.0), Vec2::new(5.0, 15.0), Vec2::new(15.0, 15.0), Vec2::new(15.0, 5.0)],
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
    points: vec![Vec2::new(5.0, 5.0), Vec2::new(25.0, 5.0), Vec2::new(25.0, 25.0), Vec2::new(5.0, 25.0)],
    anchors: Vec::new(),
    ..Default::default()
  };
  let inner = Contour {
    points: vec![Vec2::new(10.0, 10.0), Vec2::new(20.0, 10.0), Vec2::new(20.0, 20.0), Vec2::new(10.0, 20.0)],
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
    points: vec![Vec2::new(1.0, 1.0), Vec2::new(3.0, 1.0), Vec2::new(1.0, 3.0)],
    anchors: Vec::new(),
    ..Default::default()
  };
  r.fill_contours(&[tri]);
  let total = cov_sum(&mut r, FillRule::NonZero);
  // Triangle area = 2 px → ~510.
  assert!((total as i64 - 510).abs() < 60, "total={total}");
}

#[test]
fn capture_capacity_hint_matches_incremental_growth() {
  let mut r = Rasterizer::new(64, 64);
  let shape = Contour {
    points: vec![Vec2::new(2.25, 3.5), Vec2::new(47.75, 8.25), Vec2::new(34.5, 51.75), Vec2::new(6.0, 39.25)],
    anchors: Vec::new(),
    ..Default::default()
  };
  r.fill_contours(&[shape]);
  let expected = r.capture_capacities();
  let mut rows = Vec::new();
  let mut data = Vec::new();
  r.sweep(FillRule::NonZero, true, |y, x0, row| {
    rows.push((y, x0, row.len()));
    data.extend_from_slice(row);
  });
  assert_eq!(expected, (rows.capacity(), data.capacity()));
}
