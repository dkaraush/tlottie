use super::*;
use alloc::vec;
use alloc::vec::Vec;

#[test]
fn rounds_each_corner_of_a_closed_square() {
  let square = Contour {
    points: vec![Vec2::new(0.0, 0.0), Vec2::new(20.0, 0.0), Vec2::new(20.0, 20.0), Vec2::new(0.0, 20.0)],
    anchors: Vec::new(),
    inv_lin: None,
  };
  let rounded = round_polyline_corners(&square, true, 4.0);
  assert_eq!(rounded.points.len(), 20);
  assert!(!rounded.points.contains(&Vec2::new(0.0, 0.0)));
}
