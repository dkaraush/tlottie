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

#[cfg(feature = "cpu")]
#[test]
fn bulk_dash_bound_preserves_zero_phases_and_duplicate_vertices() {
  let points = [Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(5.0, 2.0), Vec2::new(5.0, 2.0), Vec2::new(10.0, 0.0)];
  for pattern in [&[3.0, 2.0][..], &[0.0, 0.0, 3.0, 2.0], &[0.0001, 5.0], &[5.0, 0.0]] {
    for offset in [-7.0, 0.0, 1.99999, 9.0] {
      for anchors in [&[][..], &[true, false, false, false, true][..]] {
        let expected = super::dash_polyline(&points, anchors, false, pattern, offset);
        let actual = super::dash_polyline_bounded(&points, anchors, false, pattern, offset, &crate::renderer::frame::budget::Budget::default()).unwrap();
        assert_eq!(actual, expected);
      }
    }
  }
}
