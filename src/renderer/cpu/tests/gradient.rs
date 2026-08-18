use super::*;
use alloc::vec;

#[test]
fn opacity_stops_do_not_shift_color_interpolation() {
  let stops = FloatList(vec![
    0.0, 1.0, 0.0, 0.0, // red
    0.5, 0.0, 1.0, 0.0, // green
    1.0, 0.0, 0.0, 1.0, // blue
    0.0, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 0.5, 1.0, 0.0,
  ]);
  let lut = build_gradient_lut(&stops, 3, 1.0);
  let pixel = lut[GRADIENT_LUT_SIZE / 4];
  let (a, r, g, b) = ((pixel >> 24) & 0xff, pixel & 0xff, (pixel >> 8) & 0xff, (pixel >> 16) & 0xff);
  assert_eq!(a, 255);
  assert!((126..=128).contains(&r), "red={r}");
  assert!((127..=129).contains(&g), "green={g}");
  assert_eq!(b, 0);
}
