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

#[test]
fn lut_retains_uniform_alpha_metadata_from_stops() {
  let colors = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
  let opaque = build_gradient_lut(&FloatList(colors.to_vec()), 2, 1.0);
  assert_eq!(opaque.uniform_alpha(), Some(255));

  let mut constant = colors.to_vec();
  constant.extend([0.0, 0.5, 1.0, 0.5]);
  let constant = build_gradient_lut(&FloatList(constant), 2, 1.0);
  assert_eq!(constant.uniform_alpha(), Some(128));

  let mut varying = colors.to_vec();
  varying.extend([0.0, 0.25, 1.0, 0.75]);
  let varying = build_gradient_lut(&FloatList(varying), 2, 1.0);
  assert_eq!(varying.uniform_alpha(), None);
}
