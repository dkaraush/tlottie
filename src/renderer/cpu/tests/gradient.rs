use super::*;
use alloc::vec;

#[test]
fn first_duplicate_opacity_endpoint_wins() {
  // At an exact duplicate endpoint rlottie_2019 selects the first opacity stop at
  // that position; the later duplicate must not lift it back to opaque.
  let stops = FloatList(vec![
    0.0, 0.0, 0.792, 1.0, //
    0.5, 0.0, 0.894, 1.0, //
    1.0, 0.0, 0.996, 1.0, //
    0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0,
  ]);
  let lut = build_gradient_lut(&stops, 3, 1.0);
  assert_eq!(lut[GRADIENT_LUT_SIZE - 1] >> 24, 0);
  let mid_alpha = lut[GRADIENT_LUT_SIZE / 2] >> 24;
  assert!((127..=128).contains(&mid_alpha), "mid_alpha={mid_alpha}");
}

#[test]
fn strictly_advancing_opacity_tail_is_retained() {
  // Stops beyond the last color position remain meaningful when their
  // positions advance: this is the authored fade used by LilCifer's rays.
  let stops = FloatList(vec![
    0.0, 0.553, 0.995, 1.0, // color 1
    0.636, 0.688, 0.743, 0.985, // color 2
    0.81, 0.822, 0.491, 0.97, // color 3
    0.0, 1.0, 0.408, 1.0, 0.815, 1.0, 0.907, 0.5, 1.0, 0.0,
  ]);
  let lut = build_gradient_lut(&stops, 3, 1.0);
  assert_eq!(lut[GRADIENT_LUT_SIZE - 1] >> 24, 0);
  let tail_alpha = lut[(GRADIENT_LUT_SIZE - 1) * 9 / 10] >> 24;
  assert!((135..=145).contains(&tail_alpha), "tail_alpha={tail_alpha}");
}

#[test]
fn opacity_stops_below_final_color_stop_are_retained() {
  // Emoji gradients can contain many opacity stops between only a few color
  // stops. rlottie_2019 consumes all of these while advancing to the final color.
  let stops = FloatList(vec![
    0.0, 1.0, 0.0, 0.0, // red
    0.5, 0.0, 1.0, 0.0, // green
    0.8, 0.0, 0.0, 1.0, // blue
    0.0, 1.0, 0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.75, 0.2,
  ]);
  let lut = build_gradient_lut(&stops, 3, 1.0);
  let alpha = lut[(GRADIENT_LUT_SIZE - 1) * 3 / 4] >> 24;
  assert!((50..=52).contains(&alpha), "alpha={alpha}");
}

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
