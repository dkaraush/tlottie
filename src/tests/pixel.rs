use super::{alpha8_to_rgba_slice, alpha8_to_rgba_slice_scalar, premultiplied_rgba_to_straight, premultiplied_rgba_to_straight_slice, premultiplied_rgba_to_straight_slice_scalar};

#[test]
fn converts_transparent_opaque_and_partial_pixels() {
  assert_eq!(premultiplied_rgba_to_straight(0), [0, 0, 0, 0]);
  assert_eq!(premultiplied_rgba_to_straight(0xff56_3412), [0x12, 0x34, 0x56, 0xff]);
  assert_eq!(premultiplied_rgba_to_straight(0x8000_2040), [128, 64, 0, 128]);
}

#[test]
fn slice_conversion_matches_scalar_pixels() {
  let src = [0, 0x0012_3456, 0xff12_3456, 0x8040_2000, 0x7f01_0203, 0xffffffff, 0x0101_0000];
  let mut expected = [0u8; 28];
  let mut actual = [0u8; 28];
  premultiplied_rgba_to_straight_slice_scalar(&src, &mut expected);
  premultiplied_rgba_to_straight_slice(&src, &mut actual);
  assert_eq!(actual, expected);
}

#[test]
fn alpha8_color_expansion_matches_scalar_with_a_tail() {
  let src = [0, 1, 2, 3, 17, 31, 63, 95, 127, 159, 191, 223, 252, 253, 254, 255, 73, 149, 211];
  let mut expected = [0u8; 19 * 4];
  let mut actual = [0u8; 19 * 4];
  alpha8_to_rgba_slice_scalar(&src, &mut expected, 0x12_3456);
  alpha8_to_rgba_slice(&src, &mut actual, 0x12_3456);
  assert_eq!(actual, expected);
  assert_eq!(&actual[..8], &[0x12, 0x34, 0x56, 0, 0x12, 0x34, 0x56, 1]);
}
