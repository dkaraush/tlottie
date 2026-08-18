use super::*;
use alloc::vec::Vec;

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    let mut x = self.0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.0 = x;
    x
  }

  fn bytes(&mut self, len: usize) -> Vec<u8> {
    (0..len).map(|_| self.next() as u8).collect()
  }
}

#[test]
fn alpha_span_kernels_match_scalar_oracles() {
  let mut rng = Rng(0xa1fa_8e5d_cafe_beef);
  for len in [0usize, 1, 7, 15, 16, 17, 31, 32, 63, 257] {
    for _ in 0..200 {
      let base = rng.bytes(len);
      let lhs = rng.bytes(len);
      let rhs = rng.bytes(len);
      let alpha = rng.next() as u8;
      let opacity = rng.next() as u8;

      let mut actual = base.clone();
      let mut expected = base.clone();
      alpha_blend_solid(&mut actual, &lhs, alpha);
      alpha_blend_solid_scalar(&mut expected, &lhs, alpha);
      assert_eq!(actual, expected, "solid len={len} alpha={alpha}");

      actual.clone_from(&base);
      expected.clone_from(&base);
      alpha_blend_product(&mut actual, &lhs, &rhs);
      alpha_blend_product_scalar(&mut expected, &lhs, &rhs);
      assert_eq!(actual, expected, "product len={len}");

      actual.clone_from(&base);
      expected.clone_from(&base);
      alpha_blend_uniform(&mut actual, lhs.first().copied().unwrap_or(0), alpha);
      let source = (u32::from(lhs.first().copied().unwrap_or(0)) * u32::from(alpha) + 127) / 255;
      alpha_blend_uniform_scalar(&mut expected, source);
      assert_eq!(actual, expected, "uniform len={len}");

      actual.clone_from(&base);
      expected.clone_from(&base);
      alpha_composite_over(&mut actual, &lhs, opacity);
      alpha_composite_over_scalar(&mut expected, &lhs, opacity);
      assert_eq!(actual, expected, "composite len={len} opacity={opacity}");

      actual.clone_from(&base);
      expected.clone_from(&base);
      alpha_multiply(&mut actual, &lhs);
      alpha_multiply_scalar(&mut expected, &lhs);
      assert_eq!(actual, expected, "multiply len={len}");

      for inverted in [false, true] {
        actual.clone_from(&base);
        expected.clone_from(&base);
        alpha_matte(&mut actual, &lhs, opacity, inverted);
        alpha_matte_scalar(&mut expected, &lhs, opacity, inverted);
        assert_eq!(actual, expected, "matte len={len} inverted={inverted}");

        for mode in [b'a', b's', b'i', b'f'] {
          actual.clone_from(&base);
          expected.clone_from(&base);
          alpha_mask_combine(&mut actual, &lhs, mode, inverted, opacity);
          alpha_mask_combine_scalar(&mut expected, &lhs, mode, inverted, opacity);
          assert_eq!(actual, expected, "mask len={len} mode={mode} inverted={inverted}");
        }
      }
    }
  }
}
