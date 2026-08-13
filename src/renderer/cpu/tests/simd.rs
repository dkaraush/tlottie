use super::*;
use alloc::vec;
use alloc::vec::Vec;

/// Deterministic xorshift — no dev-dependency needed.
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
}

#[test]
fn fill_span_solid_neon_matches_scalar() {
  let mut rng = Rng(0x1234_5678_9abc_def0);
  for len in [1usize, 7, 8, 15, 16, 17, 31, 64, 257] {
    for _case in 0..200 {
      let cov: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
      let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let (sr, sg, sb, sa) = (rng.next() as u32 & 0xff, rng.next() as u32 & 0xff, rng.next() as u32 & 0xff, rng.next() as u32 & 0xff);
      let mut a = base.clone();
      let mut b = base.clone();
      let mut c = base.clone();
      fill_span_solid(&mut a, &cov, sr, sg, sb, sa, true);
      fill_span_solid(&mut c, &cov, sr, sg, sb, sa, false);
      fill_span_solid_scalar(&mut b, &cov, sr, sg, sb, sa);
      assert_eq!(a, b, "len={len} sr={sr} sg={sg} sb={sb} sa={sa}");
      assert_eq!(c, b, "scalar gate: len={len} sr={sr} sg={sg} sb={sb} sa={sa}");
    }
  }
}

#[test]
fn fill_span_uniform_matches_solid_oracle() {
  // fill_span_uniform over any span must equal fill_span_solid_scalar
  // over a constant coverage row — the mode-S blend contract.
  let mut rng = Rng(0x5eed_5eed_5eed_5eed);
  for len in [1usize, 7, 8, 15, 16, 33, 257] {
    for _case in 0..200 {
      let cov = rng.next() as u8;
      let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let (sr, sg, sb, sa) = (rng.next() as u32 & 0xff, rng.next() as u32 & 0xff, rng.next() as u32 & 0xff, rng.next() as u32 & 0xff);
      let mut a = base.clone();
      let mut b = base.clone();
      fill_span_uniform(&mut a, cov, sr, sg, sb, sa);
      let cov_row = vec![cov; len];
      fill_span_solid_scalar(&mut b, &cov_row, sr, sg, sb, sa);
      assert_eq!(a, b, "len={len} cov={cov} sr={sr} sg={sg} sb={sb} sa={sa}");
    }
  }
}

#[test]
fn linear_lut_fill_neon_matches_scalar_form() {
  // NEON lanes recompute `row_base + X·dt` (X = absolute column)
  // exactly like the scalar form; results must be identical.
  let mut rng = Rng(0x2468_ace0_1357_9bdf);
  let lut: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x0101_0101)).collect();
  for len in [1usize, 4, 15, 16, 17, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let row_base = f(&mut rng) / 50.0;
        let dt = f(&mut rng) / 5000.0;
        let mut a = vec![0u32; len];
        let mut b = vec![0u32; len];
        linear_lut_fill(&mut a, &lut, row_base, dt, x_start);
        let scale = (lut.len() - 1) as f32;
        linear_lut_fill_scalar(&mut b, &lut, row_base, dt, x_start, scale);
        assert_eq!(a, b, "len={len} x_start={x_start} row_base={row_base} dt={dt}");
      }
    }
  }
}

#[test]
fn radial_lut_fill_neon_matches_scalar_form() {
  // NEON lanes recompute `dd0 + X·d` (X = absolute column) exactly
  // like the scalar form; results must be identical (both differ from
  // the OLD sequential accumulation by design — that is corpus-gated).
  let mut rng = Rng(0xabcd_ef01_2345_6789);
  let lut: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x0101_0101)).collect();
  let scale = (lut.len() - 1) as f32;
  for len in [1usize, 4, 15, 16, 17, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let (x0, y0, da, db) = (f(&mut rng), f(&mut rng), f(&mut rng) / 100.0, f(&mut rng) / 100.0);
        let inv_r = ((rng.next() % 1000) as f32 + 1.0) / 5000.0;
        let mut a = vec![0u32; len];
        let mut b = vec![0u32; len];
        radial_lut_fill(&mut a, &lut, x0, y0, da, db, inv_r, x_start);
        radial_lut_fill_scalar(&mut b, &lut, x0, y0, da, db, inv_r, x_start, scale);
        assert_eq!(a, b, "len={len} x_start={x_start} x0={x0} y0={y0} da={da} db={db}");
      }
    }
  }
}

#[test]
fn focal_lut_fill_neon_matches_scalar_form() {
  let mut rng = Rng(0x1357_9bdf_2468_ace0);
  let lut: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x0101_0101)).collect();
  let scale = (lut.len() - 1) as f32;
  for len in [1usize, 4, 16, 17, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let (gx0, gy0) = (f(&mut rng), f(&mut rng));
        let (sa, sb) = (f(&mut rng) / 100.0, f(&mut rng) / 100.0);
        let (dx, dy) = (f(&mut rng) / 10.0, f(&mut rng) / 10.0);
        let a = f(&mut rng);
        if a.abs() < 1e-6 {
          continue;
        }
        let inv2a = 1.0 / (2.0 * a);
        let r = f(&mut rng);
        let mut va = vec![0u32; len];
        let mut vb = vec![0u32; len];
        focal_lut_fill(&mut va, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start);
        focal_lut_fill_scalar(&mut vb, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
        assert_eq!(va, vb, "len={len} x_start={x_start} gx0={gx0} a={a} r={r}");
      }
    }
  }
}

#[test]
fn focal_horner_is_segmentation_invariant() {
  let lut: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x0101_0101)).collect();
  let cases = [
    // Ordinary positive/negative quadratic coefficients.
    (0.31, -0.72, 0.004, -0.003, 0.8, -0.25, 0.91, 1.0),
    (-1.2, 0.4, -0.002, 0.006, -0.3, 0.9, -0.47, 1.0),
    // Close to a degenerate cone and with the cone direction flipped.
    (0.0003, -0.0002, 0.00001, -0.00002, 0.7, 0.2, 0.00001, -1.0),
  ];
  for &(g0x, g0y, sa, sb, dx, dy, a, r) in &cases {
    let inv2a = 1.0 / (2.0 * a);
    for &x_start in &[0.0f32, 3.0, 137.0, 719.0] {
      let mut whole = vec![0u32; 257];
      focal_lut_fill(&mut whole, &lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start);

      let mut split = vec![0u32; whole.len()];
      let mut begin = 0usize;
      for end in [1usize, 4, 17, 63, 128, 199, 257] {
        focal_lut_fill(&mut split[begin..end], &lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + begin as f32);
        begin = end;
      }
      assert_eq!(whole, split, "x_start={x_start} a={a} r={r}");
    }
  }
}

#[test]
fn focal_horner_coefficients_match_f64_oracle() {
  let mut rng = Rng(0xface_cafe_dead_beef);
  for _ in 0..2000 {
    let unit = |r: &mut Rng| ((r.next() >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0;
    let g0x = (unit(&mut rng) * 2.0) as f32;
    let g0y = (unit(&mut rng) * 2.0) as f32;
    let sa = (unit(&mut rng) * 0.004) as f32;
    let sb = (unit(&mut rng) * 0.004) as f32;
    let dx = unit(&mut rng) as f32;
    let dy = unit(&mut rng) as f32;
    let a = (unit(&mut rng) * 1.5) as f32;
    let x = (rng.next() % 1440) as f32;

    let (b0, db, d0, d1, d2) = focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
    let b = b0 + x * db;
    let det = d0 + x * (d1 + x * d2);

    let gx64 = g0x as f64 + x as f64 * sa as f64;
    let gy64 = g0y as f64 + x as f64 * sb as f64;
    let b64 = 2.0 * (gx64 * dx as f64 + gy64 * dy as f64);
    let det64 = b64 * b64 + 4.0 * a as f64 * (gx64 * gx64 + gy64 * gy64);
    let b_err = (b as f64 - b64).abs();
    let det_err = (det as f64 - det64).abs();
    assert!(b_err <= 2.0e-5 * (1.0 + b64.abs()), "b={b} oracle={b64}");
    assert!(det_err <= 2.0e-4 * (1.0 + det64.abs()), "det={det} oracle={det64}");
  }
}

#[test]
fn focal_horner_preserves_invalid_root_semantics() {
  let lut = vec![0xff00_0000u32; 1024];
  let scale = (lut.len() - 1) as f32;

  // a<0 with d=0 makes D negative away from the focal point.
  let mut invalid_det = [0xdead_beefu32; 4];
  focal_lut_fill_scalar(&mut invalid_det, &lut, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0, -0.5, 1.0, 0.0, scale);
  assert_eq!(invalid_det, [0; 4]);

  // A finite root pointing behind the focal cone remains transparent.
  let mut behind = [0xdead_beefu32; 4];
  focal_lut_fill_scalar(&mut behind, &lut, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, -1.0, 0.0, scale);
  assert_eq!(behind, [0; 4]);
}

/// NEON == scalar even when positions go non-finite: NaN and ±inf must
/// take the sentinel (transparent 0) identically in both paths, at any
/// absolute column offset. Exercises the `is_finite`/`|t|<inf` parity.
#[test]
fn lut_fill_neon_matches_scalar_nonfinite() {
  let lut: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x0101_0101)).collect();
  let scale = (lut.len() - 1) as f32;
  let inf = f32::INFINITY;
  let nan = f32::NAN;
  for len in [1usize, 4, 16, 17, 64, 129] {
    for &x_start in &[0.0f32, 5.0, 200.0] {
      // Linear: non-finite base and non-finite step.
      for &(rb, dt) in &[(nan, 0.01f32), (0.3f32, inf), (0.3f32, -inf), (inf, 0.0f32), (-inf, 0.01f32)] {
        let mut a = vec![0u32; len];
        let mut b = vec![0u32; len];
        linear_lut_fill(&mut a, &lut, rb, dt, x_start);
        linear_lut_fill_scalar(&mut b, &lut, rb, dt, x_start, scale);
        assert_eq!(a, b, "linear len={len} x_start={x_start} rb={rb} dt={dt}");
      }
      // Radial: non-finite base/step and inv_r.
      for &(d0, d, ir) in &[(nan, 0.01f32, 0.1f32), (0.5f32, inf, 0.1f32), (0.5f32, 0.01f32, inf), (0.5f32, 0.01f32, nan)] {
        let mut a = vec![0u32; len];
        let mut b = vec![0u32; len];
        radial_lut_fill(&mut a, &lut, d0, 0.2, d, 0.01, ir, x_start);
        radial_lut_fill_scalar(&mut b, &lut, d0, 0.2, d, 0.01, ir, x_start, scale);
        assert_eq!(a, b, "radial len={len} x_start={x_start} d0={d0} d={d} ir={ir}");
      }
      // Focal: non-finite g0/step (det/root can go non-finite).
      for &(g0, s) in &[(nan, 0.01f32), (0.5f32, inf), (inf, 0.01f32)] {
        let a_coef = 0.7f32;
        let inv2a = 1.0 / (2.0 * a_coef);
        let mut a = vec![0u32; len];
        let mut b = vec![0u32; len];
        focal_lut_fill(&mut a, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start);
        focal_lut_fill_scalar(&mut b, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start, scale);
        assert_eq!(a, b, "focal len={len} x_start={x_start} g0={g0} s={s}");
      }
    }
  }
}

#[test]
fn composite_over_neon_matches_scalar() {
  let mut rng = Rng(0x0fed_cba9_8765_4321);
  for len in [1usize, 8, 15, 16, 33, 128, 511] {
    for _case in 0..200 {
      let src: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let k = rng.next() as u32 & 0xff;
      let mut a = base.clone();
      let mut b = base.clone();
      composite_over_span(&mut a, &src, k);
      composite_over_scalar(&mut b, &src, k);
      assert_eq!(a, b, "len={len} k={k}");
    }
  }
}

#[test]
fn source_over_uses_reference_byte_attenuation() {
  let mut pixel = [0x2121_2121];
  fill_span_uniform(&mut pixel, 255, 255, 255, 255, 33);
  // 33 + (33 * (256 - 33) >> 8) = 61. Rounded /255 would produce 62
  // and accumulates a visible bright fringe across faint overlaps.
  assert_eq!(pixel[0], 0x3d3d_3d3d);
}

/// Two-pass scalar reference for the fused linear kernel: fill src via
/// the LUT-fill scalar oracle, then source-over with k=255.
fn linear_over_ref(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32) {
  let scale = (lut.len() - 1) as f32;
  let mut src = vec![0u32; dst.len()];
  linear_lut_fill_scalar(&mut src, lut, row_base, dt, x_start, scale);
  composite_over_scalar(dst, &src, 255);
}

#[allow(clippy::too_many_arguments)]
fn radial_over_ref(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32) {
  let scale = (lut.len() - 1) as f32;
  let mut src = vec![0u32; dst.len()];
  radial_lut_fill_scalar(&mut src, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
  composite_over_scalar(dst, &src, 255);
}

#[allow(clippy::too_many_arguments)]
fn focal_over_ref(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32) {
  let scale = (lut.len() - 1) as f32;
  let mut src = vec![0u32; dst.len()];
  focal_lut_fill_scalar(&mut src, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
  composite_over_scalar(dst, &src, 255);
}

/// The whole point of fusion: `linear_lut_over` (NEON where available)
/// must equal the two-pass `linear_lut_fill` + `composite_over_span(255)`
/// byte-for-byte, at any absolute column and over dst of any content.
#[test]
fn linear_lut_over_matches_two_pass() {
  let mut rng = Rng(0x1111_2222_3333_4444);
  let lut: Vec<u32> = (0..1024u32).map(|i| premult(i.wrapping_mul(0x0193_7caf))).collect();
  for len in [1usize, 4, 15, 16, 17, 32, 33, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let row_base = f(&mut rng) / 50.0;
        let dt = f(&mut rng) / 5000.0;
        let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
        let mut a = base.clone();
        let mut b = base.clone();
        linear_lut_over(&mut a, &lut, row_base, dt, x_start);
        linear_over_ref(&mut b, &lut, row_base, dt, x_start);
        assert_eq!(a, b, "len={len} x_start={x_start} row_base={row_base} dt={dt}");
      }
    }
  }
}

#[test]
fn radial_lut_over_matches_two_pass() {
  let mut rng = Rng(0x5555_6666_7777_8888);
  let lut: Vec<u32> = (0..1024u32).map(|i| premult(i.wrapping_mul(0x0193_7caf))).collect();
  for len in [1usize, 4, 15, 16, 17, 32, 33, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let (x0, y0, da, db) = (f(&mut rng), f(&mut rng), f(&mut rng) / 100.0, f(&mut rng) / 100.0);
        let inv_r = ((rng.next() % 1000) as f32 + 1.0) / 5000.0;
        let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
        let mut a = base.clone();
        let mut b = base.clone();
        radial_lut_over(&mut a, &lut, x0, y0, da, db, inv_r, x_start);
        radial_over_ref(&mut b, &lut, x0, y0, da, db, inv_r, x_start);
        assert_eq!(a, b, "len={len} x_start={x_start} x0={x0} y0={y0} da={da} db={db}");
      }
    }
  }
}

#[test]
fn focal_lut_over_matches_two_pass() {
  let mut rng = Rng(0x9999_aaaa_bbbb_cccc);
  let lut: Vec<u32> = (0..1024u32).map(|i| premult(i.wrapping_mul(0x0193_7caf))).collect();
  for len in [1usize, 4, 16, 17, 32, 33, 64, 257] {
    for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
      for _ in 0..80 {
        let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
        let (gx0, gy0) = (f(&mut rng), f(&mut rng));
        let (sa, sb) = (f(&mut rng) / 100.0, f(&mut rng) / 100.0);
        let (dx, dy) = (f(&mut rng) / 10.0, f(&mut rng) / 10.0);
        let a = f(&mut rng);
        if a.abs() < 1e-6 {
          continue;
        }
        let inv2a = 1.0 / (2.0 * a);
        let r = f(&mut rng);
        let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
        let mut va = base.clone();
        let mut vb = base.clone();
        focal_lut_over(&mut va, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start);
        focal_over_ref(&mut vb, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start);
        assert_eq!(va, vb, "len={len} x_start={x_start} gx0={gx0} a={a} r={r}");
      }
    }
  }
}

/// Fused kernels must match the two-pass reference even when positions go
/// non-finite (sentinel → transparent src → dst unchanged), at any column.
#[test]
fn lut_over_matches_two_pass_nonfinite() {
  let lut: Vec<u32> = (0..1024u32).map(|i| premult(i.wrapping_mul(0x0193_7caf))).collect();
  let mut rng = Rng(0xdead_beef_0bad_f00d);
  let inf = f32::INFINITY;
  let nan = f32::NAN;
  for len in [1usize, 4, 16, 17, 64, 129] {
    for &x_start in &[0.0f32, 5.0, 200.0] {
      let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      for &(rb, dt) in &[(nan, 0.01f32), (0.3, inf), (0.3, -inf), (inf, 0.0), (-inf, 0.01)] {
        let mut a = base.clone();
        let mut b = base.clone();
        linear_lut_over(&mut a, &lut, rb, dt, x_start);
        linear_over_ref(&mut b, &lut, rb, dt, x_start);
        assert_eq!(a, b, "linear len={len} x_start={x_start} rb={rb} dt={dt}");
      }
      for &(d0, d, ir) in &[(nan, 0.01f32, 0.1f32), (0.5, inf, 0.1), (0.5, 0.01, inf), (0.5, 0.01, nan)] {
        let mut a = base.clone();
        let mut b = base.clone();
        radial_lut_over(&mut a, &lut, d0, 0.2, d, 0.01, ir, x_start);
        radial_over_ref(&mut b, &lut, d0, 0.2, d, 0.01, ir, x_start);
        assert_eq!(a, b, "radial len={len} x_start={x_start} d0={d0} d={d} ir={ir}");
      }
      for &(g0, s) in &[(nan, 0.01f32), (0.5, inf), (inf, 0.01)] {
        let a_coef = 0.7f32;
        let inv2a = 1.0 / (2.0 * a_coef);
        let mut a = base.clone();
        let mut b = base.clone();
        focal_lut_over(&mut a, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start);
        focal_over_ref(&mut b, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start);
        assert_eq!(a, b, "focal len={len} x_start={x_start} g0={g0} s={s}");
      }
    }
  }
}

/// Clamps a random word into a valid premultiplied pixel (chan <= alpha).
fn premult(w: u32) -> u32 {
  let a = (w >> 24) & 0xff;
  let r = ((w >> 16) & 0xff).min(a);
  let g = ((w >> 8) & 0xff).min(a);
  let b = (w & 0xff).min(a);
  (a << 24) | (r << 16) | (g << 8) | b
}

#[test]
fn apply_matte_alpha_matches_scalar() {
  let mut rng = Rng(0xdead_2026_beef_cafe);
  for len in [1usize, 3, 4, 7, 8, 15, 16, 17, 64, 257] {
    for _case in 0..200 {
      let src: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
      let op = rng.next() as u8;
      for &inv in &[false, true] {
        let mut a = base.clone();
        let mut b = base.clone();
        let mut c = base.clone();
        apply_matte_alpha(&mut a, &src, op, inv);
        apply_matte_alpha_scalar(&mut b, &src, op, inv);
        crate::renderer::cpu::executor::apply_matte(&mut c, &src, if inv { 2 } else { 1 }, op, crate::ChannelOrder::Rgba);
        assert_eq!(a, b, "simd vs scalar len={len} op={op} inv={inv}");
        assert_eq!(b, c, "simd scalar vs executor len={len} op={op} inv={inv}");
      }
    }
  }
}
