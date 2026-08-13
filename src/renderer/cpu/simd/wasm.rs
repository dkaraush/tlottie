//! simd128 kernels, 4 pixels per iteration. Because simd128 is enabled
//! at compile time (module-level cfg), the intrinsics are plain safe
//! calls — no `#[target_feature]`/runtime dispatch needed; only the raw
//! v128 loads/stores are `unsafe`.
//!
//! No `vld4`-style de-interleave exists in simd128, so the u16 kernels
//! work on the natural interleaved layout: one v128 = 4 pixels
//! [B,G,R,A ×4], widened to two u16x8 halves (2 pixels each). Per-pixel
//! factors (coverage, alpha) are replicated across each pixel's four
//! channel lanes with a byte swizzle. u16 lanes never overflow: every
//! product is (<=255)*(<=255) <= 65025 and div255's intermediate stays
//! <= 65407.

use core::arch::wasm32::{
  f32x4, f32x4_abs, f32x4_add, f32x4_ge, f32x4_lt, f32x4_max, f32x4_min, f32x4_mul, f32x4_neg, f32x4_splat, f32x4_sqrt, f32x4_sub, i16x8_all_true, u16x8, u16x8_add, u16x8_eq, u16x8_extend_high_u8x16, u16x8_extend_low_u8x16,
  u16x8_max, u16x8_min, u16x8_mul, u16x8_shr, u16x8_splat, u16x8_sub, u32x4_all_true, u32x4_eq, u32x4_extract_lane, u32x4_splat, u32x4_trunc_sat_f32x4, u8x16, u8x16_narrow_i16x8, u8x16_swizzle, v128,
  v128_and, v128_bitselect, v128_load, v128_store,
};

/// Swizzle pattern replicating each of the low 4 bytes across one
/// pixel's 4 channel lanes: [x0,x1,x2,x3,..] -> [x0 x4, x1 x4, ...].
#[inline]
fn rep4() -> v128 {
  u8x16(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
#[inline]
fn div255_round(n: v128) -> v128 {
  let t = u16x8_add(n, u16x8_splat(127));
  let u = u16x8_add(u16x8_add(t, u16x8_shr(t, 8)), u16x8_splat(1));
  u16x8_shr(u, 8)
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
fn over(d: v128, s: v128, inv: v128) -> v128 {
  u16x8_min(u16x8_add(s, u16x8_shr(u16x8_mul(d, u16x8_add(inv, u16x8_splat(1))), 8)), u16x8_splat(255))
}

#[inline]
fn alpha_over8(dst: v128, source: v128) -> v128 {
  over(dst, source, u16x8_sub(u16x8_splat(255), source))
}

#[inline]
fn alpha_load(bytes: &[u8]) -> v128 {
  #[allow(unsafe_code)]
  unsafe {
    v128_load(bytes.as_ptr().cast::<v128>())
  }
}

#[inline]
fn alpha_store(bytes: &mut [u8], value: v128) {
  #[allow(unsafe_code)]
  unsafe {
    v128_store(bytes.as_mut_ptr().cast::<v128>(), value)
  }
}

pub(super) fn alpha_blend_solid_wasm(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = u16x8_splat(u16::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(16).zip(coverage.chunks_exact(16)) {
    let d = alpha_load(dst);
    let c = alpha_load(coverage);
    let sl = div255_round(u16x8_mul(u16x8_extend_low_u8x16(c), alpha));
    let sh = div255_round(u16x8_mul(u16x8_extend_high_u8x16(c), alpha));
    alpha_store(dst, u8x16_narrow_i16x8(alpha_over8(u16x8_extend_low_u8x16(d), sl), alpha_over8(u16x8_extend_high_u8x16(d), sh)));
  }
}

pub(super) fn alpha_blend_product_wasm(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(16).zip(lhs.chunks_exact(16)).zip(rhs.chunks_exact(16)) {
    let d = alpha_load(dst);
    let l = alpha_load(lhs);
    let r = alpha_load(rhs);
    let sl = div255_round(u16x8_mul(u16x8_extend_low_u8x16(l), u16x8_extend_low_u8x16(r)));
    let sh = div255_round(u16x8_mul(u16x8_extend_high_u8x16(l), u16x8_extend_high_u8x16(r)));
    alpha_store(dst, u8x16_narrow_i16x8(alpha_over8(u16x8_extend_low_u8x16(d), sl), alpha_over8(u16x8_extend_high_u8x16(d), sh)));
  }
}

pub(super) fn alpha_blend_uniform_wasm(dst: &mut [u8], source: u8) {
  let source = u16x8_splat(u16::from(source));
  for dst in dst.chunks_exact_mut(16) {
    let d = alpha_load(dst);
    alpha_store(dst, u8x16_narrow_i16x8(alpha_over8(u16x8_extend_low_u8x16(d), source), alpha_over8(u16x8_extend_high_u8x16(d), source)));
  }
}

pub(super) fn alpha_composite_over_wasm(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = u16x8_splat(u16::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = alpha_load(dst);
    let s = alpha_load(src);
    let sl = div255_round(u16x8_mul(u16x8_extend_low_u8x16(s), opacity));
    let sh = div255_round(u16x8_mul(u16x8_extend_high_u8x16(s), opacity));
    alpha_store(dst, u8x16_narrow_i16x8(alpha_over8(u16x8_extend_low_u8x16(d), sl), alpha_over8(u16x8_extend_high_u8x16(d), sh)));
  }
}

pub(super) fn alpha_multiply_wasm(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(16).zip(factors.chunks_exact(16)) {
    let d = alpha_load(dst);
    let f = alpha_load(factors);
    let lo = div255_round(u16x8_mul(u16x8_extend_low_u8x16(d), u16x8_extend_low_u8x16(f)));
    let hi = div255_round(u16x8_mul(u16x8_extend_high_u8x16(d), u16x8_extend_high_u8x16(f)));
    alpha_store(dst, u8x16_narrow_i16x8(lo, hi));
  }
}

pub(super) fn alpha_matte_wasm(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = u16x8_splat(u16::from(opacity));
  let full = u16x8_splat(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = alpha_load(dst);
    let s = alpha_load(src);
    let mut fl = div255_round(u16x8_mul(u16x8_extend_low_u8x16(s), opacity));
    let mut fh = div255_round(u16x8_mul(u16x8_extend_high_u8x16(s), opacity));
    if inverted {
      fl = u16x8_sub(full, fl);
      fh = u16x8_sub(full, fh);
    }
    let lo = div255_round(u16x8_mul(u16x8_extend_low_u8x16(d), fl));
    let hi = div255_round(u16x8_mul(u16x8_extend_high_u8x16(d), fh));
    alpha_store(dst, u8x16_narrow_i16x8(lo, hi));
  }
}

pub(super) fn alpha_mask_combine_wasm(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = u16x8_splat(u16::from(opacity));
  let full = u16x8_splat(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = alpha_load(dst);
    let s = alpha_load(src);
    let (dl, dh) = (u16x8_extend_low_u8x16(d), u16x8_extend_high_u8x16(d));
    let (mut sl, mut sh) = (u16x8_extend_low_u8x16(s), u16x8_extend_high_u8x16(s));
    if inverted {
      sl = u16x8_sub(full, sl);
      sh = u16x8_sub(full, sh);
    }
    let (cl, ch) = (div255_round(u16x8_mul(sl, opacity)), div255_round(u16x8_mul(sh, opacity)));
    let combine = |old, contribution| match mode {
      b's' => div255_round(u16x8_mul(old, u16x8_sub(full, contribution))),
      b'i' => div255_round(u16x8_mul(old, contribution)),
      b'f' => u16x8_sub(u16x8_max(old, contribution), u16x8_min(old, contribution)),
      _ => u16x8_add(contribution, div255_round(u16x8_mul(u16x8_sub(full, contribution), old))),
    };
    alpha_store(dst, u8x16_narrow_i16x8(combine(dl, cl), combine(dh, ch)));
  }
}

pub(super) fn fill_span_solid_wasm(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let src = u16x8(sr as u16, sg as u16, sb as u16, 255, sr as u16, sg as u16, sb as u16, 255);
  let sa_w = u16x8_splat(sa as u16);
  let full = u16x8_splat(255);
  for (dpx, cpx) in dst.chunks_exact_mut(4).zip(cov.chunks_exact(4)) {
    let cw = u32::from_le_bytes(cpx.try_into().unwrap_or([0; 4]));
    let crep = u8x16_swizzle(u32x4_splat(cw), rep4());
    let cov_lo = u16x8_extend_low_u8x16(crep);
    let cov_hi = u16x8_extend_high_u8x16(crep);
    let ca_lo = div255_round(u16x8_mul(cov_lo, sa_w));
    let ca_hi = div255_round(u16x8_mul(cov_hi, sa_w));
    let s_lo = div255_round(u16x8_mul(src, ca_lo));
    let s_hi = div255_round(u16x8_mul(src, ca_hi));
    let inv_lo = u16x8_sub(full, ca_lo);
    let inv_hi = u16x8_sub(full, ca_hi);
    // SAFETY: chunks_exact_mut(4) guarantees exactly 4 u32
    // (16 bytes) at dpx; v128 loads/stores allow unaligned.
    #[allow(unsafe_code)]
    let d = unsafe { v128_load(dpx.as_ptr().cast::<v128>()) };
    let o_lo = over(u16x8_extend_low_u8x16(d), s_lo, inv_lo);
    let o_hi = over(u16x8_extend_high_u8x16(d), s_hi, inv_hi);
    let out = u8x16_narrow_i16x8(o_lo, o_hi);
    // SAFETY: same 16-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      v128_store(dpx.as_mut_ptr().cast::<v128>(), out)
    };
  }
}

pub(super) fn fill_span_uniform_wasm(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let s = u16x8(s_r as u16, s_g as u16, s_b as u16, ca as u16, s_r as u16, s_g as u16, s_b as u16, ca as u16);
  let inv = u16x8_splat(255 - ca as u16);
  for dpx in dst.chunks_exact_mut(4) {
    // SAFETY: chunks_exact_mut(4) guarantees exactly 4 u32
    // (16 bytes) at dpx; v128 loads/stores allow unaligned.
    #[allow(unsafe_code)]
    let d = unsafe { v128_load(dpx.as_ptr().cast::<v128>()) };
    let o_lo = over(u16x8_extend_low_u8x16(d), s, inv);
    let o_hi = over(u16x8_extend_high_u8x16(d), s, inv);
    let out = u8x16_narrow_i16x8(o_lo, o_hi);
    // SAFETY: same 16-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      v128_store(dpx.as_mut_ptr().cast::<v128>(), out)
    };
  }
}

pub(super) fn apply_matte_alpha_wasm(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let arep_pat = u8x16(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
  let opacity = u16x8_splat(u16::from(source_opacity));
  let full = u16x8_splat(255);
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    // SAFETY: chunks_exact guarantees exactly 4 u32 (16 bytes) at
    // both pointers; v128 loads/stores allow unaligned.
    #[allow(unsafe_code)]
    let s = unsafe { v128_load(spx.as_ptr().cast::<v128>()) };
    let arep = u8x16_swizzle(s, arep_pat);
    let mut fl = div255_round(u16x8_mul(u16x8_extend_low_u8x16(arep), opacity));
    let mut fh = div255_round(u16x8_mul(u16x8_extend_high_u8x16(arep), opacity));
    if inverted {
      fl = u16x8_sub(full, fl);
      fh = u16x8_sub(full, fh);
    }
    if i16x8_all_true(u16x8_eq(fl, full)) && i16x8_all_true(u16x8_eq(fh, full)) {
      continue;
    }
    // SAFETY: same 16-byte span as the source load above.
    #[allow(unsafe_code)]
    let d = unsafe { v128_load(dpx.as_ptr().cast::<v128>()) };
    let o_lo = div255_round(u16x8_mul(u16x8_extend_low_u8x16(d), fl));
    let o_hi = div255_round(u16x8_mul(u16x8_extend_high_u8x16(d), fh));
    let out = u8x16_narrow_i16x8(o_lo, o_hi);
    // SAFETY: same 16-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      v128_store(dpx.as_mut_ptr().cast::<v128>(), out)
    };
  }
}

pub(super) fn composite_over_wasm(dst: &mut [u32], src: &[u32], k: u32) {
  // Swizzle replicating each pixel's alpha byte across its 4 lanes.
  let arep_pat = u8x16(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
  let kq = u16x8_splat(k as u16);
  let full = u16x8_splat(255);
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    // SAFETY: chunks_exact guarantees exactly 4 u32 (16 bytes) at
    // both pointers; v128 loads/stores allow unaligned.
    #[allow(unsafe_code)]
    let (d, s) = unsafe { (v128_load(dpx.as_ptr().cast::<v128>()), v128_load(spx.as_ptr().cast::<v128>())) };
    if u32x4_all_true(u32x4_eq(s, u32x4_splat(0))) {
      continue;
    }
    let s_lo = div255_round(u16x8_mul(u16x8_extend_low_u8x16(s), kq));
    let s_hi = div255_round(u16x8_mul(u16x8_extend_high_u8x16(s), kq));
    let arep = u8x16_swizzle(s, arep_pat);
    let inv_lo = u16x8_sub(full, div255_round(u16x8_mul(u16x8_extend_low_u8x16(arep), kq)));
    let inv_hi = u16x8_sub(full, div255_round(u16x8_mul(u16x8_extend_high_u8x16(arep), kq)));
    let o_lo = over(u16x8_extend_low_u8x16(d), s_lo, inv_lo);
    let o_hi = over(u16x8_extend_high_u8x16(d), s_hi, inv_hi);
    let out = u8x16_narrow_i16x8(o_lo, o_hi);
    // SAFETY: same 16-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      v128_store(dpx.as_mut_ptr().cast::<v128>(), out)
    };
  }
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`, with lanes failing `valid`
/// forced to the u32::MAX sentinel (LUT miss -> transparent 0). Matches
/// the scalar `is_finite` gate + `as usize` truncation.
#[inline]
fn lut_indices(t: v128, valid: v128, scale: v128) -> v128 {
  let tc = f32x4_min(f32x4_max(t, f32x4_splat(0.0)), f32x4_splat(1.0));
  let fidx = f32x4_add(f32x4_mul(tc, scale), f32x4_splat(0.5));
  let idx = u32x4_trunc_sat_f32x4(fidx);
  v128_bitselect(idx, u32x4_splat(u32::MAX), valid)
}

/// 4 scalar LUT fetches (no gather in simd128; sentinel misses -> 0).
#[inline]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: v128) {
  let i0 = u32x4_extract_lane::<0>(idx) as usize;
  let i1 = u32x4_extract_lane::<1>(idx) as usize;
  let i2 = u32x4_extract_lane::<2>(idx) as usize;
  let i3 = u32x4_extract_lane::<3>(idx) as usize;
  if let [o0, o1, o2, o3] = chunk {
    *o0 = lut.get(i0).copied().unwrap_or(0);
    *o1 = lut.get(i1).copied().unwrap_or(0);
    *o2 = lut.get(i2).copied().unwrap_or(0);
    *o3 = lut.get(i3).copied().unwrap_or(0);
  }
}

#[inline]
fn blend4_over_k255(dpx: &mut [u32], src: &[u32; 4]) {
  // Source and destination are premultiplied RGBA bytes. k=255 means
  // source channels pass through unchanged.
  let arep_pat = u8x16(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
  let full = u16x8_splat(255);
  // SAFETY: callers pass 4-pixel chunks: exactly 16 readable/writable
  // bytes for `dpx`, and exactly 16 readable bytes for `src`.
  #[allow(unsafe_code)]
  let (d, s) = unsafe { (v128_load(dpx.as_ptr().cast::<v128>()), v128_load(src.as_ptr().cast::<v128>())) };
  let s_lo = u16x8_extend_low_u8x16(s);
  let s_hi = u16x8_extend_high_u8x16(s);
  let arep = u8x16_swizzle(s, arep_pat);
  let inv_lo = u16x8_sub(full, u16x8_extend_low_u8x16(arep));
  let inv_hi = u16x8_sub(full, u16x8_extend_high_u8x16(arep));
  let o_lo = over(u16x8_extend_low_u8x16(d), s_lo, inv_lo);
  let o_hi = over(u16x8_extend_high_u8x16(d), s_hi, inv_hi);
  let out = u8x16_narrow_i16x8(o_lo, o_hi);
  // SAFETY: same 16-byte destination span as the load above.
  #[allow(unsafe_code)]
  unsafe {
    v128_store(dpx.as_mut_ptr().cast::<v128>(), out)
  };
}

#[inline]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: v128) {
  let i0 = u32x4_extract_lane::<0>(idx) as usize;
  let i1 = u32x4_extract_lane::<1>(idx) as usize;
  let i2 = u32x4_extract_lane::<2>(idx) as usize;
  let i3 = u32x4_extract_lane::<3>(idx) as usize;
  let src = [
    lut.get(i0).copied().unwrap_or(0),
    lut.get(i1).copied().unwrap_or(0),
    lut.get(i2).copied().unwrap_or(0),
    lut.get(i3).copied().unwrap_or(0),
  ];
  blend4_over_k255(dpx, &src);
}

/// 4-lane linear gradient LUT fill: `t = row_base + X·dt` at absolute
/// device column `X = x_start + lane` (segmentation-invariant), exactly
/// the scalar form.
pub(super) fn linear_lut_fill_wasm(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  // X advances by exact integer f32 adds (exact <= 2^24); each chunk
  // recomputes `row_base + X·dt`, the same expression as scalar.
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let t0v = f32x4_splat(row_base);
  let dtv = f32x4_splat(dt);
  let four = f32x4_splat(4.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in out.chunks_exact_mut(4) {
    let t = f32x4_add(t0v, f32x4_mul(kf, dtv));
    // is_finite parity: NaN fails the compare, ±inf fails |t|<inf.
    let finite = f32x4_lt(f32x4_abs(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = f32x4_add(kf, four);
  }
}

pub(super) fn linear_lut_over_wasm(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let t0v = f32x4_splat(row_base);
  let dtv = f32x4_splat(dt);
  let four = f32x4_splat(4.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let t = f32x4_add(t0v, f32x4_mul(kf, dtv));
    let finite = f32x4_lt(f32x4_abs(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = f32x4_add(kf, four);
  }
}

/// 4-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the
/// scalar `ddx*ddx + ddy*ddy`, and f32x4_sqrt matches scalar `sqrt()`,
/// so lanes agree with the `dd0 + X·d` scalar form bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(super) fn radial_lut_fill_wasm(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let dav = f32x4_splat(da);
  let dbv = f32x4_splat(db);
  let ddx0v = f32x4_splat(dd0x);
  let ddy0v = f32x4_splat(dd0y);
  let inv_rv = f32x4_splat(inv_r);
  let four = f32x4_splat(4.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in out.chunks_exact_mut(4) {
    let ddx = f32x4_add(ddx0v, f32x4_mul(kf, dav));
    let ddy = f32x4_add(ddy0v, f32x4_mul(kf, dbv));
    let gg = f32x4_add(f32x4_mul(ddx, ddx), f32x4_mul(ddy, ddy));
    let t = f32x4_mul(f32x4_sqrt(gg), inv_rv);
    let finite = f32x4_lt(f32x4_abs(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = f32x4_add(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn radial_lut_over_wasm(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let dav = f32x4_splat(da);
  let dbv = f32x4_splat(db);
  let ddx0v = f32x4_splat(dd0x);
  let ddy0v = f32x4_splat(dd0y);
  let inv_rv = f32x4_splat(inv_r);
  let four = f32x4_splat(4.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let ddx = f32x4_add(ddx0v, f32x4_mul(kf, dav));
    let ddy = f32x4_add(ddy0v, f32x4_mul(kf, dbv));
    let gg = f32x4_add(f32x4_mul(ddx, ddx), f32x4_mul(ddy, ddy));
    let t = f32x4_mul(f32x4_sqrt(gg), inv_rv);
    let finite = f32x4_lt(f32x4_abs(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = f32x4_add(kf, four);
  }
}

/// 4-lane focal (highlight) radial LUT fill; keep the scalar absolute-X
/// Horner rounding protocol so SIMD tails cannot introduce seams.
#[allow(clippy::too_many_arguments)]
pub(super) fn focal_lut_fill_wasm(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (f32x4_splat(b0), f32x4_splat(db));
  let (d0v, d1v, d2v) = (f32x4_splat(d0), f32x4_splat(d1), f32x4_splat(d2));
  let inv2av = f32x4_splat(inv2a);
  let rv = f32x4_splat(r);
  let four = f32x4_splat(4.0);
  let zero = f32x4_splat(0.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in out.chunks_exact_mut(4) {
    let b = f32x4_add(b0v, f32x4_mul(kf, dbv));
    let det = f32x4_add(d0v, f32x4_mul(kf, f32x4_add(d1v, f32x4_mul(kf, d2v))));
    let sq = f32x4_sqrt(det);
    let nb = f32x4_neg(b);
    let root = f32x4_max(f32x4_mul(f32x4_sub(nb, sq), inv2av), f32x4_mul(f32x4_add(nb, sq), inv2av));
    let valid = v128_and(v128_and(f32x4_ge(det, zero), f32x4_ge(f32x4_mul(rv, root), zero)), f32x4_lt(f32x4_abs(root), inf));
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = f32x4_add(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn focal_lut_over_wasm(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = f32x4_add(f32x4_splat(x_start), f32x4(0.0, 1.0, 2.0, 3.0));
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (f32x4_splat(b0), f32x4_splat(db));
  let (d0v, d1v, d2v) = (f32x4_splat(d0), f32x4_splat(d1), f32x4_splat(d2));
  let inv2av = f32x4_splat(inv2a);
  let rv = f32x4_splat(r);
  let four = f32x4_splat(4.0);
  let zero = f32x4_splat(0.0);
  let inf = f32x4_splat(f32::INFINITY);
  let scalev = f32x4_splat(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let b = f32x4_add(b0v, f32x4_mul(kf, dbv));
    let det = f32x4_add(d0v, f32x4_mul(kf, f32x4_add(d1v, f32x4_mul(kf, d2v))));
    let sq = f32x4_sqrt(det);
    let nb = f32x4_neg(b);
    let root = f32x4_max(f32x4_mul(f32x4_sub(nb, sq), inv2av), f32x4_mul(f32x4_add(nb, sq), inv2av));
    let valid = v128_and(v128_and(f32x4_ge(det, zero), f32x4_ge(f32x4_mul(rv, root), zero)), f32x4_lt(f32x4_abs(root), inf));
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = f32x4_add(kf, four);
  }
}
