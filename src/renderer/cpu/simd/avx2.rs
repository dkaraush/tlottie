//! AVX2 kernels, 8 pixels (u32) / 32 pixels (Alpha8) per iteration.
//!
//! Runtime-dispatched on top of the SSE2 baseline: the module is compiled
//! unconditionally on x86_64, but every kernel carries
//! `#[target_feature(enable = "avx2")]` and is only reachable through the
//! `is_x86_feature_detected!("avx2")` gate in [`super`]. Bit-for-bit equals
//! the scalar oracle and the SSE2 kernels (same rounding protocol: rounded /255
//! source scaling, `dst * (256 - src) >> 8` attenuation).
//!
//! Layout mirrors the 128-bit kernels: one `__m256i` = 8 pixels of
//! interleaved [R,G,B,A], widened with `_mm256_cvtepu8_epi16` per
//! 128-bit half (4 pixels -> 16 u16 lanes each), narrowed back with
//! `packuswb`. Because `vpackuswb` packs within lanes, the two result halves
//! are re-ordered with one `vpermq` before the store.
//!
//! Unlike SSE2 (no gather), the gradient LUT kernels use `vpgatherdd` via
//! `_mm256_mask_i32gather_epi32`, fetching 8 LUT colors with one load.
//! Index lanes masked-off as invalid (the `u32::MAX` sentinel) read nothing
//! and land on the prefetched zero offset = transparent, exactly like the
//! scalar `lut.get(idx).unwrap_or(0)`.

use core::arch::x86_64::{
  _CMP_GE_OQ, _CMP_LT_OQ, _CMP_UNORD_Q, __m256, __m256i, _mm256_add_epi16, _mm256_add_ps, _mm256_and_ps, _mm256_and_si256, _mm256_andnot_ps, _mm256_andnot_si256,
  _mm256_castps_si256, _mm256_castsi128_si256, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_cvtepu8_epi16, _mm256_cvttps_epi32,
  _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_mask_i32gather_epi32, _mm256_max_epi16, _mm256_max_ps, _mm256_min_epi16, _mm256_min_ps, _mm256_movemask_epi8,
  _mm256_mul_ps, _mm256_mullo_epi16, _mm256_or_ps, _mm256_or_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_epi16,
  _mm256_setr_ps, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_shufflehi_epi16, _mm256_shufflelo_epi16, _mm256_sqrt_ps, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_sub_epi16,
  _mm256_sub_ps, _mm_cvtsi32_si128, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

/// Widens the low 16 bytes of `v` (first 4 pixels) to 16 u16 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn lo(v: __m256i) -> __m256i {
  _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v))
}

/// Widens the high 16 bytes of `v` (last 4 pixels) to 16 u16 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hi(v: __m256i) -> __m256i {
  _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1))
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
#[inline]
#[target_feature(enable = "avx2")]
fn div255_round(n: __m256i) -> __m256i {
  let t = _mm256_add_epi16(n, _mm256_set1_epi16(127));
  let u = _mm256_add_epi16(_mm256_add_epi16(t, _mm256_srli_epi16::<8>(t)), _mm256_set1_epi16(1));
  _mm256_srli_epi16::<8>(u)
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn over(d: __m256i, s: __m256i, inv: __m256i) -> __m256i {
  _mm256_min_epi16(
    _mm256_add_epi16(s, _mm256_srli_epi16::<8>(_mm256_mullo_epi16(d, _mm256_add_epi16(inv, _mm256_set1_epi16(1))))),
    _mm256_set1_epi16(255),
  )
}

#[inline]
#[target_feature(enable = "avx2")]
fn alpha_over8(dst: __m256i, source: __m256i) -> __m256i {
  over(dst, source, _mm256_sub_epi16(_mm256_set1_epi16(255), source))
}

/// Replicates each 16-lane half's alpha (channels 3/7/11/15) across its
/// pixel's four channel lanes — the `pshufb` alpha broadcast after widening,
/// via per-lane word shuffles.
#[inline]
#[target_feature(enable = "avx2")]
fn splat_alpha(half: __m256i) -> __m256i {
  _mm256_shufflehi_epi16::<0xFF>(_mm256_shufflelo_epi16::<0xFF>(half))
}

/// Packs two 16-u16 halves into 32 bytes with the correct interleaved order
/// (packus is lane-local, so re-order the middle 64-bit lanes).
#[inline]
#[target_feature(enable = "avx2")]
fn pack(a: __m256i, b: __m256i) -> __m256i {
  _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8)
}

/// Unaligned 32-byte load.
#[inline]
#[target_feature(enable = "avx2")]
fn load(bytes: *const u8) -> __m256i {
  // SAFETY: every caller passes a pointer to a `chunks_exact` window, so
  // 32 readable bytes are guaranteed; `loadu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm256_loadu_si256(bytes.cast())
  }
}

/// Unaligned 32-byte store.
#[inline]
#[target_feature(enable = "avx2")]
fn store(bytes: *mut u8, value: __m256i) {
  // SAFETY: every caller passes a pointer to a `chunks_exact_mut` window,
  // so 32 writable bytes are guaranteed; `storeu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm256_storeu_si256(bytes.cast(), value)
  }
}

// ---------------------------------------------------------------------------
// Alpha8 kernels — 32 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
pub(super) fn alpha_blend_solid_avx2(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = _mm256_set1_epi16(i16::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(32).zip(coverage.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let c = load(coverage.as_ptr());
    let sl = div255_round(_mm256_mullo_epi16(lo(c), alpha));
    let sh = div255_round(_mm256_mullo_epi16(hi(c), alpha));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_blend_product_avx2(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(32).zip(lhs.chunks_exact(32)).zip(rhs.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let l = load(lhs.as_ptr());
    let r = load(rhs.as_ptr());
    let sl = div255_round(_mm256_mullo_epi16(lo(l), lo(r)));
    let sh = div255_round(_mm256_mullo_epi16(hi(l), hi(r)));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_blend_uniform_avx2(dst: &mut [u8], source: u8) {
  let source = _mm256_set1_epi16(i16::from(source));
  for dst in dst.chunks_exact_mut(32) {
    let d = load(dst.as_ptr());
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), source), alpha_over8(hi(d), source)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_composite_over_avx2(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = _mm256_set1_epi16(i16::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let sl = div255_round(_mm256_mullo_epi16(lo(s), opacity));
    let sh = div255_round(_mm256_mullo_epi16(hi(s), opacity));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_multiply_avx2(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(32).zip(factors.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let f = load(factors.as_ptr());
    let l = div255_round(_mm256_mullo_epi16(lo(d), lo(f)));
    let h = div255_round(_mm256_mullo_epi16(hi(d), hi(f)));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_matte_avx2(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = _mm256_set1_epi16(i16::from(opacity));
  let full = _mm256_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let mut fl = div255_round(_mm256_mullo_epi16(lo(s), opacity));
    let mut fh = div255_round(_mm256_mullo_epi16(hi(s), opacity));
    if inverted {
      fl = _mm256_sub_epi16(full, fl);
      fh = _mm256_sub_epi16(full, fh);
    }
    let l = div255_round(_mm256_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm256_mullo_epi16(hi(d), fh));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn alpha_mask_combine_avx2(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = _mm256_set1_epi16(i16::from(opacity));
  let full = _mm256_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let (dl, dh) = (lo(d), hi(d));
    let (mut sl, mut sh) = (lo(s), hi(s));
    if inverted {
      sl = _mm256_sub_epi16(full, sl);
      sh = _mm256_sub_epi16(full, sh);
    }
    let (cl, ch) = (div255_round(_mm256_mullo_epi16(sl, opacity)), div255_round(_mm256_mullo_epi16(sh, opacity)));
    let combine = |old: __m256i, contribution: __m256i| -> __m256i {
      match mode {
        b's' => div255_round(_mm256_mullo_epi16(old, _mm256_sub_epi16(full, contribution))),
        b'i' => div255_round(_mm256_mullo_epi16(old, contribution)),
        b'f' => _mm256_sub_epi16(_mm256_max_epi16(old, contribution), _mm256_min_epi16(old, contribution)),
        _ => _mm256_add_epi16(contribution, div255_round(_mm256_mullo_epi16(_mm256_sub_epi16(full, contribution), old))),
      }
    };
    store(dst.as_mut_ptr(), pack(combine(dl, cl), combine(dh, ch)));
  }
}

// ---------------------------------------------------------------------------
// RGBA kernels — 8 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
pub(super) fn apply_matte_alpha_avx2(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = _mm256_set1_epi16(i16::from(source_opacity));
  let full = _mm256_set1_epi16(255);
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    let s = load(spx.as_ptr().cast());
    let mut fl = div255_round(_mm256_mullo_epi16(splat_alpha(lo(s)), opacity));
    let mut fh = div255_round(_mm256_mullo_epi16(splat_alpha(hi(s)), opacity));
    if inverted {
      fl = _mm256_sub_epi16(full, fl);
      fh = _mm256_sub_epi16(full, fh);
    }
    if _mm256_movemask_epi8(_mm256_cmpeq_epi16(fl, full)) == -1 && _mm256_movemask_epi8(_mm256_cmpeq_epi16(fh, full)) == -1 {
      continue;
    }
    let d = load(dpx.as_ptr().cast());
    let l = div255_round(_mm256_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm256_mullo_epi16(hi(d), fh));
    store(dpx.as_mut_ptr().cast(), pack(l, h));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn fill_span_solid_avx2(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let src = _mm256_setr_epi16(sr as i16, sg as i16, sb as i16, 255, sr as i16, sg as i16, sb as i16, 255, sr as i16, sg as i16, sb as i16, 255, sr as i16, sg as i16, sb as i16, 255);
  let sa_w = _mm256_set1_epi16(sa as i16);
  let full = _mm256_set1_epi16(255);
  for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
    // rep4 without pshufb: byte-double then word-double, per 4-byte group.
    let c0 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[0..4].try_into().unwrap_or([0; 4])) as i32);
    let crep0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c0, c0), _mm_unpacklo_epi8(c0, c0));
    let c1 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[4..8].try_into().unwrap_or([0; 4])) as i32);
    let crep1 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c1, c1), _mm_unpacklo_epi8(c1, c1));
    let crep = _mm256_inserti128_si256(_mm256_castsi128_si256(crep0), crep1, 1);
    let ca_lo = div255_round(_mm256_mullo_epi16(lo(crep), sa_w));
    let ca_hi = div255_round(_mm256_mullo_epi16(hi(crep), sa_w));
    let s_lo = div255_round(_mm256_mullo_epi16(src, ca_lo));
    let s_hi = div255_round(_mm256_mullo_epi16(src, ca_hi));
    let inv_lo = _mm256_sub_epi16(full, ca_lo);
    let inv_hi = _mm256_sub_epi16(full, ca_hi);
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn fill_span_uniform_avx2(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let s = _mm256_setr_epi16(s_r as i16, s_g as i16, s_b as i16, ca as i16, s_r as i16, s_g as i16, s_b as i16, ca as i16, s_r as i16, s_g as i16, s_b as i16, ca as i16, s_r as i16, s_g as i16, s_b as i16, ca as i16);
  let inv = _mm256_set1_epi16(255 - ca as i16);
  for dpx in dst.chunks_exact_mut(8) {
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s, inv), over(hi(d), s, inv)));
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn composite_over_avx2(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = _mm256_set1_epi16(k as i16);
  let full = _mm256_set1_epi16(255);
  let zero = _mm256_setzero_si256();
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    let d = load(dpx.as_ptr().cast());
    let s = load(spx.as_ptr().cast());
    // All eight source pixels fully transparent: `over` is the identity on
    // dst, so skip the round trip (matches the scalar `if s == 0`).
    if _mm256_movemask_epi8(_mm256_cmpeq_epi32(s, zero)) == -1 {
      continue;
    }
    let (s_lo_raw, s_hi_raw) = (lo(s), hi(s));
    let s_lo = div255_round(_mm256_mullo_epi16(s_lo_raw, kq));
    let s_hi = div255_round(_mm256_mullo_epi16(s_hi_raw, kq));
    let inv_lo = _mm256_sub_epi16(full, div255_round(_mm256_mullo_epi16(splat_alpha(s_lo_raw), kq)));
    let inv_hi = _mm256_sub_epi16(full, div255_round(_mm256_mullo_epi16(splat_alpha(s_hi_raw), kq)));
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`, with lanes failing `valid`
/// forced to the u32::MAX sentinel (LUT miss -> transparent 0). Matches
/// the scalar `is_finite` gate + `as usize` truncation.
#[inline]
#[target_feature(enable = "avx2")]
fn lut_indices(t: __m256, valid: __m256, scale: __m256) -> __m256i {
  let tc = _mm256_min_ps(_mm256_max_ps(t, _mm256_setzero_ps()), _mm256_set1_ps(1.0));
  let idx = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(tc, scale), _mm256_set1_ps(0.5)));
  let vi = _mm256_castps_si256(valid);
  _mm256_or_si256(_mm256_and_si256(vi, idx), _mm256_andnot_si256(vi, _mm256_set1_epi32(-1)))
}

/// 8-lane LUT gather: `vpgatherdd`. Index lanes with the `u32::MAX`
/// sentinel have their gather mask cleared, so they read nothing and land on
/// the zero offset = transparent, matching scalar `unwrap_or(0)`. The mask
/// lanes are inverted relative to the compare (sentinel is negative as a signed
/// i32, so `zero > idx` identifies it).
#[inline]
#[target_feature(enable = "avx2")]
fn lut_gather(lut: &[u32], idx: __m256i) -> __m256i {
  let sentinel = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), idx), _mm256_set1_epi32(-1));
  // SAFETY: the gather mask guarantees masked-off (sentinel) lanes never touch
  // memory; valid lanes are < lut.len() by construction of `lut_indices`.
  #[allow(unsafe_code)]
  unsafe {
    _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), lut.as_ptr().cast(), idx, sentinel, 4)
  }
}

#[inline]
#[target_feature(enable = "avx2")]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: __m256i) {
  store(chunk.as_mut_ptr().cast(), lut_gather(lut, idx));
}

#[inline]
#[target_feature(enable = "avx2")]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: __m256i) {
  // Source and destination are premultiplied RGBA bytes. k=255 means
  // source channels pass through unchanged.
  let full = _mm256_set1_epi16(255);
  let d = load(dpx.as_ptr().cast());
  let s = lut_gather(lut, idx);
  let (s_lo, s_hi) = (lo(s), hi(s));
  let inv_lo = _mm256_sub_epi16(full, splat_alpha(s_lo));
  let inv_hi = _mm256_sub_epi16(full, splat_alpha(s_hi));
  store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
}

/// `|v|` — mask off the sign bit, matching `f32::abs`.
#[inline]
#[target_feature(enable = "avx2")]
fn abs_ps(v: __m256) -> __m256 {
  _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)))
}

/// Lane-wise `a.max(b)` with Rust's NaN semantics (see sse2.rs).
#[inline]
#[target_feature(enable = "avx2")]
fn max_ps_rust(a: __m256, b: __m256) -> __m256 {
  let b_nan = _mm256_cmp_ps(b, b, _CMP_UNORD_Q);
  _mm256_or_ps(_mm256_and_ps(b_nan, a), _mm256_andnot_ps(b_nan, _mm256_max_ps(a, b)))
}

/// Absolute device columns for one 8-lane chunk.
#[inline]
#[target_feature(enable = "avx2")]
fn lane_columns(x_start: f32) -> __m256 {
  _mm256_add_ps(_mm256_set1_ps(x_start), _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
}

/// 8-lane linear gradient LUT fill.
#[target_feature(enable = "avx2")]
pub(super) fn linear_lut_fill_avx2(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm256_set1_ps(row_base);
  let dtv = _mm256_set1_ps(dt);
  let eight = _mm256_set1_ps(8.0);
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in out.chunks_exact_mut(8) {
    let t = _mm256_add_ps(t0v, _mm256_mul_ps(kf, dtv));
    let finite = _mm256_cmp_ps(abs_ps(t), inf, _CMP_LT_OQ);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

#[target_feature(enable = "avx2")]
pub(super) fn linear_lut_over_avx2(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm256_set1_ps(row_base);
  let dtv = _mm256_set1_ps(dt);
  let eight = _mm256_set1_ps(8.0);
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let t = _mm256_add_ps(t0v, _mm256_mul_ps(kf, dtv));
    let finite = _mm256_cmp_ps(abs_ps(t), inf, _CMP_LT_OQ);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

/// 8-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the scalar
/// `ddx*ddx + ddy*ddy`, and `sqrtps` matches scalar `sqrt()`, so lanes
/// agree with the `dd0 + X·d` scalar form bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(super) fn radial_lut_fill_avx2(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm256_set1_ps(da), _mm256_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm256_set1_ps(dd0x), _mm256_set1_ps(dd0y));
  let inv_rv = _mm256_set1_ps(inv_r);
  let eight = _mm256_set1_ps(8.0);
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in out.chunks_exact_mut(8) {
    let ddx = _mm256_add_ps(ddx0v, _mm256_mul_ps(kf, dav));
    let ddy = _mm256_add_ps(ddy0v, _mm256_mul_ps(kf, dbv));
    let gg = _mm256_add_ps(_mm256_mul_ps(ddx, ddx), _mm256_mul_ps(ddy, ddy));
    let t = _mm256_mul_ps(_mm256_sqrt_ps(gg), inv_rv);
    let finite = _mm256_cmp_ps(abs_ps(t), inf, _CMP_LT_OQ);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(super) fn radial_lut_over_avx2(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm256_set1_ps(da), _mm256_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm256_set1_ps(dd0x), _mm256_set1_ps(dd0y));
  let inv_rv = _mm256_set1_ps(inv_r);
  let eight = _mm256_set1_ps(8.0);
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let ddx = _mm256_add_ps(ddx0v, _mm256_mul_ps(kf, dav));
    let ddy = _mm256_add_ps(ddy0v, _mm256_mul_ps(kf, dbv));
    let gg = _mm256_add_ps(_mm256_mul_ps(ddx, ddx), _mm256_mul_ps(ddy, ddy));
    let t = _mm256_mul_ps(_mm256_sqrt_ps(gg), inv_rv);
    let finite = _mm256_cmp_ps(abs_ps(t), inf, _CMP_LT_OQ);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

/// 8-lane focal (highlight) radial LUT fill.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(super) fn focal_lut_fill_avx2(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm256_set1_ps(b0), _mm256_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm256_set1_ps(d0), _mm256_set1_ps(d1), _mm256_set1_ps(d2));
  let inv2av = _mm256_set1_ps(inv2a);
  let rv = _mm256_set1_ps(r);
  let eight = _mm256_set1_ps(8.0);
  let zero = _mm256_setzero_ps();
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in out.chunks_exact_mut(8) {
    let b = _mm256_add_ps(b0v, _mm256_mul_ps(kf, dbv));
    let det = _mm256_add_ps(d0v, _mm256_mul_ps(kf, _mm256_add_ps(d1v, _mm256_mul_ps(kf, d2v))));
    let sq = _mm256_sqrt_ps(det);
    let nb = _mm256_sub_ps(zero, b);
    let root = max_ps_rust(_mm256_mul_ps(_mm256_sub_ps(nb, sq), inv2av), _mm256_mul_ps(_mm256_add_ps(nb, sq), inv2av));
    let valid = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(det, zero, _CMP_GE_OQ), _mm256_cmp_ps(_mm256_mul_ps(rv, root), zero, _CMP_GE_OQ)), _mm256_cmp_ps(abs_ps(root), inf, _CMP_LT_OQ));
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(super) fn focal_lut_over_avx2(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm256_set1_ps(b0), _mm256_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm256_set1_ps(d0), _mm256_set1_ps(d1), _mm256_set1_ps(d2));
  let inv2av = _mm256_set1_ps(inv2a);
  let rv = _mm256_set1_ps(r);
  let eight = _mm256_set1_ps(8.0);
  let zero = _mm256_setzero_ps();
  let inf = _mm256_set1_ps(f32::INFINITY);
  let scalev = _mm256_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let b = _mm256_add_ps(b0v, _mm256_mul_ps(kf, dbv));
    let det = _mm256_add_ps(d0v, _mm256_mul_ps(kf, _mm256_add_ps(d1v, _mm256_mul_ps(kf, d2v))));
    let sq = _mm256_sqrt_ps(det);
    let nb = _mm256_sub_ps(zero, b);
    let root = max_ps_rust(_mm256_mul_ps(_mm256_sub_ps(nb, sq), inv2av), _mm256_mul_ps(_mm256_add_ps(nb, sq), inv2av));
    let valid = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(det, zero, _CMP_GE_OQ), _mm256_cmp_ps(_mm256_mul_ps(rv, root), zero, _CMP_GE_OQ)), _mm256_cmp_ps(abs_ps(root), inf, _CMP_LT_OQ));
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm256_add_ps(kf, eight);
  }
}

/// Opaque (sa=255) 8-pixel solid fill: if every coverage byte in the chunk is
/// 255 the output is exactly `color` (a plain vector store, matching the scalar
/// `run.fill(color)`), otherwise each pixel falls back to the exact scalar formula.
/// Splits the caller's span into SIMD_MIN_SPAN-aligned chunks; the caller handles
/// the scalar tail. Bit-exact: all-255 chunks store the source color verbatim,
/// and mixed chunks run the identical rounded scalar math.
#[target_feature(enable = "avx2")]
pub(super) fn fill_span_opaque_avx2(dst: &mut [u32], cov: &[u8], color: u32) {
  let colorv = _mm256_set1_epi32(color as i32);
  for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
    // The coverage test reads the chunk as one u64: exactly the 8 bytes this
    // iteration owns. A 256-bit load takes in the next 24 as well, which runs
    // past the slice on the final chunk, and demands all 32 be 255 before the
    // fast path can fire.
    let all255 = u64::from_le_bytes(cpx.try_into().unwrap_or([0; 8])) == u64::MAX;
    if all255 {
      #[allow(unsafe_code)]
      unsafe {
        _mm256_storeu_si256(dpx.as_mut_ptr().cast(), colorv)
      }
    } else {
      super::fill_span_solid_scalar(
        dpx,
        cpx,
        (color >> 0) & 0xff,
        (color >> 8) & 0xff,
        (color >> 16) & 0xff,
        255,
      );
    }
  }
}
