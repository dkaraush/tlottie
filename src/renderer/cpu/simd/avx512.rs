//! AVX-512 kernels, 16 pixels (u32) / 64 pixels (Alpha8) per iteration.
//!
//! Runtime-dispatched on top of the AVX2/SSE2 baseline: compiled
//! unconditionally on x86_64, every kernel carries
//! `#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]`
//! and is only reachable through the `avx512f`+`avx512bw` detection gate in
//! [`super`]. Bit-for-bit equals the scalar oracle and the 128/256-bit kernels
//! (same rounding protocol: rounded `/255` source scaling,
//! `dst * (256 - src) >> 8` attenuation).
//!
//! Layout: one `__m512i` = 16 pixels of interleaved [R,G,B,A] (64 bytes) or
//! 64 Alpha8 bytes. The wide vector is split into two 32-byte halves, each
//! widened to 32 u16 lanes with `vpmovzxbw`, computed independently, then
//! narrowed back with `vpmovuswb` and reassembled by [`pack`].
//!
//! Gradient LUT kernels use `vpgatherdd` via `_mm512_mask_i32gather_epi32`,
//! fetching 16 LUT colors with one load; lanes failing the finite check keep the
//! u32::MAX sentinel with their gather mask cleared, so they read nothing and land on
//! the zero source = transparent, exactly like the scalar `lut.get(idx).unwrap_or(0)`.

use core::arch::x86_64::{
  _CMP_GE_OQ, _CMP_LT_OQ, _CMP_UNORD_Q, __m512, __m512i, _mm256_setr_epi16, _mm512_add_epi16, _mm512_add_ps, _mm512_and_ps, _mm512_castps_si512, _mm512_castsi128_si512,
  _mm512_castsi256_si512, _mm512_castsi512_ps, _mm512_castsi512_si256, _mm512_cmpeq_epi16_mask, _mm512_cmpeq_epi32_mask, _mm512_cmp_ps_mask, _mm512_cvtepu8_epi16, _mm512_cvttps_epi32, _mm512_cvtusepi16_epi8,
  _mm512_extracti64x4_epi64, _mm512_inserti32x4, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_mask_i32gather_epi32, _mm512_mask_mov_epi32, _mm512_max_epi16, _mm512_max_ps,
  _mm512_min_epi16, _mm512_min_ps, _mm512_mul_ps, _mm512_mullo_epi16, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setr_ps, _mm512_setzero_ps, _mm512_setzero_si512,
  _mm512_shufflehi_epi16, _mm512_shufflelo_epi16, _mm512_sqrt_ps, _mm512_srli_epi16, _mm512_storeu_si512, _mm512_sub_epi16, _mm512_sub_ps, _mm_cvtsi32_si128, _mm_unpacklo_epi8,
  _mm_unpacklo_epi16,
};

/// Widens the low 32 bytes of `v` (8 RGBA pixels / 32 Alpha8) to 32 u16 lanes.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lo(v: __m512i) -> __m512i {
  _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v))
}

/// Widens the high 32 bytes of `v` (pixels 8..16 / Alpha8 32..64).
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn hi(v: __m512i) -> __m512i {
  _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v))
}

/// Narrows two 32-u16 halves back to 64 bytes in the original byte order.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn pack(a: __m512i, b: __m512i) -> __m512i {
  _mm512_inserti64x4::<1>(_mm512_castsi256_si512(_mm512_cvtusepi16_epi8(a)), _mm512_cvtusepi16_epi8(b))
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn div255_round(n: __m512i) -> __m512i {
  let t = _mm512_add_epi16(n, _mm512_set1_epi16(127));
  let u = _mm512_add_epi16(_mm512_add_epi16(t, _mm512_srli_epi16::<8>(t)), _mm512_set1_epi16(1));
  _mm512_srli_epi16::<8>(u)
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn over(d: __m512i, s: __m512i, inv: __m512i) -> __m512i {
  _mm512_min_epi16(
    _mm512_add_epi16(s, _mm512_srli_epi16::<8>(_mm512_mullo_epi16(d, _mm512_add_epi16(inv, _mm512_set1_epi16(1))))),
    _mm512_set1_epi16(255),
  )
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn alpha_over8(dst: __m512i, source: __m512i) -> __m512i {
  over(dst, source, _mm512_sub_epi16(_mm512_set1_epi16(255), source))
}

/// Replicates each pixel's alpha across its four channel lanes via per-lane word
/// shuffles.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn splat_alpha(half: __m512i) -> __m512i {
  _mm512_shufflehi_epi16::<0xFF>(_mm512_shufflelo_epi16::<0xFF>(half))
}

/// Unaligned 64-byte load.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn load(bytes: *const u8) -> __m512i {
  // SAFETY: every caller passes a pointer to a `chunks_exact` window, so
  // 64 readable bytes are guaranteed; `loadu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm512_loadu_si512(bytes.cast())
  }
}

/// Unaligned 64-byte store.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn store(bytes: *mut u8, value: __m512i) {
  // SAFETY: every caller passes a pointer to a `chunks_exact_mut` window,
  // so 64 writable bytes are guaranteed; `storeu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm512_storeu_si512(bytes.cast(), value)
  }
}

// ---------------------------------------------------------------------------
// Alpha8 kernels — 64 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_blend_solid_avx512(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = _mm512_set1_epi16(i16::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(64).zip(coverage.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let c = load(coverage.as_ptr());
    let sl = div255_round(_mm512_mullo_epi16(lo(c), alpha));
    let sh = div255_round(_mm512_mullo_epi16(hi(c), alpha));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_blend_product_avx512(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(64).zip(lhs.chunks_exact(64)).zip(rhs.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let l = load(lhs.as_ptr());
    let r = load(rhs.as_ptr());
    let sl = div255_round(_mm512_mullo_epi16(lo(l), lo(r)));
    let sh = div255_round(_mm512_mullo_epi16(hi(l), hi(r)));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_blend_uniform_avx512(dst: &mut [u8], source: u8) {
  let source = _mm512_set1_epi16(i16::from(source));
  for dst in dst.chunks_exact_mut(64) {
    let d = load(dst.as_ptr());
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), source), alpha_over8(hi(d), source)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_composite_over_avx512(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = _mm512_set1_epi16(i16::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let sl = div255_round(_mm512_mullo_epi16(lo(s), opacity));
    let sh = div255_round(_mm512_mullo_epi16(hi(s), opacity));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_multiply_avx512(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(64).zip(factors.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let f = load(factors.as_ptr());
    let l = div255_round(_mm512_mullo_epi16(lo(d), lo(f)));
    let h = div255_round(_mm512_mullo_epi16(hi(d), hi(f)));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_matte_avx512(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = _mm512_set1_epi16(i16::from(opacity));
  let full = _mm512_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let mut fl = div255_round(_mm512_mullo_epi16(lo(s), opacity));
    let mut fh = div255_round(_mm512_mullo_epi16(hi(s), opacity));
    if inverted {
      fl = _mm512_sub_epi16(full, fl);
      fh = _mm512_sub_epi16(full, fh);
    }
    let l = div255_round(_mm512_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm512_mullo_epi16(hi(d), fh));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn alpha_mask_combine_avx512(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = _mm512_set1_epi16(i16::from(opacity));
  let full = _mm512_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let (dl, dh) = (lo(d), hi(d));
    let (mut sl, mut sh) = (lo(s), hi(s));
    if inverted {
      sl = _mm512_sub_epi16(full, sl);
      sh = _mm512_sub_epi16(full, sh);
    }
    let (cl, ch) = (div255_round(_mm512_mullo_epi16(sl, opacity)), div255_round(_mm512_mullo_epi16(sh, opacity)));
    let combine = |old: __m512i, contribution: __m512i| -> __m512i {
      match mode {
        b's' => div255_round(_mm512_mullo_epi16(old, _mm512_sub_epi16(full, contribution))),
        b'i' => div255_round(_mm512_mullo_epi16(old, contribution)),
        b'f' => _mm512_sub_epi16(_mm512_max_epi16(old, contribution), _mm512_min_epi16(old, contribution)),
        _ => _mm512_add_epi16(contribution, div255_round(_mm512_mullo_epi16(_mm512_sub_epi16(full, contribution), old))),
      }
    };
    store(dst.as_mut_ptr(), pack(combine(dl, cl), combine(dh, ch)));
  }
}

// ---------------------------------------------------------------------------
// RGBA kernels — 16 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn apply_matte_alpha_avx512(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = _mm512_set1_epi16(i16::from(source_opacity));
  let full = _mm512_set1_epi16(255);
  for (dpx, spx) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let s = load(spx.as_ptr().cast());
    let mut fl = div255_round(_mm512_mullo_epi16(splat_alpha(lo(s)), opacity));
    let mut fh = div255_round(_mm512_mullo_epi16(splat_alpha(hi(s)), opacity));
    if inverted {
      fl = _mm512_sub_epi16(full, fl);
      fh = _mm512_sub_epi16(full, fh);
    }
    if _mm512_cmpeq_epi16_mask(fl, full) == 0xFFFFFFFF && _mm512_cmpeq_epi16_mask(fh, full) == 0xFFFFFFFF {
      continue;
    }
    let d = load(dpx.as_ptr().cast());
    let l = div255_round(_mm512_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm512_mullo_epi16(hi(d), fh));
    store(dpx.as_mut_ptr().cast(), pack(l, h));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn fill_span_solid_avx512(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let src_pat = pat4(sr as i16, sg as i16, sb as i16, 255);
  let sa_w = _mm512_set1_epi16(sa as i16);
  let full = _mm512_set1_epi16(255);
  for (dpx, cpx) in dst.chunks_exact_mut(16).zip(cov.chunks_exact(16)) {
    // rep4 without pshufb: byte-double then word-double.
    let c0 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[0..4].try_into().unwrap_or([0; 4])) as i32);
    let crep0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c0, c0), _mm_unpacklo_epi8(c0, c0));
    let c1 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[4..8].try_into().unwrap_or([0; 4])) as i32);
    let crep1 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c1, c1), _mm_unpacklo_epi8(c1, c1));
    let c2 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[8..12].try_into().unwrap_or([0; 4])) as i32);
    let crep2 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c2, c2), _mm_unpacklo_epi8(c2, c2));
    let c3 = _mm_cvtsi32_si128(u32::from_le_bytes(cpx[12..16].try_into().unwrap_or([0; 4])) as i32);
    let crep3 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c3, c3), _mm_unpacklo_epi8(c3, c3));
    let crep = _mm512_inserti32x4::<3>(
      _mm512_inserti32x4::<2>(
        _mm512_inserti32x4::<1>(_mm512_castsi128_si512(crep0), crep1),
        crep2,
      ),
      crep3,
    );
    let cl_l = div255_round(_mm512_mullo_epi16(lo(crep), sa_w));
    let cl_h = div255_round(_mm512_mullo_epi16(hi(crep), sa_w));
    let s_l = div255_round(_mm512_mullo_epi16(src_pat, cl_l));
    let s_h = div255_round(_mm512_mullo_epi16(src_pat, cl_h));
    let inv_l = _mm512_sub_epi16(full, cl_l);
    let inv_h = _mm512_sub_epi16(full, cl_h);
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_l, inv_l), over(hi(d), s_h, inv_h)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn fill_span_uniform_avx512(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let s = pat4(s_r as i16, s_g as i16, s_b as i16, ca as i16);
  let inv = _mm512_set1_epi16(255 - ca as i16);
  for dpx in dst.chunks_exact_mut(16) {
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s, inv), over(hi(d), s, inv)));
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn composite_over_avx512(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = _mm512_set1_epi16(k as i16);
  let full = _mm512_set1_epi16(255);
  let zero = _mm512_setzero_si512();
  for (dpx, spx) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dpx.as_ptr().cast());
    let s = load(spx.as_ptr().cast());
    // All 16 source pixels fully transparent: skip the round trip (matches the
    // scalar `if s == 0`). `_mm512_cmpeq_epi32_mask` gives a 16-bit mask.
    if _mm512_cmpeq_epi32_mask(s, zero) == 0xFFFF {
      continue;
    }
    let (s_lo, s_hi) = (lo(s), hi(s));
    let sl = div255_round(_mm512_mullo_epi16(s_lo, kq));
    let sh = div255_round(_mm512_mullo_epi16(s_hi, kq));
    let inv_l = _mm512_sub_epi16(full, div255_round(_mm512_mullo_epi16(splat_alpha(s_lo), kq)));
    let inv_h = _mm512_sub_epi16(full, div255_round(_mm512_mullo_epi16(splat_alpha(s_hi), kq)));
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), sl, inv_l), over(hi(d), sh, inv_h)));
  }
}

/// `|v|` — mask off the sign bit, matching `f32::abs`.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn abs_ps(v: __m512) -> __m512 {
  _mm512_and_ps(v, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fff_ffff)))
}

/// 32-u16 pattern `[a,b,c,d]` repeated 8x, as used for uniform solid fills.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn pat4(a: i16, b: i16, c: i16, d: i16) -> __m512i {
  let p = _mm256_setr_epi16(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d);
  _mm512_inserti64x4::<1>(_mm512_castsi256_si512(p), p)
}

/// Lane-wise `a.max(b)` with Rust's NaN semantics (see sse2.rs).
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn max_ps_rust(a: __m512, b: __m512) -> __m512 {
  let b_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(b, b);
  let r = _mm512_mask_mov_epi32(_mm512_castps_si512(_mm512_max_ps(a, b)), b_nan, _mm512_castps_si512(a));
  _mm512_castsi512_ps(r)
}

/// Absolute device columns for one 16-lane chunk.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lane_columns(x_start: f32) -> __m512 {
  _mm512_add_ps(_mm512_set1_ps(x_start), _mm512_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0))
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`; lanes failing `valid` keep the
/// u32::MAX sentinel (via masked move), so the gather mask clears them. Matches the
/// scalar `is_finite` gate + `as usize` truncation.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lut_indices(t: __m512, valid: u16, scale: __m512) -> __m512i {
  let tc = _mm512_min_ps(_mm512_max_ps(t, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
  let idx = _mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(tc, scale), _mm512_set1_ps(0.5)));
  // `valid` is an AVX-512 compare mask (all-ones for finite lanes); complement
  // selects invalid lanes -> overwrite with the u32::MAX sentinel.
  _mm512_mask_mov_epi32(idx, valid ^ 0xFFFF, _mm512_set1_epi32(-1))
}

/// 16-lane LUT gather: `vpgatherdd`. Sentinel lanes (u32::MAX = -1i32) have
/// their gather mask cleared, so they read nothing and land on `src` = zero.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lut_gather(lut: &[u32], idx: __m512i) -> __m512i {
  let sentinel = _mm512_cmpeq_epi32_mask(idx, _mm512_set1_epi32(-1));
  let guard = 0xFFFF ^ sentinel;
  // SAFETY: guard clears every sentinel lane; valid lanes are < lut.len() by
  // construction of `lut_indices`.
  #[allow(unsafe_code)]
  unsafe {
    _mm512_mask_i32gather_epi32::<4>(_mm512_setzero_si512(), guard, idx, lut.as_ptr().cast())
  }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: __m512i) {
  store(chunk.as_mut_ptr().cast(), lut_gather(lut, idx));
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: __m512i) {
  // Source and destination are premultiplied RGBA bytes. k=255 means source
  // channels pass through unchanged.
  let full = _mm512_set1_epi16(255);
  let d = load(dpx.as_ptr().cast());
  let s = lut_gather(lut, idx);
  let (s_lo, s_hi) = (lo(s), hi(s));
  let inv_l = _mm512_sub_epi16(full, splat_alpha(s_lo));
  let inv_h = _mm512_sub_epi16(full, splat_alpha(s_hi));
  store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_l), over(hi(d), s_hi, inv_h)));
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn linear_lut_fill_avx512(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm512_set1_ps(row_base);
  let dtv = _mm512_set1_ps(dt);
  let sixteen = _mm512_set1_ps(16.0);
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in out.chunks_exact_mut(16) {
    let t = _mm512_add_ps(t0v, _mm512_mul_ps(kf, dtv));
    let finite = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn linear_lut_over_avx512(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm512_set1_ps(row_base);
  let dtv = _mm512_set1_ps(dt);
  let sixteen = _mm512_set1_ps(16.0);
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(16) {
    let t = _mm512_add_ps(t0v, _mm512_mul_ps(kf, dtv));
    let finite = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

/// 16-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the scalar.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn radial_lut_fill_avx512(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm512_set1_ps(da), _mm512_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm512_set1_ps(dd0x), _mm512_set1_ps(dd0y));
  let inv_rv = _mm512_set1_ps(inv_r);
  let sixteen = _mm512_set1_ps(16.0);
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in out.chunks_exact_mut(16) {
    let ddx = _mm512_add_ps(ddx0v, _mm512_mul_ps(kf, dav));
    let ddy = _mm512_add_ps(ddy0v, _mm512_mul_ps(kf, dbv));
    let gg = _mm512_add_ps(_mm512_mul_ps(ddx, ddx), _mm512_mul_ps(ddy, ddy));
    let t = _mm512_mul_ps(_mm512_sqrt_ps(gg), inv_rv);
    let finite = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn radial_lut_over_avx512(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm512_set1_ps(da), _mm512_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm512_set1_ps(dd0x), _mm512_set1_ps(dd0y));
  let inv_rv = _mm512_set1_ps(inv_r);
  let sixteen = _mm512_set1_ps(16.0);
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(16) {
    let ddx = _mm512_add_ps(ddx0v, _mm512_mul_ps(kf, dav));
    let ddy = _mm512_add_ps(ddy0v, _mm512_mul_ps(kf, dbv));
    let gg = _mm512_add_ps(_mm512_mul_ps(ddx, ddx), _mm512_mul_ps(ddy, ddy));
    let t = _mm512_mul_ps(_mm512_sqrt_ps(gg), inv_rv);
    let finite = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

/// 16-lane focal (highlight) radial LUT fill.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn focal_lut_fill_avx512(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm512_set1_ps(b0), _mm512_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm512_set1_ps(d0), _mm512_set1_ps(d1), _mm512_set1_ps(d2));
  let inv2av = _mm512_set1_ps(inv2a);
  let rv = _mm512_set1_ps(r);
  let sixteen = _mm512_set1_ps(16.0);
  let zero = _mm512_setzero_ps();
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in out.chunks_exact_mut(16) {
    let b = _mm512_add_ps(b0v, _mm512_mul_ps(kf, dbv));
    let det = _mm512_add_ps(d0v, _mm512_mul_ps(kf, _mm512_add_ps(d1v, _mm512_mul_ps(kf, d2v))));
    let sq = _mm512_sqrt_ps(det);
    let nb = _mm512_sub_ps(zero, b);
    let root = max_ps_rust(_mm512_mul_ps(_mm512_sub_ps(nb, sq), inv2av), _mm512_mul_ps(_mm512_add_ps(nb, sq), inv2av));
    let valid = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(det, zero) & _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_mul_ps(rv, root), zero) & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(root), inf);
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn focal_lut_over_avx512(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm512_set1_ps(b0), _mm512_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm512_set1_ps(d0), _mm512_set1_ps(d1), _mm512_set1_ps(d2));
  let inv2av = _mm512_set1_ps(inv2a);
  let rv = _mm512_set1_ps(r);
  let sixteen = _mm512_set1_ps(16.0);
  let zero = _mm512_setzero_ps();
  let inf = _mm512_set1_ps(f32::INFINITY);
  let scalev = _mm512_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(16) {
    let b = _mm512_add_ps(b0v, _mm512_mul_ps(kf, dbv));
    let det = _mm512_add_ps(d0v, _mm512_mul_ps(kf, _mm512_add_ps(d1v, _mm512_mul_ps(kf, d2v))));
    let sq = _mm512_sqrt_ps(det);
    let nb = _mm512_sub_ps(zero, b);
    let root = max_ps_rust(_mm512_mul_ps(_mm512_sub_ps(nb, sq), inv2av), _mm512_mul_ps(_mm512_add_ps(nb, sq), inv2av));
    let valid = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(det, zero) & _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_mul_ps(rv, root), zero) & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_ps(root), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm512_add_ps(kf, sixteen);
  }
}

/// Opaque-source solid fill run detector, 16 pixels/iteration.
///
/// `sa == 255` makes full-coverage pixels exactly the source color; a 64-byte
/// chunk whose bytes are all 0xFF becomes a plain 16-pixel store (`vpmovuswb`
/// layout is [R,G,B,A]*16 == 64 bytes, so the all-255 coverage test exactly
/// matches the stored pattern). Chunks with any AA edge fall back to the exact
/// general kernel, identical to the AVX2/NEON opaque paths.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl")]
pub(super) fn fill_span_opaque_avx512(dst: &mut [u32], cov: &[u8], color: u32) {
  let colorv = _mm512_set1_epi32(color as i32);
  for (dpx, cpx) in dst.chunks_exact_mut(16).zip(cov.chunks_exact(16)) {
    // Reads the chunk as one u128: exactly the 16 bytes this iteration owns.
    // A 512-bit load pulls in the next 48 as well, running past the slice on
    // the final chunk and demanding all 64 be 255.
    let all255 = u128::from_le_bytes(cpx.try_into().unwrap_or([0; 16])) == u128::MAX;
    if all255 {
      #[allow(unsafe_code)]
      unsafe {
        _mm512_storeu_si512(dpx.as_mut_ptr().cast(), colorv)
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
