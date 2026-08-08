//! SSE2 kernels, 4 pixels (u32) / 16 pixels (Alpha8) per iteration.
//!
//! SSE2 is architecturally mandatory on x86_64, so — as with NEON on
//! aarch64 — there is no runtime dispatch: the module is compiled in
//! whenever the feature is statically present and the kernels are called
//! unconditionally.
//!
//! Structurally this mirrors [`super::wasm128`], not the NEON backend:
//! SSE2 has no `vld4`-style de-interleave (and `pshufb` is SSSE3, outside
//! the x86_64 baseline), so the u16 kernels work on the natural
//! interleaved layout — one `__m128i` = 4 pixels [R,G,B,A ×4], widened to
//! two u16x8 halves (2 pixels each) with `punpck?bw` against zero and
//! narrowed back with `packuswb`. Per-pixel factors (coverage, alpha) are
//! replicated across each pixel's four channel lanes with `punpck`
//! doubling and `pshuf?w`, both SSE2.
//!
//! Two signedness notes, since SSE2's integer min/max are signed:
//! - `_mm_mullo_epi16` returns the low 16 product bits, so the u16 values
//!   are bit-correct even where they exceed `i16::MAX` (`d * (inv+1)` peaks
//!   at 65280); the following `_mm_srli_epi16` is a LOGICAL shift and
//!   recovers the right magnitude.
//! - `_mm_min_epi16`/`_mm_max_epi16` are signed, but every value they see
//!   is already `<= 510` (`over` clamps `s + (d*(inv+1)>>8)`, and the mask
//!   combine operates on 0..=255 alphas), so signed and unsigned agree.
//!
//! u16 lanes never overflow: every product is `(<=255)*(<=255) <= 65025`
//! and div255's intermediate stays `<= 65407`.
//!
//! Unsafe is confined to [`load`], [`store`] and [`lut_gather`] — the only
//! places that touch a raw pointer. Every other intrinsic here is
//! register-to-register and is a safe call from inside a
//! `#[target_feature(enable = "sse2")]` function.

use core::arch::x86_64::{
  __m128, __m128i, _mm_add_epi16, _mm_add_ps, _mm_and_ps, _mm_and_si128, _mm_andnot_ps, _mm_andnot_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpge_ps, _mm_cmplt_ps,
  _mm_cmpunord_ps, _mm_cvtsi32_si128, _mm_cvttps_epi32, _mm_loadu_si128, _mm_max_epi16, _mm_max_ps, _mm_min_epi16, _mm_min_ps, _mm_movemask_epi8, _mm_mul_ps, _mm_mullo_epi16, _mm_or_ps, _mm_or_si128,
  _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_ps, _mm_setr_epi16, _mm_setr_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_sqrt_ps, _mm_srli_epi16,
  _mm_storeu_si128, _mm_sub_epi16, _mm_sub_ps, _mm_unpackhi_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};

/// Widens the low 8 bytes to u16 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn lo(v: __m128i) -> __m128i {
  _mm_unpacklo_epi8(v, _mm_setzero_si128())
}

/// Widens the high 8 bytes to u16 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn hi(v: __m128i) -> __m128i {
  _mm_unpackhi_epi8(v, _mm_setzero_si128())
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
#[inline]
#[target_feature(enable = "sse2")]
fn div255_round(n: __m128i) -> __m128i {
  let t = _mm_add_epi16(n, _mm_set1_epi16(127));
  let u = _mm_add_epi16(_mm_add_epi16(t, _mm_srli_epi16::<8>(t)), _mm_set1_epi16(1));
  _mm_srli_epi16::<8>(u)
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn over(d: __m128i, s: __m128i, inv: __m128i) -> __m128i {
  _mm_min_epi16(_mm_add_epi16(s, _mm_srli_epi16::<8>(_mm_mullo_epi16(d, _mm_add_epi16(inv, _mm_set1_epi16(1))))), _mm_set1_epi16(255))
}

#[inline]
#[target_feature(enable = "sse2")]
fn alpha_over8(dst: __m128i, source: __m128i) -> __m128i {
  over(dst, source, _mm_sub_epi16(_mm_set1_epi16(255), source))
}

/// Replicates a widened pixel-pair's alpha lanes (3 and 7) across each
/// pixel's four channel lanes — the `pshufb` alpha broadcast, done after
/// widening so plain SSE2 word shuffles suffice.
#[inline]
#[target_feature(enable = "sse2")]
fn splat_alpha(half: __m128i) -> __m128i {
  _mm_shufflehi_epi16::<0xFF>(_mm_shufflelo_epi16::<0xFF>(half))
}

/// Unaligned 16-byte load.
#[inline]
#[target_feature(enable = "sse2")]
fn load(bytes: *const u8) -> __m128i {
  // SAFETY: every caller passes a pointer to a `chunks_exact` window, so
  // 16 readable bytes are guaranteed; `loadu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm_loadu_si128(bytes.cast())
  }
}

/// Unaligned 16-byte store.
#[inline]
#[target_feature(enable = "sse2")]
fn store(bytes: *mut u8, value: __m128i) {
  // SAFETY: every caller passes a pointer to a `chunks_exact_mut` window,
  // so 16 writable bytes are guaranteed; `storeu` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm_storeu_si128(bytes.cast(), value)
  }
}

// ---------------------------------------------------------------------------
// Alpha8 kernels — 16 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "sse2")]
pub(super) fn alpha_blend_solid_sse2(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = _mm_set1_epi16(i16::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(16).zip(coverage.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let c = load(coverage.as_ptr());
    let sl = div255_round(_mm_mullo_epi16(lo(c), alpha));
    let sh = div255_round(_mm_mullo_epi16(hi(c), alpha));
    store(dst.as_mut_ptr(), _mm_packus_epi16(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn apply_matte_alpha_sse2(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = _mm_set1_epi16(i16::from(source_opacity));
  let full = _mm_set1_epi16(255);
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    let s = load(spx.as_ptr().cast());
    // `splat_alpha` replicates each pixel's alpha (lane 3/7 of the half)
    // across its four channel lanes, then `div255(alpha * op)` yields the
    // per-pixel factor on every channel lane; the scalar's factor==255 ("leave
    // dst unchanged") fast path is recovered chunk-wide for all-opaque chunks, and
    // factor==0 is reproduced exactly by the general rounding below.
    let mut fl = div255_round(_mm_mullo_epi16(splat_alpha(lo(s)), opacity));
    let mut fh = div255_round(_mm_mullo_epi16(splat_alpha(hi(s)), opacity));
    if inverted {
      fl = _mm_sub_epi16(full, fl);
      fh = _mm_sub_epi16(full, fh);
    }
    if _mm_movemask_epi8(_mm_cmpeq_epi16(fl, full)) == 0xFFFF && _mm_movemask_epi8(_mm_cmpeq_epi16(fh, full)) == 0xFFFF {
      continue;
    }
    let d = load(dpx.as_ptr().cast());
    let l = div255_round(_mm_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm_mullo_epi16(hi(d), fh));
    store(dpx.as_mut_ptr().cast(), _mm_packus_epi16(l, h));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_blend_product_sse2(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(16).zip(lhs.chunks_exact(16)).zip(rhs.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let l = load(lhs.as_ptr());
    let r = load(rhs.as_ptr());
    let sl = div255_round(_mm_mullo_epi16(lo(l), lo(r)));
    let sh = div255_round(_mm_mullo_epi16(hi(l), hi(r)));
    store(dst.as_mut_ptr(), _mm_packus_epi16(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_blend_uniform_sse2(dst: &mut [u8], source: u8) {
  let source = _mm_set1_epi16(i16::from(source));
  for dst in dst.chunks_exact_mut(16) {
    let d = load(dst.as_ptr());
    store(dst.as_mut_ptr(), _mm_packus_epi16(alpha_over8(lo(d), source), alpha_over8(hi(d), source)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_composite_over_sse2(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = _mm_set1_epi16(i16::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let sl = div255_round(_mm_mullo_epi16(lo(s), opacity));
    let sh = div255_round(_mm_mullo_epi16(hi(s), opacity));
    store(dst.as_mut_ptr(), _mm_packus_epi16(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_multiply_sse2(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(16).zip(factors.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let f = load(factors.as_ptr());
    let l = div255_round(_mm_mullo_epi16(lo(d), lo(f)));
    let h = div255_round(_mm_mullo_epi16(hi(d), hi(f)));
    store(dst.as_mut_ptr(), _mm_packus_epi16(l, h));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_matte_sse2(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = _mm_set1_epi16(i16::from(opacity));
  let full = _mm_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let mut fl = div255_round(_mm_mullo_epi16(lo(s), opacity));
    let mut fh = div255_round(_mm_mullo_epi16(hi(s), opacity));
    if inverted {
      fl = _mm_sub_epi16(full, fl);
      fh = _mm_sub_epi16(full, fh);
    }
    let l = div255_round(_mm_mullo_epi16(lo(d), fl));
    let h = div255_round(_mm_mullo_epi16(hi(d), fh));
    store(dst.as_mut_ptr(), _mm_packus_epi16(l, h));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn alpha_mask_combine_sse2(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = _mm_set1_epi16(i16::from(opacity));
  let full = _mm_set1_epi16(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let (dl, dh) = (lo(d), hi(d));
    let (mut sl, mut sh) = (lo(s), hi(s));
    if inverted {
      sl = _mm_sub_epi16(full, sl);
      sh = _mm_sub_epi16(full, sh);
    }
    let (cl, ch) = (div255_round(_mm_mullo_epi16(sl, opacity)), div255_round(_mm_mullo_epi16(sh, opacity)));
    let combine = |old, contribution| match mode {
      b's' => div255_round(_mm_mullo_epi16(old, _mm_sub_epi16(full, contribution))),
      b'i' => div255_round(_mm_mullo_epi16(old, contribution)),
      b'f' => _mm_sub_epi16(_mm_max_epi16(old, contribution), _mm_min_epi16(old, contribution)),
      _ => _mm_add_epi16(contribution, div255_round(_mm_mullo_epi16(_mm_sub_epi16(full, contribution), old))),
    };
    store(dst.as_mut_ptr(), _mm_packus_epi16(combine(dl, cl), combine(dh, ch)));
  }
}

// ---------------------------------------------------------------------------
// RGBA kernels — 4 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "sse2")]
pub(super) fn fill_span_solid_sse2(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let src = _mm_setr_epi16(sr as i16, sg as i16, sb as i16, 255, sr as i16, sg as i16, sb as i16, 255);
  let sa_w = _mm_set1_epi16(sa as i16);
  let full = _mm_set1_epi16(255);
  for (dpx, cpx) in dst.chunks_exact_mut(4).zip(cov.chunks_exact(4)) {
    let cw = u32::from_le_bytes(cpx.try_into().unwrap_or([0; 4]));
    // rep4 without pshufb: byte-double then word-double turns the four
    // coverage bytes into [c0 x4, c1 x4, c2 x4, c3 x4].
    let c = _mm_cvtsi32_si128(cw as i32);
    let doubled = _mm_unpacklo_epi8(c, c);
    let crep = _mm_unpacklo_epi16(doubled, doubled);
    let ca_lo = div255_round(_mm_mullo_epi16(lo(crep), sa_w));
    let ca_hi = div255_round(_mm_mullo_epi16(hi(crep), sa_w));
    let s_lo = div255_round(_mm_mullo_epi16(src, ca_lo));
    let s_hi = div255_round(_mm_mullo_epi16(src, ca_hi));
    let inv_lo = _mm_sub_epi16(full, ca_lo);
    let inv_hi = _mm_sub_epi16(full, ca_hi);
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), _mm_packus_epi16(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn fill_span_uniform_sse2(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let s = _mm_setr_epi16(s_r as i16, s_g as i16, s_b as i16, ca as i16, s_r as i16, s_g as i16, s_b as i16, ca as i16);
  let inv = _mm_set1_epi16(255 - ca as i16);
  for dpx in dst.chunks_exact_mut(4) {
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), _mm_packus_epi16(over(lo(d), s, inv), over(hi(d), s, inv)));
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn composite_over_sse2(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = _mm_set1_epi16(k as i16);
  let full = _mm_set1_epi16(255);
  let zero = _mm_setzero_si128();
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    let d = load(dpx.as_ptr().cast());
    let s = load(spx.as_ptr().cast());
    // All four source pixels fully transparent: `over` is the identity on
    // dst, so skip the round trip (matches the scalar `if s == 0`).
    if _mm_movemask_epi8(_mm_cmpeq_epi32(s, zero)) == 0xFFFF {
      continue;
    }
    let (s_lo_raw, s_hi_raw) = (lo(s), hi(s));
    let s_lo = div255_round(_mm_mullo_epi16(s_lo_raw, kq));
    let s_hi = div255_round(_mm_mullo_epi16(s_hi_raw, kq));
    let inv_lo = _mm_sub_epi16(full, div255_round(_mm_mullo_epi16(splat_alpha(s_lo_raw), kq)));
    let inv_hi = _mm_sub_epi16(full, div255_round(_mm_mullo_epi16(splat_alpha(s_hi_raw), kq)));
    store(dpx.as_mut_ptr().cast(), _mm_packus_epi16(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`, with lanes failing `valid`
/// forced to the u32::MAX sentinel (LUT miss -> transparent 0). Matches
/// the scalar `is_finite` gate + `as usize` truncation.
#[inline]
#[target_feature(enable = "sse2")]
fn lut_indices(t: __m128, valid: __m128, scale: __m128) -> __m128i {
  let tc = _mm_min_ps(_mm_max_ps(t, _mm_setzero_ps()), _mm_set1_ps(1.0));
  let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(tc, scale), _mm_set1_ps(0.5)));
  let vi = _mm_castps_si128(valid);
  _mm_or_si128(_mm_and_si128(vi, idx), _mm_andnot_si128(vi, _mm_set1_epi32(-1)))
}

/// 4 scalar LUT fetches (no gather in SSE2; sentinel misses -> 0).
#[inline]
#[target_feature(enable = "sse2")]
fn lut_gather(lut: &[u32], idx: __m128i) -> [u32; 4] {
  let mut raw = [0u32; 4];
  // SAFETY: `raw` is exactly four u32 = 16 writable bytes, and `storeu`
  // permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    _mm_storeu_si128(raw.as_mut_ptr().cast(), idx)
  };
  let [i0, i1, i2, i3] = raw;
  [
    lut.get(i0 as usize).copied().unwrap_or(0),
    lut.get(i1 as usize).copied().unwrap_or(0),
    lut.get(i2 as usize).copied().unwrap_or(0),
    lut.get(i3 as usize).copied().unwrap_or(0),
  ]
}

#[inline]
#[target_feature(enable = "sse2")]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: __m128i) {
  let [s0, s1, s2, s3] = lut_gather(lut, idx);
  if let [o0, o1, o2, o3] = chunk {
    (*o0, *o1, *o2, *o3) = (s0, s1, s2, s3);
  }
}

#[inline]
#[target_feature(enable = "sse2")]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: __m128i) {
  // Source and destination are premultiplied RGBA bytes. k=255 means
  // source channels pass through unchanged.
  let src = lut_gather(lut, idx);
  let full = _mm_set1_epi16(255);
  let d = load(dpx.as_ptr().cast());
  let s = load(src.as_ptr().cast());
  let (s_lo, s_hi) = (lo(s), hi(s));
  let inv_lo = _mm_sub_epi16(full, splat_alpha(s_lo));
  let inv_hi = _mm_sub_epi16(full, splat_alpha(s_hi));
  store(dpx.as_mut_ptr().cast(), _mm_packus_epi16(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
}

/// `|v|` — mask off the sign bit, matching `f32::abs`.
#[inline]
#[target_feature(enable = "sse2")]
fn abs_ps(v: __m128) -> __m128 {
  _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff)))
}

/// Lane-wise `a.max(b)` with Rust's NaN semantics: `f32::max` returns the
/// OTHER operand when one is NaN, while `maxps` unconditionally returns
/// its second. The focal solver feeds this straight into an `is_finite`
/// gate, so the difference is observable — select explicitly.
#[inline]
#[target_feature(enable = "sse2")]
fn max_ps_rust(a: __m128, b: __m128) -> __m128 {
  let b_nan = _mm_cmpunord_ps(b, b);
  _mm_or_ps(_mm_and_ps(b_nan, a), _mm_andnot_ps(b_nan, _mm_max_ps(a, b)))
}

/// Absolute device columns for one 4-lane chunk.
#[inline]
#[target_feature(enable = "sse2")]
fn lane_columns(x_start: f32) -> __m128 {
  _mm_add_ps(_mm_set1_ps(x_start), _mm_setr_ps(0.0, 1.0, 2.0, 3.0))
}

/// 4-lane linear gradient LUT fill: `t = row_base + X·dt` at absolute
/// device column `X = x_start + lane` (segmentation-invariant), exactly
/// the scalar form.
#[target_feature(enable = "sse2")]
pub(super) fn linear_lut_fill_sse2(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm_set1_ps(row_base);
  let dtv = _mm_set1_ps(dt);
  let four = _mm_set1_ps(4.0);
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in out.chunks_exact_mut(4) {
    let t = _mm_add_ps(t0v, _mm_mul_ps(kf, dtv));
    // is_finite parity: NaN fails the compare, ±inf fails |t|<inf.
    let finite = _mm_cmplt_ps(abs_ps(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm_add_ps(kf, four);
  }
}

#[target_feature(enable = "sse2")]
pub(super) fn linear_lut_over_sse2(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = _mm_set1_ps(row_base);
  let dtv = _mm_set1_ps(dt);
  let four = _mm_set1_ps(4.0);
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let t = _mm_add_ps(t0v, _mm_mul_ps(kf, dtv));
    let finite = _mm_cmplt_ps(abs_ps(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm_add_ps(kf, four);
  }
}

/// 4-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the
/// scalar `ddx*ddx + ddy*ddy`, and `sqrtps` matches scalar `sqrt()`, so
/// lanes agree with the `dd0 + X·d` scalar form bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse2")]
pub(super) fn radial_lut_fill_sse2(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm_set1_ps(da), _mm_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm_set1_ps(dd0x), _mm_set1_ps(dd0y));
  let inv_rv = _mm_set1_ps(inv_r);
  let four = _mm_set1_ps(4.0);
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in out.chunks_exact_mut(4) {
    let ddx = _mm_add_ps(ddx0v, _mm_mul_ps(kf, dav));
    let ddy = _mm_add_ps(ddy0v, _mm_mul_ps(kf, dbv));
    let gg = _mm_add_ps(_mm_mul_ps(ddx, ddx), _mm_mul_ps(ddy, ddy));
    let t = _mm_mul_ps(_mm_sqrt_ps(gg), inv_rv);
    let finite = _mm_cmplt_ps(abs_ps(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm_add_ps(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse2")]
pub(super) fn radial_lut_over_sse2(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (_mm_set1_ps(da), _mm_set1_ps(db));
  let (ddx0v, ddy0v) = (_mm_set1_ps(dd0x), _mm_set1_ps(dd0y));
  let inv_rv = _mm_set1_ps(inv_r);
  let four = _mm_set1_ps(4.0);
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let ddx = _mm_add_ps(ddx0v, _mm_mul_ps(kf, dav));
    let ddy = _mm_add_ps(ddy0v, _mm_mul_ps(kf, dbv));
    let gg = _mm_add_ps(_mm_mul_ps(ddx, ddx), _mm_mul_ps(ddy, ddy));
    let t = _mm_mul_ps(_mm_sqrt_ps(gg), inv_rv);
    let finite = _mm_cmplt_ps(abs_ps(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = _mm_add_ps(kf, four);
  }
}

/// 4-lane focal (highlight) radial LUT fill; keep the scalar absolute-X
/// Horner rounding protocol so SIMD tails cannot introduce seams.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse2")]
pub(super) fn focal_lut_fill_sse2(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm_set1_ps(b0), _mm_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm_set1_ps(d0), _mm_set1_ps(d1), _mm_set1_ps(d2));
  let inv2av = _mm_set1_ps(inv2a);
  let rv = _mm_set1_ps(r);
  let four = _mm_set1_ps(4.0);
  let zero = _mm_setzero_ps();
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in out.chunks_exact_mut(4) {
    let b = _mm_add_ps(b0v, _mm_mul_ps(kf, dbv));
    let det = _mm_add_ps(d0v, _mm_mul_ps(kf, _mm_add_ps(d1v, _mm_mul_ps(kf, d2v))));
    let sq = _mm_sqrt_ps(det);
    let nb = _mm_sub_ps(zero, b);
    let root = max_ps_rust(_mm_mul_ps(_mm_sub_ps(nb, sq), inv2av), _mm_mul_ps(_mm_add_ps(nb, sq), inv2av));
    let valid = _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(det, zero), _mm_cmpge_ps(_mm_mul_ps(rv, root), zero)), _mm_cmplt_ps(abs_ps(root), inf));
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm_add_ps(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse2")]
pub(super) fn focal_lut_over_sse2(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (_mm_set1_ps(b0), _mm_set1_ps(db));
  let (d0v, d1v, d2v) = (_mm_set1_ps(d0), _mm_set1_ps(d1), _mm_set1_ps(d2));
  let inv2av = _mm_set1_ps(inv2a);
  let rv = _mm_set1_ps(r);
  let four = _mm_set1_ps(4.0);
  let zero = _mm_setzero_ps();
  let inf = _mm_set1_ps(f32::INFINITY);
  let scalev = _mm_set1_ps(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let b = _mm_add_ps(b0v, _mm_mul_ps(kf, dbv));
    let det = _mm_add_ps(d0v, _mm_mul_ps(kf, _mm_add_ps(d1v, _mm_mul_ps(kf, d2v))));
    let sq = _mm_sqrt_ps(det);
    let nb = _mm_sub_ps(zero, b);
    let root = max_ps_rust(_mm_mul_ps(_mm_sub_ps(nb, sq), inv2av), _mm_mul_ps(_mm_add_ps(nb, sq), inv2av));
    let valid = _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(det, zero), _mm_cmpge_ps(_mm_mul_ps(rv, root), zero)), _mm_cmplt_ps(abs_ps(root), inf));
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = _mm_add_ps(kf, four);
  }
}
