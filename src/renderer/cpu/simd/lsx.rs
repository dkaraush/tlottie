//! LSX kernels, 4 pixels (u32) / 16 pixels (Alpha8) per iteration.

use core::arch::loongarch64::{
  lsx_bnz_h, lsx_bz_v, lsx_vadd_h, lsx_vand_v, lsx_vbitclri_w, lsx_vfadd_s, lsx_vfcmp_cle_s, lsx_vfcmp_clt_s, lsx_vfmax_s, lsx_vfmin_s, lsx_vfmul_s, lsx_vfsqrt_s, lsx_vfsub_s, lsx_vftintrz_w_s,
  lsx_vilvh_b, lsx_vilvl_b, lsx_vilvl_h, lsx_vld, lsx_vldi, lsx_vmax_h, lsx_vmin_h, lsx_vmuh_hu, lsx_vmul_h, lsx_vorn_v, lsx_vpickev_b, lsx_vreplgr2vr_b, lsx_vreplgr2vr_d, lsx_vreplgr2vr_h,
  lsx_vreplgr2vr_w, lsx_vrepli_h, lsx_vrepli_w, lsx_vseq_h, lsx_vshuf4i_h, lsx_vsrli_h, lsx_vst, lsx_vsub_h, lsx_vxor_v, m128, m128i,
};

/// Widens the low 8 bytes of `v` (first 2 pixels) to 8 u16 lanes. Interleaving
/// with zero measures faster than `vsllwil` on LA664.
#[inline]
#[target_feature(enable = "lsx")]
fn lo(v: m128i) -> m128i {
  lsx_vilvl_b(lsx_vldi::<0>(), v)
}

/// Widens the high 8 bytes of `v` (last 2 pixels) to 8 u16 lanes.
#[inline]
#[target_feature(enable = "lsx")]
fn hi(v: m128i) -> m128i {
  lsx_vilvh_b(lsx_vldi::<0>(), v)
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025): `(x * 0x8081) >> 23`
/// with `x = n + 127`, the multiply-high form LLVM itself emits for the scalar
/// division. It equals `x / 255` for every `x < 66299`, and `x` stays
/// `<= 65152` here, so the result is bit-identical to the scalar oracle.
#[inline]
#[target_feature(enable = "lsx")]
fn div255_round(n: m128i) -> m128i {
  lsx_vsrli_h::<7>(lsx_vmuh_hu(lsx_vadd_h(n, lsx_vrepli_h(127)), lsx_vreplgr2vr_h(0x8081)))
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
#[target_feature(enable = "lsx")]
fn over(d: m128i, s: m128i, inv: m128i) -> m128i {
  lsx_vmin_h(lsx_vadd_h(s, lsx_vsrli_h::<8>(lsx_vmul_h(d, lsx_vadd_h(inv, lsx_vrepli_h(1))))), lsx_vrepli_h(255))
}

#[inline]
#[target_feature(enable = "lsx")]
fn alpha_over8(dst: m128i, source: m128i) -> m128i {
  over(dst, source, lsx_vsub_h(lsx_vrepli_h(255), source))
}

/// Replicates each 8-lane half's alpha (channels 3/7) across its pixel's four
/// channel lanes
#[inline]
#[target_feature(enable = "lsx")]
fn splat_alpha(half: m128i) -> m128i {
  lsx_vshuf4i_h::<0xFF>(half)
}

/// Packs two 8-u16 halves into 16 bytes by keeping each lane's low byte.
/// Every caller passes lanes that are already `<= 255` (`div255_round`
/// results, `over`'s clamp, or sums/differences bounded by them), so this
/// truncation is exact and cheaper than a saturating `vssrani`. `vpickev`
/// takes its low half from the second operand, so the halves are swapped.
#[inline]
#[target_feature(enable = "lsx")]
fn pack(a: m128i, b: m128i) -> m128i {
  lsx_vpickev_b(b, a)
}

/// Unaligned 16-byte load.
#[inline]
#[target_feature(enable = "lsx")]
fn load(bytes: *const u8) -> m128i {
  // SAFETY: every caller passes a pointer to a `chunks_exact` window, so
  // 16 readable bytes are guaranteed; `vld` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lsx_vld(bytes.cast(), 0)
  }
}

/// Unaligned 16-byte store.
#[inline]
#[target_feature(enable = "lsx")]
fn store(bytes: *mut u8, value: m128i) {
  // SAFETY: every caller passes a pointer to a `chunks_exact_mut` window,
  // so 16 writable bytes are guaranteed; `vst` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lsx_vst(value, bytes.cast(), 0)
  }
}

// ---------------------------------------------------------------------------
// Alpha8 kernels — 16 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "lsx")]
pub(super) fn alpha_blend_solid_lsx(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = lsx_vreplgr2vr_h(i32::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(16).zip(coverage.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let c = load(coverage.as_ptr());
    let sl = div255_round(lsx_vmul_h(lo(c), alpha));
    let sh = div255_round(lsx_vmul_h(hi(c), alpha));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_blend_product_lsx(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(16).zip(lhs.chunks_exact(16)).zip(rhs.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let l = load(lhs.as_ptr());
    let r = load(rhs.as_ptr());
    let sl = div255_round(lsx_vmul_h(lo(l), lo(r)));
    let sh = div255_round(lsx_vmul_h(hi(l), hi(r)));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_blend_uniform_lsx(dst: &mut [u8], source: u8) {
  let source = lsx_vreplgr2vr_h(i32::from(source));
  for dst in dst.chunks_exact_mut(16) {
    let d = load(dst.as_ptr());
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), source), alpha_over8(hi(d), source)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_composite_over_lsx(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = lsx_vreplgr2vr_h(i32::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let sl = div255_round(lsx_vmul_h(lo(s), opacity));
    let sh = div255_round(lsx_vmul_h(hi(s), opacity));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_multiply_lsx(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(16).zip(factors.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let f = load(factors.as_ptr());
    let l = div255_round(lsx_vmul_h(lo(d), lo(f)));
    let h = div255_round(lsx_vmul_h(hi(d), hi(f)));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_matte_lsx(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = lsx_vreplgr2vr_h(i32::from(opacity));
  // `255 - f == f ^ 255` for `f <= 255`: inverting by xor keeps the loop free
  // of a per-chunk select on `inverted`.
  let invert = lsx_vreplgr2vr_h(if inverted { 255 } else { 0 });
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let fl = lsx_vxor_v(div255_round(lsx_vmul_h(lo(s), opacity)), invert);
    let fh = lsx_vxor_v(div255_round(lsx_vmul_h(hi(s), opacity)), invert);
    let l = div255_round(lsx_vmul_h(lo(d), fl));
    let h = div255_round(lsx_vmul_h(hi(d), fh));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn alpha_mask_combine_lsx(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = lsx_vreplgr2vr_h(i32::from(opacity));
  // `255 - s == s ^ 0xff` on bytes, applied before widening.
  let invert = lsx_vreplgr2vr_b(if inverted { -1 } else { 0 });
  let full = lsx_vrepli_h(255);
  // Dispatch on `mode` once, so each pixel loop has no mode branch.
  match mode {
    b's' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| div255_round(lsx_vmul_h(old, lsx_vsub_h(full, contribution)))),
    b'i' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| div255_round(lsx_vmul_h(old, contribution))),
    b'f' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| lsx_vsub_h(lsx_vmax_h(old, contribution), lsx_vmin_h(old, contribution))),
    _ => mask_combine_loop(dst, src, invert, opacity, |old, contribution| {
      lsx_vadd_h(contribution, div255_round(lsx_vmul_h(lsx_vsub_h(full, contribution), old)))
    }),
  }
}

#[inline]
#[target_feature(enable = "lsx")]
fn mask_combine_loop(dst: &mut [u8], src: &[u8], invert: m128i, opacity: m128i, combine: impl Fn(m128i, m128i) -> m128i) {
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    let d = load(dst.as_ptr());
    let s = lsx_vxor_v(load(src.as_ptr()), invert);
    let (cl, ch) = (div255_round(lsx_vmul_h(lo(s), opacity)), div255_round(lsx_vmul_h(hi(s), opacity)));
    store(dst.as_mut_ptr(), pack(combine(lo(d), cl), combine(hi(d), ch)));
  }
}

// ---------------------------------------------------------------------------
// RGBA kernels — 4 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "lsx")]
pub(super) fn apply_matte_alpha_lsx(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = lsx_vreplgr2vr_h(i32::from(source_opacity));
  let full = lsx_vrepli_h(255);
  // `255 - f == f ^ 255` for `f <= 255`, without a per-chunk select.
  let invert = lsx_vreplgr2vr_h(if inverted { 255 } else { 0 });
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    let s = load(spx.as_ptr().cast());
    let fl = lsx_vxor_v(div255_round(lsx_vmul_h(splat_alpha(lo(s)), opacity)), invert);
    let fh = lsx_vxor_v(div255_round(lsx_vmul_h(splat_alpha(hi(s)), opacity)), invert);
    if lsx_bnz_h(lsx_vseq_h(fl, full)) == 1 && lsx_bnz_h(lsx_vseq_h(fh, full)) == 1 {
      continue;
    }
    let d = load(dpx.as_ptr().cast());
    let l = div255_round(lsx_vmul_h(lo(d), fl));
    let h = div255_round(lsx_vmul_h(hi(d), fh));
    store(dpx.as_mut_ptr().cast(), pack(l, h));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn fill_span_solid_lsx(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let pat = sr as u64 | (sg as u64) << 16 | (sb as u64) << 32 | 255u64 << 48;
  let src = lsx_vreplgr2vr_d(pat as i64);
  let sa_w = lsx_vreplgr2vr_h(sa as i32);
  let full = lsx_vrepli_h(255);
  for (dpx, cpx) in dst.chunks_exact_mut(4).zip(cov.chunks_exact(4)) {
    // rep4: byte-double then word-double turns the four coverage bytes into
    // [c0 x4, c1 x4, c2 x4, c3 x4].
    let c = lsx_vreplgr2vr_w(u32::from_le_bytes(cpx.try_into().unwrap_or([0; 4])) as i32);
    let crep = lsx_vilvl_h(lsx_vilvl_b(c, c), lsx_vilvl_b(c, c));
    let ca_lo = div255_round(lsx_vmul_h(lo(crep), sa_w));
    let ca_hi = div255_round(lsx_vmul_h(hi(crep), sa_w));
    let s_lo = div255_round(lsx_vmul_h(src, ca_lo));
    let s_hi = div255_round(lsx_vmul_h(src, ca_hi));
    let inv_lo = lsx_vsub_h(full, ca_lo);
    let inv_hi = lsx_vsub_h(full, ca_hi);
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn fill_span_uniform_lsx(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let pat = s_r as u64 | (s_g as u64) << 16 | (s_b as u64) << 32 | (ca as u64) << 48;
  let s = lsx_vreplgr2vr_d(pat as i64);
  let inv = lsx_vreplgr2vr_h(255 - ca as i32);
  for dpx in dst.chunks_exact_mut(4) {
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s, inv), over(hi(d), s, inv)));
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn composite_over_lsx(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = lsx_vreplgr2vr_h(k as i32);
  let full = lsx_vrepli_h(255);
  for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
    let d = load(dpx.as_ptr().cast());
    let s = load(spx.as_ptr().cast());
    // All four source pixels fully transparent: `over` is the identity on
    // dst, so skip the round trip (matches the scalar `if s == 0`).
    if lsx_bz_v(s) == 1 {
      continue;
    }
    let (s_lo_raw, s_hi_raw) = (lo(s), hi(s));
    let s_lo = div255_round(lsx_vmul_h(s_lo_raw, kq));
    let s_hi = div255_round(lsx_vmul_h(s_hi_raw, kq));
    let inv_lo = lsx_vsub_h(full, div255_round(lsx_vmul_h(splat_alpha(s_lo_raw), kq)));
    let inv_hi = lsx_vsub_h(full, div255_round(lsx_vmul_h(splat_alpha(s_hi_raw), kq)));
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

/// LSX has no float broadcast instruction: broadcast the bit pattern, then
/// reinterpret it.
#[inline]
#[target_feature(enable = "lsx")]
fn splat_float(x: f32) -> m128 {
  #[allow(unsafe_code)]
  unsafe {
    core::mem::transmute(lsx_vreplgr2vr_w(x.to_bits() as i32))
  }
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`, with lanes failing `valid`
/// forced to the u32::MAX sentinel (LUT miss -> transparent 0). Matches
/// the scalar `is_finite` gate + `as usize` truncation.
#[inline]
#[target_feature(enable = "lsx")]
fn lut_indices(t: m128, valid: m128i, scale: m128) -> m128i {
  #[allow(unsafe_code)]
  let zero: m128 = unsafe { core::mem::transmute(lsx_vrepli_w(0)) };
  let tc = lsx_vfmin_s(lsx_vfmax_s(t, zero), splat_float(1.0));
  let idx = lsx_vftintrz_w_s(lsx_vfadd_s(lsx_vfmul_s(tc, scale), splat_float(0.5)));
  // (vi & idx) | (~vi & -1)  ==  idx | ~vi
  lsx_vorn_v(idx, valid)
}

/// 4-lane LUT gather LSX don't support gather.
#[inline]
#[target_feature(enable = "lsx")]
fn lut_gather(lut: &[u32], idx: m128i) -> m128i {
  let mut raw = [0u32; 4];
  // SAFETY: `raw` is exactly four u32 = 16 writable bytes, and `vst`
  // permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lsx_vst(idx, raw.as_mut_ptr().cast(), 0)
  };
  let raw = raw.map(|i| lut.get(i as usize).copied().unwrap_or(0));
  #[allow(unsafe_code)]
  unsafe {
    lsx_vld(raw.as_ptr().cast(), 0)
  }
}

#[inline]
#[target_feature(enable = "lsx")]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: m128i) {
  store(chunk.as_mut_ptr().cast(), lut_gather(lut, idx));
}

#[inline]
#[target_feature(enable = "lsx")]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: m128i) {
  // Source and destination are premultiplied RGBA bytes. k=255 means
  // source channels pass through unchanged.
  let full = lsx_vrepli_h(255);
  let d = load(dpx.as_ptr().cast());
  let s = lut_gather(lut, idx);
  let (s_lo, s_hi) = (lo(s), hi(s));
  let inv_lo = lsx_vsub_h(full, splat_alpha(s_lo));
  let inv_hi = lsx_vsub_h(full, splat_alpha(s_hi));
  store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
}

/// `|v|` — mask off the sign bit, matching `f32::abs`.
#[inline]
#[target_feature(enable = "lsx")]
fn abs_float(v: m128) -> m128 {
  #[allow(unsafe_code)]
  unsafe {
    core::mem::transmute(lsx_vbitclri_w::<31>(core::mem::transmute(v)))
  }
}

/// Absolute device columns for one 4-lane chunk.
#[inline]
#[target_feature(enable = "lsx")]
fn lane_columns(x_start: f32) -> m128 {
  #[allow(unsafe_code)]
  let lanes: m128 = unsafe { core::mem::transmute([0.0f32, 1.0, 2.0, 3.0]) };
  lsx_vfadd_s(splat_float(x_start), lanes)
}

/// 4-lane linear gradient LUT fill.
#[target_feature(enable = "lsx")]
pub(super) fn linear_lut_fill_lsx(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = splat_float(row_base);
  let dtv = splat_float(dt);
  let four = splat_float(4.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(4) {
    let t = lsx_vfadd_s(t0v, lsx_vfmul_s(kf, dtv));
    let finite = lsx_vfcmp_clt_s(abs_float(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}

#[target_feature(enable = "lsx")]
pub(super) fn linear_lut_over_lsx(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = splat_float(row_base);
  let dtv = splat_float(dt);
  let four = splat_float(4.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let t = lsx_vfadd_s(t0v, lsx_vfmul_s(kf, dtv));
    let finite = lsx_vfcmp_clt_s(abs_float(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}

/// 4-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the scalar
/// `ddx*ddx + ddy*ddy`, and `vfsqrt` matches scalar `sqrt()`, so lanes
/// agree with the `dd0 + X·d` scalar form bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lsx")]
pub(super) fn radial_lut_fill_lsx(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (splat_float(da), splat_float(db));
  let (ddx0v, ddy0v) = (splat_float(dd0x), splat_float(dd0y));
  let inv_rv = splat_float(inv_r);
  let four = splat_float(4.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(4) {
    let ddx = lsx_vfadd_s(ddx0v, lsx_vfmul_s(kf, dav));
    let ddy = lsx_vfadd_s(ddy0v, lsx_vfmul_s(kf, dbv));
    let gg = lsx_vfadd_s(lsx_vfmul_s(ddx, ddx), lsx_vfmul_s(ddy, ddy));
    let t = lsx_vfmul_s(lsx_vfsqrt_s(gg), inv_rv);
    let finite = lsx_vfcmp_clt_s(abs_float(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lsx")]
pub(super) fn radial_lut_over_lsx(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (splat_float(da), splat_float(db));
  let (ddx0v, ddy0v) = (splat_float(dd0x), splat_float(dd0y));
  let inv_rv = splat_float(inv_r);
  let four = splat_float(4.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let ddx = lsx_vfadd_s(ddx0v, lsx_vfmul_s(kf, dav));
    let ddy = lsx_vfadd_s(ddy0v, lsx_vfmul_s(kf, dbv));
    let gg = lsx_vfadd_s(lsx_vfmul_s(ddx, ddx), lsx_vfmul_s(ddy, ddy));
    let t = lsx_vfmul_s(lsx_vfsqrt_s(gg), inv_rv);
    let finite = lsx_vfcmp_clt_s(abs_float(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}

/// 4-lane focal (highlight) radial LUT fill.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lsx")]
pub(super) fn focal_lut_fill_lsx(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (splat_float(b0), splat_float(db));
  let (d0v, d1v, d2v) = (splat_float(d0), splat_float(d1), splat_float(d2));
  let inv2av = splat_float(inv2a);
  let rv = splat_float(r);
  let four = splat_float(4.0);
  let zero = splat_float(0.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(4) {
    let b = lsx_vfadd_s(b0v, lsx_vfmul_s(kf, dbv));
    let det = lsx_vfadd_s(d0v, lsx_vfmul_s(kf, lsx_vfadd_s(d1v, lsx_vfmul_s(kf, d2v))));
    let sq = lsx_vfsqrt_s(det);
    let nb = lsx_vfsub_s(zero, b);
    // LSX vfmax_s follows IEEE 754-2008 maxNum semantics, so just use it.
    let root = lsx_vfmax_s(lsx_vfmul_s(lsx_vfsub_s(nb, sq), inv2av), lsx_vfmul_s(lsx_vfadd_s(nb, sq), inv2av));
    let valid = lsx_vand_v(
      lsx_vand_v(lsx_vfcmp_cle_s(zero, det), lsx_vfcmp_cle_s(zero, lsx_vfmul_s(rv, root))),
      lsx_vfcmp_clt_s(abs_float(root), inf),
    );
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lsx")]
pub(super) fn focal_lut_over_lsx(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (splat_float(b0), splat_float(db));
  let (d0v, d1v, d2v) = (splat_float(d0), splat_float(d1), splat_float(d2));
  let inv2av = splat_float(inv2a);
  let rv = splat_float(r);
  let four = splat_float(4.0);
  let zero = splat_float(0.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(4) {
    let b = lsx_vfadd_s(b0v, lsx_vfmul_s(kf, dbv));
    let det = lsx_vfadd_s(d0v, lsx_vfmul_s(kf, lsx_vfadd_s(d1v, lsx_vfmul_s(kf, d2v))));
    let sq = lsx_vfsqrt_s(det);
    let nb = lsx_vfsub_s(zero, b);
    let root = lsx_vfmax_s(lsx_vfmul_s(lsx_vfsub_s(nb, sq), inv2av), lsx_vfmul_s(lsx_vfadd_s(nb, sq), inv2av));
    let valid = lsx_vand_v(
      lsx_vand_v(lsx_vfcmp_cle_s(zero, det), lsx_vfcmp_cle_s(zero, lsx_vfmul_s(rv, root))),
      lsx_vfcmp_clt_s(abs_float(root), inf),
    );
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = lsx_vfadd_s(kf, four);
  }
}
