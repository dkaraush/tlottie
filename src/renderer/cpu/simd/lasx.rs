//! LASX kernels, 8 pixels (u32) / 32 pixels (Alpha8) per iteration.

// Opaque kernel not implemented, on 3B6000 the chunked LSX/LASX
// store lost to the scalar run-length fill, because any AA edge
// sends its whole chunk to the per-pixel formula while the scalar
// path fills maximal runs.

use core::arch::loongarch64::{
  lasx_xbnz_h, lasx_xbz_v, lasx_xvadd_h, lasx_xvand_v, lasx_xvbitclri_w, lasx_xvfadd_s, lasx_xvfcmp_cle_s, lasx_xvfcmp_clt_s, lasx_xvfmax_s, lasx_xvfmin_s, lasx_xvfmul_s, lasx_xvfsqrt_s,
  lasx_xvfsub_s, lasx_xvftintrz_w_s, lasx_xvilvh_b, lasx_xvilvl_b, lasx_xvilvl_h, lasx_xvinsgr2vr_w, lasx_xvld, lasx_xvldi, lasx_xvmax_h, lasx_xvmin_h, lasx_xvmuh_hu, lasx_xvmul_h, lasx_xvorn_v,
  lasx_xvpickev_b, lasx_xvreplgr2vr_b, lasx_xvreplgr2vr_d, lasx_xvreplgr2vr_h, lasx_xvreplgr2vr_w, lasx_xvrepli_h, lasx_xvrepli_w, lasx_xvseq_h, lasx_xvshuf4i_h, lasx_xvsrli_h, lasx_xvst,
  lasx_xvsub_h, lasx_xvxor_v, m256, m256i,
};

/// Widens the low 8 bytes of each 128-bit lane (pixels 0-1 and 4-5) to 16 u16.
#[inline]
#[target_feature(enable = "lasx")]
fn lo(v: m256i) -> m256i {
  lasx_xvilvl_b(lasx_xvldi::<0>(), v)
}

/// Widens the high 8 bytes of each 128-bit lane (pixels 2-3 and 6-7) to 16 u16.
#[inline]
#[target_feature(enable = "lasx")]
fn hi(v: m256i) -> m256i {
  lasx_xvilvh_b(lasx_xvldi::<0>(), v)
}

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025): `(x * 0x8081) >> 23`
/// with `x = n + 127`, this comes from what LLVM emits for the scalar
/// division. It equals `x / 255` for every `x < 66299`, and `x` stays
/// `<= 65152` here.
#[inline]
#[target_feature(enable = "lasx")]
fn div255_round(n: m256i) -> m256i {
  lasx_xvsrli_h::<7>(lasx_xvmuh_hu(lasx_xvadd_h(n, lasx_xvrepli_h(127)), lasx_xvreplgr2vr_h(0x8081)))
}

/// Premultiplied source-over on u16 channel lanes.
#[inline]
#[target_feature(enable = "lasx")]
fn over(d: m256i, s: m256i, inv: m256i) -> m256i {
  lasx_xvmin_h(lasx_xvadd_h(s, lasx_xvsrli_h::<8>(lasx_xvmul_h(d, lasx_xvadd_h(inv, lasx_xvrepli_h(1))))), lasx_xvrepli_h(255))
}

#[inline]
#[target_feature(enable = "lasx")]
fn alpha_over8(dst: m256i, source: m256i) -> m256i {
  over(dst, source, lasx_xvsub_h(lasx_xvrepli_h(255), source))
}

/// Replicates each half's alpha (channels 3/7 of every 128-bit lane) across
/// its pixel's four channel lanes
#[inline]
#[target_feature(enable = "lasx")]
fn splat_alpha(half: m256i) -> m256i {
  lasx_xvshuf4i_h::<0xFF>(half)
}

/// Packs two 16-u16 halves into 32 bytes by keeping each lane's low byte.
/// [`lo`]/[`hi`] are lane-local, so the lane-local `xvpickev` lands every byte
/// back in place. Every caller passes lanes that are already `<= 255`
/// (`div255_round` results, `over`'s clamp, or sums/differences bounded by
/// them), so this truncation is exact and cheaper than a saturating
/// `xvssrani`. `xvpickev` takes its low half from the second operand, so the
/// halves are swapped.
#[inline]
#[target_feature(enable = "lasx")]
fn pack(a: m256i, b: m256i) -> m256i {
  lasx_xvpickev_b(b, a)
}

/// Unaligned 32-byte load.
#[inline]
#[target_feature(enable = "lasx")]
fn load(bytes: *const u8) -> m256i {
  // SAFETY: every caller passes a pointer to a `chunks_exact` window, so
  // 32 readable bytes are guaranteed; `xvld` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lasx_xvld(bytes.cast(), 0)
  }
}

/// Unaligned 32-byte store.
#[inline]
#[target_feature(enable = "lasx")]
fn store(bytes: *mut u8, value: m256i) {
  // SAFETY: every caller passes a pointer to a `chunks_exact_mut` window,
  // so 32 writable bytes are guaranteed; `xvst` permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lasx_xvst(value, bytes.cast(), 0)
  }
}

// ---------------------------------------------------------------------------
// Alpha8 kernels — 32 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "lasx")]
pub(super) fn alpha_blend_solid_lasx(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = lasx_xvreplgr2vr_h(i32::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(32).zip(coverage.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let c = load(coverage.as_ptr());
    let sl = div255_round(lasx_xvmul_h(lo(c), alpha));
    let sh = div255_round(lasx_xvmul_h(hi(c), alpha));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_blend_product_lasx(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(32).zip(lhs.chunks_exact(32)).zip(rhs.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let l = load(lhs.as_ptr());
    let r = load(rhs.as_ptr());
    let sl = div255_round(lasx_xvmul_h(lo(l), lo(r)));
    let sh = div255_round(lasx_xvmul_h(hi(l), hi(r)));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_blend_uniform_lasx(dst: &mut [u8], source: u8) {
  let source = lasx_xvreplgr2vr_h(i32::from(source));
  for dst in dst.chunks_exact_mut(32) {
    let d = load(dst.as_ptr());
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), source), alpha_over8(hi(d), source)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_composite_over_lasx(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = lasx_xvreplgr2vr_h(i32::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let sl = div255_round(lasx_xvmul_h(lo(s), opacity));
    let sh = div255_round(lasx_xvmul_h(hi(s), opacity));
    store(dst.as_mut_ptr(), pack(alpha_over8(lo(d), sl), alpha_over8(hi(d), sh)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_multiply_lasx(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(32).zip(factors.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let f = load(factors.as_ptr());
    let l = div255_round(lasx_xvmul_h(lo(d), lo(f)));
    let h = div255_round(lasx_xvmul_h(hi(d), hi(f)));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_matte_lasx(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = lasx_xvreplgr2vr_h(i32::from(opacity));
  // `255 - f == f ^ 255` for `f <= 255`: inverting by xor keeps the loop free
  // of a per-chunk select on `inverted`.
  let invert = lasx_xvreplgr2vr_h(if inverted { 255 } else { 0 });
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = load(src.as_ptr());
    let fl = lasx_xvxor_v(div255_round(lasx_xvmul_h(lo(s), opacity)), invert);
    let fh = lasx_xvxor_v(div255_round(lasx_xvmul_h(hi(s), opacity)), invert);
    let l = div255_round(lasx_xvmul_h(lo(d), fl));
    let h = div255_round(lasx_xvmul_h(hi(d), fh));
    store(dst.as_mut_ptr(), pack(l, h));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn alpha_mask_combine_lasx(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = lasx_xvreplgr2vr_h(i32::from(opacity));
  // `255 - s == s ^ 0xff` on bytes, applied before widening.
  let invert = lasx_xvreplgr2vr_b(if inverted { -1 } else { 0 });
  let full = lasx_xvrepli_h(255);
  // Dispatch on `mode` once, so each pixel loop has no mode branch.
  match mode {
    b's' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| div255_round(lasx_xvmul_h(old, lasx_xvsub_h(full, contribution)))),
    b'i' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| div255_round(lasx_xvmul_h(old, contribution))),
    b'f' => mask_combine_loop(dst, src, invert, opacity, |old, contribution| {
      lasx_xvsub_h(lasx_xvmax_h(old, contribution), lasx_xvmin_h(old, contribution))
    }),
    _ => mask_combine_loop(dst, src, invert, opacity, |old, contribution| {
      lasx_xvadd_h(contribution, div255_round(lasx_xvmul_h(lasx_xvsub_h(full, contribution), old)))
    }),
  }
}

#[inline]
#[target_feature(enable = "lasx")]
fn mask_combine_loop(dst: &mut [u8], src: &[u8], invert: m256i, opacity: m256i, combine: impl Fn(m256i, m256i) -> m256i) {
  for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
    let d = load(dst.as_ptr());
    let s = lasx_xvxor_v(load(src.as_ptr()), invert);
    let (cl, ch) = (div255_round(lasx_xvmul_h(lo(s), opacity)), div255_round(lasx_xvmul_h(hi(s), opacity)));
    store(dst.as_mut_ptr(), pack(combine(lo(d), cl), combine(hi(d), ch)));
  }
}

// ---------------------------------------------------------------------------
// RGBA kernels — 8 pixels per iteration.
// ---------------------------------------------------------------------------

#[target_feature(enable = "lasx")]
pub(super) fn apply_matte_alpha_lasx(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = lasx_xvreplgr2vr_h(i32::from(source_opacity));
  let full = lasx_xvrepli_h(255);
  // `255 - f == f ^ 255` for `f <= 255`, without a per-chunk select.
  let invert = lasx_xvreplgr2vr_h(if inverted { 255 } else { 0 });
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    let s = load(spx.as_ptr().cast());
    let fl = lasx_xvxor_v(div255_round(lasx_xvmul_h(splat_alpha(lo(s)), opacity)), invert);
    let fh = lasx_xvxor_v(div255_round(lasx_xvmul_h(splat_alpha(hi(s)), opacity)), invert);
    if lasx_xbnz_h(lasx_xvseq_h(fl, full)) == 1 && lasx_xbnz_h(lasx_xvseq_h(fh, full)) == 1 {
      continue;
    }
    let d = load(dpx.as_ptr().cast());
    let l = div255_round(lasx_xvmul_h(lo(d), fl));
    let h = div255_round(lasx_xvmul_h(hi(d), fh));
    store(dpx.as_mut_ptr().cast(), pack(l, h));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn fill_span_solid_lasx(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  // Source pattern per pixel is [R,G,B,255]: the 255 alpha lane makes
  // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
  // s_r/s_g/s_b AND s_a = ca in their lanes.
  let pat = sr as u64 | (sg as u64) << 16 | (sb as u64) << 32 | 255u64 << 48;
  let src = lasx_xvreplgr2vr_d(pat as i64);
  let sa_w = lasx_xvreplgr2vr_h(sa as i32);
  let full = lasx_xvrepli_h(255);
  for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
    // rep4: byte-double then word-double, per 4-byte group. xvilvl is per
    // 128-bit lane, so `crep` keeps the coverage in memory order and lines up
    // with `lo`/`hi` of the destination.
    let c0 = lasx_xvreplgr2vr_w(u32::from_le_bytes(cpx[0..4].try_into().unwrap_or([0; 4])) as i32);
    let c1 = lasx_xvinsgr2vr_w::<4>(c0, u32::from_le_bytes(cpx[4..8].try_into().unwrap_or([0; 4])) as i32);
    let crep = lasx_xvilvl_h(lasx_xvilvl_b(c1, c1), lasx_xvilvl_b(c1, c1));
    let ca_lo = div255_round(lasx_xvmul_h(lo(crep), sa_w));
    let ca_hi = div255_round(lasx_xvmul_h(hi(crep), sa_w));
    let s_lo = div255_round(lasx_xvmul_h(src, ca_lo));
    let s_hi = div255_round(lasx_xvmul_h(src, ca_hi));
    let inv_lo = lasx_xvsub_h(full, ca_lo);
    let inv_hi = lasx_xvsub_h(full, ca_hi);
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn fill_span_uniform_lasx(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let pat = s_r as u64 | (s_g as u64) << 16 | (s_b as u64) << 32 | (ca as u64) << 48;
  let s = lasx_xvreplgr2vr_d(pat as i64);
  let inv = lasx_xvreplgr2vr_h(255 - ca as i32);
  for dpx in dst.chunks_exact_mut(8) {
    let d = load(dpx.as_ptr().cast());
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s, inv), over(hi(d), s, inv)));
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn composite_over_lasx(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = lasx_xvreplgr2vr_h(k as i32);
  let full = lasx_xvrepli_h(255);
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    let d = load(dpx.as_ptr().cast());
    let s = load(spx.as_ptr().cast());
    // All eight source pixels fully transparent: `over` is the identity on
    // dst, so skip the round trip (matches the scalar `if s == 0`).
    if lasx_xbz_v(s) == 1 {
      continue;
    }
    let (s_lo_raw, s_hi_raw) = (lo(s), hi(s));
    let s_lo = div255_round(lasx_xvmul_h(s_lo_raw, kq));
    let s_hi = div255_round(lasx_xvmul_h(s_hi_raw, kq));
    let inv_lo = lasx_xvsub_h(full, div255_round(lasx_xvmul_h(splat_alpha(s_lo_raw), kq)));
    let inv_hi = lasx_xvsub_h(full, div255_round(lasx_xvmul_h(splat_alpha(s_hi_raw), kq)));
    store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
  }
}

/// LASX has no float broadcast instruction: broadcast the bit pattern, then
/// reinterpret it.
#[inline]
#[target_feature(enable = "lasx")]
fn splat_float(x: f32) -> m256 {
  #[allow(unsafe_code)]
  unsafe {
    core::mem::transmute(lasx_xvreplgr2vr_w(x.to_bits() as i32))
  }
}

/// Clamp + LUT index conversion shared by the gradient kernels:
/// `idx = trunc(clamp(t,0,1)*scale + 0.5)`, with lanes failing `valid`
/// forced to the u32::MAX sentinel (LUT miss -> transparent 0). Matches
/// the scalar `is_finite` gate + `as usize` truncation.
#[inline]
#[target_feature(enable = "lasx")]
fn lut_indices(t: m256, valid: m256i, scale: m256) -> m256i {
  #[allow(unsafe_code)]
  let zero: m256 = unsafe { core::mem::transmute(lasx_xvrepli_w(0)) };
  let tc = lasx_xvfmin_s(lasx_xvfmax_s(t, zero), splat_float(1.0));
  let idx = lasx_xvftintrz_w_s(lasx_xvfadd_s(lasx_xvfmul_s(tc, scale), splat_float(0.5)));
  // (vi & idx) | (~vi & -1)  ==  idx | ~vi
  lasx_xvorn_v(idx, valid)
}

/// 8-lane LUT gather LASX don't support gather.
#[inline]
#[target_feature(enable = "lasx")]
fn lut_gather(lut: &[u32], idx: m256i) -> m256i {
  let mut raw = [0u32; 8];
  // SAFETY: `raw` is exactly four u32 = 16 writable bytes, and `xvst`
  // permits unaligned access.
  #[allow(unsafe_code)]
  unsafe {
    lasx_xvst(idx, raw.as_mut_ptr().cast(), 0)
  };
  let raw = raw.map(|i| lut.get(i as usize).copied().unwrap_or(0));
  #[allow(unsafe_code)]
  unsafe {
    lasx_xvld(raw.as_ptr().cast(), 0)
  }
}

#[inline]
#[target_feature(enable = "lasx")]
fn lut_store(chunk: &mut [u32], lut: &[u32], idx: m256i) {
  store(chunk.as_mut_ptr().cast(), lut_gather(lut, idx));
}

#[inline]
#[target_feature(enable = "lasx")]
fn lut_blend_over_k255(dpx: &mut [u32], lut: &[u32], idx: m256i) {
  // Source and destination are premultiplied RGBA bytes. k=255 means
  // source channels pass through unchanged.
  let full = lasx_xvrepli_h(255);
  let d = load(dpx.as_ptr().cast());
  let s = lut_gather(lut, idx);
  let (s_lo, s_hi) = (lo(s), hi(s));
  let inv_lo = lasx_xvsub_h(full, splat_alpha(s_lo));
  let inv_hi = lasx_xvsub_h(full, splat_alpha(s_hi));
  store(dpx.as_mut_ptr().cast(), pack(over(lo(d), s_lo, inv_lo), over(hi(d), s_hi, inv_hi)));
}

/// `|v|` — mask off the sign bit, matching `f32::abs`.
#[inline]
#[target_feature(enable = "lasx")]
fn abs_float(v: m256) -> m256 {
  #[allow(unsafe_code)]
  unsafe {
    core::mem::transmute(lasx_xvbitclri_w::<31>(core::mem::transmute(v)))
  }
}

/// Absolute device columns for one 8-lane chunk.
#[inline]
#[target_feature(enable = "lasx")]
fn lane_columns(x_start: f32) -> m256 {
  #[allow(unsafe_code)]
  let lanes: m256 = unsafe { core::mem::transmute([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) };
  lasx_xvfadd_s(splat_float(x_start), lanes)
}

/// 8-lane linear gradient LUT fill.
#[target_feature(enable = "lasx")]
pub(super) fn linear_lut_fill_lasx(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = splat_float(row_base);
  let dtv = splat_float(dt);
  let eight = splat_float(8.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(8) {
    let t = lasx_xvfadd_s(t0v, lasx_xvfmul_s(kf, dtv));
    let finite = lasx_xvfcmp_clt_s(abs_float(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}

#[target_feature(enable = "lasx")]
pub(super) fn linear_lut_over_lasx(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let t0v = splat_float(row_base);
  let dtv = splat_float(dt);
  let eight = splat_float(8.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let t = lasx_xvfadd_s(t0v, lasx_xvfmul_s(kf, dtv));
    let finite = lasx_xvfcmp_clt_s(abs_float(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}

/// 8-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the scalar
/// `ddx*ddx + ddy*ddy`, and `sqrtps` matches scalar `sqrt()`, so lanes
/// agree with the `dd0 + X·d` scalar form bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lasx")]
pub(super) fn radial_lut_fill_lasx(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (splat_float(da), splat_float(db));
  let (ddx0v, ddy0v) = (splat_float(dd0x), splat_float(dd0y));
  let inv_rv = splat_float(inv_r);
  let eight = splat_float(8.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(8) {
    let ddx = lasx_xvfadd_s(ddx0v, lasx_xvfmul_s(kf, dav));
    let ddy = lasx_xvfadd_s(ddy0v, lasx_xvfmul_s(kf, dbv));
    let gg = lasx_xvfadd_s(lasx_xvfmul_s(ddx, ddx), lasx_xvfmul_s(ddy, ddy));
    let t = lasx_xvfmul_s(lasx_xvfsqrt_s(gg), inv_rv);
    let finite = lasx_xvfcmp_clt_s(abs_float(t), inf);
    lut_store(chunk, lut, lut_indices(t, finite, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lasx")]
pub(super) fn radial_lut_over_lasx(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (dav, dbv) = (splat_float(da), splat_float(db));
  let (ddx0v, ddy0v) = (splat_float(dd0x), splat_float(dd0y));
  let inv_rv = splat_float(inv_r);
  let eight = splat_float(8.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let ddx = lasx_xvfadd_s(ddx0v, lasx_xvfmul_s(kf, dav));
    let ddy = lasx_xvfadd_s(ddy0v, lasx_xvfmul_s(kf, dbv));
    let gg = lasx_xvfadd_s(lasx_xvfmul_s(ddx, ddx), lasx_xvfmul_s(ddy, ddy));
    let t = lasx_xvfmul_s(lasx_xvfsqrt_s(gg), inv_rv);
    let finite = lasx_xvfcmp_clt_s(abs_float(t), inf);
    lut_blend_over_k255(chunk, lut, lut_indices(t, finite, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}

/// 8-lane focal (highlight) radial LUT fill.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lasx")]
pub(super) fn focal_lut_fill_lasx(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (splat_float(b0), splat_float(db));
  let (d0v, d1v, d2v) = (splat_float(d0), splat_float(d1), splat_float(d2));
  let inv2av = splat_float(inv2a);
  let rv = splat_float(r);
  let eight = splat_float(8.0);
  let zero = splat_float(0.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in out.chunks_exact_mut(8) {
    let b = lasx_xvfadd_s(b0v, lasx_xvfmul_s(kf, dbv));
    let det = lasx_xvfadd_s(d0v, lasx_xvfmul_s(kf, lasx_xvfadd_s(d1v, lasx_xvfmul_s(kf, d2v))));
    let sq = lasx_xvfsqrt_s(det);
    let nb = lasx_xvfsub_s(zero, b);
    // LASX xvfmax_s follows IEEE 754-2008 maxNum semantics, so just use it.
    let root = lasx_xvfmax_s(lasx_xvfmul_s(lasx_xvfsub_s(nb, sq), inv2av), lasx_xvfmul_s(lasx_xvfadd_s(nb, sq), inv2av));
    let valid = lasx_xvand_v(
      lasx_xvand_v(lasx_xvfcmp_cle_s(zero, det), lasx_xvfcmp_cle_s(zero, lasx_xvfmul_s(rv, root))),
      lasx_xvfcmp_clt_s(abs_float(root), inf),
    );
    lut_store(chunk, lut, lut_indices(root, valid, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "lasx")]
pub(super) fn focal_lut_over_lasx(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let mut kf = lane_columns(x_start);
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (splat_float(b0), splat_float(db));
  let (d0v, d1v, d2v) = (splat_float(d0), splat_float(d1), splat_float(d2));
  let inv2av = splat_float(inv2a);
  let rv = splat_float(r);
  let eight = splat_float(8.0);
  let zero = splat_float(0.0);
  let inf = splat_float(f32::INFINITY);
  let scalev = splat_float(scale);
  for chunk in dst.chunks_exact_mut(8) {
    let b = lasx_xvfadd_s(b0v, lasx_xvfmul_s(kf, dbv));
    let det = lasx_xvfadd_s(d0v, lasx_xvfmul_s(kf, lasx_xvfadd_s(d1v, lasx_xvfmul_s(kf, d2v))));
    let sq = lasx_xvfsqrt_s(det);
    let nb = lasx_xvfsub_s(zero, b);
    let root = lasx_xvfmax_s(lasx_xvfmul_s(lasx_xvfsub_s(nb, sq), inv2av), lasx_xvfmul_s(lasx_xvfadd_s(nb, sq), inv2av));
    let valid = lasx_xvand_v(
      lasx_xvand_v(lasx_xvfcmp_cle_s(zero, det), lasx_xvfcmp_cle_s(zero, lasx_xvfmul_s(rv, root))),
      lasx_xvfcmp_clt_s(abs_float(root), inf),
    );
    lut_blend_over_k255(chunk, lut, lut_indices(root, valid, scalev));
    kf = lasx_xvfadd_s(kf, eight);
  }
}
