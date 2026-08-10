//! NEON kernels, 8 pixels per iteration. u16 lanes never overflow:
//! every product is (<=255)*(<=255) <= 65025 and div255's intermediate
//! stays <= 65407.

use core::arch::aarch64::{
  uint16x8_t, uint32x4_t, uint8x16_t, uint8x8x4_t, vabsq_f32, vaddq_f32, vaddq_u16, vandq_u32, vbslq_u32, vceqq_u16, vcgeq_f32, vcltq_f32, vcombine_u8, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_u16, vdupq_n_u32,
  vget_high_u8, vget_low_u8, vgetq_lane_u32, vld1_u8, vld1q_f32, vld1q_u8, vld4_u8, vmaxq_f32, vmaxq_u16, vminq_f32, vminq_u16, vminvq_u16, vminv_u8, vmovl_u8, vmovn_u16, vmulq_f32, vmulq_u16, vnegq_f32,
  vshrq_n_u16, vsqrtq_f32, vst1q_u32, vst1q_u8, vst4_u8, vsubq_f32, vsubq_u16,
};

/// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
#[inline]
#[target_feature(enable = "neon")]
fn div255_round(n: uint16x8_t) -> uint16x8_t {
  let t = vaddq_u16(n, vdupq_n_u16(127));
  let u = vaddq_u16(vaddq_u16(t, vshrq_n_u16::<8>(t)), vdupq_n_u16(1));
  vshrq_n_u16::<8>(u)
}

#[inline]
#[target_feature(enable = "neon")]
fn alpha_unpack(v: uint8x16_t) -> (uint16x8_t, uint16x8_t) {
  (vmovl_u8(vget_low_u8(v)), vmovl_u8(vget_high_u8(v)))
}

#[inline]
#[target_feature(enable = "neon")]
fn alpha_pack(lo: uint16x8_t, hi: uint16x8_t) -> uint8x16_t {
  vcombine_u8(vmovn_u16(lo), vmovn_u16(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn alpha_over8(dst: uint16x8_t, source: uint16x8_t) -> uint16x8_t {
  over(dst, source, vsubq_u16(vdupq_n_u16(255), source))
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_blend_solid_neon(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  let alpha = vdupq_n_u16(u16::from(alpha));
  for (dst, coverage) in dst.chunks_exact_mut(16).zip(coverage.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, c) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(coverage.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (cl, ch) = alpha_unpack(c);
    let sl = div255_round(vmulq_u16(cl, alpha));
    let sh = div255_round(vmulq_u16(ch, alpha));
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(alpha_over8(dl, sl), alpha_over8(dh, sh)))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_blend_product_neon(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, lhs), rhs) in dst.chunks_exact_mut(16).zip(lhs.chunks_exact(16)).zip(rhs.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, l, r) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(lhs.as_ptr()), vld1q_u8(rhs.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (ll, lh) = alpha_unpack(l);
    let (rl, rh) = alpha_unpack(r);
    let sl = div255_round(vmulq_u16(ll, rl));
    let sh = div255_round(vmulq_u16(lh, rh));
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(alpha_over8(dl, sl), alpha_over8(dh, sh)))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_blend_uniform_neon(dst: &mut [u8], source: u8) {
  let source = vdupq_n_u16(u16::from(source));
  for dst in dst.chunks_exact_mut(16) {
    #[allow(unsafe_code)]
    let d = unsafe { vld1q_u8(dst.as_ptr()) };
    let (dl, dh) = alpha_unpack(d);
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(alpha_over8(dl, source), alpha_over8(dh, source)))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_composite_over_neon(dst: &mut [u8], src: &[u8], opacity: u8) {
  let opacity = vdupq_n_u16(u16::from(opacity));
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, s) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(src.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (sl, sh) = alpha_unpack(s);
    let sl = div255_round(vmulq_u16(sl, opacity));
    let sh = div255_round(vmulq_u16(sh, opacity));
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(alpha_over8(dl, sl), alpha_over8(dh, sh)))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_multiply_neon(dst: &mut [u8], factors: &[u8]) {
  for (dst, factors) in dst.chunks_exact_mut(16).zip(factors.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, f) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(factors.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (fl, fh) = alpha_unpack(f);
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(div255_round(vmulq_u16(dl, fl)), div255_round(vmulq_u16(dh, fh))))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_matte_neon(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let opacity = vdupq_n_u16(u16::from(opacity));
  let full = vdupq_n_u16(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, s) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(src.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (sl, sh) = alpha_unpack(s);
    let mut fl = div255_round(vmulq_u16(sl, opacity));
    let mut fh = div255_round(vmulq_u16(sh, opacity));
    if inverted {
      fl = vsubq_u16(full, fl);
      fh = vsubq_u16(full, fh);
    }
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(div255_round(vmulq_u16(dl, fl)), div255_round(vmulq_u16(dh, fh))))
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn alpha_mask_combine_neon(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let opacity = vdupq_n_u16(u16::from(opacity));
  let full = vdupq_n_u16(255);
  for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
    #[allow(unsafe_code)]
    let (d, s) = unsafe { (vld1q_u8(dst.as_ptr()), vld1q_u8(src.as_ptr())) };
    let (dl, dh) = alpha_unpack(d);
    let (mut sl, mut sh) = alpha_unpack(s);
    if inverted {
      sl = vsubq_u16(full, sl);
      sh = vsubq_u16(full, sh);
    }
    let cl = div255_round(vmulq_u16(sl, opacity));
    let ch = div255_round(vmulq_u16(sh, opacity));
    let combine = |old, contribution| match mode {
      b's' => div255_round(vmulq_u16(old, vsubq_u16(full, contribution))),
      b'i' => div255_round(vmulq_u16(old, contribution)),
      b'f' => vsubq_u16(vmaxq_u16(old, contribution), vminq_u16(old, contribution)),
      _ => vaddq_u16(contribution, div255_round(vmulq_u16(vsubq_u16(full, contribution), old))),
    };
    #[allow(unsafe_code)]
    unsafe {
      vst1q_u8(dst.as_mut_ptr(), alpha_pack(combine(dl, cl), combine(dh, ch)))
    };
  }
}

/// 4-lane radial gradient LUT fill for full-coverage runs. Lane math:
/// `mul + add` (NOT fma) to mirror the scalar `ddx*ddx + ddy*ddy`;
/// FSQRT/vcvt(truncate) match scalar `sqrt()`/`as usize` exactly, so
/// NEON and the `dd0 + X·d` scalar form agree lane-for-lane. `X` is the
/// absolute device column (`x_start + lane`), making positions
/// segmentation-invariant.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) fn radial_lut_fill_neon(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array. `x_start` and the
  // lane offsets are small exact integers (< 2^24), so `x_start+lane`
  // is the exact absolute column, matching the scalar `(X as f32)`.
  #[allow(unsafe_code)]
  let kv = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let dav = vdupq_n_f32(da);
  let dbv = vdupq_n_f32(db);
  let ddx0v = vdupq_n_f32(dd0x);
  let ddy0v = vdupq_n_f32(dd0y);
  // X advances by exact integer f32 adds (exact ≤ 2^24), and each
  // chunk recomputes `dd = dd0 + X·d` — the same expression as the
  // scalar form, so lanes match it bit-for-bit at any X.
  let mut kf = kv;
  let four = vdupq_n_f32(4.0);
  let inv_rv = vdupq_n_f32(inv_r);
  let scalev = vdupq_n_f32(scale);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let sentinel = vdupq_n_u32(u32::MAX);
  for chunk in out.chunks_exact_mut(4) {
    let ddx = vaddq_f32(ddx0v, vmulq_f32(kf, dav));
    let ddy = vaddq_f32(ddy0v, vmulq_f32(kf, dbv));
    let gg = vaddq_f32(vmulq_f32(ddx, ddx), vmulq_f32(ddy, ddy));
    let t = vmulq_f32(vsqrtq_f32(gg), inv_rv);
    // clamp; NaN lanes fail t==t and take the sentinel (→ 0 pixel).
    let tc = vminq_f32(vmaxq_f32(t, zero), one);
    let fidx = vaddq_f32(vmulq_f32(tc, scalev), half);
    let idx = vcvtq_u32_f32(fidx); // truncates like `as usize`
                                   // is_finite parity: NaN fails the compare, ±inf fails |t|<inf.
    let finite = vcltq_f32(vabsq_f32(t), vdupq_n_f32(f32::INFINITY));
    let idx = vbslq_u32(finite, idx, sentinel);
    let i0 = vgetq_lane_u32::<0>(idx) as usize;
    let i1 = vgetq_lane_u32::<1>(idx) as usize;
    let i2 = vgetq_lane_u32::<2>(idx) as usize;
    let i3 = vgetq_lane_u32::<3>(idx) as usize;
    // No NEON gather: 4 scalar LUT fetches (sentinel misses → 0).
    if let [o0, o1, o2, o3] = chunk {
      *o0 = lut.get(i0).copied().unwrap_or(0);
      *o1 = lut.get(i1).copied().unwrap_or(0);
      *o2 = lut.get(i2).copied().unwrap_or(0);
      *o3 = lut.get(i3).copied().unwrap_or(0);
    }
    kf = vaddq_f32(kf, four);
  }
}

/// 4-lane linear gradient LUT fill: `t = row_base + X·dt` at absolute
/// device column `X = x_start + lane` (segmentation-invariant), clamp,
/// convert, 4 scalar LUT fetches (no gather on NEON).
#[target_feature(enable = "neon")]
pub(super) fn linear_lut_fill_neon(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array. `x_start + lane` are
  // exact integer columns (< 2^24), matching the scalar `(X as f32)`.
  #[allow(unsafe_code)]
  let mut kf = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let t0v = vdupq_n_f32(row_base);
  let dtv = vdupq_n_f32(dt);
  let four = vdupq_n_f32(4.0);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let scalev = vdupq_n_f32(scale);
  let inf = vdupq_n_f32(f32::INFINITY);
  let sentinel = vdupq_n_u32(u32::MAX);
  for chunk in out.chunks_exact_mut(4) {
    let t = vaddq_f32(t0v, vmulq_f32(kf, dtv));
    let tc = vminq_f32(vmaxq_f32(t, zero), one);
    let idx = vcvtq_u32_f32(vaddq_f32(vmulq_f32(tc, scalev), half));
    // is_finite parity: NaN fails the compare, ±inf fails |t|<inf.
    let finite = vcltq_f32(vabsq_f32(t), inf);
    let idx = vbslq_u32(finite, idx, sentinel);
    let i0 = vgetq_lane_u32::<0>(idx) as usize;
    let i1 = vgetq_lane_u32::<1>(idx) as usize;
    let i2 = vgetq_lane_u32::<2>(idx) as usize;
    let i3 = vgetq_lane_u32::<3>(idx) as usize;
    if let [o0, o1, o2, o3] = chunk {
      *o0 = lut.get(i0).copied().unwrap_or(0);
      *o1 = lut.get(i1).copied().unwrap_or(0);
      *o2 = lut.get(i2).copied().unwrap_or(0);
      *o3 = lut.get(i3).copied().unwrap_or(0);
    }
    kf = vaddq_f32(kf, four);
  }
}

/// 4-lane focal (highlight) radial LUT fill. B/determinant use the
/// absolute-X Horner form from `focal_lut_fill_scalar` (mul/add, no fma).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) fn focal_lut_fill_neon(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array. `x_start + lane` are
  // exact integer columns (< 2^24), matching the scalar `(X as f32)`.
  #[allow(unsafe_code)]
  let mut kf = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (vdupq_n_f32(b0), vdupq_n_f32(db));
  let (d0v, d1v, d2v) = (vdupq_n_f32(d0), vdupq_n_f32(d1), vdupq_n_f32(d2));
  let inv2av = vdupq_n_f32(inv2a);
  let rv = vdupq_n_f32(r);
  let four = vdupq_n_f32(4.0);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let inf = vdupq_n_f32(f32::INFINITY);
  let scalev = vdupq_n_f32(scale);
  let sentinel = vdupq_n_u32(u32::MAX);
  for chunk in out.chunks_exact_mut(4) {
    let b = vaddq_f32(b0v, vmulq_f32(kf, dbv));
    let det = vaddq_f32(d0v, vmulq_f32(kf, vaddq_f32(d1v, vmulq_f32(kf, d2v))));
    let sq = vsqrtq_f32(det);
    let nb = vnegq_f32(b);
    let root = vmaxq_f32(vmulq_f32(vsubq_f32(nb, sq), inv2av), vmulq_f32(vaddq_f32(nb, sq), inv2av));
    let valid = vandq_u32(vandq_u32(vcgeq_f32(det, zero), vcgeq_f32(vmulq_f32(rv, root), zero)), vcltq_f32(vabsq_f32(root), inf));
    let tc = vminq_f32(vmaxq_f32(root, zero), one);
    let idx = vcvtq_u32_f32(vaddq_f32(vmulq_f32(tc, scalev), half));
    let idx = vbslq_u32(valid, idx, sentinel);
    let i0 = vgetq_lane_u32::<0>(idx) as usize;
    let i1 = vgetq_lane_u32::<1>(idx) as usize;
    let i2 = vgetq_lane_u32::<2>(idx) as usize;
    let i3 = vgetq_lane_u32::<3>(idx) as usize;
    if let [o0, o1, o2, o3] = chunk {
      *o0 = lut.get(i0).copied().unwrap_or(0);
      *o1 = lut.get(i1).copied().unwrap_or(0);
      *o2 = lut.get(i2).copied().unwrap_or(0);
      *o3 = lut.get(i3).copied().unwrap_or(0);
    }
    kf = vaddq_f32(kf, four);
  }
}

/// Premultiplied source-over of scaled source planes into dst planes.
#[inline]
#[target_feature(enable = "neon")]
fn over(d: uint16x8_t, s: uint16x8_t, inv: uint16x8_t) -> uint16x8_t {
  vminq_u16(vaddq_u16(s, vshrq_n_u16::<8>(vmulq_u16(d, vaddq_u16(inv, vdupq_n_u16(1))))), vdupq_n_u16(255))
}

#[target_feature(enable = "neon")]
pub(super) fn fill_span_solid_neon(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  let (sr, sg, sb, sa) = (vdupq_n_u16(sr as u16), vdupq_n_u16(sg as u16), vdupq_n_u16(sb as u16), vdupq_n_u16(sa as u16));
  let full = vdupq_n_u16(255);
  for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
    // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes) at
    // dpx and 8 bytes at cpx; vld4_u8/vld1_u8/vst4_u8 read/write
    // exactly those spans.
    #[allow(unsafe_code)]
    let (planes, c) = unsafe { (vld4_u8(dpx.as_ptr().cast::<u8>()), vld1_u8(cpx.as_ptr())) };
    let cw = vmovl_u8(c);
    let ca = div255_round(vmulq_u16(cw, sa));
    let s_r = div255_round(vmulq_u16(sr, ca));
    let s_g = div255_round(vmulq_u16(sg, ca));
    let s_b = div255_round(vmulq_u16(sb, ca));
    let inv = vsubq_u16(full, ca);
    let d_r = vmovl_u8(planes.0);
    let d_g = vmovl_u8(planes.1);
    let d_b = vmovl_u8(planes.2);
    let d_a = vmovl_u8(planes.3);
    let out = uint8x8x4_t(
      vmovn_u16(over(d_r, s_r, inv)),
      vmovn_u16(over(d_g, s_g, inv)),
      vmovn_u16(over(d_b, s_b, inv)),
      vmovn_u16(over(d_a, ca, inv)),
    );
    // SAFETY: same 32-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      vst4_u8(dpx.as_mut_ptr().cast::<u8>(), out)
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn fill_span_opaque_neon(dst: &mut [u32], cov: &[u8], color: u32, sr: u32, sg: u32, sb: u32) {
  let color4 = vdupq_n_u32(color);
  for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
    // SAFETY: both chunks have the exact sizes used by these loads/stores.
    #[allow(unsafe_code)]
    let c = unsafe { vld1_u8(cpx.as_ptr()) };
    if vminv_u8(c) == 255 {
      #[allow(unsafe_code)]
      unsafe {
        vst1q_u32(dpx.as_mut_ptr(), color4);
        vst1q_u32(dpx.as_mut_ptr().add(4), color4);
      }
    } else {
      fill_span_solid_neon(dpx, cpx, sr, sg, sb, 255);
    }
  }
}

#[target_feature(enable = "neon")]
pub(super) fn fill_span_uniform_neon(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let sa = vdupq_n_u16(ca as u16);
  let sr = vdupq_n_u16(s_r as u16);
  let sg = vdupq_n_u16(s_g as u16);
  let sb = vdupq_n_u16(s_b as u16);
  let inv = vsubq_u16(vdupq_n_u16(255), sa);
  for dpx in dst.chunks_exact_mut(8) {
    // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes).
    #[allow(unsafe_code)]
    let planes = unsafe { vld4_u8(dpx.as_ptr().cast::<u8>()) };
    let out = uint8x8x4_t(
      vmovn_u16(over(vmovl_u8(planes.0), sr, inv)),
      vmovn_u16(over(vmovl_u8(planes.1), sg, inv)),
      vmovn_u16(over(vmovl_u8(planes.2), sb, inv)),
      vmovn_u16(over(vmovl_u8(planes.3), sa, inv)),
    );
    // SAFETY: same 32-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      vst4_u8(dpx.as_mut_ptr().cast::<u8>(), out)
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn composite_over_neon(dst: &mut [u32], src: &[u32], k: u32) {
  let kq = vdupq_n_u16(k as u16);
  let full = vdupq_n_u16(255);
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes) at
    // both pointers.
    #[allow(unsafe_code)]
    let (d4, s4) = unsafe { (vld4_u8(dpx.as_ptr().cast::<u8>()), vld4_u8(spx.as_ptr().cast::<u8>())) };
    let s_b = div255_round(vmulq_u16(vmovl_u8(s4.0), kq));
    let s_g = div255_round(vmulq_u16(vmovl_u8(s4.1), kq));
    let s_r = div255_round(vmulq_u16(vmovl_u8(s4.2), kq));
    let s_a = div255_round(vmulq_u16(vmovl_u8(s4.3), kq));
    let inv = vsubq_u16(full, s_a);
    let out = uint8x8x4_t(
      vmovn_u16(over(vmovl_u8(d4.0), s_b, inv)),
      vmovn_u16(over(vmovl_u8(d4.1), s_g, inv)),
      vmovn_u16(over(vmovl_u8(d4.2), s_r, inv)),
      vmovn_u16(over(vmovl_u8(d4.3), s_a, inv)),
    );
    // SAFETY: same 32-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      vst4_u8(dpx.as_mut_ptr().cast::<u8>(), out)
    };
  }
}

#[target_feature(enable = "neon")]
pub(super) fn apply_matte_alpha_neon(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let opacity = vdupq_n_u16(u16::from(source_opacity));
  let full = vdupq_n_u16(255);
  for (dpx, spx) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
    // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes) at
    // both pointers.
    #[allow(unsafe_code)]
    let s4 = unsafe { vld4_u8(spx.as_ptr().cast::<u8>()) };
    let mut f = div255_round(vmulq_u16(vmovl_u8(s4.3), opacity));
    if inverted {
      f = vsubq_u16(full, f);
    }
    if vminvq_u16(vceqq_u16(f, full)) == 0xFFFF {
      continue;
    }
    // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes) at the pointer.
    #[allow(unsafe_code)]
    let d4 = unsafe { vld4_u8(dpx.as_ptr().cast::<u8>()) };
    let out = uint8x8x4_t(
      vmovn_u16(div255_round(vmulq_u16(vmovl_u8(d4.0), f))),
      vmovn_u16(div255_round(vmulq_u16(vmovl_u8(d4.1), f))),
      vmovn_u16(div255_round(vmulq_u16(vmovl_u8(d4.2), f))),
      vmovn_u16(div255_round(vmulq_u16(vmovl_u8(d4.3), f))),
    );
    // SAFETY: same 32-byte span as the load above.
    #[allow(unsafe_code)]
    unsafe {
      vst4_u8(dpx.as_mut_ptr().cast::<u8>(), out)
    };
  }
}

/// Gathers 4 LUT words for lanes `idx` into `out` (len 4). Sentinel
/// (u32::MAX from a non-finite lane) misses → transparent 0, exactly as
/// the `*_lut_fill_neon` kernels do.
#[inline]
#[target_feature(enable = "neon")]
fn gather4(lut: &[u32], idx: uint32x4_t, out: &mut [u32]) {
  let i0 = vgetq_lane_u32::<0>(idx) as usize;
  let i1 = vgetq_lane_u32::<1>(idx) as usize;
  let i2 = vgetq_lane_u32::<2>(idx) as usize;
  let i3 = vgetq_lane_u32::<3>(idx) as usize;
  if let [o0, o1, o2, o3] = out {
    *o0 = lut.get(i0).copied().unwrap_or(0);
    *o1 = lut.get(i1).copied().unwrap_or(0);
    *o2 = lut.get(i2).copied().unwrap_or(0);
    *o3 = lut.get(i3).copied().unwrap_or(0);
  }
}

/// Source-over of 8 premultiplied `src` words over `dst` (32 bytes each),
/// k=255. This is exactly [`composite_over_neon`]'s per-8-chunk body with
/// `k = 255`: `div255_round(chan*255) == chan`, so the source channels
/// pass through unscaled and the blended bytes are identical.
#[inline]
#[target_feature(enable = "neon")]
fn blend8_over_k255(dpx: &mut [u32], src: &[u32; 8]) {
  // SAFETY: dpx is a chunks_exact_mut(8) slice (32 bytes); src is 8
  // u32 (32 bytes). vld4_u8/vst4_u8 read/write exactly those spans.
  #[allow(unsafe_code)]
  let (d4, s4) = unsafe { (vld4_u8(dpx.as_ptr().cast::<u8>()), vld4_u8(src.as_ptr().cast::<u8>())) };
  let s_b = vmovl_u8(s4.0);
  let s_g = vmovl_u8(s4.1);
  let s_r = vmovl_u8(s4.2);
  let s_a = vmovl_u8(s4.3);
  let inv = vsubq_u16(vdupq_n_u16(255), s_a);
  let out = uint8x8x4_t(
    vmovn_u16(over(vmovl_u8(d4.0), s_b, inv)),
    vmovn_u16(over(vmovl_u8(d4.1), s_g, inv)),
    vmovn_u16(over(vmovl_u8(d4.2), s_r, inv)),
    vmovn_u16(over(vmovl_u8(d4.3), s_a, inv)),
  );
  // SAFETY: same 32-byte span as the load above.
  #[allow(unsafe_code)]
  unsafe {
    vst4_u8(dpx.as_mut_ptr().cast::<u8>(), out)
  };
}

/// FUSED linear generate+blend, 8 dst pixels per iteration: two 4-lane
/// LUT index computations (identical to [`linear_lut_fill_neon`]) fill a
/// 32-byte stack buffer, then [`blend8_over_k255`] source-overs it. The
/// buffer never reaches DRAM — that eliminated round-trip is the win.
#[target_feature(enable = "neon")]
pub(super) fn linear_lut_over_neon(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array; `x_start + lane` are
  // exact integer columns (< 2^24), matching the scalar `(X as f32)`.
  #[allow(unsafe_code)]
  let mut kf = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let t0v = vdupq_n_f32(row_base);
  let dtv = vdupq_n_f32(dt);
  let four = vdupq_n_f32(4.0);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let scalev = vdupq_n_f32(scale);
  let inf = vdupq_n_f32(f32::INFINITY);
  let sentinel = vdupq_n_u32(u32::MAX);
  let mut src = [0u32; 8];
  for dpx in dst.chunks_exact_mut(8) {
    for h in 0..2 {
      let t = vaddq_f32(t0v, vmulq_f32(kf, dtv));
      let tc = vminq_f32(vmaxq_f32(t, zero), one);
      let idx = vcvtq_u32_f32(vaddq_f32(vmulq_f32(tc, scalev), half));
      let finite = vcltq_f32(vabsq_f32(t), inf);
      let idx = vbslq_u32(finite, idx, sentinel);
      if let Some(out) = src.get_mut(h * 4..h * 4 + 4) {
        gather4(lut, idx, out);
      }
      kf = vaddq_f32(kf, four);
    }
    blend8_over_k255(dpx, &src);
  }
}

/// FUSED radial generate+blend; LUT sampling identical to
/// [`radial_lut_fill_neon`], blend via [`blend8_over_k255`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) fn radial_lut_over_neon(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array; `x_start + lane` are
  // exact integer columns (< 2^24).
  #[allow(unsafe_code)]
  let mut kf = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let dav = vdupq_n_f32(da);
  let dbv = vdupq_n_f32(db);
  let ddx0v = vdupq_n_f32(dd0x);
  let ddy0v = vdupq_n_f32(dd0y);
  let four = vdupq_n_f32(4.0);
  let inv_rv = vdupq_n_f32(inv_r);
  let scalev = vdupq_n_f32(scale);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let inf = vdupq_n_f32(f32::INFINITY);
  let sentinel = vdupq_n_u32(u32::MAX);
  let mut src = [0u32; 8];
  for dpx in dst.chunks_exact_mut(8) {
    for h in 0..2 {
      let ddx = vaddq_f32(ddx0v, vmulq_f32(kf, dav));
      let ddy = vaddq_f32(ddy0v, vmulq_f32(kf, dbv));
      let gg = vaddq_f32(vmulq_f32(ddx, ddx), vmulq_f32(ddy, ddy));
      let t = vmulq_f32(vsqrtq_f32(gg), inv_rv);
      let tc = vminq_f32(vmaxq_f32(t, zero), one);
      let idx = vcvtq_u32_f32(vaddq_f32(vmulq_f32(tc, scalev), half));
      let finite = vcltq_f32(vabsq_f32(t), inf);
      let idx = vbslq_u32(finite, idx, sentinel);
      if let Some(out) = src.get_mut(h * 4..h * 4 + 4) {
        gather4(lut, idx, out);
      }
      kf = vaddq_f32(kf, four);
    }
    blend8_over_k255(dpx, &src);
  }
}

/// FUSED focal (highlight) generate+blend; LUT sampling identical to
/// [`focal_lut_fill_neon`], blend via [`blend8_over_k255`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) fn focal_lut_over_neon(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let lanes = [0.0f32, 1.0, 2.0, 3.0];
  // SAFETY: `lanes` is a 16-byte readable array; `x_start + lane` are
  // exact integer columns (< 2^24).
  #[allow(unsafe_code)]
  let mut kf = unsafe { vaddq_f32(vdupq_n_f32(x_start), vld1q_f32(lanes.as_ptr())) };
  let (b0, db, d0, d1, d2) = super::focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  let (b0v, dbv) = (vdupq_n_f32(b0), vdupq_n_f32(db));
  let (d0v, d1v, d2v) = (vdupq_n_f32(d0), vdupq_n_f32(d1), vdupq_n_f32(d2));
  let inv2av = vdupq_n_f32(inv2a);
  let rv = vdupq_n_f32(r);
  let four = vdupq_n_f32(4.0);
  let half = vdupq_n_f32(0.5);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let inf = vdupq_n_f32(f32::INFINITY);
  let scalev = vdupq_n_f32(scale);
  let sentinel = vdupq_n_u32(u32::MAX);
  let mut src = [0u32; 8];
  for dpx in dst.chunks_exact_mut(8) {
    for h in 0..2 {
      let b = vaddq_f32(b0v, vmulq_f32(kf, dbv));
      let det = vaddq_f32(d0v, vmulq_f32(kf, vaddq_f32(d1v, vmulq_f32(kf, d2v))));
      let sq = vsqrtq_f32(det);
      let nb = vnegq_f32(b);
      let root = vmaxq_f32(vmulq_f32(vsubq_f32(nb, sq), inv2av), vmulq_f32(vaddq_f32(nb, sq), inv2av));
      let valid = vandq_u32(vandq_u32(vcgeq_f32(det, zero), vcgeq_f32(vmulq_f32(rv, root), zero)), vcltq_f32(vabsq_f32(root), inf));
      let tc = vminq_f32(vmaxq_f32(root, zero), one);
      let idx = vcvtq_u32_f32(vaddq_f32(vmulq_f32(tc, scalev), half));
      let idx = vbslq_u32(valid, idx, sentinel);
      if let Some(out) = src.get_mut(h * 4..h * 4 + 4) {
        gather4(lut, idx, out);
      }
      kf = vaddq_f32(kf, four);
    }
    blend8_over_k255(dpx, &src);
  }
}
