//! Vector span kernels with tightly scoped unsafe implementations:
//! blits — NEON on aarch64, simd128 on wasm32 — each with a scalar
//! implementation that is the bit-exact oracle. Tests compare the two on
//! randomized inputs, and every other build (x86_64 emulators, 32-bit arm
//! without the feature, wasm without `-Ctarget-feature=+simd128`) uses the
//! scalar path unconditionally.
//!
//! Pixel format everywhere: premultiplied RGBA8 words (0xAABBGGRR), which
//! in little-endian memory is byte order [R, G, B, A] — exactly what
//! `vld4_u8` de-interleaves into planes. The wasm kernels skip the
//! de-interleave: the blend math is per-channel independent, so they widen
//! the interleaved bytes to u16 lanes in place and swizzle per-pixel
//! factors (coverage, alpha) across each pixel's four channel lanes.
//!
//! Source/coverage scaling uses rounded `/255`. Destination attenuation
//! follows the byte blend used by the reference renderers:
//! `dst * (256 - src_alpha) >> 8`.
//!
//! Spans shorter than [`SIMD_MIN_SPAN`] stay scalar: measured lesson from
//! the bench project — tiny emoji spans lose to vector setup cost.

/// Minimum span length for the vector path.
const SIMD_MIN_SPAN: usize = 16;

/// Cached (once per process) runtime AVX2 availability. CPUs without AVX2
/// fall back to the SSE2 kernels; every x86_64 core has at least SSE2.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn use_avx2() -> bool {
  static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *CACHE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// `is_x86_feature_detected!` is a `std` macro, so `no_std` reads CPUID
/// directly. See [`cpuid`] — a `std` test pins the two to the same answer.
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
fn use_avx2() -> bool {
  cpuid::avx2()
}

/// Cached (once per process) runtime AVX-512 availability. Must match the full
/// `#[target_feature(enable = ...)]` set the AVX-512 kernels declare — `avx2`,
/// `avx512f`, `avx512bw`, `avx512dq`, `avx512vl` — so every dispatched
/// function is safe on the running CPU; otherwise the AVX2 kernels run.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn use_avx512() -> bool {
  static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  *CACHE.get_or_init(|| {
    std::is_x86_feature_detected!("avx2")
      && std::is_x86_feature_detected!("avx512f")
      && std::is_x86_feature_detected!("avx512bw")
      && std::is_x86_feature_detected!("avx512dq")
      && std::is_x86_feature_detected!("avx512vl")
  })
}

/// See [`use_avx2`]: CPUID directly when there is no `std` to ask.
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
fn use_avx512() -> bool {
  cpuid::avx512()
}
/// Coverage-modulated solid source-over: for each pixel,
/// `ca = (cov*sa+127)/255`, source channels scaled by `ca`, then
/// premultiplied source-over into `dst`. `sr/sg/sb/sa` are 0..=255.
/// Mirrors (and on NEON must match bit-for-bit) the scalar loop.
pub(crate) fn fill_span_solid(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32, large_canvas: bool) {
  // Opaque source: full-coverage pixels are EXACTLY the source color
  // (ca=255 -> s_x=x, inv=0 -> o=s), so interior runs become plain stores
  // — large fills are memory-bound and interiors dominate at 320/720px.
  if sa == 255 {
    let color = crate::pixel::pack_premultiplied_rgba(sr, sg, sb, 255);
    #[cfg(target_arch = "aarch64")]
    if large_canvas && dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(cov.len());
      let full = n - n % 8;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (cov_v, cov_tail) = cov.split_at(full);
      // SAFETY: NEON is mandatory on aarch64. The kernel stores opaque
      // 8-pixel interiors directly and falls back to the exact general
      // blend only for chunks containing AA edges.
      #[allow(unsafe_code)]
      unsafe {
        neon::fill_span_opaque_neon(dst_v, cov_v, color, sr, sg, sb)
      };
      fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
      return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if large_canvas && dst.len() >= SIMD_MIN_SPAN && use_avx2() {
      if use_avx512() {
        let n = dst.len().min(cov.len());
        let full = n - n % 16;
        let (dst_v, dst_tail) = dst.split_at_mut(full);
        let (cov_v, cov_tail) = cov.split_at(full);
        // SAFETY: `use_avx512()` gates on avx2/avx512f/avx512bw/avx512dq/avx512vl.
        #[allow(unsafe_code)]
        unsafe {
          avx512::fill_span_opaque_avx512(dst_v, cov_v, color)
        }
        fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
        return;
      }
      let n = dst.len().min(cov.len());
      let full = n - n % 8;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (cov_v, cov_tail) = cov.split_at(full);
      // SAFETY: `use_avx2()` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::fill_span_opaque_avx2(dst_v, cov_v, color)
      }
      fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
      return;
    }
    let n = dst.len().min(cov.len());
    let mut i = 0usize;
    while i < n {
      if cov.get(i).copied() == Some(255) {
        let mut j = i + 1;
        while j < n && cov.get(j).copied() == Some(255) {
          j += 1;
        }
        if let Some(run) = dst.get_mut(i..j) {
          run.fill(color);
        }
        i = j;
      } else {
        // AA edge pixel: identical scalar formulas as the slow path.
        if let (Some(d), Some(&c)) = (dst.get_mut(i), cov.get(i)) {
          fill_span_solid_scalar(core::slice::from_mut(d), &[c], sr, sg, sb, sa);
        }
        i += 1;
      }
    }
    return;
  }
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(cov.len());
      let full = n - n % 8;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (cov_v, cov_tail) = cov.split_at(full.min(cov.len()));
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64
      // (ARMv8.0 baseline); the feature is always present.
      #[allow(unsafe_code)]
      unsafe {
        neon::fill_span_solid_neon(dst_v, cov_v, sr, sg, sb, sa)
      };
      fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(cov.len());
      let full = n - n % 4;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (cov_v, cov_tail) = cov.split_at(full.min(cov.len()));
      wasm128::fill_span_solid_wasm(dst_v, cov_v, sr, sg, sb, sa);
      fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(cov.len());
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = n - n % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (cov_v, cov_tail) = cov.split_at(full.min(cov.len()));
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::fill_span_solid_avx512(dst_v, cov_v, sr, sg, sb, sa)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::fill_span_solid_avx2(dst_v, cov_v, sr, sg, sb, sa)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline and the cfg above
        // additionally requires it to be statically enabled.
        #[allow(unsafe_code)]
        unsafe {
          sse2::fill_span_solid_sse2(dst_v, cov_v, sr, sg, sb, sa)
        };
      }
      fill_span_solid_scalar(dst_tail, cov_tail, sr, sg, sb, sa);
      return;
    }
  }
  fill_span_solid_scalar(dst, cov, sr, sg, sb, sa);
}

fn fill_span_solid_scalar(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
  for (dst, &cov) in dst.iter_mut().zip(cov.iter()) {
    if cov == 0 {
      continue;
    }
    let ca = (u32::from(cov) * sa + 127) / 255;
    if ca == 0 {
      continue;
    }
    let s_a = ca;
    let s_r = (sr * ca + 127) / 255;
    let s_g = (sg * ca + 127) / 255;
    let s_b = (sb * ca + 127) / 255;
    let d = *dst;
    let inv = 256 - s_a;
    let o_a = s_a + (((d >> 24) & 0xff) * inv >> 8);
    let o_r = s_r + ((d & 0xff) * inv >> 8);
    let o_g = s_g + (((d >> 8) & 0xff) * inv >> 8);
    let o_b = s_b + (((d >> 16) & 0xff) * inv >> 8);
    *dst = crate::pixel::pack_premultiplied_rgba(o_r.min(255), o_g.min(255), o_b.min(255), o_a.min(255));
  }
}

/// Solid source-over of a UNIFORM-coverage span (mode-S rasterizer output:
/// every pixel shares one coverage byte). Bit-exact with
/// [`fill_span_solid`] over a constant coverage row: the source pixel and
/// blend factor are computed ONCE; interiors (cov 255, sa 255) collapse to
/// a plain store.
pub(crate) fn fill_span_uniform_opaque(dst: &mut [u32], cov: u8, sr: u32, sg: u32, sb: u32) {
  if cov == 0 {
    return;
  }
  if cov == 255 {
    dst.fill(crate::pixel::pack_premultiplied_rgba(sr, sg, sb, 255));
    return;
  }
  let ca = u32::from(cov);
  let s_r = (sr * ca + 127) / 255;
  let s_g = (sg * ca + 127) / 255;
  let s_b = (sb * ca + 127) / 255;
  if let [dst] = dst {
    fill_one_uniform(dst, ca, s_r, s_g, s_b);
    return;
  }
  fill_span_uniform_scalar(dst, ca, s_r, s_g, s_b);
}

pub(crate) fn fill_span_uniform(dst: &mut [u32], cov: u8, sr: u32, sg: u32, sb: u32, sa: u32) {
  if cov == 0 {
    return;
  }
  let ca = (u32::from(cov) * sa + 127) / 255;
  if ca == 0 {
    return;
  }
  let s_r = (sr * ca + 127) / 255;
  let s_g = (sg * ca + 127) / 255;
  let s_b = (sb * ca + 127) / 255;
  if ca == 255 {
    // Fully opaque source: over() degenerates to the source itself.
    dst.fill(crate::pixel::pack_premultiplied_rgba(s_r, s_g, s_b, 255));
    return;
  }
  if let [dst] = dst {
    fill_one_uniform(dst, ca, s_r, s_g, s_b);
    return;
  }
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 8;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::fill_span_uniform_neon(dst_v, ca, s_r, s_g, s_b)
      };
      fill_span_uniform_scalar(dst_tail, ca, s_r, s_g, s_b);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 4;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      wasm128::fill_span_uniform_wasm(dst_v, ca, s_r, s_g, s_b);
      fill_span_uniform_scalar(dst_tail, ca, s_r, s_g, s_b);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = dst.len() - dst.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::fill_span_uniform_avx512(dst_v, ca, s_r, s_g, s_b)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::fill_span_uniform_avx2(dst_v, ca, s_r, s_g, s_b)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::fill_span_uniform_sse2(dst_v, ca, s_r, s_g, s_b)
        };
      }
      fill_span_uniform_scalar(dst_tail, ca, s_r, s_g, s_b);
      return;
    }
  }
  fill_span_uniform_scalar(dst, ca, s_r, s_g, s_b);
}

#[inline]
fn fill_one_uniform(dst: &mut u32, ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let value = *dst;
  let inv = 256 - ca;
  let o_a = ca + (((value >> 24) & 0xff) * inv >> 8);
  let o_r = s_r + ((value & 0xff) * inv >> 8);
  let o_g = s_g + (((value >> 8) & 0xff) * inv >> 8);
  let o_b = s_b + (((value >> 16) & 0xff) * inv >> 8);
  *dst = crate::pixel::pack_premultiplied_rgba(o_r.min(255), o_g.min(255), o_b.min(255), o_a.min(255));
}

fn fill_span_uniform_scalar(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
  let inv = 256 - ca;
  for dst in dst.iter_mut() {
    let d = *dst;
    let o_a = ca + (((d >> 24) & 0xff) * inv >> 8);
    let o_r = s_r + ((d & 0xff) * inv >> 8);
    let o_g = s_g + (((d >> 8) & 0xff) * inv >> 8);
    let o_b = s_b + (((d >> 16) & 0xff) * inv >> 8);
    *dst = crate::pixel::pack_premultiplied_rgba(o_r.min(255), o_g.min(255), o_b.min(255), o_a.min(255));
  }
}

/// Fills `out` with LUT colors for a FULL-COVERAGE linear gradient run,
/// sampled like `lut_sample`. Positions are SEGMENTATION-INVARIANT: each
/// pixel evaluates `t(X) = row_base + X·dt` as one rounded expression from
/// the absolute device column `X` (`x_start` is the absolute column of
/// `out[0]`), so the same pixel yields identical bits regardless of the
/// span/row/sub-run it is reached through. Same `base + X·step` exactness
/// protocol as the radial/focal kernels.
pub(crate) fn linear_lut_fill(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::linear_lut_fill_neon(head, lut, row_base, dt, x_start, scale)
      };
      linear_lut_fill_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      wasm128::linear_lut_fill_wasm(head, lut, row_base, dt, x_start, scale);
      linear_lut_fill_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = out.len() - out.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = out.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::linear_lut_fill_avx512(head, lut, row_base, dt, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::linear_lut_fill_avx2(head, lut, row_base, dt, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::linear_lut_fill_sse2(head, lut, row_base, dt, x_start, scale)
        };
      }
      linear_lut_fill_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  linear_lut_fill_scalar(out, lut, row_base, dt, x_start, scale);
}

fn linear_lut_fill_scalar(out: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  for (k, s) in out.iter_mut().enumerate() {
    let t = row_base + (x_start + k as f32) * dt;
    *s = if t.is_finite() {
      let idx = (t.clamp(0.0, 1.0) * scale + 0.5) as usize;
      lut.get(idx).copied().unwrap_or(0)
    } else {
      0
    };
  }
}

/// Fills `out` with LUT colors for a FULL-COVERAGE radial gradient run:
/// `t(X) = sqrt((dd0x + X·da)² + (dd0y + X·db)²) · inv_r`, sampled at
/// `lut[(clamp(t,0,1)·(len−1) + 0.5) as usize]`; non-finite t → 0
/// (transparent), matching `lut_sample`.
///
/// Positions are SEGMENTATION-INVARIANT: `dd0x`/`dd0y` are the row-origin
/// (device column 0) deltas and `X = x_start + lane` is the absolute device
/// column of each pixel (`x_start` = absolute column of `out[0]`), so each
/// pixel's `dd0 + X·d` is one rounded expression independent of which
/// span/row the run came from. NOT bit-exact with the historical
/// sequential-accumulation loop (float association differs), same protocol
/// as the round-2 gradient restructure: corpus-gated, not byte-gated. The
/// scalar path here uses the SAME `dd0 + X·d` form so NEON and scalar
/// builds of this function agree with each other lane-for-lane.
pub(crate) fn radial_lut_fill(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::radial_lut_fill_neon(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
      };
      radial_lut_fill_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      wasm128::radial_lut_fill_wasm(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
      radial_lut_fill_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = out.len() - out.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = out.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::radial_lut_fill_avx512(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::radial_lut_fill_avx2(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::radial_lut_fill_sse2(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      }
      radial_lut_fill_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  radial_lut_fill_scalar(out, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
}

#[allow(clippy::too_many_arguments)]
fn radial_lut_fill_scalar(out: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  for (k, s) in out.iter_mut().enumerate() {
    let xf = x_start + k as f32;
    let ddx = dd0x + xf * da;
    let ddy = dd0y + xf * db;
    let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
    *s = if t.is_finite() {
      let idx = (t.clamp(0.0, 1.0) * scale + 0.5) as usize;
      lut.get(idx).copied().unwrap_or(0)
    } else {
      0
    };
  }
}

/// Fills `out` with LUT colors for a FULL-COVERAGE focal (highlight)
/// radial run — rlottie's fetch_radial_gradient quadratic: solve
/// `a·s² + b·s − |g|² = 0` with `b = 2(g·d)`, take the larger root; no
/// real root, `r·s < 0`, or non-finite → transparent. Positions are
/// SEGMENTATION-INVARIANT: `g0x`/`g0y` are the row-origin (column 0)
/// deltas and each pixel uses `g0 + X·step` at its absolute device column
/// `X = x_start + lane` (see radial_lut_fill's note).
#[allow(clippy::too_many_arguments)]
pub(crate) fn focal_lut_fill(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::focal_lut_fill_neon(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
      };
      focal_lut_fill_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let full = out.len() - out.len() % 4;
      let (head, tail) = out.split_at_mut(full);
      wasm128::focal_lut_fill_wasm(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
      focal_lut_fill_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if out.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = out.len() - out.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = out.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::focal_lut_fill_avx512(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::focal_lut_fill_avx2(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::focal_lut_fill_sse2(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      }
      focal_lut_fill_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  focal_lut_fill_scalar(out, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
}

/// Absolute-column quadratic coefficients for a focal gradient row.
///
/// With `g(X) = g0 + X*s`, the quadratic solver uses
/// `B(X) = 2*g(X).d` and `D(X) = B(X)^2 + 4*a*|g(X)|^2`. Both are
/// polynomials in the absolute device column `X`:
///
/// `B(X) = b0 + X*db`, `D(X) = d0 + X*(d1 + X*d2)`.
///
/// Keeping the evaluation anchored to absolute X makes results independent
/// of how a coverage row is split into spans. Deliberately use separate
/// multiply/add expressions throughout; changing these to `mul_add` changes
/// the rounding protocol shared by the scalar and SIMD implementations.
#[inline]
#[allow(clippy::too_many_arguments)]
fn focal_row_coefficients(g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32) -> (f32, f32, f32, f32, f32) {
  let b0 = 2.0 * (g0x * dx + g0y * dy);
  let db = 2.0 * (sa * dx + sb * dy);
  let four_a = 4.0 * a;
  let d0 = b0 * b0 + four_a * (g0x * g0x + g0y * g0y);
  let d1 = 2.0 * b0 * db + (8.0 * a) * (g0x * sa + g0y * sb);
  let d2 = db * db + four_a * (sa * sa + sb * sb);
  (b0, db, d0, d1, d2)
}

#[allow(clippy::too_many_arguments)]
fn focal_lut_fill_scalar(out: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let (b0, db, d0, d1, d2) = focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  for (k, s) in out.iter_mut().enumerate() {
    let xf = x_start + k as f32;
    let b = b0 + xf * db;
    let det = d0 + xf * (d1 + xf * d2);
    *s = 0;
    if det >= 0.0 {
      let sq = det.sqrt();
      let root = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
      if r * root >= 0.0 && root.is_finite() {
        let idx = (root.clamp(0.0, 1.0) * scale + 0.5) as usize;
        *s = lut.get(idx).copied().unwrap_or(0);
      }
    }
  }
}

/// Composites a premultiplied `src` plane over `dst`, with `src` alpha and
/// color additionally scaled by `k` (0..=255). Used for offscreen layer
/// composition (`composite_over`).
pub(crate) fn composite_over_span(dst: &mut [u32], src: &[u32], k: u32) {
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(src.len());
      let full = n - n % 8;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (src_v, src_tail) = src.split_at(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::composite_over_neon(dst_v, src_v, k)
      };
      composite_over_scalar(dst_tail, src_tail, k);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(src.len());
      let full = n - n % 4;
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (src_v, src_tail) = src.split_at(full);
      wasm128::composite_over_wasm(dst_v, src_v, k);
      composite_over_scalar(dst_tail, src_tail, k);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let n = dst.len().min(src.len());
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = n - n % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (dst_v, dst_tail) = dst.split_at_mut(full);
      let (src_v, src_tail) = src.split_at(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::composite_over_avx512(dst_v, src_v, k)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::composite_over_avx2(dst_v, src_v, k)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::composite_over_sse2(dst_v, src_v, k)
        };
      }
      composite_over_scalar(dst_tail, src_tail, k);
      return;
    }
  }
  composite_over_scalar(dst, src, k);
}

fn composite_over_scalar(dst: &mut [u32], src: &[u32], k: u32) {
  for (d, &s) in dst.iter_mut().zip(src.iter()) {
    if s == 0 {
      continue;
    }
    let sa = (((s >> 24) & 0xff) * k + 127) / 255;
    if sa == 0 && s & 0x00ff_ffff == 0 {
      continue;
    }
    let sr = (((s >> 16) & 0xff) * k + 127) / 255;
    let sg = (((s >> 8) & 0xff) * k + 127) / 255;
    let sb = ((s & 0xff) * k + 127) / 255;
    let p = *d;
    let inv = 256 - sa;
    let a = sa + (((p >> 24) & 0xff) * inv >> 8);
    let r = sr + (((p >> 16) & 0xff) * inv >> 8);
    let g = sg + (((p >> 8) & 0xff) * inv >> 8);
    let b = sb + ((p & 0xff) * inv >> 8);
    *d = (a.min(255) << 24) | (r.min(255) << 16) | (g.min(255) << 8) | b.min(255);
  }
}

/// Source-over of ONE premultiplied `src` word over `dst`, k=255 — the
/// per-pixel body of [`composite_over_scalar`] with `k = 255` (which is
/// exactly identity on the source channels, since `(x*255+127)/255 == x`).
/// The fused `*_lut_over` scalar paths use this so their output is
/// bit-for-bit the two-pass `*_lut_fill` + `composite_over_span(255)`.
#[inline]
fn over_px_k255(d: &mut u32, s: u32) {
  if s == 0 {
    return;
  }
  let sa = (s >> 24) & 0xff;
  let sr = (s >> 16) & 0xff;
  let sg = (s >> 8) & 0xff;
  let sb = s & 0xff;
  let p = *d;
  let inv = 256 - sa;
  let a = sa + (((p >> 24) & 0xff) * inv >> 8);
  let r = sr + (((p >> 16) & 0xff) * inv >> 8);
  let g = sg + (((p >> 8) & 0xff) * inv >> 8);
  let b = sb + ((p & 0xff) * inv >> 8);
  *d = (a.min(255) << 24) | (r.min(255) << 16) | (g.min(255) << 8) | b.min(255);
}

/// FUSED linear gradient generate+blend: for each pixel computes the LUT
/// color exactly like [`linear_lut_fill`] and immediately source-overs it
/// into `dst` (k=255), in a single pass with no materialized src plane.
/// Bit-for-bit identical to `linear_lut_fill(buf, ..)` followed by
/// `composite_over_span(dst, buf, 255)` — both stages are per-pixel
/// deterministic on the same inputs, so fusing them preserves every byte.
pub(crate) fn linear_lut_over(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 8;
      let (head, tail) = dst.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::linear_lut_over_neon(head, lut, row_base, dt, x_start, scale)
      };
      linear_lut_over_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 4;
      let (head, tail) = dst.split_at_mut(full);
      wasm128::linear_lut_over_wasm(head, lut, row_base, dt, x_start, scale);
      linear_lut_over_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = dst.len() - dst.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = dst.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::linear_lut_over_avx512(head, lut, row_base, dt, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::linear_lut_over_avx2(head, lut, row_base, dt, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::linear_lut_over_sse2(head, lut, row_base, dt, x_start, scale)
        };
      }
      linear_lut_over_scalar(tail, lut, row_base, dt, x_start + full as f32, scale);
      return;
    }
  }
  linear_lut_over_scalar(dst, lut, row_base, dt, x_start, scale);
}

fn linear_lut_over_scalar(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32, scale: f32) {
  for (k, d) in dst.iter_mut().enumerate() {
    let t = row_base + (x_start + k as f32) * dt;
    let s = if t.is_finite() {
      let idx = (t.clamp(0.0, 1.0) * scale + 0.5) as usize;
      lut.get(idx).copied().unwrap_or(0)
    } else {
      0
    };
    over_px_k255(d, s);
  }
}

/// FUSED radial gradient generate+blend; see [`linear_lut_over`]. LUT
/// sampling mirrors [`radial_lut_fill`] exactly, blend is k=255 source-over.
#[allow(clippy::too_many_arguments)]
pub(crate) fn radial_lut_over(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 8;
      let (head, tail) = dst.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::radial_lut_over_neon(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
      };
      radial_lut_over_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 4;
      let (head, tail) = dst.split_at_mut(full);
      wasm128::radial_lut_over_wasm(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
      radial_lut_over_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = dst.len() - dst.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = dst.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::radial_lut_over_avx512(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::radial_lut_over_avx2(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::radial_lut_over_sse2(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale)
        };
      }
      radial_lut_over_scalar(tail, lut, dd0x, dd0y, da, db, inv_r, x_start + full as f32, scale);
      return;
    }
  }
  radial_lut_over_scalar(dst, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
}

#[allow(clippy::too_many_arguments)]
fn radial_lut_over_scalar(dst: &mut [u32], lut: &[u32], dd0x: f32, dd0y: f32, da: f32, db: f32, inv_r: f32, x_start: f32, scale: f32) {
  for (k, d) in dst.iter_mut().enumerate() {
    let xf = x_start + k as f32;
    let ddx = dd0x + xf * da;
    let ddy = dd0y + xf * db;
    let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
    let s = if t.is_finite() {
      let idx = (t.clamp(0.0, 1.0) * scale + 0.5) as usize;
      lut.get(idx).copied().unwrap_or(0)
    } else {
      0
    };
    over_px_k255(d, s);
  }
}

/// FUSED focal (highlight) radial gradient generate+blend; see
/// [`linear_lut_over`]. LUT sampling mirrors [`focal_lut_fill`] exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn focal_lut_over(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32) {
  let scale = (lut.len().saturating_sub(1)) as f32;
  #[cfg(target_arch = "aarch64")]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 8;
      let (head, tail) = dst.split_at_mut(full);
      // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
      #[allow(unsafe_code)]
      unsafe {
        neon::focal_lut_over_neon(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
      };
      focal_lut_over_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let full = dst.len() - dst.len() % 4;
      let (head, tail) = dst.split_at_mut(full);
      wasm128::focal_lut_over_wasm(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
      focal_lut_over_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  {
    if dst.len() >= SIMD_MIN_SPAN {
      let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
      let full = dst.len() - dst.len() % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
      let (head, tail) = dst.split_at_mut(full);
      if use_avx512 {
        // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
        #[allow(unsafe_code)]
        unsafe {
          avx512::focal_lut_over_avx512(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      } else if use_avx2 {
        // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
        #[allow(unsafe_code)]
        unsafe {
          avx2::focal_lut_over_avx2(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      } else {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        #[allow(unsafe_code)]
        unsafe {
          sse2::focal_lut_over_sse2(head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale)
        };
      }
      focal_lut_over_scalar(tail, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start + full as f32, scale);
      return;
    }
  }
  focal_lut_over_scalar(dst, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale);
}

#[allow(clippy::too_many_arguments)]
fn focal_lut_over_scalar(dst: &mut [u32], lut: &[u32], g0x: f32, g0y: f32, sa: f32, sb: f32, dx: f32, dy: f32, a: f32, inv2a: f32, r: f32, x_start: f32, scale: f32) {
  let (b0, db, d0, d1, d2) = focal_row_coefficients(g0x, g0y, sa, sb, dx, dy, a);
  for (k, d) in dst.iter_mut().enumerate() {
    let xf = x_start + k as f32;
    let b = b0 + xf * db;
    let det = d0 + xf * (d1 + xf * d2);
    let mut s = 0u32;
    if det >= 0.0 {
      let sq = det.sqrt();
      let root = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
      if r * root >= 0.0 && root.is_finite() {
        let idx = (root.clamp(0.0, 1.0) * scale + 0.5) as usize;
        s = lut.get(idx).copied().unwrap_or(0);
      }
    }
    over_px_k255(d, s);
  }
}

// Alpha8 span kernels. These intentionally use the same rounding protocol as
// the scalar backend: source scaling is rounded /255, while source-over uses
// dst * (256 - source) >> 8. Long spans process 16 pixels per iteration on
// both NEON and wasm simd128; short spans retain the cheaper scalar loops.

#[inline]
fn alpha_over(dst: u8, source: u32) -> u8 {
  (source + ((u32::from(dst) * (256 - source)) >> 8)).min(255) as u8
}

pub(crate) fn alpha_blend_solid(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  if alpha == 0 {
    return;
  }
  let n = dst.len().min(coverage.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_blend_solid_neon(head, &coverage[..full], alpha)
    };
    alpha_blend_solid_scalar(tail, &coverage[full..n], alpha);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_blend_solid_wasm(head, &coverage[..full], alpha);
    alpha_blend_solid_scalar(tail, &coverage[full..n], alpha);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_blend_solid_avx512(head, &coverage[..full], alpha)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_blend_solid_avx2(head, &coverage[..full], alpha)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_blend_solid_sse2(head, &coverage[..full], alpha)
      };
    }
    alpha_blend_solid_scalar(tail, &coverage[full..n], alpha);
    return;
  }
  alpha_blend_solid_scalar(&mut dst[..n], &coverage[..n], alpha);
}

fn alpha_blend_solid_scalar(dst: &mut [u8], coverage: &[u8], alpha: u8) {
  for (dst, &coverage) in dst.iter_mut().zip(coverage) {
    let source = (u32::from(alpha) * u32::from(coverage) + 127) / 255;
    *dst = alpha_over(*dst, source);
  }
}

pub(crate) fn alpha_blend_product(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  let n = dst.len().min(lhs.len()).min(rhs.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_blend_product_neon(head, &lhs[..full], &rhs[..full])
    };
    alpha_blend_product_scalar(tail, &lhs[full..n], &rhs[full..n]);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_blend_product_wasm(head, &lhs[..full], &rhs[..full]);
    alpha_blend_product_scalar(tail, &lhs[full..n], &rhs[full..n]);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_blend_product_avx512(head, &lhs[..full], &rhs[..full])
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_blend_product_avx2(head, &lhs[..full], &rhs[..full])
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_blend_product_sse2(head, &lhs[..full], &rhs[..full])
      };
    }
    alpha_blend_product_scalar(tail, &lhs[full..n], &rhs[full..n]);
    return;
  }
  alpha_blend_product_scalar(&mut dst[..n], &lhs[..n], &rhs[..n]);
}

fn alpha_blend_product_scalar(dst: &mut [u8], lhs: &[u8], rhs: &[u8]) {
  for ((dst, &lhs), &rhs) in dst.iter_mut().zip(lhs).zip(rhs) {
    let source = (u32::from(lhs) * u32::from(rhs) + 127) / 255;
    *dst = alpha_over(*dst, source);
  }
}

pub(crate) fn alpha_blend_uniform(dst: &mut [u8], coverage: u8, alpha: u8) {
  let source = (u32::from(alpha) * u32::from(coverage) + 127) / 255;
  if source == 0 {
    return;
  }
  if source == 255 {
    dst.fill(255);
    return;
  }
  #[cfg(target_arch = "aarch64")]
  if dst.len() >= SIMD_MIN_SPAN {
    let full = dst.len() - dst.len() % 16;
    let (head, tail) = dst.split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_blend_uniform_neon(head, source as u8)
    };
    alpha_blend_uniform_scalar(tail, source);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if dst.len() >= SIMD_MIN_SPAN {
    let full = dst.len() - dst.len() % 16;
    let (head, tail) = dst.split_at_mut(full);
    wasm128::alpha_blend_uniform_wasm(head, source as u8);
    alpha_blend_uniform_scalar(tail, source);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if dst.len() >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = dst.len() - dst.len() % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst.split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_blend_uniform_avx512(head, source as u8)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_blend_uniform_avx2(head, source as u8)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_blend_uniform_sse2(head, source as u8)
      };
    }
    alpha_blend_uniform_scalar(tail, source);
    return;
  }
  alpha_blend_uniform_scalar(dst, source);
}

fn alpha_blend_uniform_scalar(dst: &mut [u8], source: u32) {
  for dst in dst {
    *dst = alpha_over(*dst, source);
  }
}

pub(crate) fn alpha_composite_over(dst: &mut [u8], src: &[u8], opacity: u8) {
  if opacity == 0 {
    return;
  }
  let n = dst.len().min(src.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_composite_over_neon(head, &src[..full], opacity)
    };
    alpha_composite_over_scalar(tail, &src[full..n], opacity);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_composite_over_wasm(head, &src[..full], opacity);
    alpha_composite_over_scalar(tail, &src[full..n], opacity);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_composite_over_avx512(head, &src[..full], opacity)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_composite_over_avx2(head, &src[..full], opacity)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_composite_over_sse2(head, &src[..full], opacity)
      };
    }
    alpha_composite_over_scalar(tail, &src[full..n], opacity);
    return;
  }
  alpha_composite_over_scalar(&mut dst[..n], &src[..n], opacity);
}

fn alpha_composite_over_scalar(dst: &mut [u8], src: &[u8], opacity: u8) {
  for (dst, &src) in dst.iter_mut().zip(src) {
    let source = (u32::from(src) * u32::from(opacity) + 127) / 255;
    *dst = alpha_over(*dst, source);
  }
}

pub(crate) fn alpha_multiply(dst: &mut [u8], factors: &[u8]) {
  let n = dst.len().min(factors.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_multiply_neon(head, &factors[..full])
    };
    alpha_multiply_scalar(tail, &factors[full..n]);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_multiply_wasm(head, &factors[..full]);
    alpha_multiply_scalar(tail, &factors[full..n]);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_multiply_avx512(head, &factors[..full])
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_multiply_avx2(head, &factors[..full])
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_multiply_sse2(head, &factors[..full])
      };
    }
    alpha_multiply_scalar(tail, &factors[full..n]);
    return;
  }
  alpha_multiply_scalar(&mut dst[..n], &factors[..n]);
}

fn alpha_multiply_scalar(dst: &mut [u8], factors: &[u8]) {
  for (dst, &factor) in dst.iter_mut().zip(factors) {
    *dst = ((u32::from(*dst) * u32::from(factor) + 127) / 255) as u8;
  }
}

pub(crate) fn alpha_matte(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  let n = dst.len().min(src.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_matte_neon(head, &src[..full], opacity, inverted)
    };
    alpha_matte_scalar(tail, &src[full..n], opacity, inverted);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_matte_wasm(head, &src[..full], opacity, inverted);
    alpha_matte_scalar(tail, &src[full..n], opacity, inverted);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_matte_avx512(head, &src[..full], opacity, inverted)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_matte_avx2(head, &src[..full], opacity, inverted)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_matte_sse2(head, &src[..full], opacity, inverted)
      };
    }
    alpha_matte_scalar(tail, &src[full..n], opacity, inverted);
    return;
  }
  alpha_matte_scalar(&mut dst[..n], &src[..n], opacity, inverted);
}

fn alpha_matte_scalar(dst: &mut [u8], src: &[u8], opacity: u8, inverted: bool) {
  for (dst, &src) in dst.iter_mut().zip(src) {
    let scaled = (u32::from(src) * u32::from(opacity) + 127) / 255;
    let factor = if inverted { 255 - scaled } else { scaled };
    *dst = ((u32::from(*dst) * factor + 127) / 255) as u8;
  }
}

pub(crate) fn alpha_mask_combine(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  let n = dst.len().min(src.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    #[allow(unsafe_code)]
    unsafe {
      neon::alpha_mask_combine_neon(head, &src[..full], mode, inverted, opacity)
    };
    alpha_mask_combine_scalar(tail, &src[full..n], mode, inverted, opacity);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 16;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::alpha_mask_combine_wasm(head, &src[..full], mode, inverted, opacity);
    alpha_mask_combine_scalar(tail, &src[full..n], mode, inverted, opacity);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
      let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 64 } else if use_avx2 { 32 } else { 16 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::alpha_mask_combine_avx512(head, &src[..full], mode, inverted, opacity)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::alpha_mask_combine_avx2(head, &src[..full], mode, inverted, opacity)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::alpha_mask_combine_sse2(head, &src[..full], mode, inverted, opacity)
      };
    }
    alpha_mask_combine_scalar(tail, &src[full..n], mode, inverted, opacity);
    return;
  }
  alpha_mask_combine_scalar(&mut dst[..n], &src[..n], mode, inverted, opacity);
}

fn alpha_mask_combine_scalar(dst: &mut [u8], src: &[u8], mode: u8, inverted: bool, opacity: u8) {
  for (dst, &src) in dst.iter_mut().zip(src) {
    let sample = if inverted { 255 - u32::from(src) } else { u32::from(src) };
    let contribution = (sample * u32::from(opacity) + 127) / 255;
    let old = u32::from(*dst);
    *dst = match mode {
      b's' => ((old * (255 - contribution) + 127) / 255) as u8,
      b'i' => ((old * contribution + 127) / 255) as u8,
      b'f' => old.abs_diff(contribution) as u8,
      _ => (contribution + ((255 - contribution) * old + 127) / 255) as u8,
    };
  }
}

/// Alpha matting of a `u32` RGBA surface: every channel of `dst` is scaled by
/// `(alpha * opacity + 127) / 255` (or its inverse when `inverted`), where
/// `alpha` is the source pixel's alpha. Matches `executor::apply_matte` kinds
/// 1/2 exactly: the scalar's `factor == 255` (no-op) and `factor == 0`
/// (zero) short-circuits are reproduced by the general `(ch * factor + 127) / 255`
/// rounding, so the SIMD path runs the general formula with no branches and stays
/// bit-for-bit identical. Channel-order independent (all channels share one factor).
pub(crate) fn apply_matte_alpha(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  let n = dst.len().min(src.len());
  #[cfg(target_arch = "aarch64")]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 8;
    let (head, tail) = dst[..n].split_at_mut(full);
    // SAFETY: NEON is mandatory on aarch64; the kernel processes whole 8-pixel chunks.
    #[allow(unsafe_code)]
    unsafe {
      neon::apply_matte_alpha_neon(head, &src[..full], source_opacity, inverted)
    };
    apply_matte_alpha_scalar(tail, &src[full..n], source_opacity, inverted);
    return;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  if n >= SIMD_MIN_SPAN {
    let full = n - n % 4;
    let (head, tail) = dst[..n].split_at_mut(full);
    wasm128::apply_matte_alpha_wasm(head, &src[..full], source_opacity, inverted);
    apply_matte_alpha_scalar(tail, &src[full..n], source_opacity, inverted);
    return;
  }
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  if n >= SIMD_MIN_SPAN {
    let use_avx2 = use_avx2();
    let use_avx512 = use_avx512();
    let full = n - n % if use_avx512 { 16 } else if use_avx2 { 8 } else { 4 };
    let (head, tail) = dst[..n].split_at_mut(full);
    if use_avx512 {
      // SAFETY: `use_avx512` gates on `is_x86_feature_detected!("avx2"|"avx512f"|"avx512bw"|"avx512dq"|"avx512vl")`.
      #[allow(unsafe_code)]
      unsafe {
        avx512::apply_matte_alpha_avx512(head, &src[..full], source_opacity, inverted)
      };
    } else if use_avx2 {
      // SAFETY: `use_avx2` gates on `is_x86_feature_detected!("avx2")`.
      #[allow(unsafe_code)]
      unsafe {
        avx2::apply_matte_alpha_avx2(head, &src[..full], source_opacity, inverted)
      };
    } else {
      // SAFETY: SSE2 is part of the x86_64 baseline.
      #[allow(unsafe_code)]
      unsafe {
        sse2::apply_matte_alpha_sse2(head, &src[..full], source_opacity, inverted)
      };
    }
    apply_matte_alpha_scalar(tail, &src[full..n], source_opacity, inverted);
    return;
  }
  apply_matte_alpha_scalar(&mut dst[..n], &src[..n], source_opacity, inverted);
}

fn apply_matte_alpha_scalar(dst: &mut [u32], src: &[u32], source_opacity: u8, inverted: bool) {
  for (d, &s) in dst.iter_mut().zip(src) {
    let alpha = (s >> 24) & 0xff;
    let scaled = (alpha * u32::from(source_opacity) + 127) / 255;
    let factor = if inverted { 255 - scaled } else { scaled };
    let p = *d;
    let a = (((p >> 24) & 0xff) * factor + 127) / 255;
    let r = (((p >> 16) & 0xff) * factor + 127) / 255;
    let g = (((p >> 8) & 0xff) * factor + 127) / 255;
    let b = ((p & 0xff) * factor + 127) / 255;
    *d = (a << 24) | (r << 16) | (g << 8) | b;
  }
}

#[cfg(target_arch = "aarch64")]
#[path = "simd/neon.rs"]
mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[path = "simd/wasm.rs"]
mod wasm128;

#[cfg(target_arch = "x86_64")]
#[path = "simd/cpuid.rs"]
mod cpuid;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[path = "simd/sse2.rs"]
mod sse2;

#[cfg(target_arch = "x86_64")]
#[path = "simd/avx2.rs"]
mod avx2;

#[cfg(target_arch = "x86_64")]
#[path = "simd/avx512.rs"]
mod avx512;

#[cfg(test)]
#[path = "tests/simd.rs"]
mod tests;

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
#[cfg(test)]
#[path = "tests/alpha_simd.rs"]
mod alpha_tests;
