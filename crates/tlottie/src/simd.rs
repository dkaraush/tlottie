//! The one sanctioned SIMD module (GOALS safety contract): vector span
//! blits — NEON on aarch64, simd128 on wasm32 — each with a scalar
//! implementation that is the bit-exact oracle. Tests compare the two on
//! randomized inputs, and every other build (x86_64 emulators, 32-bit arm
//! without the feature, wasm without `-Ctarget-feature=+simd128`) uses the
//! scalar path unconditionally.
//!
//! Pixel format everywhere: premultiplied ARGB32 words (0xAARRGGBB), which
//! in little-endian memory is byte order [B, G, R, A] — exactly what
//! `vld4_u8` de-interleaves into planes. The wasm kernels skip the
//! de-interleave: the blend math is per-channel independent, so they widen
//! the interleaved bytes to u16 lanes in place and swizzle per-pixel
//! factors (coverage, alpha) across each pixel's four channel lanes.
//!
//! All `(x*a + 127) / 255` scalar steps map to the exact vector identity
//! `div255(n) = (n + (n >> 8) + 1) >> 8` applied to `n = x*a + 127`
//! (exact for n <= 65407; here n <= 255*255 + 127).
//!
//! Spans shorter than [`SIMD_MIN_SPAN`] stay scalar: measured lesson from
//! the bench project — tiny emoji spans lose to vector setup cost.

/// Minimum span length for the vector path.
const SIMD_MIN_SPAN: usize = 16;

/// Coverage-modulated solid source-over: for each pixel,
/// `ca = (cov*sa+127)/255`, source channels scaled by `ca`, then
/// premultiplied source-over into `dst`. `sr/sg/sb/sa` are 0..=255.
/// Mirrors (and on NEON must match bit-for-bit) the scalar loop.
pub(crate) fn fill_span_solid(dst: &mut [u32], cov: &[u8], sr: u32, sg: u32, sb: u32, sa: u32) {
    // Opaque source: full-coverage pixels are EXACTLY the source color
    // (ca=255 -> s_x=x, inv=0 -> o=s), so interior runs become plain stores
    // — large fills are memory-bound and interiors dominate at 320/720px.
    if sa == 255 {
        let color = (255u32 << 24) | (sr << 16) | (sg << 8) | sb;
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
        let inv = 255 - s_a;
        let o_a = s_a + (((d >> 24) & 0xff) * inv + 127) / 255;
        let o_r = s_r + (((d >> 16) & 0xff) * inv + 127) / 255;
        let o_g = s_g + (((d >> 8) & 0xff) * inv + 127) / 255;
        let o_b = s_b + ((d & 0xff) * inv + 127) / 255;
        *dst = (o_a.min(255) << 24) | (o_r.min(255) << 16) | (o_g.min(255) << 8) | o_b.min(255);
    }
}

/// Solid source-over of a UNIFORM-coverage span (mode-S rasterizer output:
/// every pixel shares one coverage byte). Bit-exact with
/// [`fill_span_solid`] over a constant coverage row: the source pixel and
/// blend factor are computed ONCE; interiors (cov 255, sa 255) collapse to
/// a plain store.
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
        dst.fill((255u32 << 24) | (s_r << 16) | (s_g << 8) | s_b);
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
    fill_span_uniform_scalar(dst, ca, s_r, s_g, s_b);
}

fn fill_span_uniform_scalar(dst: &mut [u32], ca: u32, s_r: u32, s_g: u32, s_b: u32) {
    let inv = 255 - ca;
    for dst in dst.iter_mut() {
        let d = *dst;
        let o_a = ca + (((d >> 24) & 0xff) * inv + 127) / 255;
        let o_r = s_r + (((d >> 16) & 0xff) * inv + 127) / 255;
        let o_g = s_g + (((d >> 8) & 0xff) * inv + 127) / 255;
        let o_b = s_b + ((d & 0xff) * inv + 127) / 255;
        *dst = (o_a.min(255) << 24) | (o_r.min(255) << 16) | (o_g.min(255) << 8) | o_b.min(255);
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
    linear_lut_fill_scalar(out, lut, row_base, dt, x_start, scale);
}

fn linear_lut_fill_scalar(
    out: &mut [u32],
    lut: &[u32],
    row_base: f32,
    dt: f32,
    x_start: f32,
    scale: f32,
) {
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
pub(crate) fn radial_lut_fill(
    out: &mut [u32],
    lut: &[u32],
    dd0x: f32,
    dd0y: f32,
    da: f32,
    db: f32,
    inv_r: f32,
    x_start: f32,
) {
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
            radial_lut_fill_scalar(
                tail,
                lut,
                dd0x,
                dd0y,
                da,
                db,
                inv_r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if out.len() >= SIMD_MIN_SPAN {
            let full = out.len() - out.len() % 4;
            let (head, tail) = out.split_at_mut(full);
            wasm128::radial_lut_fill_wasm(head, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
            radial_lut_fill_scalar(
                tail,
                lut,
                dd0x,
                dd0y,
                da,
                db,
                inv_r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    radial_lut_fill_scalar(out, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
}

#[allow(clippy::too_many_arguments)]
fn radial_lut_fill_scalar(
    out: &mut [u32],
    lut: &[u32],
    dd0x: f32,
    dd0y: f32,
    da: f32,
    db: f32,
    inv_r: f32,
    x_start: f32,
    scale: f32,
) {
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
pub(crate) fn focal_lut_fill(
    out: &mut [u32],
    lut: &[u32],
    g0x: f32,
    g0y: f32,
    sa: f32,
    sb: f32,
    dx: f32,
    dy: f32,
    a: f32,
    inv2a: f32,
    r: f32,
    x_start: f32,
) {
    let scale = (lut.len().saturating_sub(1)) as f32;
    #[cfg(target_arch = "aarch64")]
    {
        if out.len() >= SIMD_MIN_SPAN {
            let full = out.len() - out.len() % 4;
            let (head, tail) = out.split_at_mut(full);
            // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
            #[allow(unsafe_code)]
            unsafe {
                neon::focal_lut_fill_neon(
                    head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
                )
            };
            focal_lut_fill_scalar(
                tail,
                lut,
                g0x,
                g0y,
                sa,
                sb,
                dx,
                dy,
                a,
                inv2a,
                r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if out.len() >= SIMD_MIN_SPAN {
            let full = out.len() - out.len() % 4;
            let (head, tail) = out.split_at_mut(full);
            wasm128::focal_lut_fill_wasm(
                head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
            );
            focal_lut_fill_scalar(
                tail,
                lut,
                g0x,
                g0y,
                sa,
                sb,
                dx,
                dy,
                a,
                inv2a,
                r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    focal_lut_fill_scalar(
        out, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
    );
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
fn focal_row_coefficients(
    g0x: f32,
    g0y: f32,
    sa: f32,
    sb: f32,
    dx: f32,
    dy: f32,
    a: f32,
) -> (f32, f32, f32, f32, f32) {
    let b0 = 2.0 * (g0x * dx + g0y * dy);
    let db = 2.0 * (sa * dx + sb * dy);
    let four_a = 4.0 * a;
    let d0 = b0 * b0 + four_a * (g0x * g0x + g0y * g0y);
    let d1 = 2.0 * b0 * db + (8.0 * a) * (g0x * sa + g0y * sb);
    let d2 = db * db + four_a * (sa * sa + sb * sb);
    (b0, db, d0, d1, d2)
}

#[allow(clippy::too_many_arguments)]
fn focal_lut_fill_scalar(
    out: &mut [u32],
    lut: &[u32],
    g0x: f32,
    g0y: f32,
    sa: f32,
    sb: f32,
    dx: f32,
    dy: f32,
    a: f32,
    inv2a: f32,
    r: f32,
    x_start: f32,
    scale: f32,
) {
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
        let inv = 255 - sa;
        let a = sa + (((p >> 24) & 0xff) * inv + 127) / 255;
        let r = sr + (((p >> 16) & 0xff) * inv + 127) / 255;
        let g = sg + (((p >> 8) & 0xff) * inv + 127) / 255;
        let b = sb + ((p & 0xff) * inv + 127) / 255;
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
    let inv = 255 - sa;
    let a = sa + (((p >> 24) & 0xff) * inv + 127) / 255;
    let r = sr + (((p >> 16) & 0xff) * inv + 127) / 255;
    let g = sg + (((p >> 8) & 0xff) * inv + 127) / 255;
    let b = sb + ((p & 0xff) * inv + 127) / 255;
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
    linear_lut_over_scalar(dst, lut, row_base, dt, x_start, scale);
}

fn linear_lut_over_scalar(
    dst: &mut [u32],
    lut: &[u32],
    row_base: f32,
    dt: f32,
    x_start: f32,
    scale: f32,
) {
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
pub(crate) fn radial_lut_over(
    dst: &mut [u32],
    lut: &[u32],
    dd0x: f32,
    dd0y: f32,
    da: f32,
    db: f32,
    inv_r: f32,
    x_start: f32,
) {
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
            radial_lut_over_scalar(
                tail,
                lut,
                dd0x,
                dd0y,
                da,
                db,
                inv_r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    radial_lut_over_scalar(dst, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
}

#[allow(clippy::too_many_arguments)]
fn radial_lut_over_scalar(
    dst: &mut [u32],
    lut: &[u32],
    dd0x: f32,
    dd0y: f32,
    da: f32,
    db: f32,
    inv_r: f32,
    x_start: f32,
    scale: f32,
) {
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
pub(crate) fn focal_lut_over(
    dst: &mut [u32],
    lut: &[u32],
    g0x: f32,
    g0y: f32,
    sa: f32,
    sb: f32,
    dx: f32,
    dy: f32,
    a: f32,
    inv2a: f32,
    r: f32,
    x_start: f32,
) {
    let scale = (lut.len().saturating_sub(1)) as f32;
    #[cfg(target_arch = "aarch64")]
    {
        if dst.len() >= SIMD_MIN_SPAN {
            let full = dst.len() - dst.len() % 8;
            let (head, tail) = dst.split_at_mut(full);
            // SAFETY: NEON/AdvSIMD is architecturally mandatory on aarch64.
            #[allow(unsafe_code)]
            unsafe {
                neon::focal_lut_over_neon(
                    head, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
                )
            };
            focal_lut_over_scalar(
                tail,
                lut,
                g0x,
                g0y,
                sa,
                sb,
                dx,
                dy,
                a,
                inv2a,
                r,
                x_start + full as f32,
                scale,
            );
            return;
        }
    }
    focal_lut_over_scalar(
        dst, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
    );
}

#[allow(clippy::too_many_arguments)]
fn focal_lut_over_scalar(
    dst: &mut [u32],
    lut: &[u32],
    g0x: f32,
    g0y: f32,
    sa: f32,
    sb: f32,
    dx: f32,
    dy: f32,
    a: f32,
    inv2a: f32,
    r: f32,
    x_start: f32,
    scale: f32,
) {
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

#[cfg(target_arch = "aarch64")]
mod neon {
    //! NEON kernels, 8 pixels per iteration. u16 lanes never overflow:
    //! every product is (<=255)*(<=255) <= 65025 and div255's intermediate
    //! stays <= 65407.

    use core::arch::aarch64::{
        uint16x8_t, uint32x4_t, uint8x8x4_t, vabsq_f32, vaddq_f32, vaddq_u16, vandq_u32, vbslq_u32,
        vcgeq_f32, vcltq_f32, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_u16, vdupq_n_u32, vgetq_lane_u32,
        vld1_u8, vld1q_f32, vld4_u8, vmaxq_f32, vminq_f32, vminq_u16, vmovl_u8, vmovn_u16,
        vmulq_f32, vmulq_u16, vnegq_f32, vshrq_n_u16, vsqrtq_f32, vst4_u8, vsubq_f32, vsubq_u16,
    };

    /// Exact `(n + 127) / 255` on u16 lanes (n <= 65025).
    #[inline]
    #[target_feature(enable = "neon")]
    fn div255_round(n: uint16x8_t) -> uint16x8_t {
        let t = vaddq_u16(n, vdupq_n_u16(127));
        let u = vaddq_u16(vaddq_u16(t, vshrq_n_u16::<8>(t)), vdupq_n_u16(1));
        vshrq_n_u16::<8>(u)
    }

    /// 4-lane radial gradient LUT fill for full-coverage runs. Lane math:
    /// `mul + add` (NOT fma) to mirror the scalar `ddx*ddx + ddy*ddy`;
    /// FSQRT/vcvt(truncate) match scalar `sqrt()`/`as usize` exactly, so
    /// NEON and the `dd0 + X·d` scalar form agree lane-for-lane. `X` is the
    /// absolute device column (`x_start + lane`), making positions
    /// segmentation-invariant.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    pub(super) fn radial_lut_fill_neon(
        out: &mut [u32],
        lut: &[u32],
        dd0x: f32,
        dd0y: f32,
        da: f32,
        db: f32,
        inv_r: f32,
        x_start: f32,
        scale: f32,
    ) {
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
    pub(super) fn linear_lut_fill_neon(
        out: &mut [u32],
        lut: &[u32],
        row_base: f32,
        dt: f32,
        x_start: f32,
        scale: f32,
    ) {
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
    pub(super) fn focal_lut_fill_neon(
        out: &mut [u32],
        lut: &[u32],
        g0x: f32,
        g0y: f32,
        sa: f32,
        sb: f32,
        dx: f32,
        dy: f32,
        a: f32,
        inv2a: f32,
        r: f32,
        x_start: f32,
        scale: f32,
    ) {
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
            let root = vmaxq_f32(
                vmulq_f32(vsubq_f32(nb, sq), inv2av),
                vmulq_f32(vaddq_f32(nb, sq), inv2av),
            );
            let valid = vandq_u32(
                vandq_u32(vcgeq_f32(det, zero), vcgeq_f32(vmulq_f32(rv, root), zero)),
                vcltq_f32(vabsq_f32(root), inf),
            );
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
        vminq_u16(
            vaddq_u16(s, div255_round(vmulq_u16(d, inv))),
            vdupq_n_u16(255),
        )
    }

    #[target_feature(enable = "neon")]
    pub(super) fn fill_span_solid_neon(
        dst: &mut [u32],
        cov: &[u8],
        sr: u32,
        sg: u32,
        sb: u32,
        sa: u32,
    ) {
        let (sr, sg, sb, sa) = (
            vdupq_n_u16(sr as u16),
            vdupq_n_u16(sg as u16),
            vdupq_n_u16(sb as u16),
            vdupq_n_u16(sa as u16),
        );
        let full = vdupq_n_u16(255);
        for (dpx, cpx) in dst.chunks_exact_mut(8).zip(cov.chunks_exact(8)) {
            // SAFETY: chunks_exact guarantees exactly 8 u32 (32 bytes) at
            // dpx and 8 bytes at cpx; vld4_u8/vld1_u8/vst4_u8 read/write
            // exactly those spans.
            #[allow(unsafe_code)]
            let (planes, c) =
                unsafe { (vld4_u8(dpx.as_ptr().cast::<u8>()), vld1_u8(cpx.as_ptr())) };
            let cw = vmovl_u8(c);
            let ca = div255_round(vmulq_u16(cw, sa));
            let s_r = div255_round(vmulq_u16(sr, ca));
            let s_g = div255_round(vmulq_u16(sg, ca));
            let s_b = div255_round(vmulq_u16(sb, ca));
            let inv = vsubq_u16(full, ca);
            let d_b = vmovl_u8(planes.0);
            let d_g = vmovl_u8(planes.1);
            let d_r = vmovl_u8(planes.2);
            let d_a = vmovl_u8(planes.3);
            let out = uint8x8x4_t(
                vmovn_u16(over(d_b, s_b, inv)),
                vmovn_u16(over(d_g, s_g, inv)),
                vmovn_u16(over(d_r, s_r, inv)),
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
                vmovn_u16(over(vmovl_u8(planes.0), sb, inv)),
                vmovn_u16(over(vmovl_u8(planes.1), sg, inv)),
                vmovn_u16(over(vmovl_u8(planes.2), sr, inv)),
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
            let (d4, s4) = unsafe {
                (
                    vld4_u8(dpx.as_ptr().cast::<u8>()),
                    vld4_u8(spx.as_ptr().cast::<u8>()),
                )
            };
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
        let (d4, s4) = unsafe {
            (
                vld4_u8(dpx.as_ptr().cast::<u8>()),
                vld4_u8(src.as_ptr().cast::<u8>()),
            )
        };
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
    pub(super) fn linear_lut_over_neon(
        dst: &mut [u32],
        lut: &[u32],
        row_base: f32,
        dt: f32,
        x_start: f32,
        scale: f32,
    ) {
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
    pub(super) fn radial_lut_over_neon(
        dst: &mut [u32],
        lut: &[u32],
        dd0x: f32,
        dd0y: f32,
        da: f32,
        db: f32,
        inv_r: f32,
        x_start: f32,
        scale: f32,
    ) {
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
    pub(super) fn focal_lut_over_neon(
        dst: &mut [u32],
        lut: &[u32],
        g0x: f32,
        g0y: f32,
        sa: f32,
        sb: f32,
        dx: f32,
        dy: f32,
        a: f32,
        inv2a: f32,
        r: f32,
        x_start: f32,
        scale: f32,
    ) {
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
                let root = vmaxq_f32(
                    vmulq_f32(vsubq_f32(nb, sq), inv2av),
                    vmulq_f32(vaddq_f32(nb, sq), inv2av),
                );
                let valid = vandq_u32(
                    vandq_u32(vcgeq_f32(det, zero), vcgeq_f32(vmulq_f32(rv, root), zero)),
                    vcltq_f32(vabsq_f32(root), inf),
                );
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
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm128 {
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
        f32x4, f32x4_abs, f32x4_add, f32x4_ge, f32x4_lt, f32x4_max, f32x4_min, f32x4_mul,
        f32x4_neg, f32x4_splat, f32x4_sqrt, f32x4_sub, u16x8, u16x8_add, u16x8_extend_high_u8x16,
        u16x8_extend_low_u8x16, u16x8_min, u16x8_mul, u16x8_shr, u16x8_splat, u16x8_sub,
        u32x4_extract_lane, u32x4_splat, u32x4_trunc_sat_f32x4, u8x16, u8x16_narrow_i16x8,
        u8x16_swizzle, v128, v128_and, v128_bitselect, v128_load, v128_store,
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
        u16x8_min(
            u16x8_add(s, div255_round(u16x8_mul(d, inv))),
            u16x8_splat(255),
        )
    }

    pub(super) fn fill_span_solid_wasm(
        dst: &mut [u32],
        cov: &[u8],
        sr: u32,
        sg: u32,
        sb: u32,
        sa: u32,
    ) {
        // Source pattern per pixel is [B,G,R,255]: the 255 alpha lane makes
        // `div255(255*ca+127) == ca` hold exactly, so one multiply yields
        // s_r/s_g/s_b AND s_a = ca in their lanes.
        let src = u16x8(
            sb as u16, sg as u16, sr as u16, 255, sb as u16, sg as u16, sr as u16, 255,
        );
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
        let s = u16x8(
            s_b as u16, s_g as u16, s_r as u16, ca as u16, s_b as u16, s_g as u16, s_r as u16,
            ca as u16,
        );
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

    pub(super) fn composite_over_wasm(dst: &mut [u32], src: &[u32], k: u32) {
        // Swizzle replicating each pixel's alpha byte across its 4 lanes.
        let arep_pat = u8x16(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
        let kq = u16x8_splat(k as u16);
        let full = u16x8_splat(255);
        for (dpx, spx) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            // SAFETY: chunks_exact guarantees exactly 4 u32 (16 bytes) at
            // both pointers; v128 loads/stores allow unaligned.
            #[allow(unsafe_code)]
            let (d, s) = unsafe {
                (
                    v128_load(dpx.as_ptr().cast::<v128>()),
                    v128_load(spx.as_ptr().cast::<v128>()),
                )
            };
            let s_lo = div255_round(u16x8_mul(u16x8_extend_low_u8x16(s), kq));
            let s_hi = div255_round(u16x8_mul(u16x8_extend_high_u8x16(s), kq));
            let arep = u8x16_swizzle(s, arep_pat);
            let inv_lo = u16x8_sub(
                full,
                div255_round(u16x8_mul(u16x8_extend_low_u8x16(arep), kq)),
            );
            let inv_hi = u16x8_sub(
                full,
                div255_round(u16x8_mul(u16x8_extend_high_u8x16(arep), kq)),
            );
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

    /// 4-lane linear gradient LUT fill: `t = row_base + X·dt` at absolute
    /// device column `X = x_start + lane` (segmentation-invariant), exactly
    /// the scalar form.
    pub(super) fn linear_lut_fill_wasm(
        out: &mut [u32],
        lut: &[u32],
        row_base: f32,
        dt: f32,
        x_start: f32,
        scale: f32,
    ) {
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

    /// 4-lane radial gradient LUT fill; mul + add (NOT fma) mirrors the
    /// scalar `ddx*ddx + ddy*ddy`, and f32x4_sqrt matches scalar `sqrt()`,
    /// so lanes agree with the `dd0 + X·d` scalar form bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn radial_lut_fill_wasm(
        out: &mut [u32],
        lut: &[u32],
        dd0x: f32,
        dd0y: f32,
        da: f32,
        db: f32,
        inv_r: f32,
        x_start: f32,
        scale: f32,
    ) {
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

    /// 4-lane focal (highlight) radial LUT fill; keep the scalar absolute-X
    /// Horner rounding protocol so SIMD tails cannot introduce seams.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn focal_lut_fill_wasm(
        out: &mut [u32],
        lut: &[u32],
        g0x: f32,
        g0y: f32,
        sa: f32,
        sb: f32,
        dx: f32,
        dy: f32,
        a: f32,
        inv2a: f32,
        r: f32,
        x_start: f32,
        scale: f32,
    ) {
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
            let root = f32x4_max(
                f32x4_mul(f32x4_sub(nb, sq), inv2av),
                f32x4_mul(f32x4_add(nb, sq), inv2av),
            );
            let valid = v128_and(
                v128_and(f32x4_ge(det, zero), f32x4_ge(f32x4_mul(rv, root), zero)),
                f32x4_lt(f32x4_abs(root), inf),
            );
            lut_store(chunk, lut, lut_indices(root, valid, scalev));
            kf = f32x4_add(kf, four);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let (sr, sg, sb, sa) = (
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                );
                let mut a = base.clone();
                let mut b = base.clone();
                fill_span_solid(&mut a, &cov, sr, sg, sb, sa);
                fill_span_solid_scalar(&mut b, &cov, sr, sg, sb, sa);
                assert_eq!(a, b, "len={len} sr={sr} sg={sg} sb={sb} sa={sa}");
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
                let (sr, sg, sb, sa) = (
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                    rng.next() as u32 & 0xff,
                );
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
                    assert_eq!(
                        a, b,
                        "len={len} x_start={x_start} row_base={row_base} dt={dt}"
                    );
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
                    let (x0, y0, da, db) = (
                        f(&mut rng),
                        f(&mut rng),
                        f(&mut rng) / 100.0,
                        f(&mut rng) / 100.0,
                    );
                    let inv_r = ((rng.next() % 1000) as f32 + 1.0) / 5000.0;
                    let mut a = vec![0u32; len];
                    let mut b = vec![0u32; len];
                    radial_lut_fill(&mut a, &lut, x0, y0, da, db, inv_r, x_start);
                    radial_lut_fill_scalar(&mut b, &lut, x0, y0, da, db, inv_r, x_start, scale);
                    assert_eq!(
                        a, b,
                        "len={len} x_start={x_start} x0={x0} y0={y0} da={da} db={db}"
                    );
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
                    focal_lut_fill(
                        &mut va, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start,
                    );
                    focal_lut_fill_scalar(
                        &mut vb, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
                    );
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
                focal_lut_fill(
                    &mut whole, &lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start,
                );

                let mut split = vec![0u32; whole.len()];
                let mut begin = 0usize;
                for end in [1usize, 4, 17, 63, 128, 199, 257] {
                    focal_lut_fill(
                        &mut split[begin..end],
                        &lut,
                        g0x,
                        g0y,
                        sa,
                        sb,
                        dx,
                        dy,
                        a,
                        inv2a,
                        r,
                        x_start + begin as f32,
                    );
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
            assert!(
                det_err <= 2.0e-4 * (1.0 + det64.abs()),
                "det={det} oracle={det64}"
            );
        }
    }

    #[test]
    fn focal_horner_preserves_invalid_root_semantics() {
        let lut = vec![0xff00_0000u32; 1024];
        let scale = (lut.len() - 1) as f32;

        // a<0 with d=0 makes D negative away from the focal point.
        let mut invalid_det = [0xdead_beefu32; 4];
        focal_lut_fill_scalar(
            &mut invalid_det,
            &lut,
            1.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            -1.0,
            -0.5,
            1.0,
            0.0,
            scale,
        );
        assert_eq!(invalid_det, [0; 4]);

        // A finite root pointing behind the focal cone remains transparent.
        let mut behind = [0xdead_beefu32; 4];
        focal_lut_fill_scalar(
            &mut behind,
            &lut,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.5,
            -1.0,
            0.0,
            scale,
        );
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
                for &(rb, dt) in &[
                    (nan, 0.01f32),
                    (0.3f32, inf),
                    (0.3f32, -inf),
                    (inf, 0.0f32),
                    (-inf, 0.01f32),
                ] {
                    let mut a = vec![0u32; len];
                    let mut b = vec![0u32; len];
                    linear_lut_fill(&mut a, &lut, rb, dt, x_start);
                    linear_lut_fill_scalar(&mut b, &lut, rb, dt, x_start, scale);
                    assert_eq!(a, b, "linear len={len} x_start={x_start} rb={rb} dt={dt}");
                }
                // Radial: non-finite base/step and inv_r.
                for &(d0, d, ir) in &[
                    (nan, 0.01f32, 0.1f32),
                    (0.5f32, inf, 0.1f32),
                    (0.5f32, 0.01f32, inf),
                    (0.5f32, 0.01f32, nan),
                ] {
                    let mut a = vec![0u32; len];
                    let mut b = vec![0u32; len];
                    radial_lut_fill(&mut a, &lut, d0, 0.2, d, 0.01, ir, x_start);
                    radial_lut_fill_scalar(&mut b, &lut, d0, 0.2, d, 0.01, ir, x_start, scale);
                    assert_eq!(
                        a, b,
                        "radial len={len} x_start={x_start} d0={d0} d={d} ir={ir}"
                    );
                }
                // Focal: non-finite g0/step (det/root can go non-finite).
                for &(g0, s) in &[(nan, 0.01f32), (0.5f32, inf), (inf, 0.01f32)] {
                    let a_coef = 0.7f32;
                    let inv2a = 1.0 / (2.0 * a_coef);
                    let mut a = vec![0u32; len];
                    let mut b = vec![0u32; len];
                    focal_lut_fill(
                        &mut a, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start,
                    );
                    focal_lut_fill_scalar(
                        &mut b, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start,
                        scale,
                    );
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

    /// Two-pass scalar reference for the fused linear kernel: fill src via
    /// the LUT-fill scalar oracle, then source-over with k=255.
    fn linear_over_ref(dst: &mut [u32], lut: &[u32], row_base: f32, dt: f32, x_start: f32) {
        let scale = (lut.len() - 1) as f32;
        let mut src = vec![0u32; dst.len()];
        linear_lut_fill_scalar(&mut src, lut, row_base, dt, x_start, scale);
        composite_over_scalar(dst, &src, 255);
    }

    #[allow(clippy::too_many_arguments)]
    fn radial_over_ref(
        dst: &mut [u32],
        lut: &[u32],
        dd0x: f32,
        dd0y: f32,
        da: f32,
        db: f32,
        inv_r: f32,
        x_start: f32,
    ) {
        let scale = (lut.len() - 1) as f32;
        let mut src = vec![0u32; dst.len()];
        radial_lut_fill_scalar(&mut src, lut, dd0x, dd0y, da, db, inv_r, x_start, scale);
        composite_over_scalar(dst, &src, 255);
    }

    #[allow(clippy::too_many_arguments)]
    fn focal_over_ref(
        dst: &mut [u32],
        lut: &[u32],
        g0x: f32,
        g0y: f32,
        sa: f32,
        sb: f32,
        dx: f32,
        dy: f32,
        a: f32,
        inv2a: f32,
        r: f32,
        x_start: f32,
    ) {
        let scale = (lut.len() - 1) as f32;
        let mut src = vec![0u32; dst.len()];
        focal_lut_fill_scalar(
            &mut src, lut, g0x, g0y, sa, sb, dx, dy, a, inv2a, r, x_start, scale,
        );
        composite_over_scalar(dst, &src, 255);
    }

    /// The whole point of fusion: `linear_lut_over` (NEON where available)
    /// must equal the two-pass `linear_lut_fill` + `composite_over_span(255)`
    /// byte-for-byte, at any absolute column and over dst of any content.
    #[test]
    fn linear_lut_over_matches_two_pass() {
        let mut rng = Rng(0x1111_2222_3333_4444);
        let lut: Vec<u32> = (0..1024u32)
            .map(|i| premult(i.wrapping_mul(0x0193_7caf)))
            .collect();
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
                    assert_eq!(
                        a, b,
                        "len={len} x_start={x_start} row_base={row_base} dt={dt}"
                    );
                }
            }
        }
    }

    #[test]
    fn radial_lut_over_matches_two_pass() {
        let mut rng = Rng(0x5555_6666_7777_8888);
        let lut: Vec<u32> = (0..1024u32)
            .map(|i| premult(i.wrapping_mul(0x0193_7caf)))
            .collect();
        for len in [1usize, 4, 15, 16, 17, 32, 33, 64, 257] {
            for &x_start in &[0.0f32, 1.0, 137.0, 719.0] {
                for _ in 0..80 {
                    let f = |r: &mut Rng| ((r.next() % 2000) as f32 - 1000.0) / 37.0;
                    let (x0, y0, da, db) = (
                        f(&mut rng),
                        f(&mut rng),
                        f(&mut rng) / 100.0,
                        f(&mut rng) / 100.0,
                    );
                    let inv_r = ((rng.next() % 1000) as f32 + 1.0) / 5000.0;
                    let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
                    let mut a = base.clone();
                    let mut b = base.clone();
                    radial_lut_over(&mut a, &lut, x0, y0, da, db, inv_r, x_start);
                    radial_over_ref(&mut b, &lut, x0, y0, da, db, inv_r, x_start);
                    assert_eq!(
                        a, b,
                        "len={len} x_start={x_start} x0={x0} y0={y0} da={da} db={db}"
                    );
                }
            }
        }
    }

    #[test]
    fn focal_lut_over_matches_two_pass() {
        let mut rng = Rng(0x9999_aaaa_bbbb_cccc);
        let lut: Vec<u32> = (0..1024u32)
            .map(|i| premult(i.wrapping_mul(0x0193_7caf)))
            .collect();
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
                    focal_lut_over(
                        &mut va, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start,
                    );
                    focal_over_ref(
                        &mut vb, &lut, gx0, gy0, sa, sb, dx, dy, a, inv2a, r, x_start,
                    );
                    assert_eq!(va, vb, "len={len} x_start={x_start} gx0={gx0} a={a} r={r}");
                }
            }
        }
    }

    /// Fused kernels must match the two-pass reference even when positions go
    /// non-finite (sentinel → transparent src → dst unchanged), at any column.
    #[test]
    fn lut_over_matches_two_pass_nonfinite() {
        let lut: Vec<u32> = (0..1024u32)
            .map(|i| premult(i.wrapping_mul(0x0193_7caf)))
            .collect();
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        for len in [1usize, 4, 16, 17, 64, 129] {
            for &x_start in &[0.0f32, 5.0, 200.0] {
                let base: Vec<u32> = (0..len).map(|_| premult(rng.next() as u32)).collect();
                for &(rb, dt) in &[
                    (nan, 0.01f32),
                    (0.3, inf),
                    (0.3, -inf),
                    (inf, 0.0),
                    (-inf, 0.01),
                ] {
                    let mut a = base.clone();
                    let mut b = base.clone();
                    linear_lut_over(&mut a, &lut, rb, dt, x_start);
                    linear_over_ref(&mut b, &lut, rb, dt, x_start);
                    assert_eq!(a, b, "linear len={len} x_start={x_start} rb={rb} dt={dt}");
                }
                for &(d0, d, ir) in &[
                    (nan, 0.01f32, 0.1f32),
                    (0.5, inf, 0.1),
                    (0.5, 0.01, inf),
                    (0.5, 0.01, nan),
                ] {
                    let mut a = base.clone();
                    let mut b = base.clone();
                    radial_lut_over(&mut a, &lut, d0, 0.2, d, 0.01, ir, x_start);
                    radial_over_ref(&mut b, &lut, d0, 0.2, d, 0.01, ir, x_start);
                    assert_eq!(
                        a, b,
                        "radial len={len} x_start={x_start} d0={d0} d={d} ir={ir}"
                    );
                }
                for &(g0, s) in &[(nan, 0.01f32), (0.5, inf), (inf, 0.01)] {
                    let a_coef = 0.7f32;
                    let inv2a = 1.0 / (2.0 * a_coef);
                    let mut a = base.clone();
                    let mut b = base.clone();
                    focal_lut_over(
                        &mut a, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start,
                    );
                    focal_over_ref(
                        &mut b, &lut, g0, 0.2, s, 0.01, 0.3, 0.1, a_coef, inv2a, 0.5, x_start,
                    );
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
}
