//! Stroker v0: converts a flattened polyline into a set of same-winding
//! polygons (segment quads, join wedges, caps) whose nonzero-union is the
//! stroked area. The signed-area rasterizer merges the pieces seamlessly:
//! shared edges contribute complementary fractional coverage that sums to
//! exactly 1, so no seams appear.
//!
//! Curve-space offsetting (higher fidelity at large widths) is a later phase.

use crate::geometry::Contour;
use crate::math::Vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Join {
    Miter,
    Round,
    Bevel,
}

/// Ensures counter-clockwise orientation (positive signed area) so every
/// emitted piece has the same winding sign.
fn normalized(mut pts: Vec<Vec2>) -> Contour {
    let mut area = 0.0f32;
    for (i, p) in pts.iter().enumerate() {
        let q = pts
            .get(i + 1)
            .or_else(|| pts.first())
            .copied()
            .unwrap_or(*p);
        area += p.x * q.y - q.x * p.y;
    }
    if area < 0.0 {
        pts.reverse();
    }
    Contour {
        points: pts,
        anchors: Vec::new(),
        inv_lin: None,
    }
}

fn norm(v: Vec2) -> Option<Vec2> {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len > 1e-6 && len.is_finite() {
        Some(Vec2::new(v.x / len, v.y / len))
    } else {
        None
    }
}

/// Arc polygon around `center` from angle `a0` to `a1` (radians, shortest
/// direction given by sign of a1-a0), radius r, including both endpoints.
fn arc_points(out: &mut Vec<Vec2>, center: Vec2, r: f32, a0: f32, a1: f32) {
    let sweep = a1 - a0;
    let steps = ((sweep.abs() / 0.35).ceil() as usize).clamp(1, 24);
    // One exact reservation: callers start wedges at capacity 0-1 and this
    // loop otherwise re-grows the Vec 1→2→4→… with a realloc per step —
    // measured as the hottest allocator path at emoji sizes.
    out.reserve(steps + 1);
    for i in 0..=steps {
        let a = a0 + sweep * (i as f32 / steps as f32);
        out.push(Vec2::new(center.x + r * a.cos(), center.y + r * a.sin()));
    }
}

/// Strokes a flattened polyline. Returns polygons whose nonzero union is the
/// stroke area. `hw` is the half-width in device units.
#[allow(clippy::too_many_arguments)]
/// Dev-only stroke-path counters, read via tlottie::stroke_stats():
/// 0=contours outlined (open)  1=contours outlined (closed rings)
/// 2=piece fallback (open)     3=piece fallback (closed)
/// 4=pieces emitted total
pub(crate) static STROKE_STATS: [core::sync::atomic::AtomicU64; 5] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];

#[cfg(feature = "stats")]
#[inline]
fn stroke_stat(i: usize, n: usize) {
    if let Some(c) = STROKE_STATS.get(i) {
        c.fetch_add(n as u64, core::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(not(feature = "stats"))]
#[inline(always)]
fn stroke_stat(_i: usize, _n: usize) {}

/// Legacy piece-union stroker escape hatch (dev A/B only). Read once per
/// process: cached coverage is keyed by source geometry, so the stroker
/// choice must not change under a live Animation.
fn legacy_stroker() -> bool {
    #[cfg(test)]
    if FORCE_LEGACY.with(|c| c.get()) {
        return true;
    }
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| std::env::var_os("TLOTTIE_LEGACY_STROKER").is_some())
}

#[cfg(test)]
thread_local! {
    static FORCE_LEGACY: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Test-only: run the legacy piece-union stroker regardless of the process
/// flag (differential coverage tests compare the two constructions).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stroke_polyline_pieces_for_test(
    points: &[Vec2],
    point_anchors: &[bool],
    closed: bool,
    hw: f32,
    cap: Cap,
    join: Join,
    miter_limit: f32,
) -> Vec<Contour> {
    FORCE_LEGACY.with(|c| c.set(true));
    let out = stroke_polyline(
        points,
        point_anchors,
        closed,
        hw,
        cap,
        join,
        miter_limit,
        &mut Vec::new(),
        false,
    );
    FORCE_LEGACY.with(|c| c.set(false));
    out
}

pub(crate) fn stroke_polyline(
    points: &[Vec2],
    point_anchors: &[bool],
    closed: bool,
    hw: f32,
    cap: Cap,
    join: Join,
    miter_limit: f32,
    pool: &mut Vec<Vec<Vec2>>,
    solo: bool,
) -> Vec<Contour> {
    let mut out: Vec<Contour> = Vec::new();
    // Piece buffers come from (and, after the paint executes, return to)
    // the caller's pool: measured 2,683 piece allocations per 64px frame
    // on stroke-heavy files, all recyclable.
    macro_rules! piece {
        ($($p:expr),+ $(,)?) => {{
            let mut v: Vec<Vec2> = pool.pop().unwrap_or_default();
            v.clear();
            $( v.push($p); )+
            v
        }};
    }
    if !(hw > 0.0) || !hw.is_finite() {
        return out;
    }
    // Deduplicate consecutive identical points; anchors follow (a merged
    // point stays an anchor if EITHER duplicate was authored). The epsilon
    // must stay tiny: sub-1/64px OPEN contours are the "dot idiom" (a
    // near-zero-length path with thick round caps draws a disc) and a
    // coarser merge collapses them to one point, erasing the dot.
    let mut pts: Vec<Vec2> = Vec::with_capacity(points.len());
    let mut anc: Vec<bool> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        let a = point_anchors.get(i).copied().unwrap_or(true);
        if pts.last().map_or(true, |q| {
            (q.x - p.x).abs() > 1e-6 || (q.y - p.y).abs() > 1e-6
        }) {
            pts.push(*p);
            anc.push(a);
        } else if let Some(slot) = anc.last_mut() {
            *slot = *slot || a;
        }
    }
    if closed && pts.len() > 1 {
        if let (Some(first), Some(last)) = (pts.first().copied(), pts.last().copied()) {
            if (first.x - last.x).abs() < 1e-6 && (first.y - last.y).abs() < 1e-6 {
                pts.pop();
                anc.pop();
            }
        }
    }

    if pts.len() < 2 {
        // Degenerate single-vertex contour: rlottie's FreeType stroker
        // emits NOTHING for a segment-less path (no cap dots) — files carry
        // stray one-point paths that must stay invisible.
        return out;
    }

    // FT-style two-border stroker (stroke_ft.rs): the default since the
    // piece-union construction's criss-cross interior edges dominated
    // 320/720px raster cost. `TLOTTIE_LEGACY_STROKER=1` restores the piece
    // path for A/B; the choice is process-wide and fixed for the life of
    // every Animation (coverage-cache coherence — entries are keyed by
    // source geometry, not stroker version).
    if !legacy_stroker() {
        let before = out.len();
        crate::stroke_ft::stroke_outline(
            &pts,
            &anc,
            closed,
            hw,
            cap,
            join,
            miter_limit,
            pool,
            &mut out,
        );
        stroke_stat(if closed { 1 } else { 0 }, 1);
        stroke_stat(4, out.len() - before);
        return out;
    }

    let seg_count = if closed { pts.len() } else { pts.len() - 1 };

    // OUTLINE FAST PATH (v1): eligible OPEN contours emit ONE connected
    // polygon (left offsets + joins, end cap, reversed right offsets, start
    // cap) instead of a rect+wedge per segment — roughly halving the edge
    // count the rasterizer accumulates (measured 25-46% of heavy frames).
    // Nonzero winding fills self-overlaps at inner joins exactly like the
    // piece union. Any vertex that would take a tuned special case
    // (reversal, degenerate direction) falls back to the piece path, as do
    // closed contours (outer+inner ring construction is v2).
    if !closed {
        if let Some(c) = outline_open(&pts, &anc, hw, cap, join, miter_limit, pool) {
            stroke_stat(0, 1);
            out.push(c);
            return out;
        }
    } else if solo {
        // Ring-pair annulus is ONLY sound when this contour is the paint's
        // entire geometry: the inner ring's negative winding cancels any
        // OTHER overlapping +1 contour (consensus BROKEN 7->84 when applied
        // unconditionally).
        if let Some(rings) = outline_closed(&pts, &anc, hw, join, miter_limit, pool) {
            stroke_stat(1, 1);
            out.extend(rings);
            return out;
        }
    }
    stroke_stat(if closed { 3 } else { 2 }, 1);
    // One rect per segment + one join per vertex + two caps, upper bound.
    out.reserve(2 * seg_count + 2);

    // Segment rectangles.
    for s in 0..seg_count {
        let p0 = match pts.get(s) {
            Some(p) => *p,
            None => continue,
        };
        let p1 = match pts.get(if s + 1 == pts.len() { 0 } else { s + 1 }) {
            Some(p) => *p,
            None => continue,
        };
        let Some(d) = norm(Vec2::new(p1.x - p0.x, p1.y - p0.y)) else {
            continue;
        };
        let n = Vec2::new(-d.y * hw, d.x * hw);
        out.push(normalized(piece![
            Vec2::new(p0.x + n.x, p0.y + n.y),
            Vec2::new(p1.x + n.x, p1.y + n.y),
            Vec2::new(p1.x - n.x, p1.y - n.y),
            Vec2::new(p0.x - n.x, p0.y - n.y),
        ]));
    }

    // Joins at interior vertices (all vertices when closed).
    let join_range = if closed {
        0..pts.len()
    } else {
        1..pts.len() - 1
    };
    for v in join_range {
        let prev = match pts.get(if v == 0 { pts.len() - 1 } else { v - 1 }) {
            Some(p) => *p,
            None => continue,
        };
        let here = match pts.get(v) {
            Some(p) => *p,
            None => continue,
        };
        let next = match pts.get(if v + 1 == pts.len() { 0 } else { v + 1 }) {
            Some(p) => *p,
            None => continue,
        };
        let (Some(d0), Some(d1)) = (
            norm(Vec2::new(here.x - prev.x, here.y - prev.y)),
            norm(Vec2::new(next.x - here.x, next.y - here.y)),
        ) else {
            continue;
        };
        let is_anchor = anc.get(v).copied().unwrap_or(true);
        let cross = d0.x * d1.y - d0.y * d1.x;
        if cross.abs() < 1e-6 {
            let dot = d0.x * d1.x + d0.y * d1.y;
            if dot > 0.0 {
                continue; // straight continuation, nothing to fill
            }
            // 180° reversal. FreeType semantics: inside a flattened CURVE the
            // tip is rounded regardless of the join. At an AUTHORED corner the
            // configured join applies — BUT the miter limit is unconditionally
            // exceeded at a reversal (theta=pi -> thcos=cos(90 deg)=0 ->
            // sigma = miter_limit*0 = 0 < 1), so FreeType truncates the miter
            // to a stub of length ~radius (`hw`), NEVER a miter_limit*hw needle.
            // Planting the full needle here drew long spikes off sharp authored
            // corners (CookieAndMilky sunglasses; 732 stray px at f39).
            if !is_anchor || matches!(join, Join::Round) {
                let n = Vec2::new(-d0.y * hw, d0.x * hw);
                let a0 = n.y.atan2(n.x);
                let ad = d0.y.atan2(d0.x);
                let mut a1 = a0 - core::f32::consts::PI;
                let mid = (a0 + a1) * 0.5;
                if angle_diff(mid, ad).abs() > core::f32::consts::FRAC_PI_2 {
                    a1 = a0 + core::f32::consts::PI;
                }
                let mut semi: Vec<Vec2> = pool.pop().unwrap_or_default();
                semi.clear();
                arc_points(&mut semi, here, hw, a0, a1);
                out.push(normalized(semi));
            } else if matches!(join, Join::Miter) {
                let n0 = Vec2::new(-d0.y * hw, d0.x * hw);
                let tip = Vec2::new(here.x + d0.x * hw, here.y + d0.y * hw);
                out.push(normalized(piece![
                    Vec2::new(here.x + n0.x, here.y + n0.y),
                    tip,
                    Vec2::new(here.x - n0.x, here.y - n0.y),
                ]));
            }
            continue;
        }
        // Outer side normals (side away from the turn).
        let (n0, n1) = if cross > 0.0 {
            (
                Vec2::new(d0.y * hw, -d0.x * hw),
                Vec2::new(d1.y * hw, -d1.x * hw),
            )
        } else {
            (
                Vec2::new(-d0.y * hw, d0.x * hw),
                Vec2::new(-d1.y * hw, d1.x * hw),
            )
        };
        let a = Vec2::new(here.x + n0.x, here.y + n0.y);
        let b = Vec2::new(here.x + n1.x, here.y + n1.y);
        match join {
            Join::Bevel => out.push(normalized(piece![here, a, b])),
            Join::Round => {
                let mut wedge = piece![here];
                let a0 = (a.y - here.y).atan2(a.x - here.x);
                let mut a1 = (b.y - here.y).atan2(b.x - here.x);
                // Take the short way around (join wedge is < 180°).
                while a1 - a0 > core::f32::consts::PI {
                    a1 -= 2.0 * core::f32::consts::PI;
                }
                while a0 - a1 > core::f32::consts::PI {
                    a1 += 2.0 * core::f32::consts::PI;
                }
                arc_points(&mut wedge, here, hw, a0, a1);
                out.push(normalized(wedge));
            }
            Join::Miter => {
                // Miter point: intersection of the two offset lines. With φ
                // the turn angle between segment directions (interior angle
                // θ = π − φ), miter length = hw / sin(θ/2) = hw / cos(φ/2).
                let dot = d0.x * d1.x + d0.y * d1.y;
                let cos_phi = dot.clamp(-1.0, 1.0);
                // Curve-interior cusp guard: a sharp turn between two
                // flattening-artifact segments much shorter than the stroke
                // width is a cusp INSIDE a curve, which FreeType rounds —
                // a miter here plants a spike hw/cos(φ/2) long (FroggoInLove
                // blink wedge). Real corners have real-length segments.
                let l0 = {
                    let (dx, dy) = (here.x - prev.x, here.y - prev.y);
                    (dx * dx + dy * dy).sqrt()
                };
                let l1 = {
                    let (dx, dy) = (next.x - here.x, next.y - here.y);
                    (dx * dx + dy * dy).sqrt()
                };
                if !is_anchor && cos_phi < 0.5 && l0.min(l1) < hw {
                    // Quantization-scale micro-segment (< 1/64 px, FreeType's
                    // 26.6 grid): rlottie never sees it — its trim/dash cut
                    // the bezier, and a lineto whose 26.6 delta rounds to
                    // zero is a no-op (v_ft_stroker.cpp:1113). Emitting the
                    // round wedge here plants a spurious cap-sized disc at
                    // trim seams (RestrictedEmoji rainbow). Emit nothing:
                    // FT's over-limit fallback at a near-reversal line join
                    // is a bevel, and the adjoining segment rectangles
                    // already cover it.
                    if l0.min(l1) < 1.0 / 64.0 {
                        continue;
                    }
                    let mut wedge = piece![here];
                    let a0 = (a.y - here.y).atan2(a.x - here.x);
                    let mut a1 = (b.y - here.y).atan2(b.x - here.x);
                    while a1 - a0 > core::f32::consts::PI {
                        a1 -= 2.0 * core::f32::consts::PI;
                    }
                    while a0 - a1 > core::f32::consts::PI {
                        a1 += 2.0 * core::f32::consts::PI;
                    }
                    arc_points(&mut wedge, here, hw, a0, a1);
                    out.push(normalized(wedge));
                    continue;
                }
                let cos_half_sq = (1.0 + cos_phi) * 0.5;
                let cos_half = cos_half_sq.max(1e-12).sqrt();
                let miter_len = hw / cos_half;
                if miter_len <= miter_limit * hw {
                    // Direction of the miter: bisector of outer normals.
                    if let Some(bis) = norm(Vec2::new(n0.x + n1.x, n0.y + n1.y)) {
                        let mp = Vec2::new(here.x + bis.x * miter_len, here.y + bis.y * miter_len);
                        out.push(normalized(piece![here, a, mp, b]));
                    } else {
                        out.push(normalized(piece![here, a, b]));
                    }
                } else {
                    out.push(normalized(piece![here, a, b]));
                }
            }
        }
    }

    // Caps on open paths.
    if !closed {
        let ends = [
            (pts.first().copied(), pts.get(1).copied()),
            (
                pts.last().copied(),
                pts.get(pts.len().wrapping_sub(2)).copied(),
            ),
        ];
        for (end, inner) in ends {
            let (Some(end), Some(inner)) = (end, inner) else {
                continue;
            };
            let Some(d) = norm(Vec2::new(end.x - inner.x, end.y - inner.y)) else {
                continue;
            };
            let n = Vec2::new(-d.y * hw, d.x * hw);
            match cap {
                Cap::Butt => {}
                Cap::Square => {
                    let e = Vec2::new(end.x + d.x * hw, end.y + d.y * hw);
                    out.push(normalized(piece![
                        Vec2::new(end.x + n.x, end.y + n.y),
                        Vec2::new(e.x + n.x, e.y + n.y),
                        Vec2::new(e.x - n.x, e.y - n.y),
                        Vec2::new(end.x - n.x, end.y - n.y),
                    ]));
                }
                Cap::Round => {
                    // Semicircle bulging in direction d: from +n to -n.
                    let a0 = n.y.atan2(n.x);
                    let ad = d.y.atan2(d.x);
                    // Choose sweep passing through d's angle.
                    let mut a1 = a0 - core::f32::consts::PI;
                    let mid = (a0 + a1) * 0.5;
                    let diff = angle_diff(mid, ad);
                    if diff.abs() > core::f32::consts::FRAC_PI_2 {
                        a1 = a0 + core::f32::consts::PI;
                    }
                    let mut semi: Vec<Vec2> = pool.pop().unwrap_or_default();
                    semi.clear();
                    arc_points(&mut semi, end, hw, a0, a1);
                    out.push(normalized(semi));
                }
            }
        }
    }

    stroke_stat(4, out.len());
    out
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > core::f32::consts::PI {
        d -= 2.0 * core::f32::consts::PI;
    }
    while d < -core::f32::consts::PI {
        d += 2.0 * core::f32::consts::PI;
    }
    d
}

/// Emits the cap arc/segment(s) at `end` facing direction `d` (unit),
/// appending boundary points from offset +n to -n.
fn cap_points(outp: &mut Vec<Vec2>, end: Vec2, d: Vec2, hw: f32, cap: Cap) {
    let n = Vec2::new(-d.y * hw, d.x * hw);
    match cap {
        Cap::Butt => {}
        Cap::Square => {
            let e = Vec2::new(end.x + d.x * hw, end.y + d.y * hw);
            outp.push(Vec2::new(e.x + n.x, e.y + n.y));
            outp.push(Vec2::new(e.x - n.x, e.y - n.y));
        }
        Cap::Round => {
            let a0 = n.y.atan2(n.x);
            let ad = d.y.atan2(d.x);
            let mut a1 = a0 - core::f32::consts::PI;
            let mid = (a0 + a1) * 0.5;
            if angle_diff(mid, ad).abs() > core::f32::consts::FRAC_PI_2 {
                a1 = a0 + core::f32::consts::PI;
            }
            // Interior arc points only (endpoints ±n are pushed by the
            // side walkers).
            let sweep = a1 - a0;
            let steps = ((sweep.abs() / 0.35).ceil() as usize).clamp(1, 24);
            outp.reserve(steps.saturating_sub(1));
            for i in 1..steps {
                let a = a0 + sweep * (i as f32 / steps as f32);
                outp.push(Vec2::new(end.x + hw * a.cos(), end.y + hw * a.sin()));
            }
        }
    }
}

/// One-polygon outline of an OPEN polyline stroke. Returns None when any
/// vertex needs the tuned special-case handling of the piece path.
#[allow(clippy::too_many_arguments)]
fn outline_open(
    pts: &[Vec2],
    anc: &[bool],
    hw: f32,
    cap: Cap,
    join: Join,
    miter_limit: f32,
    pool: &mut Vec<Vec<Vec2>>,
) -> Option<Contour> {
    let n = pts.len();
    if n < 2 {
        return None;
    }
    // Pre-scan: all segment directions must normalize, and no reversal /
    // near-reversal vertices (those take FreeType-tuned piece handling).
    let mut dirs: Vec<Vec2> = Vec::with_capacity(n - 1);
    for s in 0..n - 1 {
        let (p0, p1) = (*pts.get(s)?, *pts.get(s + 1)?);
        dirs.push(norm(Vec2::new(p1.x - p0.x, p1.y - p0.y))?);
    }
    for v in 1..n - 1 {
        let (d0, d1) = (*dirs.get(v - 1)?, *dirs.get(v)?);
        let cross = d0.x * d1.y - d0.y * d1.x;
        let dot = d0.x * d1.x + d0.y * d1.y;
        if cross.abs() < 1e-6 && dot < 0.0 {
            return None; // reversal: piece path (FreeType-tuned handling)
        }
        // Curve-interior cusp (the round-wedge guard): the wedge covers a
        // disc sector that a single-sided outline join cannot represent —
        // and at cross≈0 the outer-side choice is numerically unstable
        // (FroggoInLove blink: worst-region 80.8 before this fallback).
        // Sharp corners (>90° turn) keep the FreeType-tuned piece handling:
        // over-limit miters, authored near-reversal needles (GameEmoji
        // impact stars: 9.1 worst-region under outline bevels) and the
        // curve-interior cusp wedge all live past this threshold.
        if dot < 0.0 {
            return None;
        }
        let is_anchor = anc.get(v).copied().unwrap_or(true);
        if !is_anchor && dot.clamp(-1.0, 1.0) < 0.5 {
            let l0 = {
                let (a, b) = (*pts.get(v - 1)?, *pts.get(v)?);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                (dx * dx + dy * dy).sqrt()
            };
            let l1 = {
                let (a, b) = (*pts.get(v)?, *pts.get(v + 1)?);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                (dx * dx + dy * dy).sqrt()
            };
            if l0.min(l1) < hw {
                return None;
            }
        }
    }

    let mut b: Vec<Vec2> = pool.pop().unwrap_or_default();
    b.clear();
    b.reserve(4 * n + 16);

    // Walk one side: `side` = +1 emits the left offsets forward, -1 emits
    // the right offsets backward. Join geometry mirrors the piece path:
    // outer side gets miter point / arc / bevel (nothing extra), inner side
    // connects directly (nonzero winding covers the overlap).
    let mut side_pass = |b: &mut Vec<Vec2>, side: f32| -> Option<()> {
        let seg_iter: Box<dyn Iterator<Item = usize>> = if side > 0.0 {
            Box::new(0..n - 1)
        } else {
            Box::new((0..n - 1).rev())
        };
        for s in seg_iter {
            let d = *dirs.get(s)?;
            let nn = Vec2::new(-d.y * hw * side, d.x * hw * side);
            let (p0, p1) = if side > 0.0 {
                (*pts.get(s)?, *pts.get(s + 1)?)
            } else {
                (*pts.get(s + 1)?, *pts.get(s)?)
            };
            b.push(Vec2::new(p0.x + nn.x, p0.y + nn.y));
            b.push(Vec2::new(p1.x + nn.x, p1.y + nn.y));
            // Join at the vertex AHEAD of this segment on this walking
            // direction (skip after the last segment of the pass).
            let (vprev, vnext) = if side > 0.0 {
                if s + 1 >= n - 1 {
                    continue;
                }
                (s, s + 1)
            } else {
                if s == 0 {
                    continue;
                }
                (s, s - 1)
            };
            let here = *pts.get(if side > 0.0 { s + 1 } else { s })?;
            let (d0, d1) = if side > 0.0 {
                (*dirs.get(vprev)?, *dirs.get(vnext)?)
            } else {
                // walking backward: directions reverse
                (
                    Vec2::new(-dirs.get(vprev)?.x, -dirs.get(vprev)?.y),
                    Vec2::new(-dirs.get(vnext)?.x, -dirs.get(vnext)?.y),
                )
            };
            let cross = d0.x * d1.y - d0.y * d1.x;
            let dot = d0.x * d1.x + d0.y * d1.y;
            if cross.abs() < 1e-6 && dot > 0.0 {
                continue; // straight: offsets already meet
            }
            let outer_is_this_side = cross < 0.0; // turn right => left side outer
            if !outer_is_this_side {
                // Inner side: offset-line intersection (zero-area needle;
                // see outline_closed) instead of a corner-cutting chord.
                let cos_half = (((1.0 + dot.clamp(-1.0, 1.0)) * 0.5).max(1e-12)).sqrt();
                let nn0 = Vec2::new(-d0.y * hw, d0.x * hw);
                let nn1 = Vec2::new(-d1.y * hw, d1.x * hw);
                if let Some(bis) = norm(Vec2::new(nn0.x + nn1.x, nn0.y + nn1.y)) {
                    let m = hw / cos_half;
                    if m <= miter_limit.max(1.5) * hw {
                        b.push(Vec2::new(here.x + bis.x * m, here.y + bis.y * m));
                    }
                }
                continue;
            }
            let idx = if side > 0.0 { s + 1 } else { s };
            let is_anchor = anc.get(idx).copied().unwrap_or(true);
            let n0 = Vec2::new(-d0.y * hw, d0.x * hw);
            let n1 = Vec2::new(-d1.y * hw, d1.x * hw);
            let a = Vec2::new(here.x + n0.x, here.y + n0.y);
            let bb = Vec2::new(here.x + n1.x, here.y + n1.y);
            let cos_phi = dot.clamp(-1.0, 1.0);
            let l0 = {
                let p = *pts.get(idx.checked_sub(1)?)?;
                let (dx, dy) = (here.x - p.x, here.y - p.y);
                (dx * dx + dy * dy).sqrt()
            };
            let l1 = {
                let p = *pts.get(idx + 1)?;
                let (dx, dy) = (here.x - p.x, here.y - p.y);
                (dx * dx + dy * dy).sqrt()
            };
            let round_arc = |b: &mut Vec<Vec2>| {
                let a0 = (a.y - here.y).atan2(a.x - here.x);
                let mut a1 = (bb.y - here.y).atan2(bb.x - here.x);
                while a1 - a0 > core::f32::consts::PI {
                    a1 -= 2.0 * core::f32::consts::PI;
                }
                while a0 - a1 > core::f32::consts::PI {
                    a1 += 2.0 * core::f32::consts::PI;
                }
                let sweep = a1 - a0;
                let steps = ((sweep.abs() / 0.35).ceil() as usize).clamp(1, 24);
                for i in 1..steps {
                    let ang = a0 + sweep * (i as f32 / steps as f32);
                    b.push(Vec2::new(here.x + hw * ang.cos(), here.y + hw * ang.sin()));
                }
            };
            match join {
                Join::Round => round_arc(b),
                Join::Bevel => {}
                Join::Miter => {
                    if !is_anchor && cos_phi < 0.5 && l0.min(l1) < hw {
                        // curve-interior cusp: round (piece-path parity),
                        // except quantization-scale stubs which emit nothing.
                        if l0.min(l1) >= 1.0 / 64.0 {
                            round_arc(b);
                        }
                    } else {
                        let cos_half = (((1.0 + cos_phi) * 0.5).max(1e-12)).sqrt();
                        let miter_len = hw / cos_half;
                        if miter_len <= miter_limit * hw {
                            // outward normals bisector (outer side of turn)
                            if let Some(bis) = norm(Vec2::new(n0.x + n1.x, n0.y + n1.y)) {
                                b.push(Vec2::new(
                                    here.x + bis.x * miter_len,
                                    here.y + bis.y * miter_len,
                                ));
                            }
                        }
                    }
                }
            }
        }
        Some(())
    };

    side_pass(&mut b, 1.0)?;
    let d_end = *dirs.last()?;
    cap_points(&mut b, *pts.last()?, d_end, hw, cap);
    side_pass(&mut b, -1.0)?;
    let d_start = *dirs.first()?;
    cap_points(
        &mut b,
        *pts.first()?,
        Vec2::new(-d_start.x, -d_start.y),
        hw,
        cap,
    );
    // Same CCW normalization as every piece: mixed winding signs between
    // overlapping contours of one paint would cancel under nonzero fill.
    Some(normalized(b))
}

/// Ring-pair outline of a CLOSED polyline stroke (solo paints only — see
/// call site). Forward-left and backward-right rings wind oppositely; the
/// nonzero fill of the pair is exactly the annulus. Same fallbacks as the
/// open outline (reversals, sharp corners, curve cusps).
fn outline_closed(
    pts: &[Vec2],
    anc: &[bool],
    hw: f32,
    join: Join,
    miter_limit: f32,
    pool: &mut Vec<Vec<Vec2>>,
) -> Option<Vec<Contour>> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    // Inset-collapse guard: when the shape is anywhere thinner than the
    // stroke, the inner ring self-inverts and its negative winding punches
    // artifacts (HalloUtya/FinanceEmoji morph frames hit 74/43 worst-region
    // before this). Bbox thinness is a cheap conservative proxy; necked
    // shapes bigger than the box bound still fall back via the corpus'
    // piece path only if they trip other guards — accepted residual risk
    // is bounded by the consensus gate.
    {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        if (x1 - x0).min(y1 - y0) < 4.0 * hw {
            return None;
        }
    }
    let mut dirs: Vec<Vec2> = Vec::with_capacity(n);
    for s in 0..n {
        let (p0, p1) = (*pts.get(s)?, *pts.get((s + 1) % n)?);
        dirs.push(norm(Vec2::new(p1.x - p0.x, p1.y - p0.y))?);
    }
    for v in 0..n {
        let d0 = *dirs.get((v + n - 1) % n)?;
        let d1 = *dirs.get(v)?;
        let dot = d0.x * d1.x + d0.y * d1.y;
        if dot < 0.0 {
            return None; // sharp corner / reversal: piece path
        }
        let is_anchor = anc.get(v).copied().unwrap_or(true);
        if !is_anchor && dot.clamp(-1.0, 1.0) < 0.5 {
            let l0 = {
                let (a, b) = (*pts.get((v + n - 1) % n)?, *pts.get(v)?);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                (dx * dx + dy * dy).sqrt()
            };
            let l1 = {
                let (a, b) = (*pts.get(v)?, *pts.get((v + 1) % n)?);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                (dx * dx + dy * dy).sqrt()
            };
            if l0.min(l1) < hw {
                return None;
            }
        }
    }

    let mut ring = |side: f32| -> Option<Vec<Vec2>> {
        let mut b: Vec<Vec2> = pool.pop().unwrap_or_default();
        b.clear();
        b.reserve(3 * n + 8);
        let order: Box<dyn Iterator<Item = usize>> = if side > 0.0 {
            Box::new(0..n)
        } else {
            Box::new((0..n).rev())
        };
        for s in order {
            let d = *dirs.get(s)?;
            let nn = Vec2::new(-d.y * hw * side, d.x * hw * side);
            let (i0, i1) = (s, (s + 1) % n);
            let (p0, p1) = if side > 0.0 {
                (*pts.get(i0)?, *pts.get(i1)?)
            } else {
                (*pts.get(i1)?, *pts.get(i0)?)
            };
            b.push(Vec2::new(p0.x + nn.x, p0.y + nn.y));
            b.push(Vec2::new(p1.x + nn.x, p1.y + nn.y));
            let v = if side > 0.0 { (s + 1) % n } else { s };
            let here = *pts.get(v)?;
            let (mut d0, mut d1) = (*dirs.get((v + n - 1) % n)?, *dirs.get(v)?);
            if side < 0.0 {
                let (r0, r1) = (Vec2::new(-d1.x, -d1.y), Vec2::new(-d0.x, -d0.y));
                d0 = r0;
                d1 = r1;
            }
            let cross = d0.x * d1.y - d0.y * d1.x;
            let dot = d0.x * d1.x + d0.y * d1.y;
            if cross.abs() < 1e-6 && dot > 0.0 {
                continue;
            }
            if !(cross < 0.0) {
                // Inner side: capped offset-line intersection (zero-area
                // needle; see outline_open's inside join).
                let cos_half = (((1.0 + dot.clamp(-1.0, 1.0)) * 0.5).max(1e-12)).sqrt();
                let nn0 = Vec2::new(-d0.y * hw, d0.x * hw);
                let nn1 = Vec2::new(-d1.y * hw, d1.x * hw);
                if let Some(bis) = norm(Vec2::new(nn0.x + nn1.x, nn0.y + nn1.y)) {
                    let m = hw / cos_half;
                    if m <= miter_limit.max(1.5) * hw {
                        b.push(Vec2::new(here.x + bis.x * m, here.y + bis.y * m));
                    }
                }
                continue;
            }
            let n0 = Vec2::new(-d0.y * hw, d0.x * hw);
            let n1 = Vec2::new(-d1.y * hw, d1.x * hw);
            let a = Vec2::new(here.x + n0.x, here.y + n0.y);
            let bb = Vec2::new(here.x + n1.x, here.y + n1.y);
            let cos_phi = dot.clamp(-1.0, 1.0);
            match join {
                Join::Round => {
                    let a0 = (a.y - here.y).atan2(a.x - here.x);
                    let mut a1 = (bb.y - here.y).atan2(bb.x - here.x);
                    while a1 - a0 > core::f32::consts::PI {
                        a1 -= 2.0 * core::f32::consts::PI;
                    }
                    while a0 - a1 > core::f32::consts::PI {
                        a1 += 2.0 * core::f32::consts::PI;
                    }
                    let sweep = a1 - a0;
                    let steps = ((sweep.abs() / 0.35).ceil() as usize).clamp(1, 24);
                    for i in 1..steps {
                        let ang = a0 + sweep * (i as f32 / steps as f32);
                        b.push(Vec2::new(here.x + hw * ang.cos(), here.y + hw * ang.sin()));
                    }
                }
                Join::Bevel => {}
                Join::Miter => {
                    let cos_half = (((1.0 + cos_phi) * 0.5).max(1e-12)).sqrt();
                    let miter_len = hw / cos_half;
                    if miter_len <= miter_limit * hw {
                        if let Some(bis) = norm(Vec2::new(n0.x + n1.x, n0.y + n1.y)) {
                            b.push(Vec2::new(
                                here.x + bis.x * miter_len,
                                here.y + bis.y * miter_len,
                            ));
                        }
                    }
                }
            }
        }
        Some(b)
    };
    let r1 = ring(1.0)?;
    let r2 = ring(-1.0)?;
    Some(vec![
        Contour {
            points: r1,
            anchors: Vec::new(),
            inv_lin: None,
        },
        Contour {
            points: r2,
            anchors: Vec::new(),
            inv_lin: None,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test shim: stroke with a throwaway pool.
    #[allow(clippy::too_many_arguments)]
    fn stroke_polyline_t(
        points: &[Vec2],
        anchors: &[bool],
        closed: bool,
        hw: f32,
        cap: Cap,
        join: Join,
        miter_limit: f32,
    ) -> Vec<Contour> {
        stroke_polyline(
            points,
            anchors,
            closed,
            hw,
            cap,
            join,
            miter_limit,
            &mut Vec::new(),
            false,
        )
    }

    #[test]
    fn straight_segment_area() {
        // Horizontal line length 10, width 4 => area 40.
        let pieces = stroke_polyline_t(
            &[Vec2::new(5.0, 10.0), Vec2::new(15.0, 10.0)],
            &[],
            false,
            2.0,
            Cap::Butt,
            Join::Miter,
            4.0,
        );
        assert_eq!(pieces.len(), 1);
        let area: f32 = pieces
            .iter()
            .map(|c| {
                let mut a = 0.0;
                for (i, p) in c.points.iter().enumerate() {
                    let q = c.points.get(i + 1).or_else(|| c.points.first()).unwrap();
                    a += p.x * q.y - q.x * p.y;
                }
                a * 0.5
            })
            .sum();
        assert!((area - 40.0).abs() < 0.1, "area={area}");
    }

    #[test]
    fn closed_square_stroke_produces_ring_pair() {
        let square = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        // FT-style stroker: a closed contour is exactly two rings with
        // canonical winding (band +1, hole -1) regardless of `solo` —
        // the sign invariant is structural (stroke_ft.rs), unlike the
        // reverted input-direction-dependent ring pairs.
        let rings = stroke_polyline_t(&square, &[], true, 1.0, Cap::Butt, Join::Miter, 4.0);
        assert_eq!(rings.len(), 2);
    }

    #[test]
    fn solo_closed_square_is_ring_pair() {
        let square = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let rings = stroke_polyline(
            &square,
            &[],
            true,
            1.0,
            Cap::Butt,
            Join::Miter,
            4.0,
            &mut Vec::new(),
            true,
        );
        assert_eq!(rings.len(), 2);
        let signed = |c: &Contour| {
            let mut a = 0.0f32;
            for (i, p) in c.points.iter().enumerate() {
                let q = c.points.get(i + 1).or_else(|| c.points.first()).unwrap();
                a += p.x * q.y - q.x * p.y;
            }
            a * 0.5
        };
        let (a0, a1) = (signed(&rings[0]), signed(&rings[1]));
        assert!(a0 * a1 < 0.0, "rings must wind oppositely: {a0} {a1}");
        assert!(
            ((a0 + a1).abs() - 80.0).abs() < 0.5,
            "annulus area: {}",
            a0 + a1
        );
    }

    #[test]
    fn degenerate_single_vertex_draws_nothing() {
        // rlottie's stroker emits nothing for a segment-less contour.
        let pieces = stroke_polyline_t(
            &[Vec2::new(3.0, 3.0)],
            &[],
            false,
            2.0,
            Cap::Round,
            Join::Round,
            4.0,
        );
        assert!(pieces.is_empty());
    }

    #[test]
    fn no_panic_on_garbage() {
        let garbage = [
            Vec2::new(f32::NAN, 0.0),
            Vec2::new(0.0, f32::INFINITY),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];
        let _ = stroke_polyline_t(&garbage, &[], true, 2.0, Cap::Round, Join::Round, 4.0);
        let _ = stroke_polyline_t(
            &garbage,
            &[],
            false,
            f32::NAN,
            Cap::Square,
            Join::Miter,
            0.0,
        );
    }
}
