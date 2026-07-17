//! FreeType-style two-border stroker (port of rlottie's vendored
//! `v_ft_stroker.cpp` semantics — see scratchpad STROKER_DESIGN.md for the
//! line-by-line mapping). One stroked subpath emits ONE closed outline
//! (open input) or TWO rings (closed input) whose nonzero union is the
//! stroke band — replacing the legacy union-of-pieces construction whose
//! criss-cross interior edges dominated 320/720px rasterization cost.
//!
//! The load-bearing property (VFT:1663-1664): closed subpaths always close
//! border 0 unreversed and border 1 reversed. Which border is geometrically
//! OUTER flips with the input direction, so the two flips compose into an
//! invariant — every band carries the same winding sign regardless of
//! authored path direction, and an inner ring's opposite sign only cancels
//! its own outer ring (containment), never a sibling's band. This is the
//! exact property both historical ring-pair attempts lacked (they tied ring
//! signs to input direction; opposite-direction siblings cancelled).
//!
//! At inside corners the border either insets to the offset-line
//! intersection (both neighbors straight and long enough, VFT:846-871) or
//! keeps the incoming end offset and appends the outgoing start offset
//! (VFT:874-894): the connecting chord doubles back across the band, so
//! self-overlap accumulates to |winding| >= 2, never 0.
//!
//! Output rings are NOT area-normalized — winding is semantic here. One
//! unconditional orientation flip at export makes bands +1 (the legacy
//! `normalized()` piece sign) so both stroker implementations compose
//! identically with the rasterizers' |sum| fill rule.

use crate::geometry::Contour;
use crate::math::Vec2;
use crate::stroke::{Cap, Join};

/// Tangent-turn threshold at non-anchor (curve-sample) vertices: below it
/// the vertex is a flattening artifact of a smooth curve (no join at all);
/// above it FT inserts a ROUND corner regardless of the configured join
/// (VFT:1385-1396, SW_FT_SMALL_CUBIC_THRESHOLD/4 = pi/32).
const COS_SMOOTH: f32 = 0.995_184_7; // cos(pi/32)

/// Straight-continuation epsilon on the turn cross product (matches the
/// legacy stroker's collinearity test).
const EPS_TURN: f32 = 1e-6;

/// Sagitta tolerance for polygonal join/cap arcs, in device px. rlottie
/// emits exact cubic arcs (flattened by its raster); a fixed 0.35-rad step
/// undercuts by 0.015*hw (0.5px at hw=32), so the step shrinks with hw.
const ARC_TOL: f32 = 0.05;

struct Border {
    pts: Vec<Vec2>,
    /// FT `movable` (VFT:392-423): the last point is a straight-segment
    /// end offset that the next corner may REPLACE instead of append.
    movable: bool,
}

impl Border {
    fn new(pts: Vec<Vec2>) -> Border {
        Border {
            pts,
            movable: false,
        }
    }

    /// FT border_lineto: replace-if-movable, else append (with the 1/32px
    /// dedupe of VFT:404-407); then adopt `movable`. CRITICAL: a deduped
    /// (skipped) lineto returns BEFORE the flag update (VFT:407 `return`),
    /// leaving the border non-movable — updating the flag on the skip
    /// path let the next lineto REPLACE a point that was never emitted
    /// (measured: ate a notch wall on GameEmoji's slit glyphs, a
    /// 2,217px phantom-coverage needle).
    fn line_to(&mut self, p: Vec2, movable: bool) {
        if self.movable {
            if let Some(last) = self.pts.last_mut() {
                *last = p;
            }
        } else {
            let dup = self
                .pts
                .last()
                .is_some_and(|q| (q.x - p.x).abs() < 0.03125 && (q.y - p.y).abs() < 0.03125);
            if dup {
                return;
            }
            self.pts.push(p);
        }
        self.movable = movable;
    }

    /// Polygonal arc around `center`, radius `r`, from `a0` sweeping
    /// `sweep` radians; the start point is assumed already present (it is
    /// the incoming segment's end offset), so emission starts at step 1.
    fn arc_to(&mut self, center: Vec2, r: f32, a0: f32, sweep: f32) {
        self.movable = false;
        let step = (8.0 * ARC_TOL / r.max(1e-3)).sqrt().clamp(0.0655, 0.35);
        let steps = ((sweep.abs() / step).ceil() as usize).clamp(1, 48);
        self.pts.reserve(steps);
        for i in 1..=steps {
            let a = a0 + sweep * (i as f32 / steps as f32);
            self.pts
                .push(Vec2::new(center.x + r * a.cos(), center.y + r * a.sin()));
        }
    }
}

/// One cleaned polyline segment.
struct Seg {
    /// Segment start point.
    p: Vec2,
    /// Unit direction.
    d: Vec2,
    /// Euclidean length.
    len: f32,
    /// FT "line status": true when both endpoints are authored anchors —
    /// the only case the inside-corner intersection optimization applies
    /// (curve chords behave as FT curve pieces, line_length = 0).
    is_line: bool,
    /// Source index of `p` (for the start vertex's anchor flag).
    start_idx: usize,
    /// A sub-1/64px segment was dropped immediately before this one: the
    /// corner spanning the gap is a trim/authoring seam that rlottie's
    /// 26.6 grid never sees — it joins the real segments with the
    /// CONFIGURED join (never the inserted cusp round; RestrictedEmoji
    /// rainbow fold tips are flat over-limit bevels in rlottie).
    gap_before: bool,
}

/// Strokes one flattened, deduplicated subpath (caller: `stroke_polyline`
/// after its sanitizer). Appends 1 contour (open) or up to 2 rings
/// (closed) to `out`. Winding: bands +1, closed inner rings -1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stroke_outline(
    pts: &[Vec2],
    anchors: &[bool],
    closed: bool,
    hw: f32,
    cap: Cap,
    join: Join,
    miter_limit: f32,
    pool: &mut Vec<Vec<Vec2>>,
    out: &mut Vec<Contour>,
) {
    let ml = if miter_limit.is_finite() {
        miter_limit.max(1.0)
    } else {
        4.0
    };
    let anchor_at = |i: usize| anchors.get(i).copied().unwrap_or(true);

    // Segment list; degenerate directions (post-dedupe pathologies) are
    // dropped, merging their neighbors.
    let n = pts.len();
    let seg_count = if closed { n } else { n.saturating_sub(1) };
    let mut segs: Vec<Seg> = Vec::with_capacity(seg_count);
    let mut dropped_gap = false;
    for s in 0..seg_count {
        let Some(&p0) = pts.get(s) else { continue };
        let j = if s + 1 == n { 0 } else { s + 1 };
        let Some(&p1) = pts.get(j) else { continue };
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        // Sub-1/64px segments are dropped: rlottie's 26.6 conversion
        // truncates them to an exact-zero delta and its LineTo is a no-op
        // (VFT:1113-1114). Font-outline "slit" contours carry 0.001px
        // steps whose noise directions must never reach corner logic.
        if !(len > 0.015625) || !len.is_finite() {
            dropped_gap = true;
            continue;
        }
        segs.push(Seg {
            p: p0,
            d: Vec2::new(dx / len, dy / len),
            len,
            is_line: anchor_at(s) && anchor_at(j),
            start_idx: s,
            gap_before: core::mem::take(&mut dropped_gap),
        });
    }
    // Trailing drops wrap onto the closed seam (or are absorbed by the
    // open end cap).
    let seam_gap = dropped_gap || segs.first().is_some_and(|s| s.gap_before);
    if segs.is_empty() {
        return;
    }

    let mut b0 = Border::new(pool.pop().unwrap_or_default());
    let mut b1 = Border::new(pool.pop().unwrap_or_default());
    b0.pts.clear();
    b1.pts.clear();
    let cap_pts = 2 + (core::f32::consts::PI / 0.0655) as usize;
    b0.pts.reserve(2 * segs.len() + cap_pts + 8);
    b1.pts.reserve(2 * segs.len() + 8);

    // Subpath start: moveto on each border (movable = false).
    let (first_d, first_p) = match segs.first() {
        Some(s) => (s.d, s.p),
        None => return,
    };
    let n0 = Vec2::new(-first_d.y * hw, first_d.x * hw);
    b0.pts.push(Vec2::new(first_p.x + n0.x, first_p.y + n0.y));
    b1.pts.push(Vec2::new(first_p.x - n0.x, first_p.y - n0.y));

    let mut prev_d = first_d;
    let mut prev_len = segs
        .first()
        .map_or(0.0, |s| if s.is_line { s.len } else { 0.0 });
    let mut prev_raw = segs.first().map_or(0.0, |s| s.len);
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            let cur_len = if seg.is_line { seg.len } else { 0.0 };
            let vertex_anchor = anchors.get(seg.start_idx).copied().unwrap_or(true);
            process_corner(
                &mut b0,
                &mut b1,
                seg.p,
                prev_d,
                seg.d,
                prev_len,
                cur_len,
                prev_raw,
                seg.len,
                vertex_anchor,
                seg.gap_before,
                hw,
                join,
                ml,
            );
        }
        // Segment end offsets (movable — the next inside intersection may
        // replace them).
        let q = Vec2::new(seg.p.x + seg.d.x * seg.len, seg.p.y + seg.d.y * seg.len);
        let nn = Vec2::new(-seg.d.y * hw, seg.d.x * hw);
        b0.line_to(Vec2::new(q.x + nn.x, q.y + nn.y), true);
        b1.line_to(Vec2::new(q.x - nn.x, q.y - nn.y), true);
        prev_d = seg.d;
        prev_len = if seg.is_line { seg.len } else { 0.0 };
        prev_raw = seg.len;
    }

    if closed {
        // Seam corner back into the first segment (VFT:1640-1660), then
        // close border 0 unreversed / border 1 reversed (VFT:1663-1664).
        let first_len = segs
            .first()
            .map_or(0.0, |s| if s.is_line { s.len } else { 0.0 });
        let first_raw = segs.first().map_or(0.0, |s| s.len);
        process_corner(
            &mut b0,
            &mut b1,
            first_p,
            prev_d,
            first_d,
            prev_len,
            first_len,
            prev_raw,
            first_raw,
            anchor_at(0),
            seam_gap,
            hw,
            join,
            ml,
        );
        ring_close(&mut b0.pts, false);
        ring_close(&mut b1.pts, true);
        // Canonical flip: FT-natural band winding is opposite the legacy
        // normalized() pieces; reverse both rings so bands land at +1.
        b0.pts.reverse();
        b1.pts.reverse();
        // Fully-inverted inset (shape thinner than the band EVERYWHERE:
        // min bbox dim <= 2*hw): no true hole can exist, but the inset
        // ring still traces an opposite-winding loop that would cancel
        // the disc interior — rlottie renders these FILLED (its inverted
        // offset cubics loop-the-loop and "cover the negative sector",
        // VFT:1527-1531 comment; the piece-era corpus, which cannot make
        // holes, graded CLEAN against it: FinanceEmoji dot discs).
        // Emit the outer (larger-|area|) ring alone in exactly that
        // regime; anything bigger keeps both rings (a broader 2*hw rule
        // measured -182 CLEAN on the corpus).
        let drop_inner = {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for p in pts {
                x0 = x0.min(p.x);
                y0 = y0.min(p.y);
                x1 = x1.max(p.x);
                y1 = y1.max(p.y);
            }
            // ... and ONLY for a genuine annulus pair (opposite ring
            // orientations). Degenerate self-retracing dot paths emit two
            // SAME-sign displaced discs whose union is the disc rlottie
            // draws — dropping one carved half-moons out of NewsEmoji's
            // planets.
            x1 >= x0
                && y1 >= y0
                && (x1 - x0).max(y1 - y0) <= 2.0 * hw
                && ring_area(&b0.pts) * ring_area(&b1.pts) < 0.0
        };
        let keep_b0 = if drop_inner {
            ring_area(&b0.pts).abs() >= ring_area(&b1.pts).abs()
        } else {
            true
        };
        for (ring, keep) in [(b0.pts, keep_b0), (b1.pts, !drop_inner || !keep_b0)] {
            if keep && ring.len() >= 3 {
                out.push(Contour {
                    points: ring,
                    anchors: Vec::new(),
                    inv_lin: None,
                });
            } else {
                pool.push(ring);
            }
        }
    } else {
        // Open assembly (VFT:1602-1628): border0 forward, end cap, border1
        // reversed, start cap — one closed loop.
        let last = match segs.last() {
            Some(s) => s,
            None => return,
        };
        let end_center = Vec2::new(
            last.p.x + last.d.x * last.len,
            last.p.y + last.d.y * last.len,
        );
        let mut loop_pts = b0.pts;
        emit_cap(&mut loop_pts, end_center, last.d, hw, cap);
        loop_pts.extend(b1.pts.iter().rev().copied());
        pool.push(b1.pts);
        emit_cap(
            &mut loop_pts,
            first_p,
            Vec2::new(-first_d.x, -first_d.y),
            hw,
            cap,
        );
        // Canonical flip (see closed case).
        loop_pts.reverse();
        if loop_pts.len() >= 3 {
            out.push(Contour {
                points: loop_pts,
                anchors: Vec::new(),
                inv_lin: None,
            });
        } else {
            pool.push(loop_pts);
        }
    }
}

/// FT process_corner (VFT:1033-1060) + inside (VFT:846-897) + outside
/// (VFT:899-1031) with rlottie's fixed enum choices (MITER_FIXED; conics
/// unreachable). `d_in`/`d_out` unit; `prev_len`/`next_len` are 0 for
/// curve chords (disables the inside intersection, VFT:861-871).
#[allow(clippy::too_many_arguments)]
fn process_corner(
    b0: &mut Border,
    b1: &mut Border,
    p: Vec2,
    d_in: Vec2,
    d_out: Vec2,
    prev_len: f32,
    next_len: f32,
    prev_raw: f32,
    next_raw: f32,
    vertex_anchor: bool,
    gap_before: bool,
    hw: f32,
    join: Join,
    ml: f32,
) {
    let cross = d_in.x * d_out.y - d_in.y * d_out.x;
    let dot = d_in.x * d_out.x + d_in.y * d_out.y;
    if cross.abs() < EPS_TURN && dot >= 0.0 {
        return; // straight continuation
    }
    // Reversal: turn == +pi by convention (VFT:776, 926-928).
    let reversal = cross.abs() < EPS_TURN && dot < 0.0;
    // turn > 0 -> inside is border 0 (VFT:1043-1049).
    let inside_is_b0 = if reversal { true } else { cross > 0.0 };

    let n_in = Vec2::new(-d_in.y, d_in.x);
    let n_out = Vec2::new(-d_out.y, d_out.x);

    // Corner classification (the legacy stroker's corpus-proven rules,
    // stroke.rs:313-352 + 395-411, mapped onto FT semantics):
    // - smooth curve samples get continuation geometry;
    // - a sharp turn between two flattening-scale segments (min raw len
    //   < hw) at a non-anchor vertex is a curve-interior CUSP — FT's
    //   inserted ROUND (VFT:1387-1396);
    // - a sharp non-anchor turn between REAL-length segments is a path
    //   corner whose anchors were lost (trim splices) — configured join;
    // - a corner spanning a dropped sub-1/64px segment is a seam rlottie
    //   never sees: configured join (over-limit miters go FLAT — the
    //   RestrictedEmoji rainbow fold tips).
    let smooth = !vertex_anchor && dot >= COS_SMOOTH && !reversal && !gap_before;
    let cusp = !vertex_anchor && !gap_before && dot < COS_SMOOTH && prev_raw.min(next_raw) < hw;
    let eff_join = if cusp { Join::Round } else { join };

    let (si, so) = if inside_is_b0 {
        (1.0f32, -1.0f32)
    } else {
        (-1.0, 1.0)
    };

    // ---- inside border ----
    {
        let bi: &mut Border = if inside_is_b0 { b0 } else { b1 };
        // Intersection needs both neighbors straight and long enough:
        // len >= |hw tan(theta)| with theta = half turn (VFT:861-871).
        let denom = 1.0 + dot;
        let can_intersect =
            !reversal && denom > 1e-6 && bi.movable && prev_len > 0.0 && next_len > 0.0 && {
                let t = (hw * cross / denom).abs();
                prev_len >= t && next_len >= t
            };
        if can_intersect {
            let k = hw / denom;
            let ip = Vec2::new(
                p.x + si * (n_in.x + n_out.x) * k,
                p.y + si * (n_in.y + n_out.y) * k,
            );
            bi.line_to(ip, false);
        } else {
            // Double-back: keep incoming end offset, append outgoing start
            // offset (VFT:874-894).
            bi.movable = false;
            bi.line_to(
                Vec2::new(p.x + si * n_out.x * hw, p.y + si * n_out.y * hw),
                false,
            );
        }
    }

    // ---- outside border ----
    {
        let bo: &mut Border = if inside_is_b0 { b1 } else { b0 };
        if smooth {
            // Offset-line intersection: the polyline analogue of FT's
            // translated-control-point bulge (overshoot ~hw*phi^2/8).
            let denom = 1.0 + dot;
            if denom > 1e-6 {
                let k = hw / denom;
                bo.line_to(
                    Vec2::new(
                        p.x + so * (n_in.x + n_out.x) * k,
                        p.y + so * (n_in.y + n_out.y) * k,
                    ),
                    false,
                );
            }
            return;
        }
        match eff_join {
            Join::Round => {
                let start = Vec2::new(so * n_in.x, so * n_in.y);
                let a0 = start.y.atan2(start.x);
                let sweep = if reversal {
                    // VFT:776 — turn == pi resolves to -rotate*2 of the
                    // outside side; in vector terms the semicircle bulges
                    // through the reversal direction.
                    -so * core::f32::consts::PI
                } else {
                    cross.atan2(dot)
                };
                bo.arc_to(p, hw, a0, sweep);
            }
            Join::Bevel | Join::Miter => {
                let denom = 1.0 + dot;
                // Over-limit test: ml * cos(theta/2) < 1, with
                // cos^2(theta/2) = (1+dot)/2 (VFT:934-938). Reversals have
                // denom -> 0: always over-limit -> bevel stub.
                let within =
                    matches!(join, Join::Miter) && denom > 1e-6 && ml * ml * denom * 0.5 >= 1.0;
                if within && matches!(eff_join, Join::Miter) {
                    // Miter tip lies ON the incoming offset line, so the
                    // replace-if-movable lineto is collinear-safe
                    // (VFT:1004-1013 keeps `movable` set).
                    let k = hw / denom;
                    bo.line_to(
                        Vec2::new(
                            p.x + so * (n_in.x + n_out.x) * k,
                            p.y + so * (n_in.y + n_out.y) * k,
                        ),
                        false,
                    );
                    if next_len <= 0.0 {
                        // Curve corner: FT appends the outgoing start
                        // offset too (VFT:1016-1025).
                        bo.line_to(
                            Vec2::new(p.x + so * n_out.x * hw, p.y + so * n_out.y * hw),
                            false,
                        );
                    }
                } else {
                    // Fixed bevel: the chord needs BOTH the incoming end
                    // offset and the outgoing start offset — clear movable
                    // FIRST so the lineto appends instead of replacing
                    // (VFT:957-958; replacing here ate a sliver of the
                    // band — the OutlineEmoji "notch" corpus signature).
                    bo.movable = false;
                    bo.line_to(
                        Vec2::new(p.x + so * n_out.x * hw, p.y + so * n_out.y * hw),
                        false,
                    );
                }
            }
        }
    }
}

/// Signed shoelace area of a ring (inner/outer + orientation-pair
/// classification for the fully-inverted-inset rule).
fn ring_area(pts: &[Vec2]) -> f32 {
    let mut a = 0.0f32;
    for (i, p) in pts.iter().enumerate() {
        let q = pts
            .get(i + 1)
            .or_else(|| pts.first())
            .copied()
            .unwrap_or(*p);
        a += p.x * q.y - q.x * p.y;
    }
    a
}

/// FT border_close (VFT:337-390): the provisional start offset (moveto) is
/// superseded by the final corner point — copy last into slot 0 and drop
/// it; optionally reverse the interior (winding canonicalization).
fn ring_close(pts: &mut Vec<Vec2>, reverse: bool) {
    if pts.len() < 3 {
        pts.clear();
        return;
    }
    if let Some(&last) = pts.last() {
        if let Some(first) = pts.first_mut() {
            *first = last;
        }
        pts.pop();
    }
    if reverse {
        if let Some(interior) = pts.get_mut(1..) {
            interior.reverse();
        }
    }
}

/// Cap boundary from the +n side to the -n side of direction `d` at `end`
/// (legacy cap_points semantics; round caps sagitta-bounded like joins).
fn emit_cap(out: &mut Vec<Vec2>, end: Vec2, d: Vec2, hw: f32, cap: Cap) {
    let n = Vec2::new(-d.y * hw, d.x * hw);
    match cap {
        Cap::Butt => {}
        Cap::Square => {
            let e = Vec2::new(end.x + d.x * hw, end.y + d.y * hw);
            out.push(Vec2::new(e.x + n.x, e.y + n.y));
            out.push(Vec2::new(e.x - n.x, e.y - n.y));
        }
        Cap::Round => {
            // Semicircle from +n to -n bulging through d: sweep sign so the
            // midpoint lands at end + d*hw.
            let a0 = n.y.atan2(n.x);
            let ad = d.y.atan2(d.x);
            let mut sweep = -core::f32::consts::PI;
            let mid = a0 + sweep * 0.5;
            if angle_delta(mid, ad).abs() > core::f32::consts::FRAC_PI_2 {
                sweep = core::f32::consts::PI;
            }
            let step = (8.0 * ARC_TOL / hw.max(1e-3)).sqrt().clamp(0.0655, 0.35);
            let steps = ((sweep.abs() / step).ceil() as usize).clamp(1, 48);
            out.reserve(steps.saturating_sub(1));
            for i in 1..steps {
                let a = a0 + sweep * (i as f32 / steps as f32);
                out.push(Vec2::new(end.x + hw * a.cos(), end.y + hw * a.sin()));
            }
        }
    }
}

#[inline]
fn angle_delta(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > core::f32::consts::PI {
        d -= 2.0 * core::f32::consts::PI;
    }
    while d < -core::f32::consts::PI {
        d += 2.0 * core::f32::consts::PI;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(
        pts: &[(f32, f32)],
        closed: bool,
        hw: f32,
        cap: Cap,
        join: Join,
        ml: f32,
    ) -> Vec<Contour> {
        let v: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
        let anchors = vec![true; v.len()];
        let mut pool = Vec::new();
        let mut out = Vec::new();
        stroke_outline(&v, &anchors, closed, hw, cap, join, ml, &mut pool, &mut out);
        out
    }

    fn shoelace(pts: &[Vec2]) -> f32 {
        let mut area = 0.0f32;
        for (i, p) in pts.iter().enumerate() {
            let q = pts
                .get(i + 1)
                .or_else(|| pts.first())
                .copied()
                .unwrap_or(*p);
            area += p.x * q.y - q.x * p.y;
        }
        area * 0.5
    }

    #[test]
    fn straight_band_area_and_sign() {
        let out = stroke(
            &[(0.0, 0.0), (10.0, 0.0)],
            false,
            2.0,
            Cap::Butt,
            Join::Miter,
            4.0,
        );
        assert_eq!(out.len(), 1);
        let a = shoelace(&out.first().map(|c| c.points.clone()).unwrap_or_default());
        // Band = 10 x 4, canonical sign POSITIVE (legacy piece sign).
        assert!((a - 40.0).abs() < 0.5, "area {a}");
    }

    #[test]
    fn closed_square_rings_sign_invariant() {
        // The design's THE regression test: both input directions must
        // yield outer ring sign == open-band sign (+), inner ring (-).
        let ccw = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let cw = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        for pts in [&ccw[..], &cw[..]] {
            let out = stroke(pts, true, 1.0, Cap::Butt, Join::Miter, 4.0);
            assert_eq!(out.len(), 2, "rings for {pts:?}");
            let mut areas: Vec<f32> = out.iter().map(|c| shoelace(&c.points)).collect();
            areas.sort_by(|x, y| {
                x.abs()
                    .partial_cmp(&y.abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let inner = areas.first().copied().unwrap_or(0.0);
            let outer = areas.last().copied().unwrap_or(0.0);
            assert!(outer > 0.0, "outer ring must be + ({outer} for {pts:?})");
            assert!(inner < 0.0, "inner ring must be - ({inner} for {pts:?})");
            // 12x12 outer minus 8x8 inner.
            assert!((outer - 144.0).abs() < 1.0, "outer {outer}");
            assert!((inner + 64.0).abs() < 1.0, "inner {inner}");
        }
    }

    #[test]
    fn reversal_miter_is_bevel_stub() {
        // A -> B -> A reversal with miter join must NOT emit a
        // miter_limit*hw needle (VFT:926-942).
        let out = stroke(
            &[(0.0, 0.0), (10.0, 0.0), (0.0, 0.001)],
            false,
            2.0,
            Cap::Butt,
            Join::Miter,
            10.0,
        );
        let max_x = out
            .iter()
            .flat_map(|c| c.points.iter())
            .fold(f32::MIN, |m, p| m.max(p.x));
        assert!(max_x <= 12.1, "reversal produced a needle: max_x {max_x}");
    }

    #[test]
    fn over_limit_miter_bevels() {
        // 170 degree turn, limit 4: sec(85deg) ~ 11.5 > 4 -> bevel.
        let out = stroke(
            &[(0.0, 0.0), (10.0, 0.0), (0.17, 0.87)],
            false,
            2.0,
            Cap::Butt,
            Join::Miter,
            4.0,
        );
        let max_x = out
            .iter()
            .flat_map(|c| c.points.iter())
            .fold(f32::MIN, |m, p| m.max(p.x));
        assert!(max_x <= 12.1, "over-limit miter not beveled: max_x {max_x}");
    }

    #[test]
    fn no_panic_garbage() {
        for pts in [
            vec![(0.0f32, 0.0f32)],
            vec![(0.0, 0.0), (0.0, 0.0)],
            vec![(f32::NAN, 0.0), (1.0, 1.0), (2.0, 0.0)],
            vec![(0.0, 0.0), (1e30, 1e30), (-1e30, 1e30)],
        ] {
            let v: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
            let anchors = vec![true; v.len()];
            let mut pool = Vec::new();
            let mut out = Vec::new();
            for closed in [false, true] {
                stroke_outline(
                    &v,
                    &anchors,
                    closed,
                    2.0,
                    Cap::Round,
                    Join::Round,
                    4.0,
                    &mut pool,
                    &mut out,
                );
            }
        }
    }
}

#[cfg(test)]
mod difftests {
    use super::*;
    use crate::model::FillRule;
    use crate::raster::Rasterizer;

    fn plane(contours: &[Contour], w: usize, h: usize) -> Vec<u8> {
        let mut r = Rasterizer::new(w, h);
        r.fill_contours(contours);
        let mut plane = vec![0u8; w * h];
        r.sweep(FillRule::NonZero, true, |y, x0, cov| {
            if let Some(d) = plane.get_mut(y * w + x0..y * w + x0 + cov.len()) {
                d.copy_from_slice(cov);
            }
        });
        plane
    }

    #[test]
    fn corner_sweep_matches_legacy() {
        // Sweep corner angle / half-width / segment length; FT vs legacy
        // piece coverage must agree except AA-level boundary differences
        // (joins differ by design: FT arcs vs legacy wedges are both
        // rlottie-family shapes; miters/bevels are identical polygons).
        let mut worst = (0.0f32, 0usize, String::new());
        for &deg in &[30.0f32, 60.0, 90.0, 120.0, 140.0, 160.0, 175.0] {
            for &hw in &[1.5f32, 4.0, 8.0] {
                for &len in &[6.0f32, 30.0] {
                    // Segments shorter than the band width: the piece
                    // union and the FT construction legitimately differ
                    // (offset points overshoot the neighboring segment).
                    // rlottie runs the FT construction, so the corpus
                    // scan is the oracle in that regime, not legacy.
                    if len < 2.5 * hw {
                        continue;
                    }
                    for &join in &[Join::Miter, Join::Round, Join::Bevel] {
                        let a = deg.to_radians();
                        let p0 = Vec2::new(32.0 - len, 32.0);
                        let p1 = Vec2::new(32.0, 32.0);
                        let p2 = Vec2::new(32.0 - len * a.cos(), 32.0 - len * a.sin());
                        let pts = [p0, p1, p2];
                        let anchors = [true, true, true];
                        let mut pool = Vec::new();
                        let mut ft = Vec::new();
                        stroke_outline(
                            &pts,
                            &anchors,
                            false,
                            hw,
                            Cap::Butt,
                            join,
                            4.0,
                            &mut pool,
                            &mut ft,
                        );
                        let legacy = crate::stroke::stroke_polyline_pieces_for_test(
                            &pts,
                            &anchors,
                            false,
                            hw,
                            Cap::Butt,
                            join,
                            4.0,
                        );
                        let pf = plane(&ft, 64, 64);
                        let pl = plane(&legacy, 64, 64);
                        let mut bad = 0usize;
                        let mut maxd = 0i32;
                        for (a, b) in pf.iter().zip(pl.iter()) {
                            let d = (i32::from(*a) - i32::from(*b)).abs();
                            maxd = maxd.max(d);
                            if d > 128 {
                                bad += 1;
                            }
                        }
                        if bad > worst.1 {
                            worst = (deg, bad, format!("deg={deg} hw={hw} len={len} join={join:?} maxd={maxd} bad={bad}"));
                        }
                    }
                }
            }
        }
        assert!(worst.1 <= 6, "coverage mismatch: {}", worst.2);
    }
}

#[cfg(test)]
mod needle_repro {
    use super::*;
    use crate::model::FillRule;
    use crate::raster::Rasterizer;

    #[test]
    fn planet_ring_mode_s_matches_d() {
        // Same 50-sample circle, FT rings rasterized by BOTH engines:
        // negative-winding rings are the first real mode-S stress
        // (design R7). Reuses the news_emoji input via regeneration.
        let mut pts = Vec::new();
        let (cx, cy, r) = (116.5f32, 116.4f32, 21.4f32);
        for k in 0..50 {
            let a = -(k as f32) * core::f32::consts::TAU / 50.0;
            pts.push(Vec2::new(cx + r * a.cos(), cy + r * a.sin()));
        }
        let anchors = vec![false; 50];
        let mut pool = Vec::new();
        let mut ft = Vec::new();
        stroke_outline(
            &pts,
            &anchors,
            true,
            10.0,
            Cap::Round,
            Join::Round,
            4.0,
            &mut pool,
            &mut ft,
        );
        let (w, h) = (256usize, 256usize);
        let mut rd = Rasterizer::new(w, h);
        rd.fill_contours(&ft);
        let mut pd = vec![0u8; w * h];
        rd.sweep(FillRule::NonZero, true, |y, x0, cov| {
            if let Some(d) = pd.get_mut(y * w + x0..y * w + x0 + cov.len()) {
                d.copy_from_slice(cov);
            }
        });
        let mut rs = crate::cells::CellRaster::new(w, h);
        rs.fill_contours(&ft);
        let mut ps = vec![0u8; w * h];
        rs.sweep_spans(FillRule::NonZero, true, |y, x0, len, cov| {
            if let Some(d) = ps.get_mut(y * w + x0..y * w + x0 + len) {
                d.fill(cov);
            }
        });
        let mut bad = 0usize;
        let mut first = None;
        for (i, (a, b)) in pd.iter().zip(ps.iter()).enumerate() {
            if (i32::from(*a) - i32::from(*b)).abs() > 8 {
                bad += 1;
                if first.is_none() {
                    first = Some((i % w, i / w, *a, *b));
                }
            }
        }
        assert!(
            bad <= 4,
            "mode S vs D on rings: {bad} px differ, first {first:?}"
        );
    }

    #[test]
    fn news_emoji_planet_ring() {
        // 50-sample circle r~21.4, hw=10, all curve samples (non-anchor):
        // NewsEmoji planet — FT half-vanished vs legacy full annulus.
        let pts = [
            Vec2::new(123.149864, 95.782646),
            Vec2::new(120.292564, 95.036507),
            Vec2::new(117.524117, 94.659683),
            Vec2::new(114.85701, 94.634865),
            Vec2::new(112.303802, 94.944786),
            Vec2::new(109.876877, 95.572075),
            Vec2::new(107.588829, 96.499504),
            Vec2::new(105.452057, 97.709702),
            Vec2::new(103.479126, 99.185394),
            Vec2::new(101.68248, 100.909271),
            Vec2::new(100.074646, 102.864044),
            Vec2::new(98.668091, 105.032379),
            Vec2::new(97.475311, 107.396996),
            Vec2::new(96.508804, 109.940567),
            Vec2::new(95.766068, 112.790001),
            Vec2::new(95.374817, 115.62674),
            Vec2::new(95.330688, 118.422279),
            Vec2::new(95.629204, 121.148026),
            Vec2::new(96.265999, 123.775406),
            Vec2::new(97.236603, 126.275864),
            Vec2::new(98.536606, 128.620819),
            Vec2::new(100.161606, 130.781723),
            Vec2::new(102.10717, 132.72998),
            Vec2::new(104.368874, 134.437073),
            Vec2::new(106.942299, 135.874405),
            Vec2::new(109.823029, 137.013397),
            Vec2::new(112.596863, 137.732559),
            Vec2::new(115.30851, 138.091019),
            Vec2::new(117.941994, 138.104919),
            Vec2::new(120.481369, 137.790527),
            Vec2::new(122.910622, 137.163925),
            Vec2::new(125.213837, 136.241379),
            Vec2::new(127.374962, 135.039001),
            Vec2::new(129.378082, 133.573029),
            Vec2::new(131.20723, 131.859619),
            Vec2::new(132.84642, 129.914963),
            Vec2::new(134.279663, 127.755226),
            Vec2::new(135.491013, 125.396591),
            Vec2::new(136.464478, 122.855263),
            Vec2::new(137.215347, 120.008461),
            Vec2::new(137.627899, 117.178604),
            Vec2::new(137.70224, 114.392876),
            Vec2::new(137.438293, 111.678345),
            Vec2::new(136.836136, 109.062195),
            Vec2::new(135.895691, 106.571533),
            Vec2::new(134.617004, 104.233505),
            Vec2::new(133.000076, 102.075249),
            Vec2::new(131.044891, 100.123901),
            Vec2::new(128.751465, 98.406601),
            Vec2::new(126.119781, 96.950462),
        ];
        let anchors = [false; 50];
        let mut pool = Vec::new();
        let mut ft = Vec::new();
        stroke_outline(
            &pts,
            &anchors,
            true,
            10.0,
            Cap::Round,
            Join::Round,
            4.0,
            &mut pool,
            &mut ft,
        );
        let legacy = crate::stroke::stroke_polyline_pieces_for_test(
            &pts,
            &anchors,
            true,
            10.0,
            Cap::Round,
            Join::Round,
            4.0,
        );
        let (w, h) = (256usize, 256usize);
        let plane = |cs: &[Contour]| {
            let mut r = Rasterizer::new(w, h);
            r.fill_contours(cs);
            let mut p = vec![0u8; w * h];
            r.sweep(FillRule::NonZero, true, |y, x0, cov| {
                if let Some(d) = p.get_mut(y * w + x0..y * w + x0 + cov.len()) {
                    d.copy_from_slice(cov);
                }
            });
            p
        };
        let pf = plane(&ft);
        let pl = plane(&legacy);
        let mut bad = 0usize;
        let mut first = None;
        for (i, (a, b)) in pf.iter().zip(pl.iter()).enumerate() {
            if (i32::from(*a) - i32::from(*b)).abs() > 128 {
                bad += 1;
                if first.is_none() {
                    first = Some((i % w, i / w, *a, *b));
                }
            }
        }
        assert!(bad <= 20, "planet ring: {bad} px differ, first {first:?}");
    }

    #[test]
    fn game_emoji_extrusion_quad() {
        // Exact input from GameEmoji BUFF needle triage (frame 0, 512px):
        // thin extrusion side face, closed, hw=3.5, round joins.
        let pts = [
            Vec2::new(189.1, 161.7),
            Vec2::new(206.8, 178.1),
            Vec2::new(206.8, 358.5),
            Vec2::new(189.1, 342.1),
        ];
        let anchors = [true; 4];
        let mut pool = Vec::new();
        let mut ft = Vec::new();
        stroke_outline(
            &pts,
            &anchors,
            true,
            3.5,
            Cap::Round,
            Join::Round,
            4.0,
            &mut pool,
            &mut ft,
        );
        let legacy = crate::stroke::stroke_polyline_pieces_for_test(
            &pts,
            &anchors,
            true,
            3.5,
            Cap::Round,
            Join::Round,
            4.0,
        );
        let (w, h) = (512usize, 512usize);
        let plane = |cs: &[Contour]| {
            let mut r = Rasterizer::new(w, h);
            r.fill_contours(cs);
            let mut p = vec![0u8; w * h];
            r.sweep(FillRule::NonZero, true, |y, x0, cov| {
                if let Some(d) = p.get_mut(y * w + x0..y * w + x0 + cov.len()) {
                    d.copy_from_slice(cov);
                }
            });
            p
        };
        let pf = plane(&ft);
        let pl = plane(&legacy);
        let mut bad = 0usize;
        let mut first = None;
        for (i, (a, b)) in pf.iter().zip(pl.iter()).enumerate() {
            if (i32::from(*a) - i32::from(*b)).abs() > 128 {
                bad += 1;
                if first.is_none() {
                    first = Some((i % w, i / w, *a, *b));
                }
            }
        }
        assert!(bad <= 8, "needle repro: {bad} px differ, first {first:?}");
    }
}
