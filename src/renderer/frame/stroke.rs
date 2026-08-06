//! Converts flattened paths into connected stroke outlines.
//!
//! Open paths emit one contour; closed paths emit an outer and inner ring
//! with opposite winding. Input cleanup lives here so every renderer sees
//! identical stroke geometry.

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::geometry::Contour;
use crate::math::Vec2;
use alloc::vec::Vec;

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

/// Do not retain pathological stroke-segment buffers between calls.
const SEGMENT_REUSE_CAP: usize = 512;

struct Border {
  pts: Vec<Vec2>,
  /// FT `movable` (VFT:392-423): the last point is a straight-segment
  /// end offset that the next corner may REPLACE instead of append.
  movable: bool,
}

impl Border {
  fn new(pts: Vec<Vec2>) -> Border {
    Border { pts, movable: false }
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
      let dup = self.pts.last().is_some_and(|q| (q.x - p.x).abs() < 0.03125 && (q.y - p.y).abs() < 0.03125);
      if dup {
        return;
      }
      self.pts.push(p);
    }
    self.movable = movable;
  }

  /// Polygonal arc around `center`, radius `r`, from unit vector `start`
  /// to unit vector `end` while sweeping `sweep` radians. The start point
  /// is already present, so emission starts at step 1.
  fn arc_to(&mut self, center: Vec2, r: f32, start: Vec2, end: Vec2, sweep: f32) {
    self.movable = false;
    let step = (8.0 * ARC_TOL / r.max(1e-3)).sqrt().clamp(0.0655, 0.35);
    let steps = ((sweep.abs() / step).ceil() as usize).clamp(1, 48);
    self.pts.reserve(steps);
    if steps == 1 {
      self.pts.push(Vec2::new(center.x + r * end.x, center.y + r * end.y));
      return;
    }
    let a0 = start.y.atan2(start.x);
    for i in 1..=steps {
      let a = a0 + sweep * (i as f32 / steps as f32);
      self.pts.push(Vec2::new(center.x + r * a.cos(), center.y + r * a.sin()));
    }
  }
}

/// One cleaned polyline segment.
pub(crate) struct StrokeSegment {
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
  segments: &mut Vec<StrokeSegment>,
  out: &mut Vec<Contour>,
) {
  let ml = if miter_limit.is_finite() { miter_limit.max(1.0) } else { 4.0 };
  let anchor_at = |i: usize| anchors.get(i).copied().unwrap_or(true);

  // Segment list; degenerate directions (post-dedupe pathologies) are
  // dropped, merging their neighbors.
  let n = pts.len();
  let seg_count = if closed { n } else { n.saturating_sub(1) };
  segments.clear();
  segments.reserve(seg_count);
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
    segments.push(StrokeSegment {
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
  let seam_gap = dropped_gap || segments.first().is_some_and(|s| s.gap_before);
  if segments.is_empty() {
    return;
  }

  let mut b0 = Border::new(pool.pop().unwrap_or_default());
  let mut b1 = Border::new(pool.pop().unwrap_or_default());
  b0.pts.clear();
  b1.pts.clear();
  let cap_pts = 2 + (core::f32::consts::PI / 0.0655) as usize;
  b0.pts.reserve(2 * segments.len() + cap_pts + 8);
  b1.pts.reserve(2 * segments.len() + 8);

  // Subpath start: moveto on each border (movable = false).
  let (first_d, first_p) = match segments.first() {
    Some(s) => (s.d, s.p),
    None => return,
  };
  let n0 = Vec2::new(-first_d.y * hw, first_d.x * hw);
  b0.pts.push(Vec2::new(first_p.x + n0.x, first_p.y + n0.y));
  b1.pts.push(Vec2::new(first_p.x - n0.x, first_p.y - n0.y));

  let mut prev_d = first_d;
  let mut prev_len = segments.first().map_or(0.0, |s| if s.is_line { s.len } else { 0.0 });
  let mut prev_raw = segments.first().map_or(0.0, |s| s.len);
  for (i, seg) in segments.iter().enumerate() {
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
    let first_len = segments.first().map_or(0.0, |s| if s.is_line { s.len } else { 0.0 });
    let first_raw = segments.first().map_or(0.0, |s| s.len);
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
      // round planets. Elongated closed strokes still need both loops.
      let width = x1 - x0;
      let height = y1 - y0;
      let max_dim = width.max(height);
      let min_dim = width.min(height);
      x1 >= x0
        && y1 >= y0
        && max_dim <= 2.0 * hw
        // Fully inverted round dots collapse to discs in rlottie. Strongly
        // elongated ellipses retain the inset loop, which is visible as the
        // nested candle rings in TenYearsOfTelegram.
        && min_dim >= 0.9 * max_dim
        && ring_area(&b0.pts) * ring_area(&b1.pts) < 0.0
    };
    let keep_b0 = if drop_inner { ring_area(&b0.pts).abs() >= ring_area(&b1.pts).abs() } else { true };
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
    let last = match segments.last() {
      Some(s) => s,
      None => return,
    };
    let end_center = Vec2::new(last.p.x + last.d.x * last.len, last.p.y + last.d.y * last.len);
    let mut loop_pts = b0.pts;
    emit_cap(&mut loop_pts, end_center, last.d, hw, cap);
    loop_pts.extend(b1.pts.iter().rev().copied());
    pool.push(b1.pts);
    emit_cap(&mut loop_pts, first_p, Vec2::new(-first_d.x, -first_d.y), hw, cap);
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

  let (si, so) = if inside_is_b0 { (1.0f32, -1.0f32) } else { (-1.0, 1.0) };

  // ---- inside border ----
  {
    let bi: &mut Border = if inside_is_b0 { b0 } else { b1 };
    // Intersection needs both neighbors straight and long enough:
    // len >= |hw tan(theta)| with theta = half turn (VFT:861-871).
    let denom = 1.0 + dot;
    let can_intersect = !reversal && denom > 1e-6 && bi.movable && prev_len > 0.0 && next_len > 0.0 && {
      let t = (hw * cross / denom).abs();
      prev_len >= t && next_len >= t
    };
    if can_intersect {
      let k = hw / denom;
      let ip = Vec2::new(p.x + si * (n_in.x + n_out.x) * k, p.y + si * (n_in.y + n_out.y) * k);
      bi.line_to(ip, false);
    } else {
      // Double-back: keep incoming end offset, append outgoing start
      // offset (VFT:874-894).
      bi.movable = false;
      bi.line_to(Vec2::new(p.x + si * n_out.x * hw, p.y + si * n_out.y * hw), false);
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
        bo.line_to(Vec2::new(p.x + so * (n_in.x + n_out.x) * k, p.y + so * (n_in.y + n_out.y) * k), false);
      }
      return;
    }
    match eff_join {
      Join::Round => {
        let start = Vec2::new(so * n_in.x, so * n_in.y);
        let end = Vec2::new(so * n_out.x, so * n_out.y);
        let sweep = if reversal {
          // VFT:776 — turn == pi resolves to -rotate*2 of the
          // outside side; in vector terms the semicircle bulges
          // through the reversal direction.
          -so * core::f32::consts::PI
        } else {
          cross.atan2(dot)
        };
        bo.arc_to(p, hw, start, end, sweep);
      }
      Join::Bevel | Join::Miter => {
        let denom = 1.0 + dot;
        // Over-limit test: ml * cos(theta/2) < 1, with
        // cos^2(theta/2) = (1+dot)/2 (VFT:934-938). Reversals have
        // denom -> 0: always over-limit -> bevel stub.
        let within = matches!(join, Join::Miter) && denom > 1e-6 && ml * ml * denom * 0.5 >= 1.0;
        if within && matches!(eff_join, Join::Miter) {
          // Miter tip lies ON the incoming offset line, so the
          // replace-if-movable lineto is collinear-safe
          // (VFT:1004-1013 keeps `movable` set).
          let k = hw / denom;
          bo.line_to(Vec2::new(p.x + so * (n_in.x + n_out.x) * k, p.y + so * (n_in.y + n_out.y) * k), false);
          if next_len <= 0.0 {
            // Curve corner: FT appends the outgoing start
            // offset too (VFT:1016-1025).
            bo.line_to(Vec2::new(p.x + so * n_out.x * hw, p.y + so * n_out.y * hw), false);
          }
        } else {
          // Fixed bevel: the chord needs BOTH the incoming end
          // offset and the outgoing start offset — clear movable
          // FIRST so the lineto appends instead of replacing
          // (VFT:957-958; replacing here ate a sliver of the
          // band — the OutlineEmoji "notch" corpus signature).
          bo.movable = false;
          bo.line_to(Vec2::new(p.x + so * n_out.x * hw, p.y + so * n_out.y * hw), false);
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
    let q = pts.get(i + 1).or_else(|| pts.first()).copied().unwrap_or(*p);
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

/// Sanitizes and strokes a flattened polyline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stroke_polyline(
  points: &[Vec2],
  point_anchors: &[bool],
  closed: bool,
  hw: f32,
  cap: Cap,
  join: Join,
  miter_limit: f32,
  pool: &mut Vec<Vec<Vec2>>,
  segments: &mut Vec<StrokeSegment>,
  out: &mut Vec<Contour>,
  _solo: bool,
) {
  if !(hw > 0.0) || !hw.is_finite() {
    return;
  }

  let mut pts = pool.pop().unwrap_or_default();
  pts.clear();
  pts.reserve(points.len());
  let mut anchors = Vec::with_capacity(points.len());
  for (index, point) in points.iter().enumerate() {
    if !(point.x.is_finite() && point.y.is_finite()) {
      continue;
    }
    let anchor = point_anchors.get(index).copied().unwrap_or(true);
    if pts.last().is_none_or(|previous: &Vec2| (previous.x - point.x).abs() > 1e-6 || (previous.y - point.y).abs() > 1e-6) {
      pts.push(*point);
      anchors.push(anchor);
    } else if let Some(previous) = anchors.last_mut() {
      *previous |= anchor;
    }
  }

  if closed
    && pts.len() > 1
    && pts
      .first()
      .zip(pts.last())
      .is_some_and(|(first, last)| (first.x - last.x).abs() < 1e-6 && (first.y - last.y).abs() < 1e-6)
  {
    pts.pop();
    anchors.pop();
  }

  if pts.len() < 2 {
    pool.push(pts);
    return;
  }

  stroke_outline(&pts, &anchors, closed, hw, cap, join, miter_limit, pool, segments, out);
  if segments.capacity() > SEGMENT_REUSE_CAP {
    *segments = Vec::new();
  }
  pts.clear();
  pool.push(pts);
}

#[cfg(test)]
#[path = "tests/stroke.rs"]
mod tests;
