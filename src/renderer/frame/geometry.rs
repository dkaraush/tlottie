//! Geometry: shape → device-space polygons (flattened cubic beziers).
//!
//! v0 uses uniform flattening with a segment count derived from the control
//! polygon length in device space; the adaptive/scale-aware flattener comes
//! with the quality/perf phases.

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::math::{Mat2x3, Vec2};
use crate::model::PathData;
use alloc::vec;
use alloc::vec::Vec;

/// One closed (or open) contour as a device-space polyline.
#[derive(Debug, Clone, Default)]
pub(crate) struct Contour {
  pub points: Vec<Vec2>,
  /// Per-point provenance: `true` = authored path vertex (anchor),
  /// `false` = point inserted by curve flattening. EMPTY means every
  /// point is an anchor (polylines built from straight segments).
  /// FreeType strokes curves directly and rounds cusps INSIDE a curve
  /// while applying the real line join at authored corners — the stroker
  /// needs this distinction (GameEmoji impact spikes vs Froggo cusps).
  pub anchors: Vec<bool>,
  /// Inverse LINEAR part (a,b,c,d — no translation; deltas only) of the
  /// matrix this contour was flattened with. Trim/dash arclength must be
  /// measured in the shape's LOCAL space (rlottie trims localPath():
  /// under non-uniform scale, device fractions diverge — HalloweenTeam /
  /// HandDrawn / SpookyLetters BROKEN family). None = measure in device
  /// space (repeater copies, synthetic contours).
  pub inv_lin: Option<[f32; 4]>,
}

/// Circle-from-cubics constant.
const KAPPA: f32 = 0.552_284_8;

/// Max line segments a single cubic can flatten to.
const MAX_SEGS: usize = 128;

/// Segment length as rlottie measures it (VLine::length, vline.h):
/// `max(|dx|,|dy|) + 0.375·min(|dx|,|dy|)` — a fast non-Euclidean hypot
/// (exact at 0°/90°, +6.8% near 20.5°, −2.7% at 45°; ~+4% around a circle).
/// Dash and trim boundaries must be placed with THIS metric or they drift
/// visibly out of phase with rlottie along curves. Geometry itself
/// (flattening, stroking) stays Euclidean.
pub(crate) fn seg_len_rlottie(dx: f32, dy: f32) -> f32 {
  let (ax, ay) = (dx.abs(), dy.abs());
  if ax > ay {
    ax + 0.375 * ay
  } else {
    ay + 0.375 * ax
  }
}

fn flatten_cubic(out: &mut Vec<Vec2>, anchors: &mut Vec<bool>, p0: Vec2, c1: Vec2, c2: Vec2, p1: Vec2, tolerance: f32) {
  // Error-bounded uniform flattening: for n segments the max deviation is
  // <= 3/4 * M / n^2 where M is the largest second-difference norm of the
  // control polygon (measured 0.74·M/n²; the oft-quoted 3/16 bound is for
  // one *subdivision level*, not n uniform segments). Solve for n at
  // tolerance 0.05 px: n = sqrt((0.75/0.05)·M) = sqrt(15·M).
  // Gentle curves get few segments, tight curves get many.
  let d2a = ((p0.x - 2.0 * c1.x + c2.x).abs()).max((p0.y - 2.0 * c1.y + c2.y).abs());
  let d2b = ((c1.x - 2.0 * c2.x + p1.x).abs()).max((c1.y - 2.0 * c2.y + p1.y).abs());
  let m_norm = d2a.max(d2b);
  let coefficient = 0.75 / tolerance.clamp(0.01, 4.0);
  let n = ((m_norm * coefficient).sqrt().ceil() as usize).clamp(2, MAX_SEGS);
  let step = 1.0 / n as f32;
  let mut t = step;
  for _ in 0..n.saturating_sub(1) {
    let omt = 1.0 - t;
    let a = omt * omt * omt;
    let b = 3.0 * omt * omt * t;
    let c = 3.0 * omt * t * t;
    let d = t * t * t;
    out.push(Vec2::new(a * p0.x + b * c1.x + c * c2.x + d * p1.x, a * p0.y + b * c1.y + c * c2.y + d * p1.y));
    anchors.push(false);
    t += step;
  }
  out.push(p1);
  anchors.push(true);
}

/// Reverses a path's winding in place: vertex order flips and each
/// vertex's in/out tangents swap (rlottie's direction()==CCW branches).
pub(crate) fn reverse_path(data: &mut PathData) {
  data.vertices.reverse();
  data.in_tangents.reverse();
  data.out_tangents.reverse();
  core::mem::swap(&mut data.in_tangents, &mut data.out_tangents);
}

/// Flattens Lottie path data (vertices + relative tangents) under `m`.
pub(crate) fn flatten_path(data: &PathData, m: &Mat2x3, tolerance: f32) -> Contour {
  flatten_path_reusing(data, m, tolerance, Contour::default())
}

/// Flattens into a recycled contour, retaining its point and anchor vector
/// capacities across frames. The output is identical to [`flatten_path`].
pub(crate) fn flatten_path_reusing(data: &PathData, m: &Mat2x3, tolerance: f32, mut contour: Contour) -> Contour {
  let n = data.vertices.len();
  contour.points.clear();
  contour.anchors.clear();
  contour.inv_lin = None;
  contour.points.reserve(n * 4);
  contour.anchors.reserve(n * 4);
  if n == 0 {
    return contour;
  }
  let points = &mut contour.points;
  let anchors = &mut contour.anchors;
  let vert = |i: usize| -> Vec2 { data.vertices.get(i).copied().unwrap_or(Vec2::ZERO) };
  let tan_in = |i: usize| -> Vec2 { data.in_tangents.get(i).copied().unwrap_or(Vec2::ZERO) };
  let tan_out = |i: usize| -> Vec2 { data.out_tangents.get(i).copied().unwrap_or(Vec2::ZERO) };

  let first = m.apply(vert(0));
  points.push(first);
  anchors.push(true);
  let segs = if data.closed { n } else { n.saturating_sub(1) };
  let mut prev_dev = first;
  for s in 0..segs {
    let i0 = s;
    let i1 = if s + 1 == n { 0 } else { s + 1 };
    let p0 = vert(i0);
    let p1 = vert(i1);
    let c1 = m.apply(Vec2::new(p0.x + tan_out(i0).x, p0.y + tan_out(i0).y));
    let c2 = m.apply(Vec2::new(p1.x + tan_in(i1).x, p1.y + tan_in(i1).y));
    let p1d = m.apply(p1);
    let segment_start = points.len().saturating_sub(1);
    // Straight-line fast path: zero tangents.
    let t_out = tan_out(i0);
    let t_in = tan_in(i1);
    if t_out.x == 0.0 && t_out.y == 0.0 && t_in.x == 0.0 && t_in.y == 0.0 {
      points.push(p1d);
      anchors.push(true);
    } else {
      flatten_cubic(points, anchors, prev_dev, c1, c2, p1d, tolerance);
      // Open stroke caps use the first/last flattened chord direction.
      // Pin those chords to the authored cubic endpoint tangents so two
      // separately-authored butt-capped curves which meet C1-continuously
      // cannot expose a triangular gap from independent flattening.
      if !data.closed && s == 0 && points.len() > segment_start + 1 {
        let tangent = if (c1.x - prev_dev.x).abs() + (c1.y - prev_dev.y).abs() > 1e-6 {
          Vec2::new(c1.x - prev_dev.x, c1.y - prev_dev.y)
        } else {
          Vec2::new(c2.x - prev_dev.x, c2.y - prev_dev.y)
        };
        align_endpoint_chord(points, segment_start, segment_start + 1, tangent, false);
      }
      if !data.closed && s + 1 == segs && points.len() > segment_start + 1 {
        let end = points.len() - 1;
        let tangent = if (p1d.x - c2.x).abs() + (p1d.y - c2.y).abs() > 1e-6 {
          Vec2::new(p1d.x - c2.x, p1d.y - c2.y)
        } else {
          Vec2::new(p1d.x - c1.x, p1d.y - c1.y)
        };
        align_endpoint_chord(points, end, end - 1, tangent, true);
      }
    }
    prev_dev = p1d;
  }
  let inv = m.inverse();
  contour.inv_lin = Some([inv.a, inv.b, inv.c, inv.d]);
  contour
}

fn align_endpoint_chord(points: &mut [Vec2], endpoint: usize, neighbor: usize, tangent: Vec2, reverse: bool) {
  let len = (tangent.x * tangent.x + tangent.y * tangent.y).sqrt();
  let Some((&end, old)) = points.get(endpoint).zip(points.get(neighbor).copied()) else {
    return;
  };
  let chord = ((old.x - end.x) * (old.x - end.x) + (old.y - end.y) * (old.y - end.y)).sqrt();
  if !(len > 1e-6 && chord > 0.0 && len.is_finite() && chord.is_finite()) {
    return;
  }
  let sign = if reverse { -1.0 } else { 1.0 };
  if let Some(point) = points.get_mut(neighbor) {
    *point = Vec2::new(end.x + sign * tangent.x * chord / len, end.y + sign * tangent.y * chord / len);
  }
}

/// Ellipse centered at `pos` with `size` (width/height), as 4 cubic arcs.
pub(crate) fn ellipse_contour(pos: Vec2, size: Vec2, reversed: bool, m: &Mat2x3, tolerance: f32) -> Contour {
  let rx = size.x * 0.5;
  let ry = size.y * 0.5;
  let (cx, cy) = (pos.x, pos.y);
  let kx = rx * KAPPA;
  let ky = ry * KAPPA;
  // Start at top, clockwise (matches AE/rlottie direction 1).
  let verts = [Vec2::new(cx, cy - ry), Vec2::new(cx + rx, cy), Vec2::new(cx, cy + ry), Vec2::new(cx - rx, cy)];
  let out_t = [Vec2::new(kx, 0.0), Vec2::new(0.0, ky), Vec2::new(-kx, 0.0), Vec2::new(0.0, -ky)];
  let in_t = [Vec2::new(-kx, 0.0), Vec2::new(0.0, -ky), Vec2::new(kx, 0.0), Vec2::new(0.0, ky)];
  let mut data = PathData {
    vertices: verts.to_vec(),
    in_tangents: in_t.to_vec(),
    out_tangents: out_t.to_vec(),
    closed: true,
  };
  if reversed {
    reverse_path(&mut data);
  }
  flatten_path(&data, m, tolerance)
}

/// Axis-aligned (in local space) rounded rect centered at `pos`.
pub(crate) fn rect_contour(pos: Vec2, size: Vec2, radius: f32, reversed: bool, m: &Mat2x3, tolerance: f32) -> Contour {
  let w = size.x.max(0.0);
  let h = size.y.max(0.0);
  let r = radius.clamp(0.0, 0.5 * w.min(h));
  let (l, t) = (pos.x - w * 0.5, pos.y - h * 0.5);
  let (rt, bt) = (l + w, t + h);
  if r <= 0.0 {
    let pts = [Vec2::new(rt, t), Vec2::new(rt, bt), Vec2::new(l, bt), Vec2::new(l, t)];
    let mut pts: Vec<Vec2> = pts.to_vec();
    if reversed {
      pts.reverse();
    }
    let inv = m.inverse();
    return Contour {
      points: pts.iter().map(|p| m.apply(*p)).collect(),
      anchors: Vec::new(),
      inv_lin: Some([inv.a, inv.b, inv.c, inv.d]),
    };
  }
  let k = r * KAPPA;
  // Clockwise, STARTING on the right edge just below the top-right corner
  // arc — rlottie's addRoundRect does moveTo(x+w, y+ry/2), and the start
  // vertex is the origin for trim/dash parametrization (Paris swoosh bug:
  // starting at the top apex rotated every trim by a quarter corner arc).
  let verts = [
    Vec2::new(rt, t + r),
    Vec2::new(rt, bt - r),
    Vec2::new(rt - r, bt),
    Vec2::new(l + r, bt),
    Vec2::new(l, bt - r),
    Vec2::new(l, t + r),
    Vec2::new(l + r, t),
    Vec2::new(rt - r, t),
  ];
  let out_t = [
    Vec2::ZERO,
    Vec2::new(0.0, k),
    Vec2::ZERO,
    Vec2::new(-k, 0.0),
    Vec2::ZERO,
    Vec2::new(0.0, -k),
    Vec2::ZERO,
    Vec2::new(k, 0.0),
  ];
  let in_t = [
    Vec2::new(0.0, -k),
    Vec2::ZERO,
    Vec2::new(k, 0.0),
    Vec2::ZERO,
    Vec2::new(0.0, k),
    Vec2::ZERO,
    Vec2::new(-k, 0.0),
    Vec2::ZERO,
  ];
  let mut data = PathData {
    vertices: verts.to_vec(),
    in_tangents: in_t.to_vec(),
    out_tangents: out_t.to_vec(),
    closed: true,
  };
  if reversed {
    reverse_path(&mut data);
  }
  flatten_path(&data, m, tolerance)
}

/// Sutherland–Hodgman clip of a polygon to an arbitrary convex quad
/// (device-space precomp viewport). Winding of the subject is preserved.
/// True when the axis-aligned box `(x0,y0)..(x1,y1)` lies fully on the
/// inside of `quad` under clip_to_quad's OWN non-strict test (same sign
/// convention, same `>= 0.0`). The edge cross product is affine in the
/// point, so its minimum over the box is at a corner: all four corners
/// inside ⟹ every contained point inside ⟹ the Sutherland–Hodgman pass
/// would return the polygon unchanged and can be skipped bit-exactly.
/// rlottie's VLine metric on a delta pulled back to LOCAL space via the
/// contour's inverse linear matrix (None = device space).
#[inline]
pub(crate) fn seg_len_measured(dx: f32, dy: f32, inv: Option<[f32; 4]>) -> f32 {
  match inv {
    Some([a, b, c, d]) => seg_len_rlottie(a * dx + c * dy, b * dx + d * dy),
    None => seg_len_rlottie(dx, dy),
  }
}

pub(crate) fn quad_contains_box(quad: &[Vec2; 4], x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
  let area = (quad[1].x - quad[0].x) * (quad[2].y - quad[0].y) - (quad[2].x - quad[0].x) * (quad[1].y - quad[0].y);
  let sign = if area >= 0.0 { 1.0f32 } else { -1.0f32 };
  for e in 0..4usize {
    let (Some(&a), Some(&b)) = (quad.get(e), quad.get((e + 1) % 4)) else {
      return false;
    };
    let (ex, ey) = (b.x - a.x, b.y - a.y);
    let inside = |px: f32, py: f32| sign * (ex * (py - a.y) - ey * (px - a.x)) >= 0.0;
    if !(inside(x0, y0) && inside(x1, y0) && inside(x0, y1) && inside(x1, y1)) {
      return false;
    }
  }
  true
}

pub(crate) fn clip_to_quad(c: &Contour, quad: &[Vec2; 4]) -> Contour {
  // Orientation of the quad decides which side is "inside".
  let area = (quad[1].x - quad[0].x) * (quad[2].y - quad[0].y) - (quad[2].x - quad[0].x) * (quad[1].y - quad[0].y);
  let sign = if area >= 0.0 { 1.0f32 } else { -1.0f32 };
  let mut pts = c.points.clone();
  for e in 0..4usize {
    if pts.len() < 3 {
      return Contour {
        points: Vec::new(),
        anchors: Vec::new(),
        inv_lin: None,
      };
    }
    let a = match quad.get(e) {
      Some(p) => *p,
      None => break,
    };
    let b = match quad.get((e + 1) % 4) {
      Some(p) => *p,
      None => break,
    };
    let (ex, ey) = (b.x - a.x, b.y - a.y);
    let inside = |p: &Vec2| -> bool { sign * (ex * (p.y - a.y) - ey * (p.x - a.x)) >= 0.0 };
    let intersect = |p: &Vec2, q: &Vec2| -> Vec2 {
      let dpx = q.x - p.x;
      let dpy = q.y - p.y;
      let denom = ex * dpy - ey * dpx;
      if denom.abs() < 1e-12 {
        return *q;
      }
      let t = (ex * (a.y - p.y) - ey * (a.x - p.x)) / denom;
      Vec2::new(p.x + dpx * t.clamp(0.0, 1.0), p.y + dpy * t.clamp(0.0, 1.0))
    };
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len() + 4);
    for (i, cur) in pts.iter().enumerate() {
      let prev = match i.checked_sub(1).and_then(|j| pts.get(j)).or_else(|| pts.last()) {
        Some(p) => p,
        None => break,
      };
      let cur_in = inside(cur);
      let prev_in = inside(prev);
      if cur_in {
        if !prev_in {
          out.push(intersect(prev, cur));
        }
        out.push(*cur);
      } else if prev_in {
        out.push(intersect(prev, cur));
      }
    }
    pts = out;
  }
  Contour {
    points: pts,
    anchors: Vec::new(),
    inv_lin: None,
  }
}

/// Sutherland–Hodgman clip of a polygon to the rect [0,w]×[0,h].
/// Keeps winding; coverage outside the viewport is discarded exactly.
pub(crate) fn clip_contour(c: &Contour, w: f32, h: f32) -> Contour {
  let mut pts = c.points.clone();
  for edge in 0..4u8 {
    if pts.len() < 3 {
      return Contour {
        points: Vec::new(),
        anchors: Vec::new(),
        inv_lin: None,
      };
    }
    let inside = |p: &Vec2| -> bool {
      match edge {
        0 => p.x >= 0.0,
        1 => p.x <= w,
        2 => p.y >= 0.0,
        _ => p.y <= h,
      }
    };
    let intersect = |a: &Vec2, b: &Vec2| -> Vec2 {
      let (bound_x, vertical) = match edge {
        0 => (0.0, true),
        1 => (w, true),
        2 => (0.0, false),
        _ => (h, false),
      };
      if vertical {
        let t = if (b.x - a.x).abs() > 1e-9 { (bound_x - a.x) / (b.x - a.x) } else { 0.0 };
        Vec2::new(bound_x, a.y + (b.y - a.y) * t)
      } else {
        let t = if (b.y - a.y).abs() > 1e-9 { (bound_x - a.y) / (b.y - a.y) } else { 0.0 };
        Vec2::new(a.x + (b.x - a.x) * t, bound_x)
      }
    };
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len() + 4);
    for (i, cur) in pts.iter().enumerate() {
      let prev = match i.checked_sub(1).and_then(|j| pts.get(j)) {
        Some(p) => p,
        None => match pts.last() {
          Some(p) => p,
          None => break,
        },
      };
      let cur_in = inside(cur);
      let prev_in = inside(prev);
      if cur_in {
        if !prev_in {
          out.push(intersect(prev, cur));
        }
        out.push(*cur);
      } else if prev_in {
        out.push(intersect(prev, cur));
      }
    }
    pts = out;
  }
  Contour {
    points: pts,
    anchors: Vec::new(),
    inv_lin: None,
  }
}

/// Extracts the sub-polyline covering arclength range [l0, l1] of a contour.
/// On closed contours the range may wrap past the end (walk is modulo total
/// length). Returns an open polyline plus per-point anchor flags (cut
/// points are non-anchors; copied original points inherit their flag).
pub(crate) fn extract_by_length(points: &[Vec2], anchors: &[bool], closed: bool, l0: f32, l1: f32, inv: Option<[f32; 4]>) -> (Vec<Vec2>, Vec<bool>) {
  let mut out = Vec::new();
  let mut out_anchor: Vec<bool> = Vec::new();
  let anchor_of = |i: usize| anchors.get(i).copied().unwrap_or(true);
  if points.len() < 2 || !(l1 > l0) {
    return (out, out_anchor);
  }
  // Segment list (closed contours include the wrap segment).
  let seg_count = if closed { points.len() } else { points.len() - 1 };
  let mut lengths = Vec::with_capacity(seg_count);
  let mut total = 0.0f32;
  for i in 0..seg_count {
    let a = match points.get(i) {
      Some(p) => *p,
      None => break,
    };
    let b = match points.get(if i + 1 == points.len() { 0 } else { i + 1 }) {
      Some(p) => *p,
      None => break,
    };
    // Trim arclength is measured in LOCAL space (rlottie trims
    // localPath()): under non-uniform scale device fractions diverge.
    let len = seg_len_measured(b.x - a.x, b.y - a.y, inv);
    lengths.push(len);
    total += len;
  }
  if !(total > 1e-6) || !total.is_finite() {
    return (out, out_anchor);
  }
  let span = (l1 - l0).min(total);
  let start = if closed { l0.rem_euclid(total) } else { l0.clamp(0.0, total) };
  let end = if closed { start + span } else { l1.clamp(0.0, total) };
  if !(end > start) {
    return (out, out_anchor);
  }

  let mut walked = 0.0f32;
  let mut emitted = 0.0f32;
  // Walk up to twice around for wrapped ranges on closed contours.
  let rounds = if closed { 2 } else { 1 };
  'outer: for round in 0..rounds {
    for i in 0..seg_count {
      let seg_len = match lengths.get(i) {
        Some(&l) if l > 0.0 => l,
        _ => continue,
      };
      let a = match points.get(i) {
        Some(p) => *p,
        None => continue,
      };
      let b = match points.get(if i + 1 == points.len() { 0 } else { i + 1 }) {
        Some(p) => *p,
        None => continue,
      };
      let seg_start = walked;
      let seg_end = walked + seg_len;
      walked = seg_end;
      let global_off = round as f32 * total;
      let s0 = (start - global_off).max(seg_start - global_off) + global_off;
      // Overlap of [start,end] with this segment occurrence.
      let occ_start = seg_start + 0.0;
      let occ_end = seg_end;
      let lo = start.max(occ_start);
      let hi = end.min(occ_end);
      let _ = (s0, emitted);
      if hi <= lo {
        if occ_start > end {
          break 'outer;
        }
        continue;
      }
      let t0 = (lo - occ_start) / seg_len;
      let t1 = (hi - occ_start) / seg_len;
      let p0 = Vec2::new(a.x + (b.x - a.x) * t0, a.y + (b.y - a.y) * t0);
      let p1 = Vec2::new(a.x + (b.x - a.x) * t1, a.y + (b.y - a.y) * t1);
      if out.is_empty() {
        out.push(p0);
        out_anchor.push(t0 <= 1e-6 && anchor_of(i));
      }
      out.push(p1);
      out_anchor.push(t1 >= 1.0 - 1e-6 && anchor_of(if i + 1 == points.len() { 0 } else { i + 1 }));
      emitted += hi - lo;
      if end <= occ_end {
        break 'outer;
      }
    }
    if !closed {
      break;
    }
    // Second round continues walking with accumulated `walked`.
  }
  (out, out_anchor)
}

/// Builds a star/polygon path (lottie-web / rlottie algorithm), local space.
#[allow(clippy::too_many_arguments)]
pub(crate) fn polystar_path(
  star: bool,
  reversed: bool,
  points: f32,
  position: Vec2,
  rotation_deg: f32,
  inner_radius: f32,
  outer_radius: f32,
  inner_roundness_pct: f32,
  outer_roundness_pct: f32,
) -> PathData {
  let mut data = PathData { closed: true, ..PathData::default() };
  let num_points = points.clamp(2.0, 256.0);
  let n = (if star { num_points * 2.0 } else { num_points }) as usize;
  if n == 0 {
    return data;
  }
  let pi = core::f32::consts::PI;
  let half_angle = pi / num_points;
  let angle_per = if star { half_angle } else { 2.0 * half_angle };
  // rlottie applies the polystar rotation TWICE (addPolystar bakes it,
  // then the item matrix rotates again — lottieitem.cpp:1290-1298); this
  // double-rotation is present in both trees, so it is the contract.
  let mut angle = -pi / 2.0 + 2.0 * rotation_deg.to_radians();
  let out_round = outer_roundness_pct / 100.0;
  let in_round = inner_roundness_pct / 100.0;
  // Tangent length factor per lottie-web: r * roundness * perimeter_seg / 4
  // where perimeter_seg = 2*pi*r/(2*num_points) for stars.
  for i in 0..n {
    let outer_vertex = !star || i % 2 == 0;
    let (radius, roundness) = if outer_vertex { (outer_radius, out_round) } else { (inner_radius, in_round) };
    let (sin_a, cos_a) = angle.sin_cos();
    let v = Vec2::new(position.x + radius * cos_a, position.y + radius * sin_a);
    // Perpendicular (tangential direction along the circle).
    let perp = Vec2::new(-sin_a, cos_a);
    let seg = 2.0 * pi * radius / (n as f32);
    let tan_len = roundness * seg * 0.25;
    data.vertices.push(v);
    data.in_tangents.push(Vec2::new(-perp.x * tan_len, -perp.y * tan_len));
    data.out_tangents.push(Vec2::new(perp.x * tan_len, perp.y * tan_len));
    angle += angle_per;
  }
  if reversed {
    reverse_path(&mut data);
  }
  data
}

/// Rounds sharp corners of a device-space polyline in place: each corner is
/// replaced by two points pulled back along the adjacent segments plus a
/// short arc. Smooth (flattened-curve) corners are near-collinear and are
/// left alone.
pub(crate) fn round_polyline_corners(c: &Contour, closed: bool, radius: f32) -> Contour {
  let pts = &c.points;
  let n = pts.len();
  if radius <= 0.0 || n < 3 {
    return c.clone();
  }
  let mut out: Vec<Vec2> = Vec::with_capacity(n * 2);
  let range = if closed { 0..n } else { 1..n - 1 };
  if !closed {
    if let Some(p) = pts.first() {
      out.push(*p);
    }
  }
  for i in range {
    let prev = match pts.get(if i == 0 { n - 1 } else { i - 1 }) {
      Some(p) => *p,
      None => continue,
    };
    let here = match pts.get(i) {
      Some(p) => *p,
      None => continue,
    };
    let next = match pts.get(if i + 1 == n { 0 } else { i + 1 }) {
      Some(p) => *p,
      None => continue,
    };
    let d0 = Vec2::new(here.x - prev.x, here.y - prev.y);
    let d1 = Vec2::new(next.x - here.x, next.y - here.y);
    let l0 = (d0.x * d0.x + d0.y * d0.y).sqrt();
    let l1 = (d1.x * d1.x + d1.y * d1.y).sqrt();
    if l0 < 1e-6 || l1 < 1e-6 {
      out.push(here);
      continue;
    }
    let cos_turn = (d0.x * d1.x + d0.y * d1.y) / (l0 * l1);
    if cos_turn > 0.985 {
      out.push(here);
      continue;
    }
    let radius = radius.min(l0 * 0.5).min(l1 * 0.5);
    let a = Vec2::new(here.x - d0.x / l0 * radius, here.y - d0.y / l0 * radius);
    let b = Vec2::new(here.x + d1.x / l1 * radius, here.y + d1.y / l1 * radius);
    out.push(a);
    for step in 1..4 {
      let t = step as f32 / 4.0;
      let omt = 1.0 - t;
      out.push(Vec2::new(
        omt * omt * a.x + 2.0 * omt * t * here.x + t * t * b.x,
        omt * omt * a.y + 2.0 * omt * t * here.y + t * t * b.y,
      ));
    }
    out.push(b);
  }
  if !closed {
    if let Some(p) = pts.last() {
      out.push(*p);
    }
  }
  Contour {
    points: out,
    anchors: Vec::new(),
    inv_lin: c.inv_lin,
  }
}

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod round_corners_tests;

/// Splits a polyline into dash sub-polylines. `pattern` is (dash, gap)
/// lengths already resolved; `offset` shifts the pattern start.
pub(crate) fn dash_polyline(points: &[Vec2], anchors: &[bool], closed: bool, pattern: &[f32], offset: f32) -> Vec<(Vec<Vec2>, Vec<bool>)> {
  let mut d = VDasher::new(points, anchors, closed, pattern, offset);
  d.run();
  d.out
}

/// Faithful port of rlottie's VDasher (vdasher.cpp) operating over the
/// flattened polyline: it walks the path ELEMENT by element (an element =
/// the span between two authored anchors, mirroring one lineTo/cubicTo in
/// rlottie's unflattened VPath) and applies the exact 1px thresholds and
/// per-element `updateActiveSegment` phase snapping. rlottie dashes on the
/// unflattened cubics, but its VBezier::length equals our flattened
/// chord-sum in the VLine metric to <0.1%, so measuring per-element on the
/// flattened sub-polyline reproduces its boundaries; what actually diverged
/// was a *continuous* arclength walk (the previous code) vs rlottie's
/// element-quantized one, which drops sub-pixel remainders at element ends
/// and snaps the phase — visible as fused dashes at a path's tail
/// (DuckEmoji chain).
struct VDasher<'a> {
  points: &'a [Vec2],
  anchors: &'a [bool],
  /// pattern as (length, gap) pairs
  pairs: Vec<(f32, f32)>,
  dash_offset: f32,
  index: usize,
  current_length: f32,
  discard: bool,
  start_new_segment: bool,
  out: Vec<(Vec<Vec2>, Vec<bool>)>,
}

const VDASH_EPS: f32 = 1e-3;

impl<'a> VDasher<'a> {
  fn new(points: &'a [Vec2], anchors: &'a [bool], _closed: bool, pattern: &[f32], offset: f32) -> Self {
    // Reconstruct (length, gap) pairs from the flat [d,g,d,g,...] list.
    let mut pairs = Vec::with_capacity(pattern.len() / 2 + 1);
    let mut i = 0;
    while i + 1 < pattern.len() {
      let l = pattern.get(i).copied().unwrap_or(0.0).max(0.0);
      let g = pattern.get(i + 1).copied().unwrap_or(0.0).max(0.0);
      pairs.push((l, g));
      i += 2;
    }
    if pairs.is_empty() {
      if let Some(first) = pattern.first() {
        pairs.push((first.max(0.0), 0.0));
      }
    }
    VDasher {
      points,
      anchors,
      pairs,
      dash_offset: offset,
      index: 0,
      current_length: 0.0,
      discard: false,
      start_new_segment: true,
      out: Vec::new(),
    }
  }

  fn anchor_of(&self, i: usize) -> bool {
    self.anchors.get(i).copied().unwrap_or(true)
  }

  /// rlottie VDasher::moveTo — resets phase at a subpath start, applying
  /// the dash offset. The offset normalizes modulo the LAST dash+gap pair
  /// only: rlottie computes `totalLength =` (assignment, not +=) inside
  /// its loop, so multi-pair patterns barely shift with animated offsets.
  fn move_to(&mut self) {
    self.discard = false;
    self.start_new_segment = true;
    self.index = 0;
    if self.dash_offset.abs() > VDASH_EPS && !self.pairs.is_empty() {
      let total_length = self.pairs.last().map(|(l, g)| l + g).unwrap_or(0.0);
      let mut n = if total_length > 0.0 { self.dash_offset % total_length } else { 0.0 };
      if n < 0.0 {
        n += total_length;
      }
      let mut placed = false;
      for (i, &(l, g)) in self.pairs.iter().enumerate() {
        if n < l {
          self.index = i;
          self.current_length = l - n;
          self.discard = false;
          placed = true;
          break;
        }
        n -= l;
        if n < g {
          self.index = i;
          self.current_length = g - n;
          self.discard = true;
          placed = true;
          break;
        }
        n -= g;
      }
      if !placed {
        self.current_length = self.pairs.first().map(|(l, _)| *l).unwrap_or(0.0);
      }
    } else {
      self.current_length = self.pairs.get(self.index).map(|(l, _)| *l).unwrap_or(0.0);
    }
    if self.current_length.abs() < 1e-4 {
      self.update_active_segment();
    }
  }

  /// rlottie VDasher::updateActiveSegment — advance to the next dash/gap.
  /// Iterative where rlottie recurses, so a hostile pattern of near-zero
  /// phases cannot overflow the stack; run() already rejected all-zero
  /// patterns, so a non-zero phase exists within one full cycle.
  fn update_active_segment(&mut self) {
    self.start_new_segment = true;
    if self.pairs.is_empty() {
      return;
    }
    for _ in 0..2 * self.pairs.len() + 2 {
      if self.discard {
        self.discard = false;
        self.index = (self.index + 1) % self.pairs.len();
        self.current_length = self.pairs.get(self.index).map(|(l, _)| *l).unwrap_or(0.0);
      } else {
        self.discard = true;
        self.current_length = self.pairs.get(self.index).map(|(_, g)| *g).unwrap_or(0.0);
      }
      if self.current_length.abs() >= 1e-4 {
        break;
      }
    }
  }

  /// rlottie VDasher::addLine/addCubic — appends `span` (points from the
  /// piece start to the split end, inclusive) to the active piece when not
  /// discarding.
  fn add_span(&mut self, span: &[(Vec2, bool)]) {
    if self.discard || span.is_empty() {
      return;
    }
    if self.start_new_segment {
      let mut piece = (Vec::new(), Vec::new());
      for &(p, a) in span {
        piece.0.push(p);
        piece.1.push(a);
      }
      self.out.push(piece);
      self.start_new_segment = false;
    } else if let Some(piece) = self.out.last_mut() {
      // shared junction vertex already present — skip span[0]
      for &(p, a) in span.iter().skip(1) {
        piece.0.push(p);
        piece.1.push(a);
      }
    }
  }

  /// Process one element: a sub-polyline `verts` (with anchor flags) whose
  /// endpoints are authored anchors; interior points are flattening
  /// vertices. Mirrors rlottie VDasher::cubicTo/lineTo on that element,
  /// measuring length with the VLine metric.
  fn process_element(&mut self, verts: &[(Vec2, bool)]) {
    let m = verts.len();
    if m < 2 {
      return;
    }
    // cumulative arclength at each vertex
    let mut cum = vec![0.0f32; m];
    for i in 1..m {
      let a = verts.get(i - 1).map(|v| v.0).unwrap_or(Vec2::new(0.0, 0.0));
      let b = verts.get(i).map(|v| v.0).unwrap_or(a);
      let prev = cum.get(i - 1).copied().unwrap_or(0.0);
      if let Some(slot) = cum.get_mut(i) {
        *slot = prev + seg_len_rlottie(b.x - a.x, b.y - a.y);
      }
    }
    let elem_len = cum.last().copied().unwrap_or(0.0);
    let first = verts.first().copied().unwrap_or((Vec2::new(0.0, 0.0), true));
    let last = verts.last().copied().unwrap_or(first);
    let interp = |s: f32| -> (Vec2, bool) {
      if s <= 0.0 {
        return first;
      }
      if s >= elem_len {
        return last;
      }
      let mut j = 1;
      while j + 1 < m && cum.get(j).copied().unwrap_or(f32::MAX) < s {
        j += 1;
      }
      let (a, _) = verts.get(j - 1).copied().unwrap_or(first);
      let (b, _) = verts.get(j).copied().unwrap_or(last);
      let c1 = cum.get(j - 1).copied().unwrap_or(0.0);
      let c2 = cum.get(j).copied().unwrap_or(elem_len);
      let seg = (c2 - c1).max(1e-9);
      let t = (s - c1) / seg;
      (Vec2::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t), false)
    };
    // collect vertices from local arclength s0..s1 inclusive, preserving
    // interior vertices strictly between.
    let collect = |s0: f32, s1: f32| -> Vec<(Vec2, bool)> {
      let mut span = Vec::new();
      span.push(interp(s0));
      for (i, v) in verts.iter().enumerate() {
        let c = cum.get(i).copied().unwrap_or(0.0);
        if c > s0 + 1e-6 && c < s1 - 1e-6 {
          span.push(*v);
        }
      }
      span.push(interp(s1));
      span
    };

    let mut local = 0.0f32;
    let mut remaining = elem_len;
    if remaining <= self.current_length {
      self.current_length -= remaining;
      self.add_span(&collect(0.0, elem_len));
    } else {
      while remaining > self.current_length {
        remaining -= self.current_length;
        let target = local + self.current_length;
        self.add_span(&collect(local, target));
        self.update_active_segment();
        local = target;
      }
      // rlottie drops a sub-pixel path remainder at an element end
      // (vdasher.cpp `if (length > 1.0f)`).
      if remaining > 1.0 {
        self.current_length -= remaining;
        self.add_span(&collect(local, elem_len));
      }
    }
    // ... and snaps a sub-pixel dash remainder to the next phase
    // (vdasher.cpp `if (mCurrentLength < 1.0f) updateActiveSegment()`).
    if self.current_length < 1.0 {
      self.update_active_segment();
    }
  }

  fn run(&mut self) {
    let n = self.points.len();
    if self.pairs.is_empty() || n < 2 {
      return;
    }
    // rlottie VDasher::dashed() early-out order (must match exactly):
    //   1. mNoLength && mNoGap → return path (solid)
    //   2. mNoLength          → return {} (empty)
    //   3. mNoGap             → return path (solid)
    let no_length = self.pairs.iter().all(|(l, _)| l.abs() < VDASH_EPS);
    let no_gap = self.pairs.iter().all(|(_, g)| g.abs() < VDASH_EPS);
    if no_length && !no_gap {
      return;
    }
    if no_length || no_gap {
      let mut piece = (Vec::new(), Vec::new());
      for (i, p) in self.points.iter().enumerate() {
        piece.0.push(*p);
        piece.1.push(self.anchor_of(i));
      }
      self.out.push(piece);
      return;
    }

    // Element boundaries: authored-anchor indices. Empty anchors ⇒ every
    // point authored ⇒ each segment is its own element (rlottie lineTo
    // per segment).
    let mut bounds: Vec<usize> = Vec::new();
    bounds.push(0);
    for i in 1..n {
      if self.anchor_of(i) {
        bounds.push(i);
      }
    }
    if bounds.last().copied() != Some(n - 1) {
      bounds.push(n - 1);
    }

    self.move_to();

    for w in bounds.windows(2) {
      let (Some(&a), Some(&b)) = (w.first(), w.get(1)) else {
        continue;
      };
      let verts: Vec<(Vec2, bool)> = (a..=b).filter_map(|i| self.points.get(i).map(|p| (*p, self.anchor_of(i)))).collect();
      self.process_element(&verts);
    }
    // NOTE: closed contours would append a wrap element back to
    // points[0]; flatten_path already emits the closing vertex for
    // closed contours, and rlottie's VPath Close is a no-op in the
    // dasher (no first/last dash stitching), so open handling matches.

    // drop degenerate 1-point pieces
    self.out.retain(|p| p.0.len() >= 2);
  }
}
