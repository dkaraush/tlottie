//! Animated properties: keyframes, temporal easing, evaluation.
//!
//! Evaluation is pure (no caching yet); the monotonic-cursor fast path and
//! interned easing LUTs come with the performance phases.

use crate::math::{Color, Vec2};
use alloc::vec::Vec;

/// Temporal easing between two keyframes: a cubic bezier through (0,0),
/// (ox,oy), (ix,iy), (1,1) mapping normalized time to normalized progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Easing {
  pub ox: f32,
  pub oy: f32,
  pub ix: f32,
  pub iy: f32,
  pub hold: bool,
}

impl Easing {
  pub const LINEAR: Easing = Easing {
    ox: 1.0 / 3.0,
    oy: 1.0 / 3.0,
    ix: 2.0 / 3.0,
    iy: 2.0 / 3.0,
    hold: false,
  };
  pub const HOLD: Easing = Easing {
    ox: 0.0,
    oy: 0.0,
    ix: 1.0,
    iy: 1.0,
    hold: true,
  };

  /// Maps normalized segment time `t` in [0,1] to eased progress.
  pub fn ease(&self, t: f32) -> f32 {
    if self.hold {
      return 0.0;
    }
    let t = t.clamp(0.0, 1.0);
    // Fast path: control points on the diagonal = linear.
    if (self.ox - self.oy).abs() < 1e-6 && (self.ix - self.iy).abs() < 1e-6 {
      return t;
    }
    let x = solve_bezier_parameter(self.ox, self.ix, t);
    cubic_at(self.oy, self.iy, x)
  }
}

/// Cubic bezier value with endpoints 0 and 1: B(t) = 3(1-t)²t·c1 + 3(1-t)t²·c2 + t³.
fn cubic_at(c1: f32, c2: f32, t: f32) -> f32 {
  let omt = 1.0 - t;
  3.0 * omt * omt * t * c1 + 3.0 * omt * t * t * c2 + t * t * t
}

/// Finds parameter `u` such that x(u) == target, via Newton with bisection fallback.
fn solve_bezier_parameter(c1: f32, c2: f32, target: f32) -> f32 {
  let mut u = target; // good initial guess for well-behaved easings
  for _ in 0..8 {
    let x = cubic_at(c1, c2, u) - target;
    if x.abs() < 1e-5 {
      return u;
    }
    // dx/du of the cubic
    let omu = 1.0 - u;
    let d = 3.0 * omu * omu * c1 + 6.0 * omu * u * (c2 - c1) + 3.0 * u * u * (1.0 - c2);
    if d.abs() < 1e-6 {
      break;
    }
    u = (u - x / d).clamp(0.0, 1.0);
  }
  // Bisection fallback: x(u) is monotonic in u for valid easing curves.
  let (mut lo, mut hi) = (0.0f32, 1.0f32);
  for _ in 0..20 {
    let mid = 0.5 * (lo + hi);
    if cubic_at(c1, c2, mid) < target {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  0.5 * (lo + hi)
}

/// One keyframe: value at frame `t`, eased toward the next keyframe.
#[derive(Debug, Clone)]
pub(crate) struct Keyframe<T> {
  pub t: f32,
  pub value: T,
  /// Explicit segment end value (legacy `e` field); `None` means the next
  /// keyframe's `value` is the end.
  pub end: Option<T>,
  pub easing: Easing,
  /// Spatial bezier tangents `[to.x, to.y, ti.x, ti.y]` (position tracks
  /// only): the value travels a cubic curve, not a straight line.
  pub spatial: Option<[f32; 4]>,
}

/// Keyframe list that is non-empty *by construction* — the first keyframe is
/// stored inline, so evaluation never faces an empty list. Keyframes are in
/// FILE order; `sorted` says whether times are non-decreasing (the normal
/// case, evaluated by binary search). Unsorted lists exist in real files and
/// must be evaluated with rlottie's file-order semantics (lottiemodel.h
/// LOTAnimInfo::value): front/back range checks first, then the FIRST
/// segment in file order containing the frame.
#[derive(Debug, Clone)]
pub(crate) struct Timeline<T> {
  pub first: Keyframe<T>,
  pub rest: Vec<Keyframe<T>>,
  pub sorted: bool,
}

/// A possibly-animated property.
#[derive(Debug, Clone)]
pub(crate) enum Property<T> {
  Static(T),
  Animated(Timeline<T>),
}

/// Types that can be interpolated between keyframes.
pub(crate) trait Lerp: Clone {
  fn lerp(&self, other: &Self, t: f32) -> Self;

  /// Interpolation honoring spatial tangents; only meaningful for Vec2
  /// (positions). Default ignores the tangents.
  fn lerp_spatial(&self, other: &Self, t: f32, _spatial: &Option<[f32; 4]>) -> Self {
    self.lerp(other, t)
  }
}

impl Lerp for f32 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    self + (other - self) * t
  }
}

impl Lerp for Vec2 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Vec2::lerp(*self, *other, t)
  }

  fn lerp_spatial(&self, other: &Self, t: f32, spatial: &Option<[f32; 4]>) -> Self {
    let Some([tox, toy, tix, tiy]) = spatial else {
      return Vec2::lerp(*self, *other, t);
    };
    // Cubic bezier P0 → P0+to → P1+ti → P1, walked by *arc length*:
    // eased progress means distance travelled along the curve, not curve
    // parameter (rlottie VBezier::tAtLength / lottie-web bezierData).
    let c1 = Vec2::new(self.x + tox, self.y + toy);
    let c2 = Vec2::new(other.x + tix, other.y + tiy);
    let at = |u: f32| -> Vec2 {
      let omu = 1.0 - u;
      let a = omu * omu * omu;
      let b = 3.0 * omu * omu * u;
      let c = 3.0 * omu * u * u;
      let d = u * u * u;
      Vec2::new(a * self.x + b * c1.x + c * c2.x + d * other.x, a * self.y + b * c1.y + c * c2.y + d * other.y)
    };
    let t = t.clamp(0.0, 1.0);
    const N: usize = 32;
    let mut cum = [0.0f32; N + 1];
    let mut prev = *self;
    let mut total = 0.0f32;
    for (i, slot) in cum.iter_mut().enumerate().skip(1) {
      let p = at(i as f32 / N as f32);
      // rlottie's VLine::length approximation, NOT Euclidean — the
      // angle-dependent bias shifts where eased progress lands on the
      // curve (measured 7.6px → <0.3px on GameEmoji fly-ins).
      total += crate::geometry::seg_len_rlottie(p.x - prev.x, p.y - prev.y);
      *slot = total;
      prev = p;
    }
    if !(total > 1e-6) || !total.is_finite() {
      return at(t);
    }
    let target = t * total;
    // First sample index whose cumulative length reaches the target.
    let hi = cum.partition_point(|&len| len < target).min(N);
    let lo = hi.saturating_sub(1);
    let (l0, l1) = (cum.get(lo).copied().unwrap_or(0.0), cum.get(hi).copied().unwrap_or(total));
    let frac = if l1 > l0 { (target - l0) / (l1 - l0) } else { 0.0 };
    at((lo as f32 + frac) / N as f32)
  }
}

impl Lerp for Color {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Color::lerp(*self, *other, t)
  }
}

impl<T: Lerp> Property<T> {
  pub fn eval(&self, frame: f32) -> T {
    match self {
      Property::Static(v) => v.clone(),
      Property::Animated(tl) => tl.eval(frame),
    }
  }

  pub(crate) fn map_values(&mut self, mut map: impl FnMut(&mut T)) {
    match self {
      Property::Static(value) => map(value),
      Property::Animated(timeline) => {
        map(&mut timeline.first.value);
        if let Some(end) = &mut timeline.first.end {
          map(end);
        }
        for keyframe in &mut timeline.rest {
          map(&mut keyframe.value);
          if let Some(end) = &mut keyframe.end {
            map(end);
          }
        }
      }
    }
  }

  pub fn is_static(&self) -> bool {
    matches!(self, Property::Static(_))
  }
}

impl<T: Lerp> Timeline<T> {
  fn eval(&self, frame: f32) -> T {
    if frame <= self.first.t {
      return self.first.value.clone();
    }
    let last = self.rest.last().unwrap_or(&self.first);
    if frame >= last.t {
      return last.value.clone();
    }
    if !self.sorted {
      return self.eval_file_order(frame, last);
    }
    // frame is strictly between first.t and last.t, so a segment exists.
    // idx = number of keyframes in `rest` with t <= frame.
    let idx = self.rest.partition_point(|k| k.t <= frame);
    let k0 = match idx.checked_sub(1).map(|i| self.rest.get(i)) {
      Some(Some(k)) => k,
      _ => &self.first,
    };
    let Some(k1) = self.rest.get(idx) else {
      return last.value.clone(); // unreachable: frame < last.t
    };
    Self::segment(k0, k1, frame)
  }

  /// rlottie semantics for non-monotonic keyframe lists: linear scan in
  /// FILE order, first segment with k0.t <= frame < k1.t wins. (With the
  /// front/back checks above this is what LOTAnimInfo::value computes;
  /// descending prefixes make their segments empty, so e.g. palindromic
  /// lists resolve to constants exactly like the reference.)
  fn eval_file_order(&self, frame: f32, last: &Keyframe<T>) -> T {
    let mut prev = &self.first;
    for k in &self.rest {
      if prev.t <= frame && frame < k.t {
        return Self::segment(prev, k, frame);
      }
      prev = k;
    }
    last.value.clone()
  }

  fn segment(k0: &Keyframe<T>, k1: &Keyframe<T>, frame: f32) -> T {
    let span = k1.t - k0.t;
    if span <= 0.0 || !span.is_finite() {
      return k1.value.clone();
    }
    let t = (frame - k0.t) / span;
    let eased = k0.easing.ease(t);
    let end = k0.end.as_ref().unwrap_or(&k1.value);
    k0.value.lerp_spatial(end, eased, &k0.spatial)
  }
}

#[cfg(test)]
#[path = "tests/property.rs"]
mod tests;
