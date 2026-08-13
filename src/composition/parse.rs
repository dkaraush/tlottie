//! Lottie JSON → model parser.
//!
//! Strategy: a single forward pass per object; values needed out of order
//! (e.g. a property's `k` whose interpretation depends on shape, a shape
//! item's `ty`) get their byte position recorded and are re-parsed from a
//! forked cursor afterwards. Field order therefore never matters.
//!
//! Unsupported shape/layer types are skipped silently for now; the
//! supported-feature report lands with a later phase.

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::composition::options::{LayerColorReplacement, ParseOptions, SourceColorReplacement};
use crate::error::{Error, JsonErrorKind, Limit, Result};
use crate::json::Cursor;
use crate::limits::Limits;
use crate::math::{Color, Vec2};
use crate::model::{
  Asset, Composition, DashElement, EllipseShape, Fill, FillRule, FloatList, GradientFill, GradientKind, GradientStroke, Group, Layer, LayerKind, Mask, PathData, PathShape, PolystarShape, Position,
  RectShape, Repeater, RoundCorners, Shape, Stroke, Transform, Trim, TrimMode,
};
use crate::property::{Easing, Keyframe, Lerp, Property, Timeline};
use crate::stroke::{Cap, Join};
use alloc::vec;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Maximum nesting of `gr` shape groups (separate from raw JSON depth).
const MAX_GROUP_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// Object / array walking helpers
// ---------------------------------------------------------------------------

fn invalid(c: &Cursor<'_>, what: &'static str) -> Error {
  Error::InvalidLottie { offset: c.pos(), what }
}

fn json_err(c: &Cursor<'_>, kind: JsonErrorKind) -> Error {
  Error::Json { offset: c.pos(), kind }
}

/// Iterates fields of a JSON object. The callback must consume each value
/// (parse it or `skip_value`).
fn for_each_field<'a>(c: &mut Cursor<'a>, mut f: impl FnMut(&mut Cursor<'a>, &'a [u8]) -> Result<()>) -> Result<()> {
  c.skip_ws();
  c.expect(b'{')?;
  c.skip_ws();
  if c.peek() == Some(b'}') {
    c.bump();
    return Ok(());
  }
  loop {
    c.skip_ws();
    let key = c.read_string_bytes()?;
    c.skip_ws();
    c.expect(b':')?;
    c.skip_ws();
    f(c, key)?;
    c.skip_ws();
    match c.bump() {
      Some(b',') => {}
      Some(b'}') => return Ok(()),
      Some(b) => return Err(json_err(c, JsonErrorKind::UnexpectedByte(b))),
      None => return Err(json_err(c, JsonErrorKind::UnexpectedEof)),
    }
  }
}

#[derive(Default)]
struct SeenFields(u64);

impl SeenFields {
  #[inline(always)]
  fn first(&mut self, bit: u32) -> bool {
    debug_assert!(bit < 64);
    let mask = 1u64 << bit;
    let first = self.0 & mask == 0;
    self.0 |= mask;
    first
  }
}

macro_rules! match_once {
  ($seen:ident, $key:expr, { $($field:literal => $body:block $(,)?) * _ => $fallback:block $(,)? }) => {
    match_once!(@build $seen, $key, 0u32, (), $($field => $body,)* _ => $fallback)
  };
  (@build $seen:ident, $key:expr, $bit:expr, ($($arms:tt)*), _ => $fallback:block) => {{
    match $key {
      $($arms)*
      _ => $fallback,
    }
  }};
  (@build $seen:ident, $key:expr, $bit:expr, ($($arms:tt)*), $field:literal => $body:block, $($rest:tt)*) => {
    match_once!(@build $seen, $key, (($bit) + 1u32), (
      $($arms)*
      $field if $seen.first($bit) => $body,
    ), $($rest)*)
  };
}

macro_rules! parse_object_once {
  ($cursor:expr, |$value:ident, $key:ident| { $($field:literal => $body:block $(,)?) * _ => $fallback:block $(,)? }) => {{
    let mut seen = SeenFields::default();
    for_each_field($cursor, |$value, $key| {
      match_once!(seen, $key, { $($field => $body,)* _ => $fallback });
      Ok(())
    })
  }};
}

/// Iterates elements of a JSON array. The callback must consume each element.
fn for_each_element<'a>(c: &mut Cursor<'a>, mut f: impl FnMut(&mut Cursor<'a>) -> Result<()>) -> Result<()> {
  c.skip_ws();
  c.expect(b'[')?;
  c.skip_ws();
  if c.peek() == Some(b']') {
    c.bump();
    return Ok(());
  }
  loop {
    c.skip_ws();
    f(c)?;
    c.skip_ws();
    match c.bump() {
      Some(b',') => {}
      Some(b']') => return Ok(()),
      Some(b) => return Err(json_err(c, JsonErrorKind::UnexpectedByte(b))),
      None => return Err(json_err(c, JsonErrorKind::UnexpectedEof)),
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar value parsers
// ---------------------------------------------------------------------------

fn parse_f32(c: &mut Cursor<'_>) -> Result<f32> {
  Ok(c.parse_f64()? as f32)
}

/// Number, or array whose first element is a number (`[12.5]`), as scalars
/// appear both ways in Lottie.
fn parse_scalar(c: &mut Cursor<'_>) -> Result<f32> {
  c.skip_ws();
  if c.peek() == Some(b'[') {
    let mut value: Option<f32> = None;
    for_each_element(c, |c| {
      if value.is_none() {
        value = Some(parse_f32(c)?);
      } else {
        // Extra scalar-array components are ignored by the model; validate
        // them as JSON without paying number parsing costs for huge tails.
        c.skip_value()?;
      }
      Ok(())
    })?;
    value.ok_or_else(|| invalid(c, "empty scalar array"))
  } else {
    parse_f32(c)
  }
}

/// `[x, y, ...]` (extra components ignored).
fn parse_vec2(c: &mut Cursor<'_>) -> Result<Vec2> {
  let mut got: [Option<f32>; 2] = [None, None];
  let mut i = 0usize;
  for_each_element(c, |c| {
    if let Some(slot) = got.get_mut(i) {
      *slot = Some(parse_f32(c)?);
    } else {
      // Vec2 tails are ignored. `skip_value` still validates JSON structure
      // but avoids str::parse on attacker-sized numeric arrays.
      c.skip_value()?;
    }
    i += 1;
    Ok(())
  })?;
  match got {
    [Some(x), Some(y)] => Ok(Vec2::new(x, y)),
    [Some(x), None] => Ok(Vec2::new(x, x)),
    _ => Err(invalid(c, "expected [x, y]")),
  }
}

/// `[r, g, b]` or `[r, g, b, a]`, components 0..=1 (255-scale tolerated).
fn parse_color(c: &mut Cursor<'_>) -> Result<Color> {
  let mut comps: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
  let mut i = 0usize;
  for_each_element(c, |c| {
    if let Some(slot) = comps.get_mut(i) {
      *slot = parse_f32(c)?;
    } else {
      c.skip_value()?;
    }
    i += 1;
    Ok(())
  })?;
  if i < 3 {
    return Err(invalid(c, "color needs at least 3 components"));
  }
  // rlottie takes components as-is (no 0..255 rescale heuristic — real
  // files carry rounding artifacts like 1.0001); clamp to the valid range.
  // The 4th array component is DISCARDED (rlottie getValue(LottieColor)
  // reads only rgb; paint alpha flows exclusively from the `o` property —
  // SunEmoji mouths carry a=0.52 that the reference never applies).
  Ok(Color {
    r: comps[0].clamp(0.0, 1.0),
    g: comps[1].clamp(0.0, 1.0),
    b: comps[2].clamp(0.0, 1.0),
    a: 1.0,
  })
}

fn parse_f32_list(c: &mut Cursor<'_>, max_len: usize, limit: Limit) -> Result<FloatList> {
  let mut out = Vec::new();
  for_each_element(c, |c| {
    if out.len() >= max_len {
      return Err(Error::LimitExceeded(limit));
    }
    out.push(parse_f32(c)?);
    Ok(())
  })?;
  Ok(FloatList(out))
}

fn parse_bool(c: &mut Cursor<'_>) -> Result<bool> {
  c.skip_ws();
  match c.peek() {
    Some(b't') | Some(b'f') => {
      let pos = c.pos();
      c.skip_value()?;
      // skip_value validated the keyword; length tells which it was.
      Ok(c.pos() - pos == 4)
    }
    Some(b'0'..=b'9' | b'-') => Ok(parse_f32(c)? != 0.0),
    _ => Err(invalid(c, "expected boolean")),
  }
}

/// Path contour object `{c, v, i, o}`. Also accepts the array-wrapped form
/// `[{...}]` used in keyframe `s` fields.
fn parse_vec2_list(c: &mut Cursor<'_>, limits: &Limits) -> Result<Vec<Vec2>> {
  let mut out = Vec::new();
  for_each_element(c, |c| {
    if out.len() >= limits.max_path_points {
      return Err(Error::LimitExceeded(Limit::PathPoints));
    }
    let point = parse_vec2(c)?;
    if point.x.abs() > limits.max_path_coordinate_abs || point.y.abs() > limits.max_path_coordinate_abs {
      return Err(Error::LimitExceeded(Limit::PathCoordinate));
    }
    out.push(point);
    Ok(())
  })?;
  Ok(out)
}

fn parse_path_value(c: &mut Cursor<'_>, limits: &Limits) -> Result<PathData> {
  c.skip_ws();
  if c.peek() == Some(b'[') {
    let mut path: Option<PathData> = None;
    for_each_element(c, |c| {
      if path.is_none() {
        path = Some(parse_path_object(c, limits)?);
      } else {
        c.skip_value()?;
      }
      Ok(())
    })?;
    return path.ok_or_else(|| invalid(c, "empty path keyframe value"));
  }
  parse_path_object(c, limits)
}

fn parse_path_object(c: &mut Cursor<'_>, limits: &Limits) -> Result<PathData> {
  let mut data = PathData::default();
  parse_object_once!(c, |c, key| {
    b"v" => { data.vertices = parse_vec2_list(c, limits)?; },
    b"i" => { data.in_tangents = parse_vec2_list(c, limits)?; },
    b"o" => { data.out_tangents = parse_vec2_list(c, limits)?; },
    b"c" => { data.closed = parse_bool(c)?; },
    _ => { c.skip_value()?; },
  })?;
  let n = data.vertices.len();
  // Tolerate missing/short tangent arrays by padding with zeros (= corners).
  data.in_tangents.resize(n, Vec2::ZERO);
  data.out_tangents.resize(n, Vec2::ZERO);
  data.in_tangents.truncate(n);
  data.out_tangents.truncate(n);
  Ok(data)
}

// ---------------------------------------------------------------------------
// Properties (static or keyframed)
// ---------------------------------------------------------------------------

/// Parses a property object `{a, k, ...}`. Whether `k` is static or animated
/// is decided by its own shape (array of objects = keyframes), not by `a`.
fn parse_property<T: Lerp + PartialEq, F: Fn(&mut Cursor<'_>) -> Result<T> + Copy>(c: &mut Cursor<'_>, limits: &Limits, parse_val: F) -> Result<Property<T>> {
  let mut property: Option<Property<T>> = None;
  parse_object_once!(c, |c, key| {
    b"k" => { property = Some(parse_property_k(c, limits, parse_val)?); },
    _ => { c.skip_value()?; },
  })?;
  property.ok_or_else(|| invalid(c, "property missing k"))
}

fn parse_property_k<T: Lerp + PartialEq, F: Fn(&mut Cursor<'_>) -> Result<T> + Copy>(kc: &mut Cursor<'_>, limits: &Limits, parse_val: F) -> Result<Property<T>> {
  kc.skip_ws();
  if kc.peek() == Some(b'[') {
    // Peek inside: array of objects → keyframes; anything else → static.
    let mut probe = kc.fork_at(kc.pos() + 1);
    probe.skip_ws();
    if probe.peek() == Some(b'{') {
      let property = parse_keyframes(kc, limits, parse_val)?;
      if matches!(property, Property::Animated(_)) {
        kc.mark_animated_property();
      }
      return Ok(property);
    }
  }
  Ok(Property::Static(parse_val(kc)?))
}

struct RawKeyframe<T> {
  t: f32,
  value: Option<T>,
  end: Option<T>,
  easing: Easing,
  spatial: Option<[f32; 4]>,
}

/// Parses `[{t, s, e?, i, o, h?}, ...]` into a non-empty Timeline.
fn parse_keyframes<T: Lerp + PartialEq, F: Fn(&mut Cursor<'_>) -> Result<T> + Copy>(c: &mut Cursor<'_>, limits: &Limits, parse_val: F) -> Result<Property<T>> {
  let mut raw: Vec<RawKeyframe<T>> = Vec::new();
  for_each_element(c, |c| {
    if raw.len() >= limits.max_keyframes {
      return Err(Error::LimitExceeded(Limit::Keyframes));
    }
    raw.push(parse_one_keyframe(c, parse_val)?);
    Ok(())
  })?;

  // Resolve values: a keyframe without `s` (common for the final keyframe)
  // takes the previous keyframe's segment end value.
  let mut kfs: Vec<Keyframe<T>> = Vec::with_capacity(raw.len());
  for rk in raw {
    let value = match rk.value {
      Some(v) => v,
      None => match kfs.last() {
        Some(prev) => prev.end.clone().unwrap_or_else(|| prev.value.clone()),
        None => return Err(invalid(c, "first keyframe has no value")),
      },
    };
    kfs.push(Keyframe {
      t: rk.t,
      value,
      end: rk.end,
      easing: rk.easing,
      spatial: rk.spatial,
    });
  }
  // Keyframes stay in FILE ORDER — rlottie never sorts them. Some real
  // files carry non-monotonic time lists (e.g. palindromic 39,36,..,36,39)
  // which rlottie's front/back checks collapse to a constant; sorting
  // would invent an animation the reference never plays.
  let sorted = kfs.windows(2).all(|w| match w {
    [a, b] => a.t <= b.t,
    _ => true,
  });
  let mut it = kfs.into_iter();
  let first = it.next().ok_or_else(|| invalid(c, "empty keyframe list"))?;
  let rest: Vec<Keyframe<T>> = it.collect();
  if timeline_is_constant(&first, &rest) {
    return Ok(Property::Static(first.value));
  }
  Ok(Property::Animated(Timeline { first, rest, sorted }))
}

fn timeline_is_constant<T: PartialEq>(first: &Keyframe<T>, rest: &[Keyframe<T>]) -> bool {
  if first.spatial.is_some() {
    return false;
  }
  let value = &first.value;
  if first.end.as_ref().is_some_and(|end| end != value) {
    return false;
  }
  rest
    .iter()
    .all(|keyframe| keyframe.spatial.is_none() && &keyframe.value == value && keyframe.end.as_ref().is_none_or(|end| end == value))
}

fn parse_one_keyframe<T: Lerp, F: Fn(&mut Cursor<'_>) -> Result<T> + Copy>(c: &mut Cursor<'_>, parse_val: F) -> Result<RawKeyframe<T>> {
  let mut t = 0.0f32;
  let mut value: Option<T> = None;
  let mut end: Option<T> = None;
  let mut hold = false;
  let mut o = (Easing::LINEAR.ox, Easing::LINEAR.oy);
  let mut i = (Easing::LINEAR.ix, Easing::LINEAR.iy);
  let mut to: Option<Vec2> = None;
  let mut ti: Option<Vec2> = None;
  parse_object_once!(c, |c, key| {
    b"t" => { t = parse_f32(c)?; },
    b"s" => { value = Some(parse_val(c)?); },
    b"e" => { end = Some(parse_val(c)?); },
    b"h" => { hold = parse_bool(c)?; },
    b"o" => { o = parse_easing_handle(c)?; },
    b"i" => { i = parse_easing_handle(c)?; },
    b"to" => { to = parse_vec2(c).ok(); },
    b"ti" => { ti = parse_vec2(c).ok(); },
    _ => { c.skip_value()?; },
  })?;
  let easing = if hold {
    Easing::HOLD
  } else {
    let [ox, oy, ix, iy] = c.intern_easing([o.0, o.1, i.0, i.1]);
    Easing { ox, oy, ix, iy, hold: false }
  };
  let spatial = match (to, ti) {
    (Some(t0), Some(t1)) if t0 != Vec2::ZERO || t1 != Vec2::ZERO => Some([t0.x, t0.y, t1.x, t1.y]),
    (Some(t0), None) if t0 != Vec2::ZERO => Some([t0.x, t0.y, 0.0, 0.0]),
    (None, Some(t1)) if t1 != Vec2::ZERO => Some([0.0, 0.0, t1.x, t1.y]),
    _ => None,
  };
  Ok(RawKeyframe { t, value, end, easing, spatial })
}

/// `{x: n|[n], y: n|[n]}` easing handle.
fn parse_easing_handle(c: &mut Cursor<'_>) -> Result<(f32, f32)> {
  let mut x = 0.0f32;
  let mut y = 0.0f32;
  parse_object_once!(c, |c, key| {
    b"x" => { x = parse_scalar(c)?; },
    b"y" => { y = parse_scalar(c)?; },
    _ => { c.skip_value()?; },
  })?;
  // X (time axis) is NOT clamped: AE exports overshoot/anticipation
  // handles with x outside [0,1] and rlottie's VInterpolator uses them
  // verbatim; clamping shifted eased progress on fast segments (TONEmoji
  // coin 12% oversize). Y stays bounded only against absurd values.
  Ok((x.clamp(-100.0, 100.0), y.clamp(-100.0, 100.0)))
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

fn parse_transform(c: &mut Cursor<'_>, limits: &Limits) -> Result<Transform> {
  let mut tf = Transform::identity();
  parse_object_once!(c, |c, key| {
    b"a" => { tf.anchor = parse_property(c, limits, parse_vec2)?; },
    b"p" => { tf.position = parse_position(c, limits)?; },
    b"s" => { tf.scale = parse_property(c, limits, parse_vec2)?; },
    b"r" => { tf.rotation = parse_property(c, limits, parse_scalar)?; },
    b"o" => { tf.opacity = parse_property(c, limits, parse_scalar)?; },
    b"sk" => { tf.skew = parse_property(c, limits, parse_scalar)?; },
    b"sa" => { tf.skew_axis = parse_property(c, limits, parse_scalar)?; },
    _ => { c.skip_value()?; },
  })?;
  Ok(tf)
}

/// Position property: either a normal `{a,k}` Vec2 property or the split
/// form `{s: true, x: {a,k}, y: {a,k}}`.
fn parse_position(c: &mut Cursor<'_>, limits: &Limits) -> Result<Position> {
  let mut split = false;
  let mut k: Option<Property<Vec2>> = None;
  let mut x: Option<Property<f32>> = None;
  let mut y: Option<Property<f32>> = None;
  let mut x_pos: Option<usize> = None;
  let mut y_pos: Option<usize> = None;
  parse_object_once!(c, |c, key| {
    b"s" => { split = parse_bool(c)?; },
    b"k" => { k = Some(parse_property_k(c, limits, parse_vec2)?); },
    b"x" => {
      if split {
        x = Some(parse_property(c, limits, parse_scalar)?);
      } else {
        x_pos = Some(c.pos());
        c.skip_value()?;
      }
    },
    b"y" => {
      if split {
        y = Some(parse_property(c, limits, parse_scalar)?);
      } else {
        y_pos = Some(c.pos());
        c.skip_value()?;
      }
    },
    _ => { c.skip_value()?; },
  })?;
  if split {
    let x = match (x, x_pos) {
      (Some(x), _) => x,
      (None, Some(pos)) => parse_property(&mut c.fork_at(pos), limits, parse_scalar)?,
      (None, None) => return Err(invalid(c, "split position missing x")),
    };
    let y = match (y, y_pos) {
      (Some(y), _) => y,
      (None, Some(pos)) => parse_property(&mut c.fork_at(pos), limits, parse_scalar)?,
      (None, None) => return Err(invalid(c, "split position missing y")),
    };
    Ok(Position::Split { x, y })
  } else {
    let p = k.ok_or_else(|| invalid(c, "position missing k"))?;
    Ok(Position::Combined(p))
  }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

enum ParsedItem {
  Shape(Shape),
  GroupTransform(Transform),
  Ignored,
}

#[derive(Default)]
struct ShapeCounts {
  items: usize,
  paints: usize,
  paint_source_items: usize,
  focal_radial_gradients: usize,
  round_corners: usize,
  trims: usize,
}

fn parse_shape_list(c: &mut Cursor<'_>, limits: &Limits, depth: usize, count: &mut ShapeCounts) -> Result<(Vec<Shape>, Option<Transform>)> {
  if depth > MAX_GROUP_DEPTH {
    return Err(Error::LimitExceeded(Limit::NestingDepth));
  }
  let mut shapes = Vec::new();
  let mut transform: Option<Transform> = None;
  let mut dashed_strokes = 0usize;
  let mut gradient_strokes = 0usize;
  let mut max_dashed_source_segment = 0.0f32;
  let mut dashed_source_len = 0.0f32;
  let mut dashed_piece_estimate = 0usize;
  let mut repeater_product = 1usize;
  let mut source_items = 0usize;
  for_each_element(c, |c| {
    count.items += 1;
    if count.items > limits.max_shapes_per_layer {
      return Err(Error::LimitExceeded(Limit::ShapesPerLayer));
    }
    match parse_shape_item(c, limits, depth, count)? {
      ParsedItem::Shape(s) => {
        if is_paint(&s) {
          count.paints += 1;
          if count.paints > limits.max_paints_per_layer {
            return Err(Error::LimitExceeded(Limit::PaintsPerLayer));
          }
          count.paint_source_items = count.paint_source_items.saturating_add(source_items);
          if count.paint_source_items > limits.max_paint_source_items_per_layer {
            return Err(Error::LimitExceeded(Limit::PaintSourceItemsPerLayer));
          }
        }
        if is_focal_radial_gradient(&s) {
          count.focal_radial_gradients += 1;
          if count.focal_radial_gradients > limits.max_focal_radial_gradients_per_layer {
            return Err(Error::LimitExceeded(Limit::FocalRadialGradientsPerLayer));
          }
        }
        if matches!(s, Shape::RoundCorners(_)) {
          count.round_corners += 1;
          if count.round_corners > limits.max_round_corners_per_layer {
            return Err(Error::LimitExceeded(Limit::RoundCornersPerLayer));
          }
        }
        if matches!(s, Shape::Trim(_)) {
          count.trims += 1;
          if count.trims > limits.max_trims_per_layer {
            return Err(Error::LimitExceeded(Limit::TrimsPerLayer));
          }
        }
        if let Some(copies) = repeater_copies(&s) {
          repeater_product = repeater_product.saturating_mul(copies.max(1));
          if repeater_product > limits.max_repeater_product_per_group {
            return Err(Error::LimitExceeded(Limit::RepeaterProductPerGroup));
          }
        }
        if is_dashed_stroke(&s) {
          dashed_strokes += 1;
          if dashed_strokes > limits.max_dashed_strokes_per_group {
            return Err(Error::LimitExceeded(Limit::DashedStrokesPerGroup));
          }
          if is_round_join_dashed_stroke(&s) && max_dashed_source_segment > limits.max_dashed_path_segment_span {
            return Err(Error::LimitExceeded(Limit::DashedPathSegment));
          }
          if let Some(pieces) = dash_piece_estimate(&s, dashed_source_len) {
            dashed_piece_estimate = dashed_piece_estimate.saturating_add(pieces);
            if dashed_piece_estimate > limits.max_dashed_piece_estimate_per_group {
              return Err(Error::LimitExceeded(Limit::DashedPiecesPerGroup));
            }
          }
        }
        if matches!(s, Shape::GradientStroke(_)) {
          gradient_strokes += 1;
          if gradient_strokes > limits.max_gradient_strokes_per_group {
            return Err(Error::LimitExceeded(Limit::GradientStrokesPerGroup));
          }
        }
        max_dashed_source_segment = max_dashed_source_segment.max(max_path_segment_span(&s));
        dashed_source_len += path_segment_span_sum(&s);
        if is_geometry_source(&s) {
          source_items += 1;
        }
        if shapes.last().is_some_and(|previous| redundant_opaque_gradient_fill(previous, &s)) {
          return Ok(());
        }
        shapes.push(s);
      }
      ParsedItem::GroupTransform(t) => transform = Some(t),
      ParsedItem::Ignored => {}
    }
    Ok(())
  })?;
  Ok((shapes, transform))
}

fn is_paint(shape: &Shape) -> bool {
  matches!(shape, Shape::Fill(_) | Shape::Stroke(_) | Shape::GradientFill(_) | Shape::GradientStroke(_))
}

fn is_geometry_source(shape: &Shape) -> bool {
  matches!(shape, Shape::Path(_) | Shape::Rect(_) | Shape::Ellipse(_) | Shape::Polystar(_) | Shape::Group(_))
}

fn is_focal_radial_gradient(shape: &Shape) -> bool {
  match shape {
    Shape::GradientFill(fill) => fill.kind == GradientKind::Radial && fill.highlight_len.eval(0.0).abs() > 0.001,
    Shape::GradientStroke(stroke) => stroke.kind == GradientKind::Radial && stroke.highlight_len.eval(0.0).abs() > 0.001,
    _ => false,
  }
}

fn redundant_opaque_gradient_fill(previous: &Shape, next: &Shape) -> bool {
  let (Shape::GradientFill(previous), Shape::GradientFill(next)) = (previous, next) else {
    return false;
  };
  previous.kind == next.kind
    && previous.rule == next.rule
    && previous.color_count == next.color_count
    && same_static_property(&previous.start, &next.start)
    && same_static_property(&previous.end, &next.end)
    && same_static_property(&previous.highlight_len, &next.highlight_len)
    && same_static_property(&previous.highlight_angle, &next.highlight_angle)
    && same_static_property(&previous.stops, &next.stops)
    && same_opaque_static_opacity(&previous.opacity, &next.opacity)
    && opaque_gradient_stops(&previous.stops, previous.color_count)
}

fn same_static_property<T: PartialEq>(a: &Property<T>, b: &Property<T>) -> bool {
  match (a, b) {
    (Property::Static(a), Property::Static(b)) => a == b,
    _ => false,
  }
}

fn same_opaque_static_opacity(a: &Property<f32>, b: &Property<f32>) -> bool {
  match (a, b) {
    (Property::Static(a), Property::Static(b)) => a == b && *a >= 100.0,
    _ => false,
  }
}

fn opaque_gradient_stops(stops: &Property<FloatList>, color_count: usize) -> bool {
  let Property::Static(stops) = stops else {
    return false;
  };
  stops.0.len() == color_count.saturating_mul(4)
}

fn is_dashed_stroke(shape: &Shape) -> bool {
  match shape {
    Shape::Stroke(stroke) => !stroke.dashes.is_empty(),
    Shape::GradientStroke(stroke) => !stroke.dashes.is_empty(),
    _ => false,
  }
}

fn is_round_join_dashed_stroke(shape: &Shape) -> bool {
  match shape {
    Shape::Stroke(stroke) => !stroke.dashes.is_empty() && stroke.join == Join::Round,
    Shape::GradientStroke(stroke) => !stroke.dashes.is_empty() && stroke.join == Join::Round,
    _ => false,
  }
}

fn repeater_copies(shape: &Shape) -> Option<usize> {
  let Shape::Repeater(repeater) = shape else {
    return None;
  };
  let copies = property_max_abs_f32(&repeater.copies)?.ceil();
  if copies.is_finite() {
    Some(copies as usize)
  } else {
    Some(usize::MAX)
  }
}

fn dash_piece_estimate(shape: &Shape, source_len: f32) -> Option<usize> {
  let dashes = match shape {
    Shape::Stroke(stroke) => &stroke.dashes,
    Shape::GradientStroke(stroke) => &stroke.dashes,
    _ => return None,
  };
  let period = dash_period(dashes)?;
  Some((source_len / period).ceil().max(0.0) as usize)
}

fn dash_period(dashes: &[DashElement]) -> Option<f32> {
  if dashes.len() < 2 {
    return None;
  }
  let mut values: Vec<f32> = dashes.iter().filter_map(|dash| property_min_abs_f32(&dash.value)).collect();
  if values.len() < 2 {
    return None;
  }
  // Match renderer dash_pattern(): role tags are ignored, an even-length list
  // gets a synthesized gap, then the final value is consumed as offset.
  if values.len() % 2 == 0 {
    let last = values.last().copied().unwrap_or(0.0);
    let prev = values.get(values.len() - 2).copied().unwrap_or(0.0);
    if let Some(slot) = values.last_mut() {
      *slot = prev;
    }
    values.push(last);
  }
  values.pop();
  let period: f32 = values.iter().map(|v| v.max(0.0)).sum();
  (period > 0.001).then_some(period)
}

fn property_min_abs_f32(property: &Property<f32>) -> Option<f32> {
  let finite_abs = |value: f32| value.is_finite().then_some(value.abs());
  match property {
    Property::Static(value) => finite_abs(*value),
    Property::Animated(timeline) => {
      let mut min = finite_abs(timeline.first.value);
      if let Some(value) = timeline.first.end.and_then(finite_abs) {
        min = Some(min.map_or(value, |current| current.min(value)));
      }
      for keyframe in &timeline.rest {
        if let Some(value) = finite_abs(keyframe.value) {
          min = Some(min.map_or(value, |current| current.min(value)));
        }
        if let Some(value) = keyframe.end.and_then(finite_abs) {
          min = Some(min.map_or(value, |current| current.min(value)));
        }
      }
      min
    }
  }
}

fn property_max_abs_f32(property: &Property<f32>) -> Option<f32> {
  let finite_abs = |value: f32| value.is_finite().then_some(value.abs());
  match property {
    Property::Static(value) => finite_abs(*value),
    Property::Animated(timeline) => {
      let mut max = finite_abs(timeline.first.value);
      if let Some(value) = timeline.first.end.and_then(finite_abs) {
        max = Some(max.map_or(value, |current| current.max(value)));
      }
      for keyframe in &timeline.rest {
        if let Some(value) = finite_abs(keyframe.value) {
          max = Some(max.map_or(value, |current| current.max(value)));
        }
        if let Some(value) = keyframe.end.and_then(finite_abs) {
          max = Some(max.map_or(value, |current| current.max(value)));
        }
      }
      max
    }
  }
}

fn max_path_segment_span(shape: &Shape) -> f32 {
  let Shape::Path(path) = shape else {
    return 0.0;
  };
  let data = path.path.eval(0.0);
  data
    .vertices
    .windows(2)
    .map(|w| match w {
      [a, b] => (b.x - a.x).abs().max((b.y - a.y).abs()),
      _ => 0.0,
    })
    .fold(0.0, f32::max)
}

fn path_segment_span_sum(shape: &Shape) -> f32 {
  let Shape::Path(path) = shape else {
    return 0.0;
  };
  let data = path.path.eval(0.0);
  data
    .vertices
    .windows(2)
    .map(|w| match w {
      [a, b] => (b.x - a.x).abs().max((b.y - a.y).abs()),
      _ => 0.0,
    })
    .sum()
}

fn parse_shape_item(c: &mut Cursor<'_>, limits: &Limits, depth: usize, count: &mut ShapeCounts) -> Result<ParsedItem> {
  // Record positions of every field we might need, dispatch after `ty` is known.
  let mut ty: Option<[u8; 2]> = None;
  let mut hidden = false;
  let mut it_pos: Option<usize> = None;
  let mut parsed_group: Option<(Vec<Shape>, Option<Transform>)> = None;
  let mut ks_pos: Option<usize> = None;
  let mut parsed_path: Option<Property<PathData>> = None;
  let mut p_pos: Option<usize> = None;
  let mut s_pos: Option<usize> = None;
  let mut r_pos: Option<usize> = None;
  let mut c_pos: Option<usize> = None;
  let mut o_pos: Option<usize> = None;
  let mut a_pos: Option<usize> = None;
  let mut w_pos: Option<usize> = None;
  let mut e_pos: Option<usize> = None;
  let mut g_pos: Option<usize> = None;
  let mut d_pos: Option<usize> = None;
  let mut pt_pos: Option<usize> = None;
  let mut ir_pos: Option<usize> = None;
  let mut or_pos: Option<usize> = None;
  let mut is_pos: Option<usize> = None;
  let mut os_pos: Option<usize> = None;
  let mut h_pos: Option<usize> = None;
  let mut tr_pos: Option<usize> = None;
  let mut star_type = 1.0f32;
  let mut grad_type = 1.0f32;
  let mut trim_mode = 1.0f32;
  let mut line_cap = 1.0f32;
  let mut line_join = 1.0f32;
  // rlottie's model default is 0; the FT stroker clamps the miter
  // limit up to a MINIMUM of 1.0 — absent or tiny ml means 1.0 (bevel
  // sharp corners), NOT the AE default of 4.
  let mut miter_limit = 0.0f32;
  let obj_start = c.pos();

  parse_object_once!(c, |c, key| {
    b"t" => {
      grad_type = parse_f32(c)?;
    }
    b"e" => {
      e_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"g" => {
      g_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"ty" => {
      let s = c.read_string_bytes()?;
      let mut tag = [0u8; 2];
      if let (Some(&b0), b1) = (s.first(), s.get(1)) {
        tag[0] = b0;
        tag[1] = b1.copied().unwrap_or(0);
      }
      ty = Some(tag);
    }
    b"hd" => {
      hidden = parse_bool(c)?;
    }
    b"it" => {
      if ty == Some(*b"gr") {
        parsed_group = Some(parse_shape_list(c, limits, depth + 1, count)?);
        it_pos = None;
      } else {
        it_pos = Some(c.pos());
        c.skip_value()?;
      }
    }
    b"ks" => {
      if ty == Some(*b"sh") {
        parsed_path = Some(parse_property(c, limits, |c| parse_path_value(c, limits))?);
        ks_pos = None;
      } else {
        ks_pos = Some(c.pos());
        c.skip_value()?;
      }
    }
    b"p" => {
      p_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"s" => {
      s_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"r" => {
      r_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"c" => {
      c_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"o" => {
      o_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"a" => {
      a_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"w" => {
      w_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"m" => {
      trim_mode = parse_f32(c)?;
    }
    b"sy" => {
      star_type = parse_f32(c)?;
    }
    b"h" => {
      h_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"tr" => {
      tr_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"d" => {
      d_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"pt" => {
      pt_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"ir" => {
      ir_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"or" => {
      or_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"is" => {
      is_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"os" => {
      os_pos = Some(c.pos());
      c.skip_value()?;
    }
    b"lc" => {
      line_cap = parse_f32(c)?;
    }
    b"lj" => {
      line_join = parse_f32(c)?;
    }
    b"ml" => {
      // Usually a plain number, but some exporters write it as an
      // animated-property object {"a":0,"k":4}; take the value at
      // frame 0 (animated miter limit is not a real-world case).
      c.skip_ws();
      if c.peek() == Some(b'{') {
        miter_limit = parse_property(c, limits, parse_f32)?.eval(0.0);
      } else {
        miter_limit = parse_f32(c)?;
      }
    }
    _ => { c.skip_value()? },
  })?;

  let Some(ty) = ty else {
    return Err(Error::InvalidLottie {
      offset: obj_start,
      what: "shape item missing ty",
    });
  };
  // rlottie's group sub-parser never reads `hd`, so hidden GROUPS still
  // render in the reference; only leaf shapes honor it (contract review).
  if hidden && &ty != b"gr" {
    return Ok(ParsedItem::Ignored);
  }

  let prop_vec2 = |c: &Cursor<'_>, pos: Option<usize>, what: &'static str| -> Result<Property<Vec2>> {
    let pos = pos.ok_or(Error::InvalidLottie { offset: obj_start, what })?;
    parse_property(&mut c.fork_at(pos), limits, parse_vec2)
  };
  let prop_scalar = |c: &Cursor<'_>, pos: Option<usize>, default: f32| -> Result<Property<f32>> {
    match pos {
      Some(pos) => parse_property(&mut c.fork_at(pos), limits, parse_scalar),
      None => Ok(Property::Static(default)),
    }
  };
  // Path direction on primitives: `d` == 3 reverses the winding. rlottie
  // supports this too, but accidentally ignores `d` if it appears before
  // `ty`; honor the authored field order-independently.
  let direction_reversed = |c: &Cursor<'_>, pos: Option<usize>| -> bool { pos.is_some_and(|p| parse_f32(&mut c.fork_at(p)).unwrap_or(1.0) == 3.0) };

  match &ty {
    b"gr" => {
      if let Some((shapes, transform)) = parsed_group {
        return Ok(ParsedItem::Shape(Shape::Group(Box::new(Group {
          transform: transform.unwrap_or_else(Transform::identity),
          shapes,
        }))));
      }
      let Some(it_pos) = it_pos else {
        return Ok(ParsedItem::Ignored); // empty group
      };
      let mut ic = c.fork_at(it_pos);
      let (shapes, transform) = parse_shape_list(&mut ic, limits, depth + 1, count)?;
      Ok(ParsedItem::Shape(Shape::Group(Box::new(Group {
        transform: transform.unwrap_or_else(Transform::identity),
        shapes,
      }))))
    }
    b"sh" => {
      let path = match parsed_path {
        Some(path) => path,
        None => {
          let ks_pos = ks_pos.ok_or(Error::InvalidLottie {
            offset: obj_start,
            what: "path shape missing ks",
          })?;
          parse_property(&mut c.fork_at(ks_pos), limits, |c| parse_path_value(c, limits))?
        }
      };
      Ok(ParsedItem::Shape(Shape::Path(PathShape { path })))
    }
    b"rc" => Ok(ParsedItem::Shape(Shape::Rect(RectShape {
      position: prop_vec2(c, p_pos, "rect missing p")?,
      size: prop_vec2(c, s_pos, "rect missing s")?,
      radius: prop_scalar(c, r_pos, 0.0)?,
      reversed: direction_reversed(c, d_pos),
    }))),
    b"el" => Ok(ParsedItem::Shape(Shape::Ellipse(EllipseShape {
      position: prop_vec2(c, p_pos, "ellipse missing p")?,
      size: prop_vec2(c, s_pos, "ellipse missing s")?,
      reversed: direction_reversed(c, d_pos),
    }))),
    b"fl" => {
      let c_pos = c_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "fill missing color",
      })?;
      let color = parse_property(&mut c.fork_at(c_pos), limits, parse_color)?;
      let opacity = prop_scalar(c, o_pos, 100.0)?;
      // In a fill item, `r` is the fill rule (int), not a property.
      let rule = match r_pos {
        Some(pos) => {
          if parse_f32(&mut c.fork_at(pos))? as i64 == 2 {
            FillRule::EvenOdd
          } else {
            FillRule::NonZero
          }
        }
        None => FillRule::NonZero,
      };
      Ok(ParsedItem::Shape(Shape::Fill(Fill { color, opacity, rule })))
    }
    b"tr" => {
      // Group transform: rebuild from recorded field positions.
      let mut tf = Transform::identity();
      if let Some(pos) = a_pos {
        tf.anchor = parse_property(&mut c.fork_at(pos), limits, parse_vec2)?;
      }
      if let Some(pos) = p_pos {
        tf.position = parse_position(&mut c.fork_at(pos), limits)?;
      }
      if let Some(pos) = s_pos {
        tf.scale = parse_property(&mut c.fork_at(pos), limits, parse_vec2)?;
      }
      if let Some(pos) = r_pos {
        tf.rotation = parse_property(&mut c.fork_at(pos), limits, parse_scalar)?;
      }
      if let Some(pos) = o_pos {
        tf.opacity = parse_property(&mut c.fork_at(pos), limits, parse_scalar)?;
      }
      Ok(ParsedItem::GroupTransform(tf))
    }
    b"st" => {
      let c_pos = c_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "stroke missing color",
      })?;
      let w_pos = w_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "stroke missing width",
      })?;
      let color = parse_property(&mut c.fork_at(c_pos), limits, parse_color)?;
      let width = parse_property(&mut c.fork_at(w_pos), limits, parse_scalar)?;
      let opacity = prop_scalar(c, o_pos, 100.0)?;
      let cap = match line_cap as i64 {
        2 => Cap::Round,
        3 => Cap::Square,
        _ => Cap::Butt,
      };
      let join = match line_join as i64 {
        2 => Join::Round,
        3 => Join::Bevel,
        _ => Join::Miter,
      };
      let miter_limit = miter_limit.max(1.0);
      let dashes = match d_pos {
        Some(pos) => parse_dashes(&mut c.fork_at(pos), limits)?,
        None => Vec::new(),
      };
      Ok(ParsedItem::Shape(Shape::Stroke(Box::new(Stroke {
        color,
        opacity,
        width,
        cap,
        join,
        miter_limit,
        dashes,
      }))))
    }
    b"gf" => {
      let s_pos_ = s_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient missing start",
      })?;
      let e_pos_ = e_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient missing end",
      })?;
      let g_pos_ = g_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient missing stops",
      })?;
      let start = parse_property(&mut c.fork_at(s_pos_), limits, parse_vec2)?;
      let end = parse_property(&mut c.fork_at(e_pos_), limits, parse_vec2)?;
      let (stops, color_count) = parse_gradient_stops(&mut c.fork_at(g_pos_), limits)?;
      let opacity = prop_scalar(c, o_pos, 100.0)?;
      let rule = match r_pos {
        Some(pos) => {
          if parse_f32(&mut c.fork_at(pos))? as i64 == 2 {
            FillRule::EvenOdd
          } else {
            FillRule::NonZero
          }
        }
        None => FillRule::NonZero,
      };
      let kind = if grad_type as i64 == 2 { GradientKind::Radial } else { GradientKind::Linear };
      let highlight_len = prop_scalar(c, h_pos, 0.0)?;
      let highlight_angle = prop_scalar(c, a_pos, 0.0)?;
      Ok(ParsedItem::Shape(Shape::GradientFill(Box::new(GradientFill {
        kind,
        start,
        end,
        highlight_len,
        highlight_angle,
        stops,
        color_count,
        opacity,
        rule,
      }))))
    }
    b"tm" => {
      let start = prop_scalar(c, s_pos, 0.0)?;
      let end = match e_pos {
        Some(pos) => parse_property(&mut c.fork_at(pos), limits, parse_scalar)?,
        None => Property::Static(100.0),
      };
      let offset = prop_scalar(c, o_pos, 0.0)?;
      let mode = if trim_mode as i64 == 2 { TrimMode::Individual } else { TrimMode::Simultaneous };
      Ok(ParsedItem::Shape(Shape::Trim(Box::new(Trim { start, end, offset, mode }))))
    }
    b"gs" => {
      let s_pos_ = s_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient stroke missing start",
      })?;
      let e_pos_ = e_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient stroke missing end",
      })?;
      let g_pos_ = g_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient stroke missing stops",
      })?;
      let w_pos_ = w_pos.ok_or(Error::InvalidLottie {
        offset: obj_start,
        what: "gradient stroke missing width",
      })?;
      let start = parse_property(&mut c.fork_at(s_pos_), limits, parse_vec2)?;
      let end = parse_property(&mut c.fork_at(e_pos_), limits, parse_vec2)?;
      let (stops, color_count) = parse_gradient_stops(&mut c.fork_at(g_pos_), limits)?;
      let width = parse_property(&mut c.fork_at(w_pos_), limits, parse_scalar)?;
      let opacity = prop_scalar(c, o_pos, 100.0)?;
      let cap = match line_cap as i64 {
        2 => Cap::Round,
        3 => Cap::Square,
        _ => Cap::Butt,
      };
      let join = match line_join as i64 {
        2 => Join::Round,
        3 => Join::Bevel,
        _ => Join::Miter,
      };
      let dashes = match d_pos {
        Some(pos) => parse_dashes(&mut c.fork_at(pos), limits)?,
        None => Vec::new(),
      };
      let kind = if grad_type as i64 == 2 { GradientKind::Radial } else { GradientKind::Linear };
      let highlight_len = prop_scalar(c, h_pos, 0.0)?;
      let highlight_angle = prop_scalar(c, a_pos, 0.0)?;
      Ok(ParsedItem::Shape(Shape::GradientStroke(Box::new(GradientStroke {
        kind,
        start,
        end,
        highlight_len,
        highlight_angle,
        stops,
        color_count,
        opacity,
        width,
        cap,
        join,
        miter_limit: miter_limit.max(1.0),
        dashes,
      }))))
    }
    b"sr" => {
      let prop_scalar_req = |pos: Option<usize>, default: f32| -> Result<Property<f32>> {
        match pos {
          Some(pos) => parse_property(&mut c.fork_at(pos), limits, parse_scalar),
          None => Ok(Property::Static(default)),
        }
      };
      let points = prop_scalar_req(pt_pos, 5.0)?;
      if points.eval(0.0).abs() > limits.max_polystar_points {
        return Err(Error::LimitExceeded(Limit::PolystarPoints));
      }
      Ok(ParsedItem::Shape(Shape::Polystar(Box::new(PolystarShape {
        star: star_type as i64 != 2,
        reversed: direction_reversed(c, d_pos),
        points,
        position: prop_vec2(c, p_pos, "polystar missing p")?,
        rotation: prop_scalar_req(r_pos, 0.0)?,
        inner_radius: prop_scalar_req(ir_pos, 0.0)?,
        outer_radius: prop_scalar_req(or_pos, 0.0)?,
        inner_roundness: prop_scalar_req(is_pos, 0.0)?,
        outer_roundness: prop_scalar_req(os_pos, 0.0)?,
      }))))
    }
    b"rd" => Ok(ParsedItem::Shape(Shape::RoundCorners(Box::new(RoundCorners { radius: prop_scalar(c, r_pos, 0.0)? })))),
    b"rp" => {
      let copies = match c_pos {
        Some(pos) => parse_property(&mut c.fork_at(pos), limits, parse_scalar)?,
        None => Property::Static(1.0),
      };
      if property_max_abs_f32(&copies).is_some_and(|copies| copies > limits.max_repeater_copies) {
        return Err(Error::LimitExceeded(Limit::RepeaterCopies));
      }
      let offset = prop_scalar(c, o_pos, 0.0)?;
      let (transform, start_opacity, end_opacity) = match tr_pos {
        Some(pos) => parse_repeater_transform(&mut c.fork_at(pos), limits)?,
        None => (Transform::identity(), Property::Static(100.0), Property::Static(100.0)),
      };
      Ok(ParsedItem::Shape(Shape::Repeater(Box::new(Repeater {
        copies,
        offset,
        transform,
        start_opacity,
        end_opacity,
      }))))
    }
    // rp/mm/... : later phases.
    _ => Ok(ParsedItem::Ignored),
  }
}

/// Dash array on strokes: `[{n: "d"|"g"|"o", v: {a,k}}, ...]`.
fn parse_dashes(c: &mut Cursor<'_>, limits: &Limits) -> Result<Vec<DashElement>> {
  let mut out = Vec::new();
  for_each_element(c, |c| {
    let mut kind = 0u8;
    let mut v_pos: Option<usize> = None;
    for_each_field(c, |c, key| {
      match key {
        b"n" => {
          let raw = c.read_string_bytes()?;
          kind = raw.first().copied().unwrap_or(0);
        }
        b"v" => {
          v_pos = Some(c.pos());
          c.skip_value()?;
        }
        _ => c.skip_value()?,
      }
      Ok(())
    })?;
    if let (b'd' | b'g' | b'o', Some(pos)) = (kind, v_pos) {
      if out.len() >= limits.max_dash_elements {
        return Err(Error::LimitExceeded(Limit::DashElements));
      }
      let value = parse_property(&mut c.fork_at(pos), limits, parse_scalar)?;
      out.push(DashElement { value });
    }
    Ok(())
  })?;
  Ok(out)
}

/// `g` object of a gradient: `{p: colorStopCount, k: {a, k: [floats]}}`.
fn parse_gradient_stops(c: &mut Cursor<'_>, limits: &Limits) -> Result<(Property<FloatList>, usize)> {
  let mut count = 0usize;
  let mut stops: Option<Property<FloatList>> = None;
  parse_object_once!(c, |c, key| {
    b"p" => { count = parse_f32(c)? as usize; },
    b"k" => { stops = Some(parse_property(c, limits, |c| parse_f32_list(c, limits.max_gradient_stop_values, Limit::GradientStopValues))?); },
    _ => { c.skip_value()?; },
  })?;
  let stops = stops.ok_or_else(|| invalid(c, "gradient stops missing k"))?;
  if count == 0 {
    // Infer: all floats are color stops.
    if let Property::Static(FloatList(v)) = &stops {
      count = v.len() / 4;
    }
  }
  Ok((stops, count.min(64)))
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

fn parse_layer(c: &mut Cursor<'_>, limits: &Limits, total_masks: &mut usize, total_painted_shape_layers: &mut usize, total_solid_layers: &mut usize) -> Result<Layer> {
  let mut ty = 255u8;
  let mut index = 0i32;
  let mut parent: Option<i32> = None;
  let mut in_point = 0.0f32;
  let mut out_point = f32::MAX;
  let mut start_time = 0.0f32;
  let mut time_stretch = 1.0f32;
  let mut hidden = false;
  let mut transform = Transform::identity();
  let mut shapes_pos: Option<usize> = None;
  let mut parsed_shapes: Option<Vec<Shape>> = None;
  let mut ref_id: Option<String> = None;
  let mut layer_w: Option<f32> = None;
  let mut layer_h: Option<f32> = None;
  let mut masks_pos: Option<usize> = None;
  let mut has_mask = false;
  let mut paint_count = 0usize;
  let mut matte: Option<u8> = None;
  let mut matte_src = false;
  let mut solid_w = 0.0f32;
  let mut solid_h = 0.0f32;
  let mut solid_color: Option<Color> = None;
  let mut time_remap_pos: Option<usize> = None;
  let mut auto_orient = false;
  let mut name = String::new();

  parse_object_once!(c, |c, key| {
      b"ty" => {
        ty = parse_f32(c)? as u8;
      }
      b"nm" => {
        name = String::from_utf8_lossy(c.read_string_bytes()?).into_owned();
      }
      b"ind" => {
        index = parse_f32(c)? as i32;
      }
      b"parent" => {
        parent = Some(parse_f32(c)? as i32);
      }
      b"ip" => {
        in_point = parse_f32(c)?.round();
      }
      b"op" => {
        let authored = parse_f32(c)?;
        let rounded = authored.round();
        // rlottie rounds layer boundaries to integers and considers the
        // rounded out frame visible. Preserve half-open behavior for exact
        // integer `op` values (the Lottie contract), but include that final
        // rounded frame for fractional 29.97-fps exports such as 34.034.
        out_point = rounded + if authored != rounded { 1.0 } else { 0.0 };
      }
      // rlottie stores start_time as an int (mStartFrame); fractional
      // st truncates, shifting precomp child frames by one otherwise.
      b"st" => {
        start_time = parse_f32(c)?.trunc();
      }
      b"sr" => {
        time_stretch = parse_f32(c)?;
      }
      b"hd" => {
        hidden = parse_bool(c)?;
      }
      b"w" => {
        layer_w = Some(parse_f32(c)?);
      }
      b"h" => {
        layer_h = Some(parse_f32(c)?);
      }
      b"refId" => {
        let raw = c.read_string_bytes()?;
        ref_id = Some(String::from_utf8_lossy(raw).into_owned());
      }
      b"masksProperties" => {
        masks_pos = Some(c.pos());
        c.skip_value()?;
      }
      b"hasMask" => {
        has_mask = parse_bool(c)?;
      }
      b"tm" => {
        let pos = c.pos();
        c.skip_value()?;
        time_remap_pos = Some(pos);
      }
      b"tt" => {
        let v = parse_f32(c)? as u8;
        if (1..=4).contains(&v) {
          matte = Some(v);
        }
      }
      b"td" => {
        matte_src = parse_f32(c)? != 0.0;
      }
      b"ao" => {
        auto_orient = parse_bool(c)?;
      }
      b"sw" => {
        solid_w = parse_f32(c)?;
      }
      b"sh" => {
        solid_h = parse_f32(c)?;
      }
      b"sc" => {
        let raw = c.read_string_bytes()?;
        solid_color = parse_hex_color(raw);
      }
      b"ks" => {
        transform = parse_transform(c, limits)?;
      }
      b"shapes" => {
        if ty == 4 {
          let mut count = ShapeCounts::default();
          let (list, _) = parse_shape_list(c, limits, 0, &mut count)?;
          paint_count = count.paints;
          parsed_shapes = Some(list);
          shapes_pos = None;
        } else {
          shapes_pos = Some(c.pos());
          c.skip_value()?;
        }
      }
      _ => { c.skip_value()? },
  })?;

  let kind = match ty {
    4 => LayerKind::Shape,
    3 => LayerKind::Null,
    0 => LayerKind::Precomp,
    1 => LayerKind::Solid,
    other => LayerKind::Other(other),
  };
  let mut shapes = parsed_shapes.unwrap_or_default();
  if kind == LayerKind::Shape && shapes_pos.is_some() {
    if let Some(pos) = shapes_pos {
      let mut count = ShapeCounts::default();
      let (list, _) = parse_shape_list(&mut c.fork_at(pos), limits, 0, &mut count)?;
      paint_count = count.paints;
      shapes = list;
    }
  }
  if kind == LayerKind::Shape && paint_count > 0 {
    *total_painted_shape_layers += 1;
    if *total_painted_shape_layers > limits.max_painted_shape_layers {
      return Err(Error::LimitExceeded(Limit::PaintedShapeLayers));
    }
  }
  let precomp_size = match (kind, layer_w, layer_h) {
    (LayerKind::Precomp, Some(w), Some(h)) if w > 0.0 && h > 0.0 => Some((w, h)),
    _ => None,
  };
  // rlottie only builds LOTLayerMaskItem when the layer's `hasMask` flag
  // is true (LOTLayerItem ctor) — masksProperties without the flag are
  // parsed but never applied.
  let masks = match masks_pos {
    Some(pos) if has_mask => parse_masks(&mut c.fork_at(pos), limits)?,
    _ => Vec::new(),
  };
  *total_masks = total_masks.saturating_add(masks.len());
  if *total_masks > limits.max_masks {
    return Err(Error::LimitExceeded(Limit::Masks));
  }
  let solid = match (kind, solid_color) {
    (LayerKind::Solid, Some(color)) if solid_w > 0.0 && solid_h > 0.0 => Some((solid_w, solid_h, color)),
    _ => None,
  };
  if solid.is_some() {
    *total_solid_layers += 1;
    if *total_solid_layers > limits.max_solid_layers {
      return Err(Error::LimitExceeded(Limit::SolidLayers));
    }
  }
  let time_remap = match (kind, time_remap_pos) {
    (LayerKind::Precomp, Some(pos)) => Some(parse_property(&mut c.fork_at(pos), limits, parse_scalar)?),
    _ => None,
  };
  let layer = Layer {
    name,
    color_override: None,
    kind,
    index,
    parent,
    in_point,
    out_point,
    start_time,
    time_stretch: if time_stretch > 0.0 { time_stretch } else { 1.0 },
    hidden,
    transform,
    shapes,
    ref_id,
    precomp_size,
    masks,
    matte,
    matte_src,
    solid,
    time_remap,
    auto_orient,
  };
  Ok(layer)
}

fn argb_color(argb: u32) -> Color {
  Color {
    r: ((argb >> 16) & 0xff) as f32 / 255.0,
    g: ((argb >> 8) & 0xff) as f32 / 255.0,
    b: (argb & 0xff) as f32 / 255.0,
    a: ((argb >> 24) & 0xff) as f32 / 255.0,
  }
}

fn apply_layer_color(layer: &mut Layer, replacement: &LayerColorReplacement) {
  layer.color_override = Some(argb_color(replacement.color));
}

#[derive(Clone, Copy)]
struct FitzEntry {
  original: u32,
  replacements: [u32; 5],
}

fn parse_fitz_entries(c: &mut Cursor<'_>, entries: &mut Vec<FitzEntry>, limits: &Limits) -> Result<()> {
  for_each_element(c, |c| {
    if entries.len() >= limits.max_fitz_entries {
      return Err(Error::LimitExceeded(Limit::FitzEntries));
    }
    let mut entry = FitzEntry { original: 0, replacements: [0; 5] };
    for_each_field(c, |c, key| {
      let target = match key {
        b"o" => Some(&mut entry.original),
        b"f12" => entry.replacements.get_mut(0),
        b"f3" => entry.replacements.get_mut(1),
        b"f4" => entry.replacements.get_mut(2),
        b"f5" => entry.replacements.get_mut(3),
        b"f6" => entry.replacements.get_mut(4),
        _ => None,
      };
      if let Some(target) = target {
        *target = c.parse_f64()? as u32;
      } else {
        c.skip_value()?;
      }
      Ok(())
    })?;
    entries.push(entry);
    Ok(())
  })
}

fn fitz_key(color: Color) -> u32 {
  let r = (color.r * 255.0).round() as u32 & 0xff;
  let g = (color.g * 255.0).round() as u32 & 0xff;
  let b = (color.b * 255.0).round() as u32 & 0xff;
  (r << 16) | (g << 8) | b
}

fn fitz_color(encoded: u32) -> Color {
  Color {
    r: ((encoded >> 16) & 0xff) as f32 / 255.0,
    g: ((encoded >> 8) & 0xff) as f32 / 255.0,
    b: (encoded & 0xff) as f32 / 255.0,
    a: 1.0,
  }
}

fn replace_source_color(color: &mut Color, replacements: &[SourceColorReplacement]) {
  let key = fitz_key(*color);
  if let Some(replacement) = replacements.iter().find(|replacement| replacement.source_color & 0x00ff_ffff == key) {
    let alpha = color.a;
    *color = fitz_color(replacement.target_color & 0x00ff_ffff);
    color.a = alpha;
  }
}

fn replace_gradient_colors(stops: &mut FloatList, color_count: usize, replacements: &[SourceColorReplacement]) {
  for stop in stops.0.chunks_exact_mut(4).take(color_count) {
    let mut color = Color {
      r: stop[1],
      g: stop[2],
      b: stop[3],
      a: 1.0,
    };
    replace_source_color(&mut color, replacements);
    stop[1] = color.r;
    stop[2] = color.g;
    stop[3] = color.b;
  }
}

fn apply_source_colors_shapes(shapes: &mut [Shape], replacements: &[SourceColorReplacement]) {
  for shape in shapes {
    match shape {
      Shape::Group(group) => apply_source_colors_shapes(&mut group.shapes, replacements),
      Shape::Fill(fill) => fill.color.map_values(|color| replace_source_color(color, replacements)),
      Shape::Stroke(stroke) => stroke.color.map_values(|color| replace_source_color(color, replacements)),
      Shape::GradientFill(fill) => fill.stops.map_values(|stops| replace_gradient_colors(stops, fill.color_count, replacements)),
      Shape::GradientStroke(stroke) => stroke.stops.map_values(|stops| replace_gradient_colors(stops, stroke.color_count, replacements)),
      _ => {}
    }
  }
}

fn apply_source_colors_layers(layers: &mut [Layer], replacements: &[SourceColorReplacement]) {
  for layer in layers {
    apply_source_colors_shapes(&mut layer.shapes, replacements);
    if let Some((_, _, color)) = &mut layer.solid {
      replace_source_color(color, replacements);
    }
  }
}

fn apply_source_colors(layers: &mut [Layer], assets: &mut [Asset], replacements: &[SourceColorReplacement]) {
  apply_source_colors_layers(layers, replacements);
  for asset in assets {
    apply_source_colors_layers(&mut asset.layers, replacements);
  }
}

fn replace_fitz_color(color: &mut Color, entries: &[FitzEntry], index: usize) {
  let key = fitz_key(*color);
  if let Some(replacement) = entries
    .iter()
    .find(|entry| entry.original == key)
    .and_then(|entry| entry.replacements.get(index))
    .copied()
    .filter(|value| *value != 0)
  {
    let alpha = color.a;
    *color = fitz_color(replacement);
    color.a = alpha;
  }
}

fn apply_fitz_shapes(shapes: &mut [Shape], entries: &[FitzEntry], index: usize) {
  for shape in shapes {
    match shape {
      Shape::Group(group) => apply_fitz_shapes(&mut group.shapes, entries, index),
      Shape::Fill(fill) => fill.color.map_values(|color| replace_fitz_color(color, entries, index)),
      Shape::Stroke(stroke) => stroke.color.map_values(|color| replace_fitz_color(color, entries, index)),
      _ => {}
    }
  }
}

fn apply_fitz_layers(layers: &mut [Layer], entries: &[FitzEntry], index: usize) {
  for layer in layers {
    apply_fitz_shapes(&mut layer.shapes, entries, index);
    if let Some((_, _, color)) = &mut layer.solid {
      replace_fitz_color(color, entries, index);
    }
  }
}

fn apply_fitz(layers: &mut [Layer], assets: &mut [Asset], entries: &[FitzEntry], index: usize) {
  apply_fitz_layers(layers, entries, index);
  for asset in assets {
    apply_fitz_layers(&mut asset.layers, entries, index);
  }
}

fn apply_layer_replacements(layers: &mut [Layer], replacements: &[LayerColorReplacement]) {
  for layer in layers {
    if let Some(replacement) = replacements.iter().find(|replacement| layer.name.starts_with(&replacement.layer_name_prefix)) {
      apply_layer_color(layer, replacement);
    }
  }
}

#[derive(Clone, Copy, Default)]
struct ExpansionCost {
  layers: usize,
  focal_radial_gradients: usize,
}

impl ExpansionCost {
  fn add_layer(&mut self, layer: &Layer) {
    self.layers = self.layers.saturating_add(1);
    if layer.kind == LayerKind::Shape {
      self.focal_radial_gradients = self.focal_radial_gradients.saturating_add(focal_radial_gradients_in_shapes(&layer.shapes));
    }
  }

  fn add(&mut self, other: Self) {
    self.layers = self.layers.saturating_add(other.layers);
    self.focal_radial_gradients = self.focal_radial_gradients.saturating_add(other.focal_radial_gradients);
  }

  fn validate(&self, limits: &Limits) -> Result<()> {
    if self.layers > limits.max_precomp_expansion {
      return Err(Error::LimitExceeded(Limit::PrecompExpansion));
    }
    if self.focal_radial_gradients > limits.max_focal_radial_gradient_expansion {
      return Err(Error::LimitExceeded(Limit::FocalRadialGradientExpansion));
    }
    Ok(())
  }
}

fn focal_radial_gradients_in_shapes(shapes: &[Shape]) -> usize {
  shapes.iter().map(focal_radial_gradients_in_shape).sum()
}

fn focal_radial_gradients_in_shape(shape: &Shape) -> usize {
  match shape {
    Shape::Group(group) => focal_radial_gradients_in_shapes(&group.shapes),
    shape if is_focal_radial_gradient(shape) => 1,
    _ => 0,
  }
}

fn validate_precomp_expansion(layers: &[Layer], assets: &[Asset], limits: &Limits) -> Result<()> {
  let mut asset_by_id = crate::compat::HashMap::new();
  for (index, asset) in assets.iter().enumerate() {
    asset_by_id.entry(asset.id.as_str()).or_insert(index);
  }
  let mut memo: Vec<Option<ExpansionCost>> = vec![None; assets.len()];
  let mut visiting = vec![false; assets.len()];
  let total = layer_list_expansion(layers, assets, &asset_by_id, &mut memo, &mut visiting, limits)?;
  total.validate(limits)
}

fn layer_list_expansion<'a>(
  layers: &[Layer],
  assets: &'a [Asset],
  asset_by_id: &crate::compat::HashMap<&'a str, usize>,
  memo: &mut [Option<ExpansionCost>],
  visiting: &mut [bool],
  limits: &Limits,
) -> Result<ExpansionCost> {
  let mut total = ExpansionCost::default();
  for layer in layers {
    total.add_layer(layer);
    if let (LayerKind::Precomp, Some(ref_id)) = (layer.kind, layer.ref_id.as_deref()) {
      if let Some(&asset_index) = asset_by_id.get(ref_id) {
        total.add(asset_expansion(asset_index, assets, asset_by_id, memo, visiting, limits)?);
      }
    }
    total.validate(limits)?;
  }
  Ok(total)
}

fn asset_expansion<'a>(
  asset_index: usize,
  assets: &'a [Asset],
  asset_by_id: &crate::compat::HashMap<&'a str, usize>,
  memo: &mut [Option<ExpansionCost>],
  visiting: &mut [bool],
  limits: &Limits,
) -> Result<ExpansionCost> {
  if let Some(total) = memo.get(asset_index).copied().flatten() {
    return Ok(total);
  }
  let Some(slot) = visiting.get_mut(asset_index) else {
    return Ok(ExpansionCost::default());
  };
  if *slot {
    return Err(Error::LimitExceeded(Limit::PrecompExpansion));
  }
  *slot = true;
  let total = layer_list_expansion(&assets[asset_index].layers, assets, asset_by_id, memo, visiting, limits)?;
  visiting[asset_index] = false;
  if let Some(slot) = memo.get_mut(asset_index) {
    *slot = Some(total);
  }
  Ok(total)
}

fn validate_layer_parent_chains(layers: &[Layer], limits: &Limits) -> Result<()> {
  if !layers.iter().any(|layer| layer.parent.is_some()) {
    return Ok(());
  }

  let lookup = ParentLookup::new(layers);
  let mut depths: Vec<Option<usize>> = vec![None; layers.len()];
  let mut visiting = vec![false; layers.len()];

  let mut total_depth = 0usize;
  for slot in 0..layers.len() {
    total_depth = total_depth.saturating_add(parent_chain_depth(slot, layers, &lookup, &mut depths, &mut visiting, limits)?);
    if total_depth > limits.max_parent_chain_total_depth {
      return Err(Error::LimitExceeded(Limit::ParentChainTotalDepth));
    }
  }
  Ok(())
}

enum ParentLookup<'a> {
  Linear(&'a [Layer]),
  Sorted(Vec<(i32, usize)>),
}

impl<'a> ParentLookup<'a> {
  fn new(layers: &'a [Layer]) -> Self {
    if layers.len() <= 256 {
      return Self::Linear(layers);
    }

    let mut by_index: Vec<(i32, usize)> = layers.iter().enumerate().map(|(slot, layer)| (layer.index, slot)).collect();
    by_index.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    Self::Sorted(by_index)
  }

  #[inline]
  fn first_layer_with_index(&self, parent: i32) -> Option<usize> {
    match self {
      Self::Linear(layers) => layers.iter().position(|layer| layer.index == parent),
      Self::Sorted(by_index) => {
        let pos = by_index.partition_point(|&(index, _)| index < parent);
        by_index.get(pos).and_then(|&(index, slot)| (index == parent).then_some(slot))
      }
    }
  }
}

fn parent_chain_depth(slot: usize, layers: &[Layer], lookup: &ParentLookup<'_>, depths: &mut [Option<usize>], visiting: &mut [bool], limits: &Limits) -> Result<usize> {
  if let Some(depth) = depths[slot] {
    return Ok(depth);
  }
  if visiting[slot] {
    return Err(Error::LimitExceeded(Limit::ParentChainDepth));
  }

  visiting[slot] = true;
  let depth = match layers[slot].parent.and_then(|parent| lookup.first_layer_with_index(parent)) {
    Some(parent) => parent_chain_depth(parent, layers, lookup, depths, visiting, limits)?.saturating_add(1),
    None => 0,
  };
  visiting[slot] = false;

  if depth > limits.max_parent_chain_depth {
    return Err(Error::LimitExceeded(Limit::ParentChainDepth));
  }
  depths[slot] = Some(depth);
  Ok(depth)
}

fn parse_layer_list(
  c: &mut Cursor<'_>,
  limits: &Limits,
  total_layers: &mut usize,
  total_masks: &mut usize,
  total_painted_shape_layers: &mut usize,
  total_solid_layers: &mut usize,
) -> Result<Vec<Layer>> {
  let mut layers = Vec::new();
  for_each_element(c, |c| {
    *total_layers += 1;
    if *total_layers > limits.max_layers {
      return Err(Error::LimitExceeded(Limit::Layers));
    }
    layers.push(parse_layer(c, limits, total_masks, total_painted_shape_layers, total_solid_layers)?);
    Ok(())
  })?;
  validate_layer_parent_chains(&layers, limits)?;
  Ok(layers)
}

/// `#rrggbb` or `#rrggbbaa` solid-layer color.
fn parse_hex_color(raw: &[u8]) -> Option<Color> {
  let hex = raw.strip_prefix(b"#").unwrap_or(raw);
  let nib = |b: u8| -> Option<u32> {
    match b {
      b'0'..=b'9' => Some(u32::from(b - b'0')),
      b'a'..=b'f' => Some(u32::from(b - b'a' + 10)),
      b'A'..=b'F' => Some(u32::from(b - b'A' + 10)),
      _ => None,
    }
  };
  let byte = |i: usize| -> Option<f32> {
    let hi = nib(*hex.get(i)?)?;
    let lo = nib(*hex.get(i + 1)?)?;
    Some((hi * 16 + lo) as f32 / 255.0)
  };
  Some(Color {
    r: byte(0)?,
    g: byte(2)?,
    b: byte(4)?,
    a: byte(6).unwrap_or(1.0),
  })
}

/// Repeater `tr`: a normal transform plus `so`/`eo` copy-opacity ramps.
#[allow(clippy::type_complexity)]
fn parse_repeater_transform(c: &mut Cursor<'_>, limits: &Limits) -> Result<(Transform, Property<f32>, Property<f32>)> {
  let mut tf = Transform::identity();
  let mut so = Property::Static(100.0);
  let mut eo = Property::Static(100.0);
  parse_object_once!(c, |c, key| {
    b"a" => { tf.anchor = parse_property(c, limits, parse_vec2)?; },
    b"p" => { tf.position = parse_position(c, limits)?; },
    b"s" => { tf.scale = parse_property(c, limits, parse_vec2)?; },
    b"r" => { tf.rotation = parse_property(c, limits, parse_scalar)?; },
    b"o" => { tf.opacity = parse_property(c, limits, parse_scalar)?; },
    b"so" => { so = parse_property(c, limits, parse_scalar)?; },
    b"eo" => { eo = parse_property(c, limits, parse_scalar)?; },
    _ => { c.skip_value()?; },
  })?;
  Ok((tf, so, eo))
}

/// `masksProperties` array.
fn parse_masks(c: &mut Cursor<'_>, limits: &Limits) -> Result<Vec<Mask>> {
  let mut out = Vec::new();
  for_each_element(c, |c| {
    if out.len() >= limits.max_masks_per_layer {
      return Err(Error::LimitExceeded(Limit::MasksPerLayer));
    }
    let mut mode = b'a';
    let mut invert = false;
    let mut pt_pos: Option<usize> = None;
    let mut o_pos: Option<usize> = None;
    for_each_field(c, |c, key| {
      match key {
        b"mode" => {
          let raw = c.read_string_bytes()?;
          mode = raw.first().copied().unwrap_or(b'a');
        }
        b"inv" => invert = parse_bool(c)?,
        b"pt" => {
          pt_pos = Some(c.pos());
          c.skip_value()?;
        }
        b"o" => {
          o_pos = Some(c.pos());
          c.skip_value()?;
        }
        _ => c.skip_value()?,
      }
      Ok(())
    })?;
    let Some(pt_pos) = pt_pos else { return Ok(()) };
    if mode == b'n' {
      return Ok(());
    }
    let path = parse_property(&mut c.fork_at(pt_pos), limits, |c| parse_path_value(c, limits))?;
    if path_property_exceeds_points(&path, limits.max_mask_path_points) {
      return Err(Error::LimitExceeded(Limit::MaskPathPoints));
    }
    let opacity = match o_pos {
      Some(pos) => parse_property(&mut c.fork_at(pos), limits, parse_scalar)?,
      None => Property::Static(100.0),
    };
    out.push(Mask { mode, invert, path, opacity });
    Ok(())
  })?;
  Ok(out)
}

fn path_property_exceeds_points(path: &Property<PathData>, max_points: usize) -> bool {
  match path {
    Property::Static(data) => data.vertices.len() > max_points,
    Property::Animated(timeline) => {
      timeline.first.value.vertices.len() > max_points
        || timeline.first.end.as_ref().is_some_and(|data| data.vertices.len() > max_points)
        || timeline
          .rest
          .iter()
          .any(|kf| kf.value.vertices.len() > max_points || kf.end.as_ref().is_some_and(|data| data.vertices.len() > max_points))
    }
  }
}

/// One entry of the top-level `assets` array. Only precomp assets (those
/// with a `layers` list) are kept; image assets are ignored.
fn parse_asset(
  c: &mut Cursor<'_>,
  limits: &Limits,
  total_layers: &mut usize,
  total_masks: &mut usize,
  total_painted_shape_layers: &mut usize,
  total_solid_layers: &mut usize,
) -> Result<Option<Asset>> {
  let mut id = String::new();
  let mut layers = Vec::new();
  let mut has_layers = false;
  parse_object_once!(c, |c, key| {
      b"id" => {
        let raw = c.read_string_bytes()?;
        id = String::from_utf8_lossy(raw).into_owned();
      }
      b"layers" => {
        has_layers = true;
        layers = parse_layer_list(c, limits, total_layers, total_masks, total_painted_shape_layers, total_solid_layers)?;
      }
      _ => { c.skip_value()? },
  })?;
  if !has_layers {
    return Ok(None);
  }
  Ok(Some(Asset { id, layers }))
}

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

pub(crate) fn parse_composition(bytes: &[u8], limits: &Limits, options: &ParseOptions) -> Result<Composition> {
  if bytes.len() > limits.max_input_bytes {
    return Err(Error::LimitExceeded(Limit::InputBytes));
  }

  let mut c = Cursor::new(bytes, limits.max_nesting_depth);
  let mut width: Option<f64> = None;
  let mut height: Option<f64> = None;
  let mut frame_rate: Option<f64> = None;
  let mut in_point: Option<f64> = None;
  let mut out_point: Option<f64> = None;
  let mut layers: Vec<Layer> = Vec::new();
  let mut assets: Vec<Asset> = Vec::new();
  let mut total_layers = 0usize;
  let mut total_masks = 0usize;
  let mut total_painted_shape_layers = 0usize;
  let mut total_solid_layers = 0usize;
  let mut fitz_entries = Vec::new();

  parse_object_once!(&mut c, |c, key| {
      b"w" => { width = Some(c.parse_f64()?); },
      b"h" => { height = Some(c.parse_f64()?); },
      b"fr" => { frame_rate = Some(c.parse_f64()?); },
      b"ip" => { in_point = Some(c.parse_f64()?); },
      b"op" => { out_point = Some(c.parse_f64()?); },
      b"layers" => {
        layers = parse_layer_list(c, limits, &mut total_layers, &mut total_masks, &mut total_painted_shape_layers, &mut total_solid_layers)?;
      }
      b"assets" => {
        for_each_element(c, |c| {
          if assets.len() >= limits.max_assets {
            return Err(Error::LimitExceeded(Limit::Assets));
          }
          if let Some(asset) = parse_asset(c, limits, &mut total_layers, &mut total_masks, &mut total_painted_shape_layers, &mut total_solid_layers)? {
            assets.push(asset);
          }
          Ok(())
        })?;
      }
      b"fitz" => {
        if options.fitz_modifier.replacement_index().is_some() {
          parse_fitz_entries(c, &mut fitz_entries, limits)?;
        } else {
          c.skip_value()?;
        }
      }
      _ => { c.skip_value()? },
  })?;
  c.skip_ws();
  if c.peek().is_some() {
    return Err(json_err(&c, JsonErrorKind::TrailingData));
  }

  let offset = c.pos();
  let missing = |what: &'static str| Error::InvalidLottie { offset, what };

  let width = width.ok_or_else(|| missing("missing width (w)"))?;
  let height = height.ok_or_else(|| missing("missing height (h)"))?;
  let frame_rate = frame_rate.ok_or_else(|| missing("missing frame rate (fr)"))?;
  let in_point = in_point.ok_or_else(|| missing("missing in point (ip)"))?;
  let out_point = out_point.ok_or_else(|| missing("missing out point (op)"))?;

  let max_dim = f64::from(limits.max_dimension);
  if !(1.0..=max_dim).contains(&width) || !(1.0..=max_dim).contains(&height) {
    return Err(Error::LimitExceeded(Limit::CompositionSize));
  }
  if !(f64::EPSILON..=1000.0).contains(&frame_rate) {
    return Err(missing("frame rate out of range"));
  }
  if !out_point.is_finite() || !in_point.is_finite() || out_point <= in_point {
    return Err(missing("out point must be greater than in point"));
  }
  validate_precomp_expansion(&layers, &assets, limits)?;

  // Static detection piggybacks on parsing: every deferred cursor shares the
  // animated-property bit. Keep the final lifetime check flat and
  // conservative; precomps are rejected rather than recursively analyzed.
  let static_content = !layers.is_empty()
    && assets.is_empty()
    && c.properties_are_static()
    && layers
      .iter()
      .all(|layer| layer.hidden || layer.out_point <= in_point as f32 || layer.in_point >= out_point as f32 || (layer.in_point <= in_point as f32 && layer.out_point >= out_point as f32));

  if let Some(index) = options.fitz_modifier.replacement_index() {
    apply_fitz(&mut layers, &mut assets, &fitz_entries, index);
  }
  apply_source_colors(&mut layers, &mut assets, &options.source_color_replacements);
  apply_layer_replacements(&mut layers, &options.layer_color_replacements);
  for asset in &mut assets {
    apply_layer_replacements(&mut asset.layers, &options.layer_color_replacements);
  }

  let mut composition = Composition {
    width: width as u32,
    height: height as u32,
    frame_rate: frame_rate as f32,
    in_point: in_point as f32,
    out_point: out_point as f32,
    static_content,
    layers,
    assets,
    channel_order: options.channel_order,
  };
  // Last step, deliberately: color replacements above match and supply
  // `0xAARRGGBB`, so the swap has to happen after they have been resolved.
  if options.channel_order == crate::composition::options::ChannelOrder::Bgra {
    crate::composition::swizzle::swap_red_blue(&mut composition);
  }
  Ok(composition)
}

#[cfg(test)]
#[path = "tests/parse.rs"]
mod tests;
