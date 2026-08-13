//! The immutable parsed animation model. Shared (`Arc`) across render
//! instances; render-time state lives elsewhere.

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::math::{Color, Vec2};
use crate::property::{Lerp, Property};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Raw float list (gradient stop data); lerps pointwise when lengths match.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct FloatList(pub Vec<f32>);

impl Lerp for FloatList {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if self.0.len() != other.0.len() {
      return self.clone();
    }
    FloatList(self.0.iter().zip(other.0.iter()).map(|(a, b)| a + (b - a) * t).collect())
  }
}

/// A parsed Lottie composition.
#[derive(Debug, Clone)]
pub struct Composition {
  /// Composition width in points.
  pub width: u32,
  /// Composition height in points.
  pub height: u32,
  /// Frames per second.
  pub frame_rate: f32,
  /// First frame of the playable range.
  pub in_point: f32,
  /// One past the last frame of the playable range.
  pub out_point: f32,
  /// Whether parsing proved that every playable frame is identical.
  pub(crate) static_content: bool,
  /// Layers in file order (index 0 is drawn on top).
  pub(crate) layers: Vec<Layer>,
  /// Precomp assets, looked up by `Layer::ref_id`.
  pub(crate) assets: Vec<Asset>,
  /// Byte order the model's colors are stored in. Everything except the luma
  /// matte is channel-order agnostic, so this is only consulted there.
  pub(crate) channel_order: crate::composition::options::ChannelOrder,
}

impl Composition {
  /// Number of frames in the playable range (at least 1).
  pub fn frame_count(&self) -> u32 {
    if self.static_content {
      return 1;
    }
    let frames = f64::from(self.out_point) - f64::from(self.in_point);
    let frames = frames.floor().max(1.0);
    frames as u32 // saturating float→int cast
  }

  /// Returns whether parsing proved that every playable frame is identical.
  ///
  /// Detection is conservative: `false` means "possibly animated", while
  /// `true` guarantees that [`Self::frame_count`] returns one.
  pub fn is_static(&self) -> bool {
    self.static_content
  }

  /// Animation duration in seconds.
  pub fn duration_secs(&self) -> f32 {
    if self.frame_rate <= 0.0 {
      return 0.0;
    }
    (f64::from(self.frame_count()) / f64::from(self.frame_rate)) as f32
  }
}

/// Lottie layer types we distinguish. Everything else is parsed as `Other`
/// (transform still read, content ignored) so parenting chains stay intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
  /// ty=4 — vector shapes.
  Shape,
  /// ty=3 — null (transform-only, used for parenting).
  Null,
  /// ty=0 — precomposition instance.
  Precomp,
  /// ty=1 — solid color layer.
  Solid,
  /// Any other layer type; contents unsupported for now.
  Other(u8),
}

/// A precomp asset: a named reusable layer list.
#[derive(Debug, Clone)]
pub(crate) struct Asset {
  pub id: String,
  pub layers: Vec<Layer>,
}

/// One entry of a layer's `masksProperties`.
#[derive(Debug, Clone)]
pub(crate) struct Mask {
  /// b'a' add, b's' subtract, b'i' intersect, b'd' difference, b'n' none.
  pub mode: u8,
  pub invert: bool,
  pub path: Property<PathData>,
  /// Percent (100 = opaque).
  pub opacity: Property<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct Layer {
  /// `nm` — used to resolve constructor-time prefix color replacements.
  pub name: String,
  /// Constructor-time color override inherited by this layer's descendants.
  pub color_override: Option<Color>,
  pub kind: LayerKind,
  /// `ind` — this layer's id for parenting.
  pub index: i32,
  /// `parent` — `ind` of the parent layer, if any.
  pub parent: Option<i32>,
  /// In/out frames and start-time offset.
  pub in_point: f32,
  pub out_point: f32,
  pub start_time: f32,
  /// `sr` — time stretch (1.0 = none).
  pub time_stretch: f32,
  pub hidden: bool,
  pub transform: Transform,
  /// Shape tree (only for `LayerKind::Shape`).
  pub shapes: Vec<Shape>,
  /// Asset reference (only for `LayerKind::Precomp`).
  pub ref_id: Option<String>,
  /// Precomp viewport (layer w/h); content is clipped to it.
  pub precomp_size: Option<(f32, f32)>,
  /// Layer masks (`masksProperties`).
  pub masks: Vec<Mask>,
  /// `tt`: this layer is matted by the layer above (1 alpha, 2 alpha-inv,
  /// 3 luma, 4 luma-inv).
  pub matte: Option<u8>,
  /// `td`: this layer only serves as a matte source; not drawn directly.
  pub matte_src: bool,
  /// Solid layer (ty=1): width, height, color.
  pub solid: Option<(f32, f32, Color)>,
  /// Precomp time remap (`tm`), in seconds.
  pub time_remap: Option<Property<f32>>,
  /// `ao`: rotate the layer along its position path.
  pub auto_orient: bool,
}

/// Layer or group transform.
#[derive(Debug, Clone)]
pub(crate) struct Transform {
  pub anchor: Property<Vec2>,
  pub position: Position,
  /// Percent (100 = identity).
  pub scale: Property<Vec2>,
  /// Degrees.
  pub rotation: Property<f32>,
  /// Percent (100 = opaque).
  pub opacity: Property<f32>,
  /// Skew amount in degrees and skew axis in degrees.
  pub skew: Property<f32>,
  pub skew_axis: Property<f32>,
}

impl Transform {
  /// True when every track (including opacity) is a constant.
  pub fn is_static(&self) -> bool {
    self.anchor.is_static() && self.position.is_static() && self.scale.is_static() && self.rotation.is_static() && self.opacity.is_static() && self.skew.is_static() && self.skew_axis.is_static()
  }

  /// True when every geometry track is a constant; opacity may still animate.
  pub fn geometry_static(&self) -> bool {
    self.anchor.is_static() && self.position.is_static() && self.scale.is_static() && self.rotation.is_static() && self.skew.is_static() && self.skew_axis.is_static()
  }

  pub fn identity() -> Self {
    Transform {
      anchor: Property::Static(Vec2::ZERO),
      position: Position::Combined(Property::Static(Vec2::ZERO)),
      scale: Property::Static(Vec2::new(100.0, 100.0)),
      rotation: Property::Static(0.0),
      opacity: Property::Static(100.0),
      skew: Property::Static(0.0),
      skew_axis: Property::Static(0.0),
    }
  }
}

/// Position can be a combined animated Vec2 or split x/y scalar tracks.
#[derive(Debug, Clone)]
pub(crate) enum Position {
  Combined(Property<Vec2>),
  Split { x: Property<f32>, y: Property<f32> },
}

impl Position {
  pub fn eval(&self, frame: f32) -> Vec2 {
    match self {
      Position::Combined(p) => p.eval(frame),
      Position::Split { x, y } => Vec2::new(x.eval(frame), y.eval(frame)),
    }
  }

  pub fn is_static(&self) -> bool {
    match self {
      Position::Combined(p) => p.is_static(),
      Position::Split { x, y } => x.is_static() && y.is_static(),
    }
  }
}

/// One element of a shape layer's content tree.
#[derive(Debug, Clone)]
pub(crate) enum Shape {
  Group(Box<Group>),
  Path(PathShape),
  Rect(RectShape),
  Ellipse(EllipseShape),
  Fill(Fill),
  Stroke(Box<Stroke>),
  GradientFill(Box<GradientFill>),
  GradientStroke(Box<GradientStroke>),
  Trim(Box<Trim>),
  Polystar(Box<PolystarShape>),
  RoundCorners(Box<RoundCorners>),
  Repeater(Box<Repeater>),
}

impl Shape {
  /// True when the shape (and, for groups, its whole subtree) evaluates
  /// identically at every frame — the static-subtree replay condition.
  pub fn is_static(&self) -> bool {
    match self {
      Shape::Group(g) => g.transform.is_static() && shapes_static(&g.shapes),
      Shape::Path(p) => p.path.is_static(),
      Shape::Rect(r) => r.position.is_static() && r.size.is_static() && r.radius.is_static(),
      Shape::Ellipse(e) => e.position.is_static() && e.size.is_static(),
      Shape::Fill(f) => f.color.is_static() && f.opacity.is_static(),
      Shape::Stroke(st) => st.color.is_static() && st.opacity.is_static() && st.width.is_static() && st.dashes.iter().all(|d| d.value.is_static()),
      Shape::GradientFill(gf) => gf.start.is_static() && gf.end.is_static() && gf.highlight_len.is_static() && gf.highlight_angle.is_static() && gf.stops.is_static() && gf.opacity.is_static(),
      Shape::GradientStroke(gs) => {
        gs.start.is_static()
          && gs.end.is_static()
          && gs.highlight_len.is_static()
          && gs.highlight_angle.is_static()
          && gs.stops.is_static()
          && gs.opacity.is_static()
          && gs.width.is_static()
          && gs.dashes.iter().all(|d| d.value.is_static())
      }
      Shape::Trim(t) => t.start.is_static() && t.end.is_static() && t.offset.is_static(),
      Shape::Polystar(p) => {
        p.points.is_static()
          && p.position.is_static()
          && p.rotation.is_static()
          && p.inner_radius.is_static()
          && p.outer_radius.is_static()
          && p.inner_roundness.is_static()
          && p.outer_roundness.is_static()
      }
      Shape::RoundCorners(r) => r.radius.is_static(),
      Shape::Repeater(rp) => rp.copies.is_static() && rp.offset.is_static() && rp.transform.is_static() && rp.start_opacity.is_static() && rp.end_opacity.is_static(),
    }
  }
}

/// Returns true once a shape tree contains more than one paint operation.
/// A single paint can absorb its layer opacity without an offscreen surface;
/// multiple paints must be flattened first because their coverage may overlap.
#[cfg(test)]
pub(crate) fn shapes_have_multiple_paints(shapes: &[Shape]) -> bool {
  fn add_paints(shapes: &[Shape], count: &mut u8) {
    for shape in shapes {
      match shape {
        Shape::Group(group) => add_paints(&group.shapes, count),
        Shape::Fill(_) | Shape::Stroke(_) | Shape::GradientFill(_) | Shape::GradientStroke(_) => {
          *count = count.saturating_add(1);
          if *count > 1 {
            return;
          }
        }
        _ => {}
      }
      if *count > 1 {
        return;
      }
    }
  }

  let mut count = 0;
  add_paints(shapes, &mut count);
  count > 1
}

/// Counts only paints visible at this frame. Paints whose evaluated opacity
/// truncates to zero cannot contribute pixels and do not force isolation.
pub(crate) fn shapes_have_multiple_visible_paints(shapes: &[Shape], frame: f32) -> bool {
  fn visible(alpha: f32) -> bool {
    (alpha.clamp(0.0, 1.0) * 255.0) as u32 != 0
  }

  fn add_paints(shapes: &[Shape], frame: f32, inherited_opacity: f32, count: &mut u8) {
    for shape in shapes {
      let contributes = match shape {
        Shape::Group(group) => {
          let opacity = inherited_opacity * (group.transform.opacity.eval(frame) / 100.0).clamp(0.0, 1.0);
          add_paints(&group.shapes, frame, opacity, count);
          false
        }
        Shape::Fill(fill) => {
          let color = fill.color.eval(frame);
          visible(inherited_opacity * (fill.opacity.eval(frame) / 100.0) * color.a)
        }
        Shape::Stroke(stroke) => {
          let color = stroke.color.eval(frame);
          stroke.width.eval(frame) > 0.0 && visible(inherited_opacity * (stroke.opacity.eval(frame) / 100.0) * color.a)
        }
        Shape::GradientFill(fill) => visible(inherited_opacity * (fill.opacity.eval(frame) / 100.0)),
        Shape::GradientStroke(stroke) => stroke.width.eval(frame) > 0.0 && visible(inherited_opacity * (stroke.opacity.eval(frame) / 100.0)),
        _ => false,
      };
      if contributes {
        *count = count.saturating_add(1);
      }
      if *count > 1 {
        return;
      }
    }
  }

  let mut count = 0;
  add_paints(shapes, frame, 1.0, &mut count);
  count > 1
}

/// All shapes in a list static (see [`Shape::is_static`]).
pub(crate) fn shapes_static(shapes: &[Shape]) -> bool {
  shapes.iter().all(Shape::is_static)
}

/// Repeater modifier (`rp`): duplicates preceding group content.
#[derive(Debug, Clone)]
pub(crate) struct Repeater {
  pub copies: Property<f32>,
  pub offset: Property<f32>,
  pub transform: Transform,
  /// Percent opacity of first / last copy.
  pub start_opacity: Property<f32>,
  pub end_opacity: Property<f32>,
}

/// Star / polygon shape (`sr` with sy=1 star, sy=2 polygon).
#[derive(Debug, Clone)]
pub(crate) struct PolystarShape {
  pub star: bool,
  pub reversed: bool,
  pub points: Property<f32>,
  pub position: Property<Vec2>,
  pub rotation: Property<f32>,
  pub inner_radius: Property<f32>,
  pub outer_radius: Property<f32>,
  pub inner_roundness: Property<f32>,
  pub outer_roundness: Property<f32>,
}

/// Rounded-corners modifier (`rd`).
#[derive(Debug, Clone)]
pub(crate) struct RoundCorners {
  pub radius: Property<f32>,
}

/// One positional dash-pattern value.
#[derive(Debug, Clone)]
pub(crate) struct DashElement {
  pub value: Property<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct GradientStroke {
  pub kind: GradientKind,
  pub start: Property<Vec2>,
  pub end: Property<Vec2>,
  /// Radial highlight length (%) and angle (degrees).
  pub highlight_len: Property<f32>,
  pub highlight_angle: Property<f32>,
  pub stops: Property<FloatList>,
  pub color_count: usize,
  pub opacity: Property<f32>,
  pub width: Property<f32>,
  pub cap: crate::stroke::Cap,
  pub join: crate::stroke::Join,
  pub miter_limit: f32,
  pub dashes: Vec<DashElement>,
}

/// Trim-path modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrimMode {
  Simultaneous,
  Individual,
}

#[derive(Debug, Clone)]
pub(crate) struct Trim {
  /// Percent 0..=100.
  pub start: Property<f32>,
  /// Percent 0..=100.
  pub end: Property<f32>,
  /// Degrees; 360 = one full loop of offset.
  pub offset: Property<f32>,
  pub mode: TrimMode,
}

#[derive(Debug, Clone)]
pub(crate) struct Group {
  /// The group's `tr` item, applied to all children.
  pub transform: Transform,
  pub shapes: Vec<Shape>,
}

#[derive(Debug, Clone)]
pub(crate) struct PathShape {
  pub path: Property<PathData>,
}

#[derive(Debug, Clone)]
pub(crate) struct RectShape {
  pub position: Property<Vec2>,
  pub size: Property<Vec2>,
  pub radius: Property<f32>,
  /// Path direction `d` == 3 → reversed (CCW) winding.
  pub reversed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct EllipseShape {
  pub position: Property<Vec2>,
  pub size: Property<Vec2>,
  pub reversed: bool,
}

/// Fill rule, from the fill shape's `r` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FillRule {
  NonZero,
  EvenOdd,
}

#[derive(Debug, Clone)]
pub(crate) struct Fill {
  pub color: Property<Color>,
  /// Percent (100 = opaque).
  pub opacity: Property<f32>,
  pub rule: FillRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientKind {
  Linear,
  Radial,
}

#[derive(Debug, Clone)]
pub(crate) struct GradientFill {
  pub kind: GradientKind,
  pub start: Property<Vec2>,
  pub end: Property<Vec2>,
  /// Radial highlight length (%) and angle (degrees).
  pub highlight_len: Property<f32>,
  pub highlight_angle: Property<f32>,
  /// Interleaved stop data: `color_count * [offset r g b]` then optional
  /// `[offset alpha]` pairs.
  pub stops: Property<FloatList>,
  pub color_count: usize,
  /// Percent (100 = opaque).
  pub opacity: Property<f32>,
  pub rule: FillRule,
}

#[derive(Debug, Clone)]
pub(crate) struct Stroke {
  pub color: Property<Color>,
  /// Percent (100 = opaque).
  pub opacity: Property<f32>,
  /// Stroke width in local units.
  pub width: Property<f32>,
  pub cap: crate::stroke::Cap,
  pub join: crate::stroke::Join,
  pub miter_limit: f32,
  pub dashes: Vec<DashElement>,
}

/// Cubic bezier contour data as stored in Lottie `sh` shapes: vertex points
/// with per-vertex in/out tangents (tangents are relative to the vertex).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PathData {
  pub vertices: Vec<Vec2>,
  pub in_tangents: Vec<Vec2>,
  pub out_tangents: Vec<Vec2>,
  pub closed: bool,
}

impl Lerp for PathData {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    // Topology must match to interpolate; otherwise hold the start value
    // (same behavior as rlottie for mismatched path keyframes).
    if self.vertices.len() != other.vertices.len() {
      return self.clone();
    }
    let pair = |a: &[Vec2], b: &[Vec2]| -> Vec<Vec2> { a.iter().zip(b.iter()).map(|(p, q)| Vec2::lerp(*p, *q, t)).collect() };
    PathData {
      vertices: pair(&self.vertices, &other.vertices),
      in_tangents: pair(&self.in_tangents, &other.in_tangents),
      out_tangents: pair(&self.out_tangents, &other.out_tangents),
      closed: self.closed,
    }
  }
}
