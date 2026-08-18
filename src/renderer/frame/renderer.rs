//! Renderer-neutral types and backend operations for an evaluated frame.

#![allow(missing_docs)]

use crate::geometry::Contour;

pub(crate) const GRADIENT_LUT_SIZE: usize = 1024;

/// Device-space point.
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
  /// X coordinate in target pixels.
  pub x: f32,
  /// Y coordinate in target pixels.
  pub y: f32,
}

/// Fill rule for a vector paint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
  /// Non-zero winding rule.
  NonZero,
  /// Even-odd winding rule.
  EvenOdd,
}

/// Premultiplied solid color paint metadata.
#[derive(Clone, Copy, Debug)]
pub struct SolidPaint {
  /// Fill rule used by this paint.
  pub rule: Rule,
  /// Premultiplied RGBA8 color packed as `0xAABBGGRR`.
  pub rgba: u32,
  pub(crate) color: crate::math::Color,
  pub(crate) opacity: f32,
}

/// Device-to-gradient-local affine transform.
#[derive(Clone, Copy, Debug)]
pub struct GradientTransform {
  pub a: f32,
  pub b: f32,
  pub c: f32,
  pub d: f32,
  pub tx: f32,
  pub ty: f32,
}

/// Evaluated gradient coordinate parameters.
#[derive(Clone, Copy, Debug)]
pub enum GradientKind {
  Linear { sx: f32, sy: f32, dx: f32, dy: f32, inv_len_sq: f32 },
  Radial { sx: f32, sy: f32, inv_r: f32 },
  Focal { fx: f32, fy: f32, dx: f32, dy: f32, a: f32, r: f32 },
}

/// Premultiplied gradient LUT and evaluated coordinate map.
#[derive(Clone, Debug)]
pub struct GradientPaint {
  pub rule: Rule,
  pub lut: alloc::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
  pub transform: GradientTransform,
  pub kind: GradientKind,
  pub(crate) source_key: u128,
  /// Extra whole-paint alpha applied at the final source-over (e.g. an
  /// animated layer opacity deferred out of the frozen LUT). 255 = no-op and
  /// keeps the legacy bit-exact paths untouched.
  pub(crate) alpha: u8,
}

/// Borrowed device-space geometry for one draw operation.
#[derive(Clone, Copy)]
pub struct Geometry<'a> {
  contours: &'a [Contour],
  pub(crate) cache_key: u128,
  translation: Point,
}

impl<'a> Geometry<'a> {
  pub(super) fn new(contours: &'a [Contour], cache_key: u128) -> Self {
    Self {
      contours,
      cache_key,
      translation: Point::default(),
    }
  }

  pub(super) fn translated(contours: &'a [Contour], cache_key: u128, x: f32, y: f32) -> Self {
    Self {
      contours,
      cache_key,
      translation: Point { x, y },
    }
  }

  /// Number of contours participating in this draw.
  pub fn len(self) -> usize {
    self.contours.len()
  }

  /// Whether this draw has no contours.
  pub fn is_empty(self) -> bool {
    self.contours.is_empty()
  }

  /// Iterates contours without copying their point storage.
  pub fn contours(self) -> impl ExactSizeIterator<Item = ContourRef<'a>> {
    self.contours.iter().map(move |contour| ContourRef {
      contour,
      translation: self.translation,
    })
  }

  pub(crate) fn raw_contours(self) -> &'a [Contour] {
    self.contours
  }

  pub(crate) fn raw_translation(self) -> Point {
    self.translation
  }
}

/// Borrowed contour supplied to a frame renderer.
#[derive(Clone, Copy)]
pub struct ContourRef<'a> {
  contour: &'a Contour,
  translation: Point,
}

impl<'a> ContourRef<'a> {
  /// Iterates device-space points without copying them.
  pub fn points(self) -> impl ExactSizeIterator<Item = Point> + 'a {
    self.contour.points.iter().map(move |point| Point {
      x: point.x + self.translation.x,
      y: point.y + self.translation.y,
    })
  }

  /// Expanded fills and strokes currently always produce closed contours.
  pub fn closed(self) -> bool {
    true
  }
}

/// Borrowed paint data for a draw operation.
#[derive(Clone, Copy)]
pub enum Paint<'a> {
  /// Premultiplied solid paint.
  Solid(SolidPaint),
  /// Premultiplied evaluated gradient paint.
  Gradient(&'a GradientPaint),
}

/// Operation performed when ending one or more saved layers.
#[derive(Clone, Copy, Debug)]
pub enum Composite {
  /// Source-over composite of one saved layer.
  Over { opacity: u8 },
  /// Applies a saved matte source to a saved target and composites it.
  /// This consumes the two most recently saved layers.
  Matte { kind: u8, opacity: u8, source_opacity: u8 },
}

/// Backend operations called synchronously while evaluating a frame.
pub trait FrameRenderer {
  /// Starts a transparent isolated surface.
  fn save_layer(&mut self);

  /// Draws evaluated device-space geometry.
  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>);

  /// Applies one entry of the current layer's mask stack.
  #[allow(clippy::too_many_arguments)]
  fn apply_mask(&mut self, geometry: Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool);

  /// Ends saved surface state with the requested composite operation.
  fn end_layer(&mut self, composite: Composite);

  /// Reports whether geometry for `cache_key` is retained after acceptance.
  /// The walker may omit its contour payload on later frames when true.
  fn retains_geometry(&self, _cache_key: u128) -> bool {
    false
  }
}
