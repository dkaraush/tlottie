//! Small single-precision geometry and math primitives.

#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;

/// 2D point / vector.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Vec2 {
  pub x: f32,
  pub y: f32,
}

impl Vec2 {
  pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

  pub fn new(x: f32, y: f32) -> Self {
    Self { x, y }
  }

  pub fn lerp(self, other: Vec2, t: f32) -> Vec2 {
    Vec2::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
  }
}

/// 2x3 affine transform, column notation:
/// `[ a c tx ]`
/// `[ b d ty ]`
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mat2x3 {
  pub a: f32,
  pub b: f32,
  pub c: f32,
  pub d: f32,
  pub tx: f32,
  pub ty: f32,
}

impl Mat2x3 {
  pub const IDENTITY: Mat2x3 = Mat2x3 {
    a: 1.0,
    b: 0.0,
    c: 0.0,
    d: 1.0,
    tx: 0.0,
    ty: 0.0,
  };

  pub fn translate(tx: f32, ty: f32) -> Self {
    Mat2x3 {
      a: 1.0,
      b: 0.0,
      c: 0.0,
      d: 1.0,
      tx,
      ty,
    }
  }

  pub fn scale(sx: f32, sy: f32) -> Self {
    Mat2x3 {
      a: sx,
      b: 0.0,
      c: 0.0,
      d: sy,
      tx: 0.0,
      ty: 0.0,
    }
  }

  pub fn rotate(degrees: f32) -> Self {
    let rad = degrees.to_radians();
    let (sin, cos) = rad.sin_cos();
    Mat2x3 {
      a: cos,
      b: sin,
      c: -sin,
      d: cos,
      tx: 0.0,
      ty: 0.0,
    }
  }

  /// Skew by `sk` degrees along axis `sa` degrees (AE semantics):
  /// rotate into the axis frame, shear in X, rotate back.
  #[allow(dead_code)] // Reserved until skew is enabled in transform evaluation.
  pub fn skew(sk_deg: f32, sa_deg: f32) -> Self {
    let shear = Mat2x3 {
      a: 1.0,
      b: 0.0,
      c: (-sk_deg.to_radians()).tan(),
      d: 1.0,
      tx: 0.0,
      ty: 0.0,
    };
    Mat2x3::rotate(sa_deg).concat(shear).concat(Mat2x3::rotate(-sa_deg))
  }

  /// `self * rhs`: applies `rhs` first, then `self`.
  pub fn concat(self, rhs: Mat2x3) -> Mat2x3 {
    Mat2x3 {
      a: self.a * rhs.a + self.c * rhs.b,
      b: self.b * rhs.a + self.d * rhs.b,
      c: self.a * rhs.c + self.c * rhs.d,
      d: self.b * rhs.c + self.d * rhs.d,
      tx: self.a * rhs.tx + self.c * rhs.ty + self.tx,
      ty: self.b * rhs.tx + self.d * rhs.ty + self.ty,
    }
  }

  /// Inverse of this affine transform; identity fallback when singular.
  pub fn inverse(&self) -> Mat2x3 {
    let det = self.a * self.d - self.b * self.c;
    if det.abs() < 1e-12 || !det.is_finite() {
      return Mat2x3::IDENTITY;
    }
    let inv = 1.0 / det;
    let a = self.d * inv;
    let b = -self.b * inv;
    let c = -self.c * inv;
    let d = self.a * inv;
    Mat2x3 {
      a,
      b,
      c,
      d,
      tx: -(a * self.tx + c * self.ty),
      ty: -(b * self.tx + d * self.ty),
    }
  }

  pub fn apply(&self, p: Vec2) -> Vec2 {
    Vec2::new(self.a * p.x + self.c * p.y + self.tx, self.b * p.x + self.d * p.y + self.ty)
  }
}

/// Straight (non-premultiplied) RGBA color, components in 0..=1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Color {
  pub r: f32,
  pub g: f32,
  pub b: f32,
  pub a: f32,
}

impl Color {
  pub const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

  pub fn lerp(self, other: Color, t: f32) -> Color {
    Color {
      r: self.r + (other.r - self.r) * t,
      g: self.g + (other.g - self.g) * t,
      b: self.b + (other.b - self.b) * t,
      a: self.a + (other.a - self.a) * t,
    }
  }
}
