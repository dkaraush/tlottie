//! The two things `core` + `alloc` do not provide on their own.
//!
//! Kept in one place so the rest of the crate reads the same in both
//! configurations: everything else already works from `alloc` paths, which
//! are valid in `std` builds too.

#[cfg(not(feature = "std"))]
pub(crate) use hashbrown::{HashMap, HashSet};
/// `alloc` has no hash map. `std::collections::HashMap` already wraps
/// hashbrown, so the two are the same implementation and the caches keyed by
/// this type behave — and perform — identically either way.
#[cfg(feature = "std")]
pub(crate) use std::collections::{HashMap, HashSet};

/// Float methods that live in `std` because they lower to libm calls.
///
/// `core` covers the pure-arithmetic ones already (`min`, `max`, `clamp`,
/// `is_finite`, `to_radians`, …); only the transcendental and rounding
/// operations need an implementation. In `std` builds this trait is not
/// compiled at all — the inherent methods are used, so behaviour is
/// bit-identical rather than merely equivalent.
#[cfg(not(feature = "std"))]
pub(crate) use libm_shim::FloatExt;

#[cfg(not(feature = "std"))]
mod libm_shim {
  // Every symbol below is part of the C runtime already linked by any host
  // capable of running this crate; declaring them keeps the no_std build
  // dependency-free rather than pulling in a Rust libm reimplementation.
  #![allow(unsafe_code)]

  extern "C" {
    fn sqrtf(x: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn ceilf(x: f32) -> f32;
    fn roundf(x: f32) -> f32;
    fn truncf(x: f32) -> f32;
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn tanf(x: f32) -> f32;
    fn atan2f(y: f32, x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
    fn fmodf(x: f32, y: f32) -> f32;
    fn sqrt(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn ceil(x: f64) -> f64;
    fn round(x: f64) -> f64;
    fn trunc(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn tan(x: f64) -> f64;
    fn atan2(y: f64, x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn fmod(x: f64, y: f64) -> f64;
  }

  /// Rounding and transcendental float operations for `no_std` builds.
  pub(crate) trait FloatExt {
    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Largest integer not greater than self.
    fn floor(self) -> Self;
    /// Smallest integer not less than self.
    fn ceil(self) -> Self;
    /// Nearest integer, halfway cases away from zero.
    fn round(self) -> Self;
    /// Integer part.
    fn trunc(self) -> Self;
    /// Fractional part, keeping the sign of self.
    fn fract(self) -> Self;
    /// Sine of an angle in radians.
    fn sin(self) -> Self;
    /// Cosine of an angle in radians.
    fn cos(self) -> Self;
    /// Tangent of an angle in radians.
    fn tan(self) -> Self;
    /// Sine and cosine together, matching `f32::sin_cos`.
    fn sin_cos(self) -> (Self, Self)
    where
      Self: Sized;
    /// Four-quadrant arctangent of `self / other`.
    fn atan2(self, other: Self) -> Self;
    /// Self raised to a floating point power.
    fn powf(self, n: Self) -> Self;
    /// Least nonnegative remainder of `self (mod rhs)`.
    fn rem_euclid(self, rhs: Self) -> Self;
  }

  impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
      unsafe { sqrtf(self) }
    }

    // Sign-bit mask rather than a libm call: exact, branchless, and correct
    // for NaN and both zeroes.
    #[inline]
    fn abs(self) -> Self {
      Self::from_bits(self.to_bits() & 0x7fff_ffff)
    }

    #[inline]
    fn floor(self) -> Self {
      unsafe { floorf(self) }
    }

    #[inline]
    fn ceil(self) -> Self {
      unsafe { ceilf(self) }
    }

    #[inline]
    fn round(self) -> Self {
      unsafe { roundf(self) }
    }

    #[inline]
    fn trunc(self) -> Self {
      unsafe { truncf(self) }
    }

    // core's definition exactly: self - trunc(self).
    #[inline]
    fn fract(self) -> Self {
      self - FloatExt::trunc(self)
    }

    #[inline]
    fn sin(self) -> Self {
      unsafe { sinf(self) }
    }

    #[inline]
    fn cos(self) -> Self {
      unsafe { cosf(self) }
    }

    #[inline]
    fn tan(self) -> Self {
      unsafe { tanf(self) }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
      (FloatExt::sin(self), FloatExt::cos(self))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
      unsafe { atan2f(self, other) }
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
      unsafe { powf(self, n) }
    }

    // Mirrors core's definition; `%` on floats lowers to fmodf either way.
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
      let r = unsafe { fmodf(self, rhs) };
      if r < 0.0 {
        r + FloatExt::abs(rhs)
      } else {
        r
      }
    }
  }

  impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
      unsafe { sqrt(self) }
    }

    #[inline]
    fn abs(self) -> Self {
      Self::from_bits(self.to_bits() & 0x7fff_ffff_ffff_ffff)
    }

    #[inline]
    fn floor(self) -> Self {
      unsafe { floor(self) }
    }

    #[inline]
    fn ceil(self) -> Self {
      unsafe { ceil(self) }
    }

    #[inline]
    fn round(self) -> Self {
      unsafe { round(self) }
    }

    #[inline]
    fn trunc(self) -> Self {
      unsafe { trunc(self) }
    }

    #[inline]
    fn fract(self) -> Self {
      self - FloatExt::trunc(self)
    }

    #[inline]
    fn sin(self) -> Self {
      unsafe { sin(self) }
    }

    #[inline]
    fn cos(self) -> Self {
      unsafe { cos(self) }
    }

    #[inline]
    fn tan(self) -> Self {
      unsafe { tan(self) }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
      (FloatExt::sin(self), FloatExt::cos(self))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
      unsafe { atan2(self, other) }
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
      unsafe { pow(self, n) }
    }

    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
      let r = unsafe { fmod(self, rhs) };
      if r < 0.0 {
        r + FloatExt::abs(rhs)
      } else {
        r
      }
    }
  }
}
