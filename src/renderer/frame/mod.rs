//! Shared vector geometry and renderer-neutral evaluated-frame operations.

pub(crate) mod geometry;
pub(crate) mod renderer;
pub(crate) mod stroke;
#[cfg(feature = "cpu")]
mod walker;

pub(crate) use renderer::GRADIENT_LUT_SIZE;
#[doc(hidden)]
#[cfg(feature = "cpu")]
pub use renderer::{Composite, ContourRef, FrameRenderer, Geometry, GradientKind, GradientPaint, GradientTransform, Paint, Point, Rule, SolidPaint};
#[doc(hidden)]
#[cfg(feature = "cpu")]
pub use walker::{walk_frame_into, FrameWalker};
