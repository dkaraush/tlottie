//! tlottie — a Lottie evaluator with CPU and optional GPU presentation/rendering backends.
//!
//! Core contracts:
//! - malformed input yields [`Error`] rather than intentionally panicking;
//! - **no I/O**: input is JSON bytes handed in by the host; external references
//!   in the JSON (fonts, image files) are never resolved;
//! - **single-threaded**: no internal threads, no global locks; instances are
//!   independent and the host owns all concurrency;
//! - the default CPU build has no mandatory third-party dependencies.

// Unsafe is denied by default. The isolated SIMD,
// opt-in FFI, and opt-in Vulkan modules locally allow only the operations
// required at those boundaries. `deny` (not `forbid`) lets those modules opt
// in explicitly while keeping the rest of the crate safe by default.
#![deny(unsafe_code)]
#![deny(missing_docs)]

mod composition;
mod error;
mod math;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod pixel;
mod renderer;

#[cfg(any(all(feature = "c-api", not(target_arch = "wasm32")), all(feature = "wasm", target_arch = "wasm32")))]
mod bindings;

// Transitional crate-local names keep the implementation readable while the
// physical layout reflects ownership. These are not part of the public API.
pub(crate) use composition::{json, limits, model, parse, property};
#[cfg(feature = "cpu")]
pub(crate) use renderer::cpu as render;
#[cfg(feature = "cpu")]
pub(crate) use renderer::cpu::{cells, raster, simd};
pub(crate) use renderer::frame::{geometry, stroke};

#[cfg(feature = "opengl")]
pub use renderer::opengl;
#[cfg(feature = "vulkan")]
pub use renderer::vulkan;

#[doc(hidden)]
#[cfg(feature = "cpu")]
pub mod internal {
  //! Unstable renderer-neutral frame commands.
  //!
  //! This module is not part of tlottie's stable API.

  pub use crate::renderer::frame::*;
}

pub use error::{Error, JsonErrorKind, Limit, Result};
pub use limits::Limits;
pub use model::Composition;
pub use renderer::options::RenderOptions;

/// Dev-only counters for the bench CLI (gradient pixels per kind and
/// batched-kernel coverage). Not part of the stable API.
#[doc(hidden)]
#[cfg(feature = "cpu")]
pub fn mode_stats() -> [u64; 3] {
  let mut out = [0u64; 3];
  for (o, c) in out.iter_mut().zip(render::MODE_STATS.iter()) {
    *o = c.load(core::sync::atomic::Ordering::Relaxed);
  }
  out
}

#[doc(hidden)]
#[cfg(feature = "cpu")]
pub fn px_stats() -> [u64; 12] {
  let mut out = [0u64; 12];
  for (o, c) in out.iter_mut().zip(render::PX_STATS.iter()) {
    *o = c.load(core::sync::atomic::Ordering::Relaxed);
  }
  out
}

#[doc(hidden)]
#[cfg(feature = "cpu")]
pub fn stroke_stats() -> [u64; 5] {
  let mut out = [0u64; 5];
  for (o, c) in out.iter_mut().zip(stroke::STROKE_STATS.iter()) {
    *o = c.load(core::sync::atomic::Ordering::Relaxed);
  }
  out
}

#[doc(hidden)]
#[cfg(feature = "cpu")]
pub fn gradient_stats() -> [u64; 5] {
  let mut out = [0u64; 5];
  for (o, c) in out.iter_mut().zip(render::GRAD_STATS.iter()) {
    *o = c.load(core::sync::atomic::Ordering::Relaxed);
  }
  out
}

impl Composition {
  /// Parses a Lottie composition from raw JSON bytes.
  ///
  /// Never panics; any malformed, truncated, or hostile input returns an
  /// [`Error`]. Resource consumption is bounded by `limits`.
  pub fn parse(json: &[u8], limits: &Limits) -> Result<Composition> {
    parse::parse_composition(json, limits)
  }
}

#[cfg(feature = "cpu")]
pub use renderer::cpu::CPURenderer;

/// Backward-compatible name for [`CPURenderer`].
#[deprecated(note = "renamed to CPURenderer")]
#[cfg(feature = "cpu")]
pub type Animation = CPURenderer;
