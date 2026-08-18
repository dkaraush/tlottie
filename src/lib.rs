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
// Unit tests always use the standard test harness even when exercising the
// opt-in no_std feature set. Production no_std artifacts stay genuinely so.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]

// Available in std builds too, so the rest of the crate can name `alloc`
// paths unconditionally and read identically in both configurations.
extern crate alloc;

#[cfg(all(not(feature = "std"), not(feature = "no-std")))]
compile_error!("building without `std` needs a hash map: enable the `no-std` feature (see the `compat` module)");

// A standalone C ABI artifact needs these; a Rust binary linking the crate
// brings its own.
#[cfg(all(not(feature = "std"), feature = "c-api", not(target_arch = "wasm32"), not(test)))]
mod no_std_runtime;

// wasm32-unknown-unknown has no host C allocator. Use the same dlmalloc
// implementation as Rust's standard library, backed directly by memory.grow.
#[cfg(all(not(feature = "std"), feature = "wasm", target_arch = "wasm32", not(test)))]
mod wasm_runtime;

mod compat;
mod composition;
mod error;
mod math;
mod pixel;
mod renderer;

#[cfg(any(all(feature = "c-api", not(target_arch = "wasm32")), all(feature = "wasm", target_arch = "wasm32")))]
mod bindings;

// Transitional crate-local names keep the implementation readable while the
// physical layout reflects ownership. These are not part of the public API.
pub(crate) use composition::{json, limits, model, parse, property};
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

pub use composition::options::{ChannelOrder, FitzModifier, LayerColorReplacement, ParseOptions, SourceColorReplacement};
pub use error::{Error, JsonErrorKind, Limit, Result};
pub use limits::Limits;
pub use model::Composition;
pub use renderer::options::RenderOptions;

impl Composition {
  /// Parses a Lottie composition from raw JSON bytes.
  ///
  /// Never panics; any malformed, truncated, or hostile input returns an
  /// [`Error`]. Resource consumption is bounded by `limits`.
  pub fn parse(json: &[u8], limits: &Limits) -> Result<Composition> {
    parse::parse_composition(json, limits, &ParseOptions::default())
  }

  /// Parses a Lottie composition and applies constructor-time customizations.
  ///
  /// Fitz and layer-name color replacements are folded into the immutable
  /// model once; rendering does not parse or match keypaths again.
  pub fn parse_with_options(json: &[u8], limits: &Limits, options: &ParseOptions) -> Result<Composition> {
    parse::parse_composition(json, limits, options)
  }
}

#[cfg(feature = "cpu")]
pub use renderer::cpu::CPURenderer;

#[cfg(test)]
#[path = "tests/dos.rs"]
mod dos;
