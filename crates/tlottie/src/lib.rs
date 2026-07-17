//! tlottie — a CPU Lottie renderer built for the Telegram workload.
//!
//! Contracts (see GOALS.md):
//! - **never panics**: hostile or malformed input yields [`Error`], not a crash;
//! - **no I/O**: input is JSON bytes handed in by the host; external references
//!   in the JSON (fonts, image files) are never resolved;
//! - **single-threaded**: no internal threads, no global locks; instances are
//!   independent and the host owns all concurrency;
//! - **zero runtime dependencies**.

// Safety contract (GOALS.md): no unsafe anywhere EXCEPT the isolated SIMD
// blit module (simd.rs), whose few unsafe blocks are pointer-width vector
// load/stores with locally-checked bounds and a scalar bit-exact oracle.
// `deny` (not `forbid`) solely so that module can opt in explicitly.
#![deny(unsafe_code)]
#![deny(missing_docs)]

mod cells;
mod error;
mod geometry;
mod json;
mod limits;
mod math;
mod model;
mod parse;
mod property;
mod raster;
mod render;
mod simd;
mod stroke;
mod stroke_ft;

#[doc(hidden)]
pub mod internal {
    //! Unstable internals shared with experimental sibling crates.
    //!
    //! This module is not part of tlottie's stable API.

    pub use crate::render::vulkan::*;
}

pub use error::{Error, JsonErrorKind, Limit, Result};
pub use limits::Limits;
pub use model::Composition;

/// Per-frame rendering options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
    /// Enables analytic edge antialiasing. Defaults to `true`.
    ///
    /// When disabled, coverage is thresholded to fully transparent or fully
    /// opaque at the 50% mark. This is faster for some span-heavy animations,
    /// but produces visibly jagged edges.
    pub antialias: bool,
    /// Maximum device-space curve-flattening error in pixels.
    ///
    /// The accurate default is `0.05`. Larger values reduce contour points
    /// and GPU edge work at the cost of geometric accuracy.
    pub curve_tolerance: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            antialias: true,
            curve_tolerance: 0.05,
        }
    }
}

/// Dev-only counters for the bench CLI (gradient pixels per kind and
/// batched-kernel coverage). Not part of the stable API.
#[doc(hidden)]
pub fn mode_stats() -> [u64; 3] {
    let mut out = [0u64; 3];
    for (o, c) in out.iter_mut().zip(render::MODE_STATS.iter()) {
        *o = c.load(core::sync::atomic::Ordering::Relaxed);
    }
    out
}

#[doc(hidden)]
pub fn px_stats() -> [u64; 12] {
    let mut out = [0u64; 12];
    for (o, c) in out.iter_mut().zip(render::PX_STATS.iter()) {
        *o = c.load(core::sync::atomic::Ordering::Relaxed);
    }
    out
}

#[doc(hidden)]
pub fn stroke_stats() -> [u64; 5] {
    let mut out = [0u64; 5];
    for (o, c) in out.iter_mut().zip(stroke::STROKE_STATS.iter()) {
        *o = c.load(core::sync::atomic::Ordering::Relaxed);
    }
    out
}

#[doc(hidden)]
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

/// A playing instance of a composition: the immutable, shareable
/// [`Composition`] plus this instance's private render state (reused
/// buffers, memoized gradient tables). One `Animation` per on-screen
/// animation; many instances may share one parsed model via `Arc`.
///
/// Single-threaded by design — no internal threads or locks; the instance
/// is `Send`, so a host may move it between its own worker threads.
pub struct Animation {
    comp: std::sync::Arc<Composition>,
    state: render::RenderScratch,
}

impl Animation {
    /// Creates an instance owning its model.
    pub fn new(comp: Composition) -> Animation {
        Animation {
            comp: std::sync::Arc::new(comp),
            state: Default::default(),
        }
    }

    /// Creates an instance over a shared model (model-dedup across
    /// instances: parse once, play many).
    pub fn from_shared(comp: std::sync::Arc<Composition>) -> Animation {
        Animation {
            comp,
            state: Default::default(),
        }
    }

    /// The underlying composition.
    pub fn composition(&self) -> &Composition {
        &self.comp
    }

    /// Renders frame `frame` into `pixels` (row-major premultiplied ARGB32,
    /// `width * height` entries; fully overwritten). Working buffers persist
    /// inside the instance across calls.
    ///
    /// The Lottie timeline is continuous (keyframe times are floats), so
    /// `frame` may fall between authored frames for exact in-between poses
    /// (e.g. 120 Hz playback of a 60 fps file). It is clamped to
    /// `[0, frame_count - 1]`; non-finite values render frame 0.
    pub fn render(
        &mut self,
        frame: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.render_with_options(frame, pixels, width, height, RenderOptions::default())
    }

    /// Renders a frame with explicit [`RenderOptions`]. Working buffers and
    /// coverage caches remain safe when options change between calls.
    pub fn render_with_options(
        &mut self,
        frame: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
        options: RenderOptions,
    ) -> Result<()> {
        self.comp
            .render_pooled(&mut self.state, frame, pixels, width, height, options)
    }
}
