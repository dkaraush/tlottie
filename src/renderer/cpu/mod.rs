//! Stateful CPU renderer and its rasterization implementation.

mod alpha_backend;
mod backend;
pub(crate) mod cells;
pub(crate) mod executor;
mod mapped_surface;
pub(crate) mod raster;
mod renderer;
pub(crate) mod simd;

pub use renderer::CPURenderer;
