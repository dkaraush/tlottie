//! Stateful CPU renderer and its rasterization implementation.

mod backend;
pub(crate) mod cells;
pub(crate) mod executor;
pub(crate) mod raster;
mod renderer;
pub(crate) mod simd;

pub(crate) use executor::{GRAD_STATS, MODE_STATS, PX_STATS};
pub use renderer::CPURenderer;
