//! Renderer-neutral frame evaluation and the CPU rendering backend.

#[cfg(feature = "cpu")]
pub(crate) mod cpu;
pub(crate) mod frame;
pub mod options;
