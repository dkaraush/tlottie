//! Renderer-neutral frame evaluation and optional rendering backends.

#[cfg(feature = "cpu")]
pub(crate) mod cpu;
pub(crate) mod frame;
#[cfg(feature = "opengl")]
pub mod opengl;
pub mod options;
#[cfg(feature = "vulkan")]
pub mod vulkan;
