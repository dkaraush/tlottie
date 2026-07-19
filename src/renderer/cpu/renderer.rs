//! Public CPU renderer entry points.

use crate::{Composition, Result};

use super::executor::RenderScratch;

/// Stateful CPU renderer for a parsed [`Composition`].
///
/// The composition is immutable and shareable while this renderer owns the
/// reusable raster buffers, mask planes, and gradient tables needed between
/// frames. Rendering is synchronous and single-threaded by design.
pub struct CPURenderer {
  pub(super) comp: std::sync::Arc<Composition>,
  pub(super) walker: crate::renderer::frame::FrameWalker,
  pub(super) state: RenderScratch,
  pub(super) bitmap: Option<core::ptr::NonNull<[u32]>>,
  pub(super) width: usize,
  pub(super) height: usize,
  pub(super) antialias: bool,
  pub(super) bitmap_dirty: bool,
  pub(super) surfaces: Vec<Vec<u32>>,
  pub(super) surface_dirty: Vec<super::executor::DirtyBox>,
  pub(super) mask_accumulator: Option<Vec<u8>>,
}

impl CPURenderer {
  /// Creates a CPU renderer owning its composition.
  pub fn new(comp: Composition) -> Self {
    Self {
      comp: std::sync::Arc::new(comp),
      walker: Default::default(),
      state: RenderScratch::default(),
      bitmap: None,
      width: 0,
      height: 0,
      antialias: true,
      bitmap_dirty: false,
      surfaces: Vec::new(),
      surface_dirty: Vec::new(),
      mask_accumulator: None,
    }
  }

  /// Creates a CPU renderer over a shared composition.
  pub fn from_shared(comp: std::sync::Arc<Composition>) -> Self {
    Self {
      comp,
      walker: Default::default(),
      state: RenderScratch::default(),
      bitmap: None,
      width: 0,
      height: 0,
      antialias: true,
      bitmap_dirty: false,
      surfaces: Vec::new(),
      surface_dirty: Vec::new(),
      mask_accumulator: None,
    }
  }

  /// Returns the underlying composition.
  pub fn composition(&self) -> &Composition {
    &self.comp
  }

  /// Renders a frame with explicit [`crate::RenderOptions`].
  pub fn render(&mut self, frame: f32, pixels: &mut [u32], width: u32, height: u32, options: crate::RenderOptions) -> Result<()> {
    let composition = std::sync::Arc::clone(&self.comp);
    let mut walker = core::mem::take(&mut self.walker);
    let result = self.with_bitmap(pixels, width, height, options, |renderer| walker.render(&composition, frame, width, height, options, renderer));
    self.walker = walker;
    result
  }
}
