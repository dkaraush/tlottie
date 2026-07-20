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
  pub(super) alpha_only: bool,
  pub(super) bitmap_dirty: bool,
  pub(super) surfaces: Vec<super::mapped_surface::Surface>,
  pub(super) surface_dirty: Vec<super::executor::DirtyBox>,
  pub(super) surface_rows: Vec<Vec<super::executor::RowBounds>>,
  pub(super) row_bounds_pool: Vec<Vec<super::executor::RowBounds>>,
  pub(super) mask_accumulator: Option<Vec<u8>>,
  pub(super) alpha_fallback: Vec<u32>,
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
      alpha_only: false,
      bitmap_dirty: false,
      surfaces: Vec::new(),
      surface_dirty: Vec::new(),
      surface_rows: Vec::new(),
      row_bounds_pool: Vec::new(),
      mask_accumulator: None,
      alpha_fallback: Vec::new(),
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
      alpha_only: false,
      bitmap_dirty: false,
      surfaces: Vec::new(),
      surface_dirty: Vec::new(),
      surface_rows: Vec::new(),
      row_bounds_pool: Vec::new(),
      mask_accumulator: None,
      alpha_fallback: Vec::new(),
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

  /// Renders a frame directly into a one-byte-per-pixel alpha mask.
  pub fn render_alpha8(&mut self, frame: f32, alpha: &mut [u8], width: u32, height: u32, mut options: crate::RenderOptions) -> Result<()> {
    let limits = crate::Limits::default();
    if width == 0 || height == 0 || width > limits.max_dimension || height > limits.max_dimension {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "render size out of range",
      });
    }
    let Some(pixel_count) = (width as usize).checked_mul(height as usize) else {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "render size overflow",
      });
    };
    let Some(target) = alpha.get_mut(..pixel_count) else {
      return Err(crate::Error::InvalidLottie {
        offset: 0,
        what: "alpha buffer too small",
      });
    };
    // Luma mattes depend on source RGB, so they are the one case that cannot
    // be evaluated by an alpha-only backend. Keep a reusable ARGB fallback
    // for those uncommon compositions; ordinary Alpha8 rendering never
    // allocates or converts a color bitmap.
    if composition_uses_luma_matte(&self.comp) {
      let mut color = core::mem::take(&mut self.alpha_fallback);
      color.resize(pixel_count, 0);
      options.alpha_only = false;
      let result = self.render(frame, &mut color, width, height, options);
      if result.is_ok() {
        for (out, &pixel) in target.iter_mut().zip(&color) {
          *out = (pixel >> 24) as u8;
        }
      }
      self.alpha_fallback = color;
      return result;
    }
    options.alpha_only = true;
    self.state.cov_cache.set_budget_for_canvas(width as usize, height as usize);
    self.state.cov_cache.frame_tick();
    let composition = std::sync::Arc::clone(&self.comp);
    let mut walker = core::mem::take(&mut self.walker);
    let mut backend = super::alpha_backend::Alpha8Renderer::new(target, width as usize, height as usize, options.antialias, &mut self.state);
    let result = walker.render(&composition, frame, width, height, options, &mut backend);
    backend.finish();
    self.walker = walker;
    result
  }
}

fn composition_uses_luma_matte(composition: &Composition) -> bool {
  composition
    .layers
    .iter()
    .chain(composition.assets.iter().flat_map(|asset| asset.layers.iter()))
    .any(|layer| matches!(layer.matte, Some(3 | 4)))
}
