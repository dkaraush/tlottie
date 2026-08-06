//! Public CPU renderer entry points.

use crate::{Composition, Result};
use alloc::vec::Vec;

use super::executor::RenderScratch;

/// Avoid retaining an arbitrarily large second copy of the output bitmap.
const STATIC_BITMAP_CACHE_BYTES: usize = 4 * 1024 * 1024;

/// Stateful CPU renderer for a parsed [`Composition`].
///
/// The composition is immutable and shareable while this renderer owns the
/// reusable raster buffers, mask planes, and gradient tables needed between
/// frames. Rendering is synchronous and single-threaded by design.
pub struct CPURenderer {
  pub(super) comp: alloc::sync::Arc<Composition>,
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
  /// Final bitmap for a composition proven static by the parser. Repeated
  /// renders are common for players and benchmarks even when frame_count is
  /// one; replaying the complete shape tree in that case only burns CPU.
  static_bitmap: Option<StaticBitmap>,
}

struct StaticBitmap {
  width: u32,
  height: u32,
  options: crate::RenderOptions,
  pixels: Vec<u32>,
}

impl CPURenderer {
  /// Creates a CPU renderer owning its composition.
  pub fn new(comp: Composition) -> Self {
    Self {
      comp: alloc::sync::Arc::new(comp),
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
      static_bitmap: None,
    }
  }

  /// Creates a CPU renderer over a shared composition.
  pub fn from_shared(comp: alloc::sync::Arc<Composition>) -> Self {
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
      static_bitmap: None,
    }
  }

  /// Returns the underlying composition.
  pub fn composition(&self) -> &Composition {
    &self.comp
  }

  /// Renders a frame with explicit [`crate::RenderOptions`].
  pub fn render(&mut self, frame: f32, pixels: &mut [u32], width: u32, height: u32, options: crate::RenderOptions) -> Result<()> {
    if options.clear && self.comp.is_static() {
      if let Some(cached) = &self.static_bitmap {
        if cached.width == width && cached.height == height && cached.options == options {
          let Some(target) = pixels.get_mut(..cached.pixels.len()) else {
            return Err(crate::Error::InvalidLottie {
              offset: 0,
              what: "pixel buffer too small",
            });
          };
          target.copy_from_slice(&cached.pixels);
          return Ok(());
        }
      }
    }
    let composition = alloc::sync::Arc::clone(&self.comp);
    let mut walker = core::mem::take(&mut self.walker);
    let result = self.with_bitmap(pixels, width, height, options, |renderer| walker.render(&composition, frame, width, height, options, renderer));
    self.walker = walker;
    if result.is_ok() && options.clear && self.comp.is_static() {
      let pixel_count = (width as usize).saturating_mul(height as usize);
      let cache_fits = pixel_count.checked_mul(core::mem::size_of::<u32>()).is_some_and(|bytes| bytes <= STATIC_BITMAP_CACHE_BYTES);
      if cache_fits {
        let Some(rendered) = pixels.get(..pixel_count) else {
          return result;
        };
        self.static_bitmap = Some(StaticBitmap {
          width,
          height,
          options,
          pixels: rendered.to_vec(),
        });
      } else {
        self.static_bitmap = None;
      }
    }
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
    // be evaluated by an alpha-only backend. Keep a reusable RGBA fallback
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
    let composition = alloc::sync::Arc::clone(&self.comp);
    let mut walker = core::mem::take(&mut self.walker);
    let mut backend = super::alpha_backend::Alpha8Renderer::new(target, width as usize, height as usize, options.antialias, options.clear, &mut self.state);
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
