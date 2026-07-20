#![allow(unsafe_code)]

use std::alloc::{alloc, dealloc, Layout};

use crate::{CPURenderer, Composition, Limits, RenderOptions};

/// A playing instance plus its conversion buffers.
pub struct Instance {
  anim: CPURenderer,
  /// Premultiplied ARGB32 render target (what the renderer writes).
  argb: Vec<u32>,
  /// Un-premultiplied RGBA8 copy handed to the canvas `ImageData`.
  rgba: Vec<u8>,
  /// Direct one-byte-per-pixel alpha render target.
  alpha8: Vec<u8>,
}

/// Allocates `len` bytes for JS to copy input into. Returns null on failure
/// (zero `len`, or OOM).
#[no_mangle]
pub extern "C" fn tlottie_alloc(len: usize) -> *mut u8 {
  match Layout::from_size_align(len, 1) {
    Ok(layout) if len > 0 => {
      // SAFETY: layout is valid and non-zero-sized.
      unsafe { alloc(layout) }
    }
    _ => std::ptr::null_mut(),
  }
}

/// Frees a buffer from `tlottie_alloc`. `len` must match the allocation.
///
/// # Safety
/// `ptr` must come from `tlottie_alloc(len)` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn tlottie_free(ptr: *mut u8, len: usize) {
  if ptr.is_null() {
    return;
  }
  if let Ok(layout) = Layout::from_size_align(len, 1) {
    if len > 0 {
      // SAFETY: caller contract — ptr/layout match the allocation.
      unsafe { dealloc(ptr, layout) }
    }
  }
}

/// Parses `json` (plain Lottie JSON bytes) and returns an instance, or null
/// if parsing rejects the input.
///
/// # Safety
/// `json_ptr..json_ptr+json_len` must be readable (a `tlottie_alloc` buffer JS
/// filled).
#[no_mangle]
pub unsafe extern "C" fn tlottie_new(json_ptr: *const u8, json_len: usize) -> *mut Instance {
  if json_ptr.is_null() {
    return std::ptr::null_mut();
  }
  // SAFETY: caller contract — the range is a live allocation.
  let json = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
  match Composition::parse(json, &Limits::default()) {
    Ok(comp) => Box::into_raw(Box::new(Instance {
      anim: CPURenderer::new(comp),
      argb: Vec::new(),
      rgba: Vec::new(),
      alpha8: Vec::new(),
    })),
    Err(_) => std::ptr::null_mut(),
  }
}

/// Releases an instance.
///
/// # Safety
/// `inst` must come from `tlottie_new` and not have been dropped.
#[no_mangle]
pub unsafe extern "C" fn tlottie_drop(inst: *mut Instance) {
  if !inst.is_null() {
    // SAFETY: caller contract — inst is a live Box from tlottie_new.
    drop(unsafe { Box::from_raw(inst) });
  }
}

/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_width(inst: *const Instance) -> u32 {
  // SAFETY: caller contract.
  unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().width)
}

/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_height(inst: *const Instance) -> u32 {
  // SAFETY: caller contract.
  unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().height)
}

/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_frame_rate(inst: *const Instance) -> f32 {
  // SAFETY: caller contract.
  unsafe { inst.as_ref() }.map_or(0.0, |i| i.anim.composition().frame_rate)
}

/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_frame_count(inst: *const Instance) -> u32 {
  // SAFETY: caller contract.
  unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().frame_count())
}

/// Renders `frame` at `width`x`height` and returns a pointer to a
/// `width * height * 4` un-premultiplied RGBA8 buffer (row-major), or null
/// on error. The buffer is owned by the instance and overwritten by the
/// next call. `antialias != 0` enables analytic edge antialiasing.
///
/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32) -> *const u8 {
  // SAFETY: this function has the same instance contract as the extended API.
  unsafe { tlottie_render_with_options(inst, frame, width, height, antialias, RenderOptions::default().curve_tolerance) }
}

/// Like [`tlottie_render`], with an explicit positive maximum curve-flattening
/// error in device pixels. Returns null for an invalid tolerance.
///
/// # Safety
/// `inst` must be a live instance from `tlottie_new`.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_with_options(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32, curve_tolerance: f32) -> *const u8 {
  // SAFETY: caller contract.
  let Some(inst) = (unsafe { inst.as_mut() }) else {
    return std::ptr::null_mut();
  };
  if !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return std::ptr::null_mut();
  }
  let Some(px) = (width as usize).checked_mul(height as usize) else {
    return std::ptr::null_mut();
  };
  // Resize-only: render() fully overwrites all px pixels, so re-zeroing
  // an already-sized buffer is pure memset waste (profiled ~1MB/frame).
  if inst.argb.len() != px {
    inst.argb.clear();
    inst.argb.resize(px, 0);
  }
  if inst
    .anim
    .render(
      frame,
      &mut inst.argb,
      width,
      height,
      RenderOptions {
        antialias: antialias != 0,
        curve_tolerance,
        ..RenderOptions::default()
      },
    )
    .is_err()
  {
    return std::ptr::null_mut();
  }
  let Some(bytes) = px.checked_mul(4) else {
    return std::ptr::null_mut();
  };
  // Same resize-only rationale: the conversion loop below writes every
  // byte of the RGBA buffer.
  if inst.rgba.len() != bytes {
    inst.rgba.clear();
    inst.rgba.resize(bytes, 0);
  }
  crate::pixel::argb_to_rgba_slice(&inst.argb, &mut inst.rgba);
  inst.rgba.as_ptr()
}

/// Renders directly into a `width * height` Alpha8 buffer owned by the
/// instance. The returned pointer remains valid until the next render call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32) -> *const u8 {
  unsafe { tlottie_render_alpha8_with_options(inst, frame, width, height, antialias, RenderOptions::default().curve_tolerance) }
}

/// Like [`tlottie_render_alpha8`], with an explicit curve tolerance.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8_with_options(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32, curve_tolerance: f32) -> *const u8 {
  let Some(inst) = (unsafe { inst.as_mut() }) else {
    return std::ptr::null();
  };
  if !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return std::ptr::null();
  }
  let Some(px) = (width as usize).checked_mul(height as usize) else {
    return std::ptr::null();
  };
  if inst.alpha8.len() != px {
    inst.alpha8.resize(px, 0);
  }
  if inst
    .anim
    .render_alpha8(
      frame,
      &mut inst.alpha8,
      width,
      height,
      RenderOptions {
        antialias: antialias != 0,
        curve_tolerance,
        alpha_only: true,
      },
    )
    .is_err()
  {
    return std::ptr::null();
  }
  inst.alpha8.as_ptr()
}
