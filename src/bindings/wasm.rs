#![allow(unsafe_code)]

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::vec::Vec;
use alloc::alloc::{alloc, dealloc};
use core::alloc::Layout;

use crate::{CPURenderer, Composition, FitzModifier, LayerColorReplacement, Limits, ParseOptions, RenderOptions};

/// Wasm-memory representation of one layer-prefix color override.
#[repr(C)]
pub struct LayerColorReplacementAbi {
  /// Pointer to UTF-8 prefix bytes.
  pub layer_name_prefix: *const u8,
  /// Prefix byte length.
  pub layer_name_prefix_len: usize,
  /// Straight-alpha color in `0xAARRGGBB` form.
  pub color: u32,
}

/// A playing instance plus its conversion buffers.
pub struct Instance {
  anim: CPURenderer,
  /// Premultiplied RGBA8 render target (what the renderer writes).
  premultiplied_rgba: Vec<u32>,
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
    _ => core::ptr::null_mut(),
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
  unsafe { tlottie_new_with_options(json_ptr, json_len, 0, core::ptr::null(), 0) }
}

/// Parses JSON with constructor-time Fitz and layer-prefix color options.
///
/// # Safety
/// All supplied wasm-memory byte ranges and the replacement array must be
/// readable for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_new_with_options(
  json_ptr: *const u8,
  json_len: usize,
  fitz_modifier: u32,
  replacements_ptr: *const LayerColorReplacementAbi,
  replacements_len: usize,
) -> *mut Instance {
  if json_ptr.is_null() {
    return core::ptr::null_mut();
  }
  let Some(fitz_modifier) = FitzModifier::from_u32(fitz_modifier) else {
    return core::ptr::null_mut();
  };
  if replacements_len != 0 && replacements_ptr.is_null() {
    return core::ptr::null_mut();
  }
  // SAFETY: caller contract — the range is a live allocation.
  let json = unsafe { core::slice::from_raw_parts(json_ptr, json_len) };
  let raw_replacements = if replacements_len == 0 {
    &[]
  } else {
    unsafe { core::slice::from_raw_parts(replacements_ptr, replacements_len) }
  };
  let mut layer_color_replacements = Vec::with_capacity(raw_replacements.len());
  for replacement in raw_replacements {
    if replacement.layer_name_prefix_len != 0 && replacement.layer_name_prefix.is_null() {
      return core::ptr::null_mut();
    }
    let bytes = if replacement.layer_name_prefix_len == 0 {
      &[]
    } else {
      unsafe { core::slice::from_raw_parts(replacement.layer_name_prefix, replacement.layer_name_prefix_len) }
    };
    let Ok(prefix) = core::str::from_utf8(bytes) else {
      return core::ptr::null_mut();
    };
    layer_color_replacements.push(LayerColorReplacement {
      layer_name_prefix: prefix.to_owned(),
      color: replacement.color,
    });
  }
  let options = ParseOptions {
    fitz_modifier,
    layer_color_replacements,
    ..ParseOptions::default()
  };
  match Composition::parse_with_options(json, &Limits::default(), &options) {
    Ok(comp) => Box::into_raw(Box::new(Instance {
      anim: CPURenderer::new(comp),
      premultiplied_rgba: Vec::new(),
      rgba: Vec::new(),
      alpha8: Vec::new(),
    })),
    Err(_) => core::ptr::null_mut(),
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
    return core::ptr::null_mut();
  };
  if !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return core::ptr::null_mut();
  }
  let Some(px) = (width as usize).checked_mul(height as usize) else {
    return core::ptr::null_mut();
  };
  // Resize-only: render() fully overwrites all px pixels, so re-zeroing
  // an already-sized buffer is pure memset waste (profiled ~1MB/frame).
  if inst.premultiplied_rgba.len() != px {
    inst.premultiplied_rgba.clear();
    inst.premultiplied_rgba.resize(px, 0);
  }
  if inst
    .anim
    .render(
      frame,
      &mut inst.premultiplied_rgba,
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
    return core::ptr::null_mut();
  }
  let Some(bytes) = px.checked_mul(4) else {
    return core::ptr::null_mut();
  };
  // Same resize-only rationale: the conversion loop below writes every
  // byte of the RGBA buffer.
  if inst.rgba.len() != bytes {
    inst.rgba.clear();
    inst.rgba.resize(bytes, 0);
  }
  crate::pixel::premultiplied_rgba_to_straight_slice(&inst.premultiplied_rgba, &mut inst.rgba);
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
    return core::ptr::null();
  };
  if render_alpha8(inst, frame, width, height, antialias, curve_tolerance).is_none() {
    return core::ptr::null();
  }
  inst.alpha8.as_ptr()
}

/// Renders Alpha8 and expands it into straight-alpha RGBA8 using the constant
/// RGB color encoded as `0x00RRGGBB`. The returned instance-owned buffer has
/// `width * height * 4` bytes and is overwritten by the next render call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8_color(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32, color: u32) -> *const u8 {
  unsafe { tlottie_render_alpha8_color_with_options(inst, frame, width, height, antialias, color, RenderOptions::default().curve_tolerance) }
}

/// Like [`tlottie_render_alpha8_color`], with an explicit curve tolerance.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8_color_with_options(inst: *mut Instance, frame: f32, width: u32, height: u32, antialias: u32, color: u32, curve_tolerance: f32) -> *const u8 {
  let Some(inst) = (unsafe { inst.as_mut() }) else {
    return core::ptr::null();
  };
  let Some(px) = render_alpha8(inst, frame, width, height, antialias, curve_tolerance) else {
    return core::ptr::null();
  };
  let Some(bytes) = px.checked_mul(4) else {
    return core::ptr::null();
  };
  if inst.rgba.len() != bytes {
    inst.rgba.resize(bytes, 0);
  }
  crate::pixel::alpha8_to_rgba_slice(&inst.alpha8, &mut inst.rgba, color);
  inst.rgba.as_ptr()
}

fn render_alpha8(inst: &mut Instance, frame: f32, width: u32, height: u32, antialias: u32, curve_tolerance: f32) -> Option<usize> {
  if !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return None;
  }
  let px = (width as usize).checked_mul(height as usize)?;
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
        clear: true,
        antialias: antialias != 0,
        curve_tolerance,
        alpha_only: true,
      },
    )
    .is_err()
  {
    return None;
  }
  Some(px)
}
