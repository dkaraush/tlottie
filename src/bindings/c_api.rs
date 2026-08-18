//! Native C ABI for rendering into caller-owned premultiplied RGBA8 buffers.

#![allow(unsafe_code)]

use crate::{CPURenderer, ChannelOrder, Composition, FitzModifier, LayerColorReplacement, Limits, ParseOptions, RenderOptions, SourceColorReplacement};
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// One constructor-time layer-prefix color override.
#[repr(C)]
pub struct TLottieLayerColorReplacement {
  /// UTF-8 bytes for the layer-name prefix (without a trailing `**`).
  pub layer_name_prefix: *const u8,
  /// Number of bytes in `layer_name_prefix`.
  pub layer_name_prefix_len: usize,
  /// Straight-alpha color in `0xAARRGGBB` form.
  pub color: u32,
}

/// One constructor-time exact source-color replacement.
#[repr(C)]
pub struct TLottieColorReplacement {
  /// Source color in Android `0xAARRGGBB` form. The alpha byte is ignored.
  pub source_color: u32,
  /// Target color in Android `0xAARRGGBB` form. The animation's original alpha is preserved.
  pub target_color: u32,
}

/// Opaque renderer handle owned by C callers.
pub struct TLottieInstance {
  renderer: CPURenderer,
}

/// Parses Lottie JSON bytes and returns an animation handle, or null on error.
///
/// # Safety
/// `json_ptr..json_ptr+json_len` must be readable for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_new(json_ptr: *const u8, json_len: usize) -> *mut TLottieInstance {
  unsafe { tlottie_new_with_options(json_ptr, json_len, 0, core::ptr::null(), 0, core::ptr::null(), 0, 0) }
}

/// Parses Lottie JSON with a Fitz modifier and layer-prefix color overrides.
///
/// `fitz_modifier` is one of `TLOTTIE_FITZ_*`. Prefix strings are matched
/// directly against layer `nm` values and need no trailing `**`.
///
/// `channel_order` is one of `TLOTTIE_CHANNEL_*` and selects the byte order
/// of every pixel this instance renders. It is applied once here, so it costs
/// nothing per frame; color replacements are still given as `0xAARRGGBB`
/// regardless of the order chosen.
///
/// # Safety
/// The JSON and every prefix byte range must be readable for this call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_new_with_options(
  json_ptr: *const u8,
  json_len: usize,
  fitz_modifier: u32,
  replacements_ptr: *const TLottieLayerColorReplacement,
  replacements_len: usize,
  color_replacements_ptr: *const TLottieColorReplacement,
  color_replacements_len: usize,
  channel_order: u32,
) -> *mut TLottieInstance {
  if json_ptr.is_null() {
    return core::ptr::null_mut();
  }
  let Some(fitz_modifier) = FitzModifier::from_u32(fitz_modifier) else {
    return core::ptr::null_mut();
  };
  let Some(channel_order) = ChannelOrder::from_u32(channel_order) else {
    return core::ptr::null_mut();
  };
  if replacements_len != 0 && replacements_ptr.is_null() {
    return core::ptr::null_mut();
  }
  if color_replacements_len != 0 && color_replacements_ptr.is_null() {
    return core::ptr::null_mut();
  }
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
  let raw_color_replacements = if color_replacements_len == 0 {
    &[]
  } else {
    unsafe { core::slice::from_raw_parts(color_replacements_ptr, color_replacements_len) }
  };
  let source_color_replacements = raw_color_replacements
    .iter()
    .map(|replacement| SourceColorReplacement {
      source_color: replacement.source_color,
      target_color: replacement.target_color,
    })
    .collect();
  let options = ParseOptions {
    fitz_modifier,
    layer_color_replacements,
    source_color_replacements,
    channel_order,
  };
  match Composition::parse_with_options(json, &Limits::default(), &options) {
    Ok(comp) => Box::into_raw(Box::new(TLottieInstance { renderer: CPURenderer::new(comp) })),
    Err(_) => core::ptr::null_mut(),
  }
}

/// Releases an animation handle.
///
/// # Safety
/// `anim` must be null or a handle returned by [`tlottie_new`] that
/// has not already been dropped.
#[no_mangle]
pub unsafe extern "C" fn tlottie_drop(anim: *mut TLottieInstance) {
  if !anim.is_null() {
    drop(unsafe { Box::from_raw(anim) });
  }
}

/// Returns the source composition width, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_width(anim: *const TLottieInstance) -> u32 {
  unsafe { anim.as_ref() }.map_or(0, |a| a.renderer.composition().width)
}

/// Returns the source composition height, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_height(anim: *const TLottieInstance) -> u32 {
  unsafe { anim.as_ref() }.map_or(0, |a| a.renderer.composition().height)
}

/// Returns the source frame rate, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_frame_rate(anim: *const TLottieInstance) -> f32 {
  unsafe { anim.as_ref() }.map_or(0.0, |a| a.renderer.composition().frame_rate)
}

/// Returns the source frame count, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_frame_count(anim: *const TLottieInstance) -> u32 {
  unsafe { anim.as_ref() }.map_or(0, |a| a.renderer.composition().frame_count())
}

/// Renders one frame into caller-owned premultiplied RGBA8 pixels. Each word
/// is `0xAABBGGRR`, giving `[R, G, B, A]` bytes on little-endian targets.
///
/// Returns 0 on success and a negative value on error:
/// - -1: null handle or buffer
/// - -2: `out_len` is too small or dimensions overflow
/// - -3: render failed
///
/// # Safety
/// `anim` must be a live handle returned by [`tlottie_new`].
/// `out` must point to at least `out_len` writable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render(anim: *mut TLottieInstance, frame: f32, width: u32, height: u32, out: *mut u32, out_len: usize, antialias: u32) -> i32 {
  // SAFETY: this function has the same pointer contract as the extended API.
  unsafe { tlottie_render_with_options(anim, frame, width, height, out, out_len, antialias, RenderOptions::default().curve_tolerance, 1) }
}

/// Renders one frame with an explicit device-space curve tolerance.
///
/// `curve_tolerance` is the maximum curve-flattening error in pixels. Smaller
/// positive values improve geometric accuracy at a performance cost. When
/// `clear` is zero, the animation is composited over the existing
/// premultiplied pixels; nonzero preserves the default clear-first behavior.
/// Returns -1 when the tolerance is non-finite or not positive; other status
/// codes and safety requirements are identical to [`tlottie_render`].
///
/// # Safety
/// `anim` must be a live handle returned by [`tlottie_new`].
/// `out` must point to at least `out_len` writable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_with_options(
  anim: *mut TLottieInstance,
  frame: f32,
  width: u32,
  height: u32,
  out: *mut u32,
  out_len: usize,
  antialias: u32,
  curve_tolerance: f32,
  clear: u32,
) -> i32 {
  let Some(anim) = (unsafe { anim.as_mut() }) else {
    return -1;
  };
  if out.is_null() || !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return -1;
  }
  let Some(px) = (width as usize).checked_mul(height as usize) else {
    return -2;
  };
  if out_len < px {
    return -2;
  }
  let pixels = unsafe { core::slice::from_raw_parts_mut(out, px) };
  match anim.renderer.render(
    frame,
    pixels,
    width,
    height,
    RenderOptions {
      clear: clear != 0,
      antialias: antialias != 0,
      curve_tolerance,
      ..RenderOptions::default()
    },
  ) {
    Ok(()) => 0,
    Err(_) => -3,
  }
}

/// Renders one frame directly into `width * height` Alpha8 pixels.
///
/// # Safety
/// `anim` must be a live handle returned by [`tlottie_new`]. `out` must
/// point to at least `out_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8(anim: *mut TLottieInstance, frame: f32, width: u32, height: u32, out: *mut u8, out_len: usize, antialias: u32) -> i32 {
  unsafe { tlottie_render_alpha8_with_options(anim, frame, width, height, out, out_len, antialias, RenderOptions::default().curve_tolerance, 1) }
}

/// Renders directly into Alpha8 with an explicit curve tolerance. A zero
/// `clear` composites alpha over the existing values.
///
/// # Safety
/// The pointer requirements are identical to [`tlottie_render_alpha8`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_alpha8_with_options(
  anim: *mut TLottieInstance,
  frame: f32,
  width: u32,
  height: u32,
  out: *mut u8,
  out_len: usize,
  antialias: u32,
  curve_tolerance: f32,
  clear: u32,
) -> i32 {
  let Some(anim) = (unsafe { anim.as_mut() }) else {
    return -1;
  };
  if out.is_null() || !curve_tolerance.is_finite() || curve_tolerance <= 0.0 {
    return -1;
  }
  let Some(px) = (width as usize).checked_mul(height as usize) else {
    return -2;
  };
  if out_len < px {
    return -2;
  }
  let alpha = unsafe { core::slice::from_raw_parts_mut(out, px) };
  match anim.renderer.render_alpha8(
    frame,
    alpha,
    width,
    height,
    RenderOptions {
      clear: clear != 0,
      antialias: antialias != 0,
      curve_tolerance,
      alpha_only: true,
    },
  ) {
    Ok(()) => 0,
    Err(_) => -3,
  }
}
