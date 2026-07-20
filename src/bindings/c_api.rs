//! Native C ABI for rendering into caller-owned ARGB32 buffers.

#![allow(unsafe_code)]

use crate::{CPURenderer, Composition, Limits, RenderOptions};

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
  if json_ptr.is_null() {
    return core::ptr::null_mut();
  }
  let json = unsafe { core::slice::from_raw_parts(json_ptr, json_len) };
  match Composition::parse(json, &Limits::default()) {
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

/// Renders one frame into caller-owned premultiplied ARGB32 pixels.
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
  unsafe { tlottie_render_with_options(anim, frame, width, height, out, out_len, antialias, RenderOptions::default().curve_tolerance) }
}

/// Renders one frame with an explicit device-space curve tolerance.
///
/// `curve_tolerance` is the maximum curve-flattening error in pixels. Smaller
/// positive values improve geometric accuracy at a performance cost. Returns
/// -1 when the tolerance is non-finite or not positive; other status codes and
/// safety requirements are identical to [`tlottie_render`].
///
/// # Safety
/// `anim` must be a live handle returned by [`tlottie_new`].
/// `out` must point to at least `out_len` writable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tlottie_render_with_options(anim: *mut TLottieInstance, frame: f32, width: u32, height: u32, out: *mut u32, out_len: usize, antialias: u32, curve_tolerance: f32) -> i32 {
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
      antialias: antialias != 0,
      curve_tolerance,
      ..RenderOptions::default()
    },
  ) {
    Ok(()) => 0,
    Err(_) => -3,
  }
}
