//! Native C ABI over `tlottie`.
//!
//! This crate is intentionally separate from `tlottie-wasm`: desktop hosts can
//! link/load this shared library directly and render into caller-owned ARGB32
//! buffers without browser-specific RGBA conversion.

use tlottie::{Animation, Composition, Limits, RenderOptions};

/// Opaque animation handle owned by C callers.
pub struct TlottieAnimation {
    anim: Animation,
}

/// Parses Lottie JSON bytes and returns an animation handle, or null on error.
///
/// # Safety
/// `json_ptr..json_ptr+json_len` must be readable for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_new(
    json_ptr: *const u8,
    json_len: usize,
) -> *mut TlottieAnimation {
    if json_ptr.is_null() {
        return core::ptr::null_mut();
    }
    let json = unsafe { core::slice::from_raw_parts(json_ptr, json_len) };
    match Composition::parse(json, &Limits::default()) {
        Ok(comp) => Box::into_raw(Box::new(TlottieAnimation {
            anim: Animation::new(comp),
        })),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Releases an animation handle.
///
/// # Safety
/// `anim` must be null or a handle returned by [`tlottie_animation_new`] that
/// has not already been dropped.
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_drop(anim: *mut TlottieAnimation) {
    if !anim.is_null() {
        drop(unsafe { Box::from_raw(anim) });
    }
}

/// Returns the source composition width, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_animation_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_width(anim: *const TlottieAnimation) -> u32 {
    unsafe { anim.as_ref() }.map_or(0, |a| a.anim.composition().width)
}

/// Returns the source composition height, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_animation_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_height(anim: *const TlottieAnimation) -> u32 {
    unsafe { anim.as_ref() }.map_or(0, |a| a.anim.composition().height)
}

/// Returns the source frame rate, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_animation_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_frame_rate(anim: *const TlottieAnimation) -> f32 {
    unsafe { anim.as_ref() }.map_or(0.0, |a| a.anim.composition().frame_rate)
}

/// Returns the source frame count, or zero for null handles.
///
/// # Safety
/// `anim` must be null or a live handle returned by [`tlottie_animation_new`].
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_frame_count(anim: *const TlottieAnimation) -> u32 {
    unsafe { anim.as_ref() }.map_or(0, |a| a.anim.composition().frame_count())
}

/// Renders one frame into caller-owned premultiplied ARGB32 pixels.
///
/// Returns 0 on success and a negative value on error:
/// - -1: null handle or buffer
/// - -2: `out_len` is too small or dimensions overflow
/// - -3: render failed
///
/// # Safety
/// `anim` must be a live handle returned by [`tlottie_animation_new`].
/// `out` must point to at least `out_len` writable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tlottie_animation_render_argb(
    anim: *mut TlottieAnimation,
    frame: f32,
    width: u32,
    height: u32,
    out: *mut u32,
    out_len: usize,
    antialias: u32,
) -> i32 {
    let Some(anim) = (unsafe { anim.as_mut() }) else {
        return -1;
    };
    if out.is_null() {
        return -1;
    }
    let Some(px) = (width as usize).checked_mul(height as usize) else {
        return -2;
    };
    if out_len < px {
        return -2;
    }
    let pixels = unsafe { core::slice::from_raw_parts_mut(out, px) };
    match anim.anim.render_with_options(
        frame,
        pixels,
        width,
        height,
        RenderOptions {
            antialias: antialias != 0,
            ..RenderOptions::default()
        },
    ) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}
