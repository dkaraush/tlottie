//! C-ABI shim over `tlottie` for wasm32-unknown-unknown, consumed by the
//! hand-written JS loader in `tools/wasm-demo/`. No wasm-bindgen: the ABI is
//! pointers + integers only, so the loader stays a page of plain JS.
//!
//! Ownership protocol: JS allocates input buffers with `tl_alloc`/`tl_free`,
//! instances are opaque pointers created by `tl_new` and released by
//! `tl_drop`. The RGBA output buffer lives inside the instance; the pointer
//! returned by `tl_render` is valid until the next `tl_render`/`tl_drop`
//! call on that instance (or any wasm memory growth, which JS handles by
//! re-deriving views after every call).

use std::alloc::{alloc, dealloc, Layout};

use tlottie::{Animation, Composition, Limits, RenderOptions};

/// A playing instance plus its conversion buffers.
pub struct Instance {
    anim: Animation,
    /// Premultiplied ARGB32 render target (what the renderer writes).
    argb: Vec<u32>,
    /// Un-premultiplied RGBA8 copy handed to the canvas `ImageData`.
    rgba: Vec<u8>,
}

/// Allocates `len` bytes for JS to copy input into. Returns null on failure
/// (zero `len`, or OOM).
#[no_mangle]
pub extern "C" fn tl_alloc(len: usize) -> *mut u8 {
    match Layout::from_size_align(len, 1) {
        Ok(layout) if len > 0 => {
            // SAFETY: layout is valid and non-zero-sized.
            unsafe { alloc(layout) }
        }
        _ => std::ptr::null_mut(),
    }
}

/// Frees a buffer from `tl_alloc`. `len` must match the allocation.
///
/// # Safety
/// `ptr` must come from `tl_alloc(len)` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn tl_free(ptr: *mut u8, len: usize) {
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
/// `json_ptr..json_ptr+json_len` must be readable (a `tl_alloc` buffer JS
/// filled).
#[no_mangle]
pub unsafe extern "C" fn tl_new(json_ptr: *const u8, json_len: usize) -> *mut Instance {
    if json_ptr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller contract — the range is a live allocation.
    let json = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    match Composition::parse(json, &Limits::default()) {
        Ok(comp) => Box::into_raw(Box::new(Instance {
            anim: Animation::new(comp),
            argb: Vec::new(),
            rgba: Vec::new(),
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Releases an instance.
///
/// # Safety
/// `inst` must come from `tl_new` and not have been dropped.
#[no_mangle]
pub unsafe extern "C" fn tl_drop(inst: *mut Instance) {
    if !inst.is_null() {
        // SAFETY: caller contract — inst is a live Box from tl_new.
        drop(unsafe { Box::from_raw(inst) });
    }
}

/// # Safety
/// `inst` must be a live instance from `tl_new`.
#[no_mangle]
pub unsafe extern "C" fn tl_width(inst: *const Instance) -> u32 {
    // SAFETY: caller contract.
    unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().width)
}

/// # Safety
/// `inst` must be a live instance from `tl_new`.
#[no_mangle]
pub unsafe extern "C" fn tl_height(inst: *const Instance) -> u32 {
    // SAFETY: caller contract.
    unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().height)
}

/// # Safety
/// `inst` must be a live instance from `tl_new`.
#[no_mangle]
pub unsafe extern "C" fn tl_frame_rate(inst: *const Instance) -> f32 {
    // SAFETY: caller contract.
    unsafe { inst.as_ref() }.map_or(0.0, |i| i.anim.composition().frame_rate)
}

/// # Safety
/// `inst` must be a live instance from `tl_new`.
#[no_mangle]
pub unsafe extern "C" fn tl_frame_count(inst: *const Instance) -> u32 {
    // SAFETY: caller contract.
    unsafe { inst.as_ref() }.map_or(0, |i| i.anim.composition().frame_count())
}

/// Renders `frame` at `width`x`height` and returns a pointer to a
/// `width * height * 4` un-premultiplied RGBA8 buffer (row-major), or null
/// on error. The buffer is owned by the instance and overwritten by the
/// next call. `antialias != 0` enables analytic edge antialiasing.
///
/// # Safety
/// `inst` must be a live instance from `tl_new`.
#[no_mangle]
pub unsafe extern "C" fn tl_render(
    inst: *mut Instance,
    frame: f32,
    width: u32,
    height: u32,
    antialias: u32,
) -> *const u8 {
    // SAFETY: caller contract.
    let Some(inst) = (unsafe { inst.as_mut() }) else {
        return std::ptr::null_mut();
    };
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
        .render_with_options(
            frame,
            &mut inst.argb,
            width,
            height,
            RenderOptions {
                antialias: antialias != 0,
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
    for (src, dst) in inst.argb.iter().zip(inst.rgba.chunks_exact_mut(4)) {
        let a = src >> 24;
        // Fast paths for the two dominant cases (profiled at ~39% of frame
        // time as 3 divisions/px): transparent -> zeros, opaque -> identity
        // ((c*255 + 127)/255 == c exactly, so this is bit-equal).
        let (r, g, b) = if a == 0 {
            (0, 0, 0)
        } else if a == 255 {
            ((src >> 16) & 0xff, (src >> 8) & 0xff, src & 0xff)
        } else {
            // Un-premultiply with rounding; canvas ImageData expects
            // straight alpha. min() guards hostile >alpha channels.
            let un = |c: u32| (((c * 255) + a / 2) / a).min(255);
            (
                un((src >> 16) & 0xff),
                un((src >> 8) & 0xff),
                un(src & 0xff),
            )
        };
        if let [dr, dg, db, da] = dst {
            *dr = r as u8;
            *dg = g as u8;
            *db = b as u8;
            *da = a as u8;
        }
    }
    inst.rgba.as_ptr()
}
