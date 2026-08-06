//! The lang items a `no_std` staticlib has to carry.
//!
//! A library would normally leave both to the final binary, but a C or C++
//! host cannot define Rust lang items, so the `c-api` staticlib provides them
//! itself. `std` builds are unaffected — this module does not exist there.
//!
//! Both are deliberately minimal: the allocator forwards to the C runtime the
//! host already links, and panics abort. The crate is written not to panic
//! (malformed input returns [`crate::Error`], and `panic`/`unwrap`/`expect`
//! /`indexing_slicing` are denied by lint), so the handler is a backstop, not
//! a control-flow path.

#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};

extern "C" {
  fn malloc(size: usize) -> *mut u8;
  fn free(ptr: *mut u8);
  fn abort() -> !;
}

/// Forwards to the host C runtime's allocator.
///
/// `malloc` only promises fundamental alignment (16 bytes on the targets this
/// crate builds for), so anything stricter is over-allocated and hand-aligned,
/// with the original pointer stashed in the word below the aligned address.
/// The renderer never asks for more than 16 today; the slow path exists so a
/// future over-aligned type cannot silently corrupt the heap.
struct HostAlloc;

/// Fundamental alignment guaranteed by the C runtime.
const MALLOC_ALIGN: usize = 16;

unsafe impl GlobalAlloc for HostAlloc {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    if layout.align() <= MALLOC_ALIGN {
      return unsafe { malloc(layout.size()) };
    }
    // Room for the payload, the worst-case adjustment, and the saved pointer.
    let Some(padded) = layout.size().checked_add(layout.align()).and_then(|n| n.checked_add(core::mem::size_of::<usize>())) else {
      return core::ptr::null_mut();
    };
    let raw = unsafe { malloc(padded) };
    if raw.is_null() {
      return raw;
    }
    let base = raw as usize;
    let reserved = base.wrapping_add(core::mem::size_of::<usize>());
    let aligned = reserved.wrapping_add(layout.align() - 1) & !(layout.align() - 1);
    // SAFETY: `aligned` is at least one usize above `raw` and inside the
    // padded allocation, so the slot below it is ours to write.
    unsafe { (aligned as *mut usize).sub(1).write(base) };
    aligned as *mut u8
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    if layout.align() <= MALLOC_ALIGN {
      unsafe { free(ptr) };
      return;
    }
    // SAFETY: alloc wrote the original pointer immediately below the address
    // it returned, and layout.align() matches the allocating call.
    let base = unsafe { (ptr as *mut usize).sub(1).read() };
    unsafe { free(base as *mut u8) };
  }
}

#[global_allocator]
static ALLOCATOR: HostAlloc = HostAlloc;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
  // No message: formatting one would pull in core::fmt's machinery, which is
  // most of what a no_std build is trying to leave behind.
  unsafe { abort() }
}
