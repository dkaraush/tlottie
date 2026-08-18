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
/// `malloc` promises enough alignment for fundamental C types, but that value
/// differs between 32- and 64-bit targets. We conservatively rely only on
/// pointer alignment. Anything stricter is over-allocated and hand-aligned,
/// with the original pointer stashed in the word below the aligned address.
/// This keeps future over-aligned Rust types safe on 32-bit Android as well as
/// the current 64-bit targets.
struct HostAlloc;

/// Alignment every supported C allocator must provide.
const MALLOC_ALIGN: usize = core::mem::align_of::<usize>();

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
    // Keep the allocation's pointer provenance while finding the aligned
    // address; integer round-tripping is unnecessary here.
    let reserved = unsafe { raw.add(core::mem::size_of::<*mut u8>()) };
    let adjustment = reserved.align_offset(layout.align());
    if adjustment == usize::MAX {
      unsafe { free(raw) };
      return core::ptr::null_mut();
    }
    let aligned = unsafe { reserved.add(adjustment) };
    // SAFETY: `aligned` is at least one usize above `raw` and inside the
    // padded allocation, so the slot below it is ours to write.
    unsafe { (aligned as *mut *mut u8).sub(1).write(raw) };
    aligned
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    if layout.align() <= MALLOC_ALIGN {
      unsafe { free(ptr) };
      return;
    }
    // SAFETY: alloc wrote the original pointer immediately below the address
    // it returned, and layout.align() matches the allocating call.
    let base = unsafe { (ptr as *mut *mut u8).sub(1).read() };
    unsafe { free(base) };
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

/// Abort builds never unwind, but Android's precompiled `core` and `alloc`
/// still carry weak DWARF references to this symbol. It is therefore only a
/// link-time anchor and cannot be reached through Rust panic control flow.
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
