//! Runtime support for standalone `no_std` WebAssembly artifacts.
//!
//! `wasm32-unknown-unknown` has no host C runtime, so allocations grow linear
//! memory through dlmalloc (the allocator used by Rust's standard library on
//! this target). Unexpected panics trap without formatting or unwinding.

#[global_allocator]
static ALLOCATOR: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
  core::arch::wasm32::unreachable()
}
