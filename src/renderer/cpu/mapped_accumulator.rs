#![allow(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

const MAP_THRESHOLD: usize = 1 << 20;

pub(super) struct Accumulator {
  storage: Storage,
}

enum Storage {
  Heap(Vec<f32>),
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
  Mapped {
    ptr: core::ptr::NonNull<f32>,
    len: usize,
  },
}

impl Accumulator {
  pub(super) fn new_zeroed(len: usize) -> Self {
    let mut value = Self { storage: Storage::Heap(Vec::new()) };
    value.resize_zeroed(len);
    value
  }

  pub(super) fn len(&self) -> usize {
    match &self.storage {
      Storage::Heap(values) => values.len(),
      #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
      Storage::Mapped { len, .. } => *len,
    }
  }

  pub(super) fn resize_zeroed(&mut self, len: usize) {
    if self.len() >= len {
      return;
    }
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if len.saturating_mul(core::mem::size_of::<f32>()) >= MAP_THRESHOLD {
      if let Some(storage) = map_zeroed(len) {
        self.storage = storage;
        return;
      }
    }
    self.storage = Storage::Heap(vec![0.0; len]);
  }

  pub(super) fn get_mut(&mut self, range: Range<usize>) -> Option<&mut [f32]> {
    match &mut self.storage {
      Storage::Heap(values) => values.get_mut(range),
      #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
      Storage::Mapped { ptr, len } => {
        // SAFETY: mapped storage owns a writable region of `len` f32s until
        // Drop; the returned slice is tied to this exclusive borrow.
        unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), *len) }.get_mut(range)
      }
    }
  }
}

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
fn map_zeroed(len: usize) -> Option<Storage> {
  use core::ffi::c_void;

  unsafe extern "C" {
    fn mmap(address: *mut c_void, length: usize, protection: i32, flags: i32, fd: i32, offset: isize) -> *mut c_void;
  }

  const PROT_READ: i32 = 1;
  const PROT_WRITE: i32 = 2;
  const MAP_PRIVATE: i32 = 2;
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
  const MAP_ANONYMOUS: i32 = 0x1000;
  #[cfg(any(target_os = "linux", target_os = "android"))]
  const MAP_ANONYMOUS: i32 = 0x20;

  let bytes = len.checked_mul(core::mem::size_of::<f32>())?;
  // SAFETY: anonymous private mapping, no file descriptor, writable bytes.
  let raw = unsafe { mmap(core::ptr::null_mut(), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
  if raw as isize == -1 {
    return None;
  }
  core::ptr::NonNull::new(raw.cast::<f32>()).map(|ptr| Storage::Mapped { ptr, len })
}

impl Drop for Storage {
  fn drop(&mut self) {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if let Storage::Mapped { ptr, len } = self {
      use core::ffi::c_void;
      unsafe extern "C" {
        fn munmap(address: *mut c_void, length: usize) -> i32;
      }
      let bytes = len.saturating_mul(core::mem::size_of::<f32>());
      // SAFETY: this exact region was returned by mmap and remains owned.
      let _ = unsafe { munmap(ptr.as_ptr().cast::<c_void>(), bytes) };
    }
  }
}

// The mapping is exclusively owned and has no thread-affine state.
unsafe impl Send for Accumulator {}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn large_mapping_is_zeroed_and_writable() {
    let mut accumulator = Accumulator::new_zeroed(300_000);
    assert_eq!(accumulator.len(), 300_000);
    let values = accumulator.get_mut(0..300_000).expect("full mapped range");
    assert_eq!(values[0], 0.0);
    assert_eq!(values[299_999], 0.0);
    values[0] = 1.25;
    values[299_999] = -2.5;
    assert_eq!(values[0], 1.25);
    assert_eq!(values[299_999], -2.5);
  }
}
