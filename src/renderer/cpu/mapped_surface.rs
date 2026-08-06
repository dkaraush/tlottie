#![allow(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

const MAP_THRESHOLD: usize = 1 << 20;

pub(crate) struct Surface {
  storage: Storage,
}

enum Storage {
  Heap(Vec<u32>),
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
  Mapped {
    ptr: core::ptr::NonNull<u32>,
    len: usize,
    capacity: usize,
  },
}

impl Default for Surface {
  fn default() -> Self {
    Self { storage: Storage::Heap(Vec::new()) }
  }
}

impl Surface {
  pub(super) fn resize_zeroed(&mut self, len: usize) {
    match &mut self.storage {
      Storage::Heap(values) if values.capacity() >= len => {
        values.resize(len, 0);
        return;
      }
      #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
      Storage::Mapped { ptr, len: old_len, capacity } if *capacity >= len => {
        if len > *old_len {
          // SAFETY: old_len..len lies inside this exclusively owned mapping.
          unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr().add(*old_len), len - *old_len) }.fill(0);
        }
        *old_len = len;
        return;
      }
      _ => {}
    }

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if len.saturating_mul(core::mem::size_of::<u32>()) >= MAP_THRESHOLD {
      if let Some(storage) = map_zeroed(len) {
        self.storage = storage;
        return;
      }
    }
    self.storage = Storage::Heap(vec![0; len]);
  }

  pub(super) fn as_mut_slice(&mut self) -> &mut [u32] {
    &mut *self
  }
}

impl Deref for Surface {
  type Target = [u32];

  fn deref(&self) -> &Self::Target {
    match &self.storage {
      Storage::Heap(values) => values,
      #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
      Storage::Mapped { ptr, len, .. } => {
        // SAFETY: the mapping owns `capacity >= len` initialized u32 values.
        unsafe { core::slice::from_raw_parts(ptr.as_ptr(), *len) }
      }
    }
  }
}

impl DerefMut for Surface {
  fn deref_mut(&mut self) -> &mut Self::Target {
    match &mut self.storage {
      Storage::Heap(values) => values,
      #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
      Storage::Mapped { ptr, len, .. } => {
        // SAFETY: the mapping is exclusively borrowed and owns `len` values.
        unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), *len) }
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

  let bytes = len.checked_mul(core::mem::size_of::<u32>())?;
  // SAFETY: anonymous private mapping, no file descriptor, writable bytes.
  let raw = unsafe { mmap(core::ptr::null_mut(), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
  if raw as isize == -1 {
    return None;
  }
  core::ptr::NonNull::new(raw.cast::<u32>()).map(|ptr| Storage::Mapped { ptr, len, capacity: len })
}

impl Drop for Storage {
  fn drop(&mut self) {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if let Storage::Mapped { ptr, capacity, .. } = self {
      use core::ffi::c_void;
      unsafe extern "C" {
        fn munmap(address: *mut c_void, length: usize) -> i32;
      }
      let bytes = capacity.saturating_mul(core::mem::size_of::<u32>());
      // SAFETY: this exact region was returned by mmap and remains owned.
      let _ = unsafe { munmap(ptr.as_ptr().cast::<c_void>(), bytes) };
    }
  }
}

// The mapping is exclusively owned and has no thread-affine state.
unsafe impl Send for Surface {}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn large_surface_is_zeroed_writable_and_resizable() {
    let mut surface = Surface::default();
    surface.resize_zeroed(300_000);
    assert_eq!(surface.len(), 300_000);
    assert_eq!(surface[0], 0);
    assert_eq!(surface[299_999], 0);
    surface[0] = 7;
    surface[299_999] = 9;
    surface.resize_zeroed(200_000);
    assert_eq!(surface[0], 7);
    surface.resize_zeroed(300_000);
    assert_eq!(surface[0], 7);
    assert_eq!(surface[299_999], 0);
  }
}
