//! Runtime x86 and LoongArch feature detection for builds without `std`.
//!
//! `is_x86_feature_detected!` lives in `std`, so a `no_std` build would
//! otherwise be stuck on the SSE2 kernels no matter what the CPU can do.
//! This reproduces the parts of it the vector kernels need, straight from
//! `CPUID` and `XGETBV`.
//!
//! Checking the CPUID feature bit alone is not enough: the wider register
//! files only exist if the OS has enabled saving them across context
//! switches, and using them when it has not is a fault, not slow code. So
//! every query below confirms `OSXSAVE` and the matching `XCR0` bits first.
//!
//! On LoongArch, Linux sets the LSX/LASX `AT_HWCAP` bits only when both
//! hardware and kernel support them.
//!
//! A `std` test asserts this agrees with the `std` detector on
//! whatever machine runs the suite, which is what keeps the two paths from
//! drifting apart.

#![allow(unsafe_code)]
// Only the `no_std` dispatch calls into here, but the module stays compiled
// under `std` so the agreement test below can run against the very detector
// it has to match. Without that test the two paths could drift silently.
#![cfg_attr(feature = "std", allow(dead_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
use core::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const ABSENT: u8 = 1;
const PRESENT: u8 = 2;

/// `XCR0` bits 1..=2: XMM and YMM state. Required for AVX and AVX2.
#[cfg(target_arch = "x86_64")]
const XCR0_YMM: u64 = 0b0000_0110;
/// `XCR0` bits 1,2,5,6,7: the above plus opmask, ZMM_Hi256 and Hi16_ZMM.
#[cfg(target_arch = "x86_64")]
const XCR0_ZMM: u64 = 0b1110_0110;

/// Reads `XCR0`.
///
/// # Safety
/// Only callable once `OSXSAVE` is known to be set; `XGETBV` faults
/// otherwise.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "xsave")]
unsafe fn xcr0() -> u64 {
  unsafe { _xgetbv(0) }
}

/// True when the CPU reports `OSXSAVE` and the OS preserves every register
/// bank in `mask`.
#[cfg(target_arch = "x86_64")]
fn os_preserves(mask: u64) -> bool {
  // CPUID leaf 1 exists on every x86_64 part.
  let leaf1 = __cpuid(1);
  if leaf1.ecx & (1 << 27) == 0 {
    return false;
  }
  // SAFETY: OSXSAVE is set, which is exactly XGETBV's precondition.
  let xcr0 = unsafe { xcr0() };
  xcr0 & mask == mask
}

/// Extended-feature flags, or `None` when the CPU predates CPUID leaf 7.
#[cfg(target_arch = "x86_64")]
fn leaf7_ebx() -> Option<u32> {
  // Leaf 0 is always present and reports the highest valid leaf.
  let max_leaf = __cpuid(0).eax;
  if max_leaf < 7 {
    return None;
  }
  // Guarded by the max-leaf check above.
  Some(__cpuid_count(7, 0).ebx)
}

#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> bool {
  os_preserves(XCR0_YMM) && leaf7_ebx().is_some_and(|ebx| ebx & (1 << 5) != 0)
}

#[cfg(target_arch = "x86_64")]
fn detect_avx512() -> bool {
  if !os_preserves(XCR0_ZMM) {
    return false;
  }
  // Every feature the AVX-512 kernels name in their `target_feature` list:
  // avx2 (5), f (16), dq (17), bw (30), vl (31). Detecting a subset would let
  // a dispatched kernel execute an instruction the CPU does not implement.
  leaf7_ebx().is_some_and(|ebx| {
    let has = |bit: u32| ebx & (1 << bit) != 0;
    has(5) && has(16) && has(17) && has(30) && has(31)
  })
}

/// `HWCAP_LOONGARCH_L(A)SX` defined in uapi `asm/hwcap.h`
#[cfg(target_arch = "loongarch64")]
const HWCAP_LSX: core::ffi::c_ulong = 1 << 4;
#[cfg(target_arch = "loongarch64")]
const HWCAP_LASX: core::ffi::c_ulong = 1 << 5;

/// True when the kernel reports every `AT_HWCAP` bit in `mask`.
#[cfg(target_arch = "loongarch64")]
fn hwcap_has(mask: core::ffi::c_ulong) -> bool {
  extern "C" {
    fn getauxval(kind: core::ffi::c_ulong) -> core::ffi::c_ulong;
  }
  const AT_HWCAP: core::ffi::c_ulong = 16;
  // SAFETY: `getauxval` has no preconditions; unknown types return 0.
  unsafe { getauxval(AT_HWCAP) & mask == mask }
}

#[cfg(target_arch = "loongarch64")]
fn detect_lsx() -> bool {
  hwcap_has(HWCAP_LSX)
}

#[cfg(target_arch = "loongarch64")]
fn detect_lasx() -> bool {
  hwcap_has(HWCAP_LASX)
}

/// Caches one probe. A race just runs the probe twice and stores the same
/// answer, so `Relaxed` is enough — there is nothing to publish but the
/// verdict itself.
fn cached(cell: &AtomicU8, probe: fn() -> bool) -> bool {
  match cell.load(Ordering::Relaxed) {
    PRESENT => true,
    ABSENT => false,
    _ => {
      let found = probe();
      cell.store(if found { PRESENT } else { ABSENT }, Ordering::Relaxed);
      found
    }
  }
}

#[cfg(target_arch = "x86_64")]
pub(super) fn avx2() -> bool {
  static CACHE: AtomicU8 = AtomicU8::new(UNKNOWN);
  cached(&CACHE, detect_avx2)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn avx512() -> bool {
  static CACHE: AtomicU8 = AtomicU8::new(UNKNOWN);
  cached(&CACHE, detect_avx512)
}

#[cfg(target_arch = "loongarch64")]
#[cfg_attr(not(feature = "lsx"), allow(dead_code))]
pub(super) fn lsx() -> bool {
  static CACHE: AtomicU8 = AtomicU8::new(UNKNOWN);
  cached(&CACHE, detect_lsx)
}

#[cfg(target_arch = "loongarch64")]
#[cfg_attr(not(feature = "lasx"), allow(dead_code))]
pub(super) fn lasx() -> bool {
  static CACHE: AtomicU8 = AtomicU8::new(UNKNOWN);
  cached(&CACHE, detect_lasx)
}

/// The `no_std` path must reach the same verdict as `std`'s detector; if it
/// ever says yes where `std` says no, the kernels fault.
#[cfg(all(test, feature = "std", target_arch = "x86_64"))]
mod tests {
  #[test]
  fn matches_the_std_detector() {
    assert_eq!(super::avx2(), std::is_x86_feature_detected!("avx2"), "avx2");
    assert_eq!(
      super::avx512(),
      std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512vl"),
      "avx512"
    );
  }

  #[test]
  fn repeated_queries_are_stable() {
    let (a, b) = (super::avx2(), super::avx512());
    for _ in 0..8 {
      assert_eq!(super::avx2(), a);
      assert_eq!(super::avx512(), b);
    }
  }
}

/// See the x86 tests above: same contract for LSX/LASX.
#[cfg(all(test, feature = "std", target_arch = "loongarch64"))]
mod tests {
  #[test]
  fn matches_the_std_detector() {
    let lsx = std::arch::is_loongarch_feature_detected!("lsx");
    assert_eq!(super::lsx(), lsx, "lsx");
    assert_eq!(super::lasx(), lsx && std::arch::is_loongarch_feature_detected!("lasx"), "lasx");
  }
}
