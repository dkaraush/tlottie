//! Pixel-format conversion shared by browser and development-tool output.

#![allow(unsafe_code)]

/// Converts one premultiplied ARGB32 word to straight-alpha RGBA8.
#[inline]
pub(crate) fn argb_to_rgba(pixel: u32) -> [u8; 4] {
  let alpha = pixel >> 24;
  let (red, green, blue) = if alpha == 0 {
    (0, 0, 0)
  } else if alpha == 255 {
    ((pixel >> 16) & 0xff, (pixel >> 8) & 0xff, pixel & 0xff)
  } else {
    let straight = |channel: u32| (((channel * 255) + alpha / 2) / alpha).min(255);
    (straight((pixel >> 16) & 0xff), straight((pixel >> 8) & 0xff), straight(pixel & 0xff))
  };
  [red as u8, green as u8, blue as u8, alpha as u8]
}

/// Converts premultiplied ARGB32 pixels to straight-alpha RGBA8.
///
/// WebAssembly uses a SIMD fast path for four-pixel groups containing only
/// transparent and opaque pixels. Groups containing partial alpha use the
/// scalar oracle because simd128 has no integer division instruction for
/// the un-premultiplication step.
pub(crate) fn argb_to_rgba_slice(src: &[u32], dst: &mut [u8]) {
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  argb_to_rgba_slice_wasm(src, dst);
  #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
  argb_to_rgba_slice_scalar(src, dst);
}

pub(crate) fn argb_to_rgba_slice_scalar(src: &[u32], dst: &mut [u8]) {
  for (&pixel, rgba) in src.iter().zip(dst.chunks_exact_mut(4)) {
    rgba.copy_from_slice(&argb_to_rgba(pixel));
  }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn argb_to_rgba_slice_wasm(src: &[u32], dst: &mut [u8]) {
  use core::arch::wasm32::{u8x16, u8x16_swizzle, v128, v128_and, v128_load, v128_store};

  let bgra_to_rgba = u8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
  let replicate_alpha = u8x16(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
  let pixels = src.len().min(dst.len() / 4);
  let full = pixels - pixels % 4;

  if let (Some(src4), Some(dst16)) = (src.get(..full), dst.get_mut(..full * 4)) {
    for (input, output) in src4.chunks_exact(4).zip(dst16.chunks_exact_mut(16)) {
      let binary_alpha = if let [p0, p1, p2, p3] = input {
        [p0, p1, p2, p3].iter().all(|pixel| matches!(**pixel >> 24, 0 | 255))
      } else {
        false
      };
      if binary_alpha {
        // SAFETY: the exact chunks above provide 16 readable source bytes
        // and 16 writable destination bytes; wasm v128 permits unaligned
        // accesses.
        let bgra = unsafe { v128_load(input.as_ptr().cast::<v128>()) };
        let rgba = u8x16_swizzle(bgra, bgra_to_rgba);
        // Transparent pixels must become canonical zero even if a hostile
        // input word contains non-zero RGB channels with alpha zero.
        let alpha = u8x16_swizzle(rgba, replicate_alpha);
        unsafe { v128_store(output.as_mut_ptr().cast::<v128>(), v128_and(rgba, alpha)) };
      } else {
        argb_to_rgba_slice_scalar(input, output);
      }
    }
  }

  if let (Some(src_tail), Some(dst_tail)) = (src.get(full..pixels), dst.get_mut(full * 4..pixels * 4)) {
    argb_to_rgba_slice_scalar(src_tail, dst_tail);
  }
}

#[cfg(test)]
mod tests {
  use super::{argb_to_rgba, argb_to_rgba_slice, argb_to_rgba_slice_scalar};

  #[test]
  fn converts_transparent_opaque_and_partial_pixels() {
    assert_eq!(argb_to_rgba(0), [0, 0, 0, 0]);
    assert_eq!(argb_to_rgba(0xff12_3456), [0x12, 0x34, 0x56, 0xff]);
    assert_eq!(argb_to_rgba(0x8040_2000), [128, 64, 0, 128]);
  }

  #[test]
  fn slice_conversion_matches_scalar_pixels() {
    let src = [0, 0x0012_3456, 0xff12_3456, 0x8040_2000, 0x7f01_0203, 0xffffffff, 0x0101_0000];
    let mut expected = [0u8; 28];
    let mut actual = [0u8; 28];
    argb_to_rgba_slice_scalar(&src, &mut expected);
    argb_to_rgba_slice(&src, &mut actual);
    assert_eq!(actual, expected);
  }
}
