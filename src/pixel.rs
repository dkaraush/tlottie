//! Pixel-format helpers shared by renderers, browser output, and tools.

#![allow(unsafe_code)]

/// Packs premultiplied channels as `0xAABBGGRR`, which is RGBA byte order
/// in little-endian memory.
#[inline]
pub(crate) const fn pack_premultiplied_rgba(red: u32, green: u32, blue: u32, alpha: u32) -> u32 {
  (alpha << 24) | (blue << 16) | (green << 8) | red
}

/// Converts one premultiplied RGBA8 word to straight-alpha RGBA8.
#[inline]
pub(crate) fn premultiplied_rgba_to_straight(pixel: u32) -> [u8; 4] {
  let alpha = pixel >> 24;
  let (red, green, blue) = if alpha == 0 {
    (0, 0, 0)
  } else if alpha == 255 {
    (pixel & 0xff, (pixel >> 8) & 0xff, (pixel >> 16) & 0xff)
  } else {
    let straight = |channel: u32| (((channel * 255) + alpha / 2) / alpha).min(255);
    (straight(pixel & 0xff), straight((pixel >> 8) & 0xff), straight((pixel >> 16) & 0xff))
  };
  [red as u8, green as u8, blue as u8, alpha as u8]
}

/// Converts premultiplied RGBA8 pixels to straight-alpha RGBA8.
///
/// WebAssembly uses a SIMD fast path for four-pixel groups containing only
/// transparent and opaque pixels. Groups containing partial alpha use the
/// scalar oracle because simd128 has no integer division instruction for
/// the un-premultiplication step.
pub(crate) fn premultiplied_rgba_to_straight_slice(src: &[u32], dst: &mut [u8]) {
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  premultiplied_rgba_to_straight_slice_wasm(src, dst);
  #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
  premultiplied_rgba_to_straight_slice_scalar(src, dst);
}

pub(crate) fn premultiplied_rgba_to_straight_slice_scalar(src: &[u32], dst: &mut [u8]) {
  for (&pixel, rgba) in src.iter().zip(dst.chunks_exact_mut(4)) {
    rgba.copy_from_slice(&premultiplied_rgba_to_straight(pixel));
  }
}

/// Expands Alpha8 pixels into straight-alpha RGBA8 using one constant RGB
/// color encoded as `0x00RRGGBB`.
pub(crate) fn alpha8_to_rgba_slice(src: &[u8], dst: &mut [u8], color: u32) {
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  alpha8_to_rgba_slice_wasm(src, dst, color);
  #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
  alpha8_to_rgba_slice_scalar(src, dst, color);
}

pub(crate) fn alpha8_to_rgba_slice_scalar(src: &[u8], dst: &mut [u8], color: u32) {
  let red = (color >> 16) as u8;
  let green = (color >> 8) as u8;
  let blue = color as u8;
  for (&alpha, rgba) in src.iter().zip(dst.chunks_exact_mut(4)) {
    rgba.copy_from_slice(&[red, green, blue, alpha]);
  }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn alpha8_to_rgba_slice_wasm(src: &[u8], dst: &mut [u8], color: u32) {
  use core::arch::wasm32::{u8x16, u8x16_swizzle, v128, v128_load, v128_or, v128_store};

  let red = (color >> 16) as u8;
  let green = (color >> 8) as u8;
  let blue = color as u8;
  let rgb = u8x16(red, green, blue, 0, red, green, blue, 0, red, green, blue, 0, red, green, blue, 0);
  // WebAssembly swizzle indices outside 0..16 produce zero. Each mask
  // places four source alpha bytes into the A positions of four RGBA pixels.
  let alpha0 = u8x16(16, 16, 16, 0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3);
  let alpha1 = u8x16(16, 16, 16, 4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7);
  let alpha2 = u8x16(16, 16, 16, 8, 16, 16, 16, 9, 16, 16, 16, 10, 16, 16, 16, 11);
  let alpha3 = u8x16(16, 16, 16, 12, 16, 16, 16, 13, 16, 16, 16, 14, 16, 16, 16, 15);
  let pixels = src.len().min(dst.len() / 4);
  let full = pixels - pixels % 16;

  if let (Some(src16), Some(dst64)) = (src.get(..full), dst.get_mut(..full * 4)) {
    for (input, output) in src16.chunks_exact(16).zip(dst64.chunks_exact_mut(64)) {
      // SAFETY: the exact chunks provide 16 readable input bytes and four
      // disjoint 16-byte output ranges. wasm v128 permits unaligned access.
      let alpha = unsafe { v128_load(input.as_ptr().cast::<v128>()) };
      for (offset, mask) in [(0, alpha0), (16, alpha1), (32, alpha2), (48, alpha3)] {
        let rgba = v128_or(rgb, u8x16_swizzle(alpha, mask));
        unsafe { v128_store(output.as_mut_ptr().add(offset).cast::<v128>(), rgba) };
      }
    }
  }

  if let (Some(src_tail), Some(dst_tail)) = (src.get(full..pixels), dst.get_mut(full * 4..pixels * 4)) {
    alpha8_to_rgba_slice_scalar(src_tail, dst_tail, color);
  }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn premultiplied_rgba_to_straight_slice_wasm(src: &[u32], dst: &mut [u8]) {
  use core::arch::wasm32::{u8x16, u8x16_swizzle, v128, v128_and, v128_load, v128_store};
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
        let rgba = unsafe { v128_load(input.as_ptr().cast::<v128>()) };
        // Transparent pixels must become canonical zero even if a hostile
        // input word contains non-zero RGB channels with alpha zero.
        let alpha = u8x16_swizzle(rgba, replicate_alpha);
        unsafe { v128_store(output.as_mut_ptr().cast::<v128>(), v128_and(rgba, alpha)) };
      } else {
        premultiplied_rgba_to_straight_slice_scalar(input, output);
      }
    }
  }

  if let (Some(src_tail), Some(dst_tail)) = (src.get(full..pixels), dst.get_mut(full * 4..pixels * 4)) {
    premultiplied_rgba_to_straight_slice_scalar(src_tail, dst_tail);
  }
}

#[cfg(test)]
#[path = "tests/pixel.rs"]
mod tests;
