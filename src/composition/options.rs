//! Parse-time animation customizations.

/// Fitzpatrick skin-tone modifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum FitzModifier {
  /// Do not apply a skin-tone replacement.
  #[default]
  None = 0,
  /// Fitzpatrick types 1 and 2.
  Type12 = 1,
  /// Fitzpatrick type 3.
  Type3 = 2,
  /// Fitzpatrick type 4.
  Type4 = 3,
  /// Fitzpatrick type 5.
  Type5 = 4,
  /// Fitzpatrick type 6.
  Type6 = 5,
}

impl FitzModifier {
  pub(crate) fn replacement_index(self) -> Option<usize> {
    match self {
      Self::None => None,
      Self::Type12 => Some(0),
      Self::Type3 => Some(1),
      Self::Type4 => Some(2),
      Self::Type5 => Some(3),
      Self::Type6 => Some(4),
    }
  }

  pub(crate) fn from_u32(value: u32) -> Option<Self> {
    match value {
      0 => Some(Self::None),
      1 => Some(Self::Type12),
      2 => Some(Self::Type3),
      3 => Some(Self::Type4),
      4 => Some(Self::Type5),
      5 => Some(Self::Type6),
      _ => None,
    }
  }
}

/// A parse-time color override for every solid fill and stroke below a layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerColorReplacement {
  /// Match layers whose `nm` starts with this string. No trailing `*` is needed.
  pub layer_name_prefix: String,
  /// Replacement color in straight-alpha `0xAARRGGBB` form.
  pub color: u32,
}

/// A parse-time replacement for a color used by fills, strokes, gradients,
/// and solid layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceColorReplacement {
  /// Source color in `0xAARRGGBB` form. The alpha byte is ignored.
  pub source_color: u32,
  /// Target color in `0xAARRGGBB` form. The animation's original alpha is preserved.
  pub target_color: u32,
}

use alloc::string::String;
use alloc::vec::Vec;

/// Byte order of the premultiplied pixels a renderer writes.
///
/// The blend pipeline itself is channel-order agnostic — every stage reads a
/// channel at bit N and writes it back at bit N — so the order is applied
/// once, by pre-swapping the parsed model's colors. Rendering therefore costs
/// exactly the same in either order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ChannelOrder {
  /// `0xAABBGGRR` words: `[R, G, B, A]` bytes in little-endian memory.
  /// Matches Android's `Bitmap.Config.ARGB_8888`.
  #[default]
  Rgba = 0,
  /// `0xAARRGGBB` words: `[B, G, R, A]` bytes in little-endian memory.
  /// Matches Qt's `QImage::Format_ARGB32_Premultiplied`, Direct2D /
  /// DXGI `B8G8R8A8_UNORM`, and Cairo's `CAIRO_FORMAT_ARGB32` — i.e. what
  /// desktop compositors take without a conversion pass.
  Bgra = 1,
}

impl ChannelOrder {
  pub(crate) fn from_u32(value: u32) -> Option<Self> {
    match value {
      0 => Some(Self::Rgba),
      1 => Some(Self::Bgra),
      _ => None,
    }
  }
}

/// Options applied while a [`crate::Composition`] is parsed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseOptions {
  /// Telegram skin-tone selection used with the top-level `fitz` table.
  pub fitz_modifier: FitzModifier,
  /// Layer-name-prefix color overrides, resolved once during parsing.
  pub layer_color_replacements: Vec<LayerColorReplacement>,
  /// Exact source-color replacements, resolved once during parsing.
  pub source_color_replacements: Vec<SourceColorReplacement>,
  /// Byte order of the rendered pixels. Applied after every color
  /// replacement, so replacement colors are still matched as `0xAARRGGBB`.
  pub channel_order: ChannelOrder,
}
