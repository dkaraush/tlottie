//! Parse-time animation customizations.

/// Telegram's Fitzpatrick skin-tone modifier.
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

/// Options applied while a [`crate::Composition`] is parsed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseOptions {
  /// Telegram skin-tone selection used with the top-level `fitz` table.
  pub fitz_modifier: FitzModifier,
  /// Layer-name-prefix color overrides, resolved once during parsing.
  pub layer_color_replacements: Vec<LayerColorReplacement>,
}
