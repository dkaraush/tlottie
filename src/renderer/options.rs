/// Per-frame rendering options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
  /// Clears the destination before rendering. Defaults to `true`.
  ///
  /// Disable this to composite the animation over the existing
  /// premultiplied destination pixels.
  pub clear: bool,
  /// Enables analytic edge antialiasing. Defaults to `true`.
  ///
  /// When disabled, coverage is thresholded to fully transparent or fully
  /// opaque at the 50% mark. This is faster for some span-heavy animations,
  /// but produces visibly jagged edges.
  pub antialias: bool,
  /// Maximum device-space curve-flattening error in pixels.
  ///
  /// The default is `0.125`, prioritizing visible geometric accuracy.
  /// Smaller values improve accuracy at a performance cost.
  pub curve_tolerance: f32,
  /// Renders only the animation's alpha mask.
  ///
  /// Output RGB channels are zero. Paint colors are skipped where possible,
  /// while paint opacity, edge coverage, masks, mattes, and layer compositing
  /// continue to determine the alpha channel.
  pub alpha_only: bool,
}

impl Default for RenderOptions {
  fn default() -> Self {
    Self {
      clear: true,
      antialias: true,
      curve_tolerance: 0.125,
      alpha_only: false,
    }
  }
}
