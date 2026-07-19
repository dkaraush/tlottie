/// Per-frame rendering options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
  /// Enables analytic edge antialiasing. Defaults to `true`.
  ///
  /// When disabled, coverage is thresholded to fully transparent or fully
  /// opaque at the 50% mark. This is faster for some span-heavy animations,
  /// but produces visibly jagged edges.
  pub antialias: bool,
  /// Maximum device-space curve-flattening error in pixels.
  ///
  /// The default is `0.5`, which keeps flattening error below one output
  /// pixel while substantially reducing contour work. Smaller values improve
  /// geometric accuracy at a performance cost.
  pub curve_tolerance: f32,

  /// Reserved for a future single-color rendering override.
  ///
  /// This currently has no effect and remains present for API compatibility.
  pub single_color: bool,
}

impl Default for RenderOptions {
  fn default() -> Self {
    Self {
      antialias: true,
      curve_tolerance: 0.5,
      single_color: false,
    }
  }
}
