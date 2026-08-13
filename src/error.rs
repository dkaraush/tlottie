use core::fmt;

/// Everything tlottie can fail with. The library never panics: any input,
/// however malformed or hostile, must surface as one of these.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
  /// The input is not valid JSON. `offset` is a byte offset into the input.
  Json {
    /// Byte offset into the input where the error was detected.
    offset: usize,
    /// What exactly was wrong.
    kind: JsonErrorKind,
  },
  /// Valid JSON, but not a Lottie composition we can make sense of
  /// (e.g. missing width/height/frame rate, malformed keyframe structure).
  InvalidLottie {
    /// Byte offset into the input where the error was detected.
    offset: usize,
    /// Human-readable description of the problem.
    what: &'static str,
  },
  /// A hard resource limit was exceeded (see [`crate::Limits`]).
  LimitExceeded(Limit),
  /// The composition has no renderable content at all.
  Empty,
}

/// What exactly was wrong with the JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JsonErrorKind {
  /// Input ended in the middle of a value.
  UnexpectedEof,
  /// A byte that doesn't belong at this position.
  UnexpectedByte(u8),
  /// Malformed or non-finite number.
  BadNumber,
  /// Malformed string.
  BadString,
  /// Malformed escape sequence inside a string.
  BadEscape,
  /// Extra non-whitespace content after the top-level value.
  TrailingData,
}

/// Which resource limit tripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Limit {
  /// Input larger than [`crate::Limits::max_input_bytes`].
  InputBytes,
  /// JSON nesting deeper than [`crate::Limits::max_nesting_depth`].
  NestingDepth,
  /// More layers than [`crate::Limits::max_layers`].
  Layers,
  /// More shapes in one layer than [`crate::Limits::max_shapes_per_layer`].
  ShapesPerLayer,
  /// More paints in one layer than [`crate::Limits::max_paints_per_layer`].
  PaintsPerLayer,
  /// Paints in one layer reference too many cumulative geometry-source items.
  PaintSourceItemsPerLayer,
  /// More focal radial gradient paints than [`crate::Limits::max_focal_radial_gradients_per_layer`].
  FocalRadialGradientsPerLayer,
  /// Focal radial gradient paints expand beyond [`crate::Limits::max_focal_radial_gradient_expansion`].
  FocalRadialGradientExpansion,
  /// More painted shape layers than [`crate::Limits::max_painted_shape_layers`].
  PaintedShapeLayers,
  /// More solid layers than [`crate::Limits::max_solid_layers`].
  SolidLayers,
  /// More keyframes on one property than [`crate::Limits::max_keyframes`].
  Keyframes,
  /// More points in one path than [`crate::Limits::max_path_points`].
  PathPoints,
  /// A path coordinate exceeded [`crate::Limits::max_path_coordinate_abs`].
  PathCoordinate,
  /// More masks in one layer than [`crate::Limits::max_masks_per_layer`].
  MasksPerLayer,
  /// More retained masks than [`crate::Limits::max_masks`].
  Masks,
  /// A mask path exceeded [`crate::Limits::max_mask_path_points`].
  MaskPathPoints,
  /// More dash entries in one stroke than [`crate::Limits::max_dash_elements`].
  DashElements,
  /// More dashed strokes in one group than [`crate::Limits::max_dashed_strokes_per_group`].
  DashedStrokesPerGroup,
  /// More gradient strokes in one group than [`crate::Limits::max_gradient_strokes_per_group`].
  GradientStrokesPerGroup,
  /// A round-join dashed stroke source segment exceeded [`crate::Limits::max_dashed_path_segment_span`].
  DashedPathSegment,
  /// More gradient stop floats than [`crate::Limits::max_gradient_stop_values`].
  GradientStopValues,
  /// More Fitzpatrick metadata entries than [`crate::Limits::max_fitz_entries`].
  FitzEntries,
  /// Precomp references expand beyond [`crate::Limits::max_precomp_expansion`].
  PrecompExpansion,
  /// Parent chain deeper than [`crate::Limits::max_parent_chain_depth`].
  ParentChainDepth,
  /// Parent chains total more than [`crate::Limits::max_parent_chain_total_depth`].
  ParentChainTotalDepth,
  /// A polystar exceeded [`crate::Limits::max_polystar_points`].
  PolystarPoints,
  /// A repeater exceeded [`crate::Limits::max_repeater_copies`].
  RepeaterCopies,
  /// More round-corners modifiers than [`crate::Limits::max_round_corners_per_layer`].
  RoundCornersPerLayer,
  /// More trim modifiers than [`crate::Limits::max_trims_per_layer`].
  TrimsPerLayer,
  /// Estimated dash pieces exceed [`crate::Limits::max_dashed_piece_estimate_per_group`].
  DashedPiecesPerGroup,
  /// Compounded repeaters exceed [`crate::Limits::max_repeater_product_per_group`].
  RepeaterProductPerGroup,
  /// More assets than [`crate::Limits::max_assets`].
  Assets,
  /// Composition dimensions beyond [`crate::Limits::max_dimension`].
  CompositionSize,
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::Json { offset, kind } => {
        write!(f, "invalid JSON at byte {offset}: {kind:?}")
      }
      Error::InvalidLottie { offset, what } => {
        write!(f, "invalid Lottie at byte {offset}: {what}")
      }
      Error::LimitExceeded(limit) => write!(f, "resource limit exceeded: {limit:?}"),
      Error::Empty => write!(f, "composition has no renderable content"),
    }
  }
}

// core has no Error trait, so this is the one API difference between the two
// configurations. Display, Debug and the enum itself are unchanged.
#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience alias: `tlottie::Result<T>` = `Result<T, tlottie::Error>`.
pub type Result<T> = core::result::Result<T, Error>;
