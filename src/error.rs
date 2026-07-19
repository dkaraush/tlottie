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
  /// More keyframes on one property than [`crate::Limits::max_keyframes`].
  Keyframes,
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

impl std::error::Error for Error {}

/// Convenience alias: `tlottie::Result<T>` = `Result<T, tlottie::Error>`.
pub type Result<T> = core::result::Result<T, Error>;
