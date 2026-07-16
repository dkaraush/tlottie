/// Hard resource limits applied while parsing and rendering.
///
/// Defaults are sized generously against the Telegram fixture corpus
/// (16.4k real files: p99 input ~730 KB, max ~1.7 MB) while still bounding
/// hostile input. Every limit maps to an [`crate::error::Limit`] error —
/// exceeding one is a clean `Err`, never a crash or an OOM spiral.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Maximum input size in bytes.
    pub max_input_bytes: usize,
    /// Maximum JSON nesting depth (objects + arrays combined).
    pub max_nesting_depth: usize,
    /// Maximum number of layers across the composition and all precomps.
    pub max_layers: usize,
    /// Maximum number of shape elements in a single layer.
    pub max_shapes_per_layer: usize,
    /// Maximum keyframes for a single animated property.
    pub max_keyframes: usize,
    /// Maximum number of assets (precomps).
    pub max_assets: usize,
    /// Maximum composition width/height in points, and render size in pixels.
    pub max_dimension: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 16 << 20, // 16 MiB
            max_nesting_depth: 128,
            max_layers: 4096,
            max_shapes_per_layer: 4096,
            max_keyframes: 65_536,
            max_assets: 512,
            max_dimension: 8192,
        }
    }
}
