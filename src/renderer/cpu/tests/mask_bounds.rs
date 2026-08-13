//! Byte-exactness of the offscreen-dirty-box bounding in `build_mask`.
//!
//! The property under test: for any bound box `B`, `build_mask(.., B)`
//! produces bytes inside `B` identical to `build_mask(.., FULL)` (the whole
//! canvas). Since the full-canvas bound runs the accumulate over every
//! pixel with the unchanged per-pixel body, `build_mask(.., FULL)` is the
//! former (pre-bounding) result, so equality inside `B` is exactly the
//! byte-identical gate — proven here for every mask mode, inverted masks,
//! multi-mask stacks, and the geometry edge cases (mask larger than canvas,
//! empty mask, mask fully outside canvas).
use super::*;
use crate::math::Vec2;
use crate::model::{Composition, Mask, PathData, Transform};
use crate::property::Property;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

fn full_box(w: usize, h: usize) -> DirtyBox {
  DirtyBox { x0: 0, y0: 0, x1: w - 1, y1: h - 1 }
}

fn rect_path(x0: f32, y0: f32, x1: f32, y1: f32) -> PathData {
  PathData {
    vertices: vec![Vec2::new(x0, y0), Vec2::new(x1, y0), Vec2::new(x1, y1), Vec2::new(x0, y1)],
    in_tangents: vec![Vec2::ZERO; 4],
    out_tangents: vec![Vec2::ZERO; 4],
    closed: true,
  }
}

fn mask(mode: u8, invert: bool, path: PathData, opacity: f32) -> Mask {
  Mask {
    mode,
    invert,
    path: Property::Static(path),
    opacity: Property::Static(opacity),
  }
}

fn layer_with_masks(masks: Vec<Mask>) -> Layer {
  Layer {
    name: String::new(),
    color_override: None,
    kind: LayerKind::Shape,
    index: 0,
    parent: None,
    in_point: 0.0,
    out_point: 100.0,
    start_time: 0.0,
    time_stretch: 1.0,
    hidden: false,
    transform: Transform::identity(),
    shapes: Vec::new(),
    ref_id: None,
    precomp_size: None,
    masks,
    matte: None,
    matte_src: false,
    solid: None,
    time_remap: None,
    auto_orient: false,
  }
}

fn empty_comp() -> Composition {
  Composition {
    width: 0,
    height: 0,
    frame_rate: 60.0,
    in_point: 0.0,
    out_point: 100.0,
    static_content: false,
    layers: Vec::new(),
    assets: Vec::new(),
    channel_order: Default::default(),
  }
}

/// Builds the mask over `bound` with a fresh scratch (so pooled stale bytes
/// never leak between calls, making the comparison unambiguous).
fn build(layer: &Layer, w: usize, h: usize, bound: DirtyBox) -> Vec<u8> {
  let comp = empty_comp();
  let ctx = RenderCtx {
    comp: &comp,
    continuous: false,
    antialias: true,
    curve_tolerance: 0.05,
  };
  let mut scratch = RenderScratch::default();
  ctx.build_mask(&mut scratch, layer, Mat2x3::IDENTITY, 0.0, w, h, bound)
}

/// Asserts every pixel inside `b` matches the full-canvas reference.
fn assert_bound_matches(layer: &Layer, w: usize, h: usize, b: DirtyBox) {
  let full = build(layer, w, h, full_box(w, h));
  let bounded = build(layer, w, h, b);
  for y in b.y0..=b.y1 {
    for x in b.x0..=b.x1 {
      let i = y * w + x;
      assert_eq!(full.get(i), bounded.get(i), "mismatch at ({x},{y}) i={i} for bound {:?}", (b.x0, b.y0, b.x1, b.y1));
    }
  }
}

// A representative selection of bounds: interior slice of the mask, a box
// straddling the mask edge, a box entirely outside the mask geometry, a
// one-pixel box, and the full canvas. Every box is clamped into the canvas
// (a real DirtyBox is always marked from in-canvas pixel coords, so
// x1 < w and y1 < h always hold) and degenerate boxes are dropped.
fn sample_bounds(w: usize, h: usize) -> Vec<DirtyBox> {
  [
    full_box(w, h),
    DirtyBox { x0: 20, y0: 20, x1: 30, y1: 30 }, // inside the mask rect
    DirtyBox { x0: 8, y0: 8, x1: 18, y1: 18 },   // straddles the edge
    DirtyBox { x0: 40, y0: 40, x1: 55, y1: 55 }, // outside the mask rect
    DirtyBox { x0: 0, y0: 0, x1: 4, y1: 4 },     // corner, outside
    DirtyBox { x0: 33, y0: 33, x1: 33, y1: 33 }, // single pixel, outside
  ]
  .into_iter()
  .filter_map(|b| {
    let (x1, y1) = (b.x1.min(w - 1), b.y1.min(h - 1));
    (b.x0 <= x1 && b.y0 <= y1).then_some(DirtyBox { x1, y1, ..b })
  })
  .collect()
}

#[test]
fn all_modes_and_inversion_are_bound_exact() {
  let (w, h) = (64, 64);
  // Mask geometry: a rect [10,10]-[35,35] (well inside the canvas).
  for &mode in &[b'a', b's', b'i', b'f'] {
    for &invert in &[false, true] {
      let layer = layer_with_masks(vec![mask(mode, invert, rect_path(10.0, 10.0, 35.0, 35.0), 100.0)]);
      for b in sample_bounds(w, h) {
        assert_bound_matches(&layer, w, h, b);
      }
    }
  }
}

#[test]
fn partial_opacity_is_bound_exact() {
  let (w, h) = (64, 64);
  for &mode in &[b'a', b's', b'i', b'f'] {
    let layer = layer_with_masks(vec![mask(mode, false, rect_path(10.0, 10.0, 35.0, 35.0), 50.0)]);
    for b in sample_bounds(w, h) {
      assert_bound_matches(&layer, w, h, b);
    }
  }
}

#[test]
fn multi_mask_stack_is_bound_exact() {
  let (w, h) = (64, 64);
  // Add, then subtract an overlapping rect, then intersect a third, with
  // one inverted member — exercises the running-`cur` fold under bounds.
  let layer = layer_with_masks(vec![
    mask(b'a', false, rect_path(8.0, 8.0, 40.0, 40.0), 100.0),
    mask(b's', false, rect_path(20.0, 20.0, 50.0, 50.0), 75.0),
    mask(b'i', true, rect_path(5.0, 5.0, 45.0, 45.0), 100.0),
    mask(b'f', false, rect_path(15.0, 25.0, 30.0, 38.0), 50.0),
  ]);
  for b in sample_bounds(w, h) {
    assert_bound_matches(&layer, w, h, b);
  }
}

#[test]
fn mask_larger_than_canvas_is_bound_exact() {
  let (w, h) = (48, 48);
  // Rect covers the whole canvas and spills far past every edge: after
  // clipping, coverage is 255 everywhere.
  for &mode in &[b'a', b's', b'i', b'f'] {
    for &invert in &[false, true] {
      let layer = layer_with_masks(vec![mask(mode, invert, rect_path(-100.0, -100.0, 200.0, 200.0), 100.0)]);
      for b in sample_bounds(w, h) {
        assert_bound_matches(&layer, w, h, b);
      }
      // Interior pixel: full coverage (t=255). Add non-inverted ⇒ 255.
      if mode == b'a' && !invert {
        let m = build(&layer, w, h, DirtyBox { x0: 24, y0: 24, x1: 24, y1: 24 });
        assert_eq!(m.get(24 * w + 24), Some(&255));
      }
    }
  }
}

#[test]
fn empty_mask_stack_is_bound_exact() {
  let (w, h) = (32, 32);
  // No effective masks (mode 'n'): acc keeps its seed (first_additive
  // defaults true ⇒ 0). Also the truly-empty vec.
  for masks in [vec![mask(b'n', false, rect_path(4.0, 4.0, 20.0, 20.0), 100.0)], Vec::new()] {
    let layer = layer_with_masks(masks);
    for b in sample_bounds(w, h).into_iter().filter(|b| b.x1 < w && b.y1 < h) {
      assert_bound_matches(&layer, w, h, b);
    }
    // Every sampled pixel is 0 (layer hidden).
    let m = build(&layer, w, h, DirtyBox { x0: 5, y0: 5, x1: 15, y1: 15 });
    for y in 5..=15 {
      for x in 5..=15 {
        assert_eq!(m.get(y * w + x), Some(&0));
      }
    }
  }
}

#[test]
fn mask_fully_outside_canvas_is_bound_exact() {
  let (w, h) = (32, 32);
  // Rect lives at (1000,1000)-(1100,1100): clips to nothing, coverage 0.
  for &mode in &[b'a', b's', b'i', b'f'] {
    for &invert in &[false, true] {
      let layer = layer_with_masks(vec![mask(mode, invert, rect_path(1000.0, 1000.0, 1100.0, 1100.0), 100.0)]);
      for b in sample_bounds(w, h).into_iter().filter(|b| b.x1 < w && b.y1 < h) {
        assert_bound_matches(&layer, w, h, b);
      }
    }
  }
}

/// Directly validates the per-mode OUTSIDE-value analysis documented on
/// `build_mask`: a pixel inside `bound` but outside all mask geometry
/// (coverage t=0) must equal the analytically derived constant O, and the
/// bounded build must reproduce it (never leaving stale/zero bytes there).
#[test]
fn outside_value_matches_per_mode_analysis() {
  let (w, h) = (64, 64);
  // op = 1.0 ⇒ at t=0: non-inverted c=0, inverted c=255. Expected O:
  //   a: noninv 0,  inv 255      s: noninv 255, inv 0
  //   i: noninv 0,  inv 255      f: noninv 0,   inv 255
  let cases: &[(u8, bool, u8)] = &[
    (b'a', false, 0),
    (b'a', true, 255),
    (b's', false, 255),
    (b's', true, 0),
    (b'i', false, 0),
    (b'i', true, 255),
    (b'f', false, 0),
    (b'f', true, 255),
  ];
  // Bound entirely outside the mask rect [10,10]-[35,35].
  let outside = DirtyBox { x0: 44, y0: 44, x1: 50, y1: 50 };
  for &(mode, invert, expected) in cases {
    let layer = layer_with_masks(vec![mask(mode, invert, rect_path(10.0, 10.0, 35.0, 35.0), 100.0)]);
    let m = build(&layer, w, h, outside);
    for y in outside.y0..=outside.y1 {
      for x in outside.x0..=outside.x1 {
        assert_eq!(m.get(y * w + x), Some(&expected), "mode {} invert {invert} outside value", mode as char);
      }
    }
    // And the same value must appear in the full build (consistency).
    let full = build(&layer, w, h, full_box(w, h));
    assert_eq!(full.get(47 * w + 47), Some(&expected));
  }
}
