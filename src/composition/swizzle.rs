//! Red/blue channel swap applied once, at parse time.
//!
//! The CPU pipeline never interprets a channel's meaning: coverage blending,
//! gradient interpolation, matte modulation and every SIMD kernel read a
//! channel at bit N and write the result back at bit N. The single exception
//! is the luma matte, whose Rec.601 weights are channel-specific; it reads
//! [`crate::model::Composition::channel_order`] instead.
//!
//! So emitting BGRA costs nothing per pixel — pre-swap the handful of colors
//! in the model and every downstream stage produces `0xAARRGGBB` words on its
//! own. This runs AFTER color replacements so callers still match and supply
//! replacement colors as `0xAARRGGBB`, exactly as they do for RGBA output.

use crate::model::{Asset, Composition, FloatList, Layer, Shape};
use crate::property::Property;

/// Swaps red and blue in every color the model carries.
pub(crate) fn swap_red_blue(comp: &mut Composition) {
  for layer in comp.layers.iter_mut() {
    swap_layer(layer);
  }
  for asset in comp.assets.iter_mut() {
    swap_asset(asset);
  }
}

fn swap_asset(asset: &mut Asset) {
  for layer in asset.layers.iter_mut() {
    swap_layer(layer);
  }
}

fn swap_layer(layer: &mut Layer) {
  if let Some((_, _, color)) = layer.solid.as_mut() {
    core::mem::swap(&mut color.r, &mut color.b);
  }
  swap_shapes(&mut layer.shapes);
}

fn swap_shapes(shapes: &mut [Shape]) {
  for shape in shapes.iter_mut() {
    match shape {
      Shape::Group(group) => swap_shapes(&mut group.shapes),
      Shape::Fill(fill) => swap_color_property(&mut fill.color),
      Shape::Stroke(stroke) => swap_color_property(&mut stroke.color),
      Shape::GradientFill(gradient) => {
        let count = gradient.color_count;
        map_property(&mut gradient.stops, &mut |stops| swap_stops(stops, count));
      }
      Shape::GradientStroke(gradient) => {
        let count = gradient.color_count;
        map_property(&mut gradient.stops, &mut |stops| swap_stops(stops, count));
      }
      // No color of their own; geometry and modifiers only. Repeater holds a
      // transform and copy counts — the shapes it repeats are its siblings.
      Shape::Path(_) | Shape::Rect(_) | Shape::Ellipse(_) | Shape::Trim(_) | Shape::Polystar(_) | Shape::RoundCorners(_) | Shape::Repeater(_) => {}
    }
  }
}

fn swap_color_property(color: &mut Property<crate::math::Color>) {
  map_property(color, &mut |value| core::mem::swap(&mut value.r, &mut value.b));
}

/// Stop data is `color_count * [offset, r, g, b]` followed by optional
/// `[offset, alpha]` pairs — only the color quads are touched, so the alpha
/// ramp that follows them is left exactly as parsed.
fn swap_stops(stops: &mut FloatList, color_count: usize) {
  for index in 0..color_count {
    let base = index.saturating_mul(4);
    let (Some(red), Some(blue)) = (base.checked_add(1), base.checked_add(3)) else {
      continue;
    };
    let (Some(&r), Some(&b)) = (stops.0.get(red), stops.0.get(blue)) else {
      continue;
    };
    if let Some(slot) = stops.0.get_mut(red) {
      *slot = b;
    }
    if let Some(slot) = stops.0.get_mut(blue) {
      *slot = r;
    }
  }
}

/// Applies `f` to every value a property can take: the static value, or each
/// keyframe's value together with its explicit segment end.
fn map_property<T>(property: &mut Property<T>, f: &mut impl FnMut(&mut T)) {
  match property {
    Property::Static(value) => f(value),
    Property::Animated(timeline) => {
      f(&mut timeline.first.value);
      if let Some(end) = timeline.first.end.as_mut() {
        f(end);
      }
      for keyframe in timeline.rest.iter_mut() {
        f(&mut keyframe.value);
        if let Some(end) = keyframe.end.as_mut() {
          f(end);
        }
      }
    }
  }
}
