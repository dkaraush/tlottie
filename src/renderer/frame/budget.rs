//! Deterministic, per-frame work limits. Charge before expanding geometry or
//! invoking a backend, including when commands replay from a cache.
use super::renderer::*;
#[cfg(not(feature = "std"))]
use crate::compat::FloatExt as _;
use crate::{Error, Limit, Limits, Result};
use core::cell::Cell;

#[derive(Default)]
pub(crate) struct Budget {
  points: Cell<usize>,
  work: Cell<usize>,
  layers: Cell<usize>,
}

impl Budget {
  pub(crate) fn charge(counter: &Cell<usize>, amount: usize, maximum: usize, limit: Limit) -> Result<()> {
    let remaining = maximum.saturating_sub(counter.get());
    if amount > remaining {
      return Err(Error::LimitExceeded(limit));
    }
    counter.set(counter.get() + amount);
    Ok(())
  }
  pub(crate) fn points(&self, amount: usize) -> Result<()> {
    Self::charge(&self.points, amount, Limits::default().max_render_points, Limit::RenderGeometry)
  }
  pub(crate) fn work(&self, amount: usize) -> Result<()> {
    Self::charge(&self.work, amount, Limits::default().max_render_work, Limit::RenderWork)
  }
  pub(crate) fn layers(&self, amount: usize) -> Result<()> {
    Self::charge(&self.layers, amount, Limits::default().max_precomp_expansion, Limit::PrecompExpansion)
  }
  pub(crate) fn path(&self, data: &crate::model::PathData, matrix: &crate::math::Mat2x3, tolerance: f32) -> Result<usize> {
    use crate::math::Vec2;
    let limits = Limits::default();
    let n = data.vertices.len();
    if n.max(data.in_tangents.len()).max(data.out_tangents.len()) > limits.max_path_points {
      return Err(Error::LimitExceeded(Limit::PathPoints));
    }
    // Reserve the flattener's proven maximum before any expansion. Usually
    // this fits without examining coordinates or calculating subdivisions.
    // Refund unused capacity after flattening. Near the allocation limit,
    // retain the precise preflight so valid complex paths still fit.
    let segments = if data.closed { n } else { n.saturating_sub(1) };
    let maximum = segments.saturating_mul(crate::geometry::MAX_SEGS).saturating_add(1);
    if maximum <= Limits::default().max_render_points.saturating_sub(self.points.get()) {
      self.points(maximum)?;
      return Ok(maximum);
    }
    let mut points = 1usize;
    let segments = if data.closed { n } else { n.saturating_sub(1) };
    for i in 0..segments {
      let j = if i + 1 == n { 0 } else { i + 1 };
      let p0 = data.vertices.get(i).copied().unwrap_or(Vec2::ZERO);
      let p1 = data.vertices.get(j).copied().unwrap_or(Vec2::ZERO);
      let t0 = data.out_tangents.get(i).copied().unwrap_or(Vec2::ZERO);
      let t1 = data.in_tangents.get(j).copied().unwrap_or(Vec2::ZERO);
      let a = matrix.apply(p0);
      let b = matrix.apply(Vec2::new(p0.x + t0.x, p0.y + t0.y));
      let c = matrix.apply(Vec2::new(p1.x + t1.x, p1.y + t1.y));
      let d = matrix.apply(p1);
      if [a, b, c, d].iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return Err(Error::LimitExceeded(Limit::PathCoordinate));
      }
      let count = if t0.x == 0.0 && t0.y == 0.0 && t1.x == 0.0 && t1.y == 0.0 {
        1
      } else {
        crate::geometry::cubic_segments(a, b, c, d, tolerance)
      };
      points = points.saturating_add(count);
    }
    self.points(points)?;
    Ok(points)
  }
  pub(crate) fn finish_expansion(&self, reserved: usize, actual: usize) {
    debug_assert!(actual <= reserved);
    self.points.set(self.points.get().saturating_sub(reserved.saturating_sub(actual.max(1))));
  }
}

pub(crate) fn canvas_size(width: u32, height: u32) -> Result<usize> {
  let limits = Limits::default();
  if width == 0 || height == 0 || width > limits.max_dimension || height > limits.max_dimension {
    return Err(Error::LimitExceeded(Limit::CompositionSize));
  }
  let pixels = (width as usize).checked_mul(height as usize).ok_or(Error::LimitExceeded(Limit::RenderMemory))?;
  // Allow for the dense raster, mask planes, fallback bitmap and pooled
  // planes before any renderer-side allocation (including Alpha8 fallback).
  if pixels > limits.max_render_bytes / 32 {
    return Err(Error::LimitExceeded(Limit::RenderMemory));
  }
  Ok(pixels)
}

/// Bounds belong to the same immutable geometry key as retained coverage.
#[derive(Clone, Copy)]
pub(crate) struct DrawCost {
  work: usize,
  pixels: usize,
  sparse: bool,
}

pub(crate) struct GuardedRenderer<'a, R> {
  inner: &'a mut R,
  width: usize,
  height: usize,
  depth: usize,
  pixels: Cell<usize>,
  pixel_limit: usize,
  budget: Budget,
  error: Option<Error>,
  pub(crate) pixel_costs: crate::compat::HashMap<u128, DrawCost>,
}
impl<'a, R: FrameRenderer> GuardedRenderer<'a, R> {
  pub(crate) fn new(inner: &'a mut R, width: u32, height: u32, pixel_costs: crate::compat::HashMap<u128, DrawCost>) -> Self {
    Self {
      inner,
      width: width as usize,
      height: height as usize,
      depth: 0,
      pixels: Cell::new(0),
      // Pixel work grows with output area even for the same authored scene.
      // Keep the small-canvas floor while allowing proportional raster work.
      pixel_limit: ((Limits::default().max_render_pixels as u64).saturating_mul((width as u64 * height as u64).max(64 * 64)) / (64 * 64)).min(usize::MAX as u64) as usize,
      budget: Budget::default(),
      error: None,
      pixel_costs,
    }
  }
  fn accept(&mut self, result: Result<()>) -> bool {
    if self.error.is_some() {
      return false;
    }
    match result {
      Ok(()) => true,
      Err(error) => {
        self.error = Some(error);
        false
      }
    }
  }
  fn pixels(&self, times: usize) -> Result<()> {
    Budget::charge(&self.pixels, self.width.saturating_mul(self.height).saturating_mul(times), self.pixel_limit, Limit::RenderWork)
  }
  fn geometry(&mut self, geometry: &mut Geometry<'_>, pixel_weight: usize) -> Result<()> {
    let translation = geometry.raw_translation();
    if !translation.x.is_finite() || !translation.y.is_finite() {
      return Err(Error::LimitExceeded(Limit::PathCoordinate));
    }
    if let Some(cost) = self.pixel_costs.get(&geometry.cache_key) {
      self.budget.work(cost.work)?;
      geometry.raster_mode = Some(cost.sparse);
      return Budget::charge(&self.pixels, cost.pixels.saturating_mul(pixel_weight), self.pixel_limit, Limit::RenderWork);
    }
    let metrics = crate::geometry::RasterMetrics::new(geometry.raw_contours());
    if !metrics.perim.is_finite() {
      return Err(Error::LimitExceeded(Limit::PathCoordinate));
    }
    // Twice the Manhattan perimeter bounds horizontal + twice vertical
    // crossings. Add fixed per-edge overhead before raster allocation.
    let work = ((metrics.perim * 2.0).ceil() as usize).saturating_add(metrics.points.saturating_mul(8));
    if work > Limits::default().max_render_points {
      return Err(Error::LimitExceeded(Limit::RenderGeometry));
    }
    self.budget.work(work)?;
    geometry.raster_mode = Some(metrics.sparse(self.width.saturating_mul(self.height)));
    let (x0, y0, x1, y1) = (metrics.x0 + translation.x, metrics.y0 + translation.y, metrics.x1 + translation.x, metrics.y1 + translation.y);
    if metrics.points > 0 && [x0, y0, x1, y1].iter().any(|v| !v.is_finite()) {
      return Err(Error::LimitExceeded(Limit::PathCoordinate));
    }
    let pixels = if x0 <= x1 && y0 <= y1 {
      let w = (x1.ceil().clamp(0.0, self.width as f32) - x0.floor().clamp(0.0, self.width as f32)).max(0.0) as usize;
      let h = (y1.ceil().clamp(0.0, self.height as f32) - y0.floor().clamp(0.0, self.height as f32)).max(0.0) as usize;
      let pixels = w.saturating_mul(h);
      if self.pixel_costs.len() >= 512 {
        self.pixel_costs.clear();
      }
      self.pixel_costs.try_reserve(1).map_err(|_| Error::LimitExceeded(Limit::RenderMemory))?;
      self.pixel_costs.insert(
        geometry.cache_key,
        DrawCost {
          work,
          pixels,
          sparse: geometry.raster_mode.unwrap_or(false),
        },
      );
      pixels
    } else {
      // Coverage replay may omit all points. Retain its previous conservative
      // bounding-box charge across frames; on a miss charge the full canvas.
      self.width.saturating_mul(self.height)
    };
    Budget::charge(&self.pixels, pixels.saturating_mul(pixel_weight), self.pixel_limit, Limit::RenderWork)
  }
}
impl<R: FrameRenderer> FrameRenderer for GuardedRenderer<'_, R> {
  fn status(&self) -> Result<()> {
    match &self.error {
      Some(error) => Err(error.clone()),
      None => self.inner.status(),
    }
  }
  fn save_layer(&mut self) {
    let bytes = self.width.saturating_mul(self.height).saturating_mul(32 + 4 * (self.depth + 1));
    let result = if bytes > Limits::default().max_render_bytes {
      Err(Error::LimitExceeded(Limit::RenderMemory))
    } else {
      self.pixels(1)
    };
    if self.accept(result) {
      self.depth += 1;
      self.inner.save_layer();
    }
  }
  fn draw(&mut self, mut geometry: Geometry<'_>, paint: Paint<'_>) {
    if self.error.is_some() {
      return;
    }
    // Gradient mapping, LUT lookup and blending cost more than a solid fill;
    // focal sampling also solves a quadratic. Charge evaluated paint kinds so
    // animation and coverage replay cannot bypass this work allowance. These
    // conservative weights are work units, not hardware timing predictions.
    let pixel_weight = match paint {
      Paint::Solid(_) => 1,
      Paint::Gradient(gradient) => match gradient.kind {
        GradientKind::Linear { .. } => 8,
        GradientKind::Radial { .. } => 16,
        GradientKind::Focal { .. } => 32,
      },
    };
    let result = self.geometry(&mut geometry, pixel_weight);
    if self.accept(result) {
      self.inner.draw(geometry, paint);
    }
  }
  fn apply_mask(&mut self, mut geometry: Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    if self.error.is_some() {
      return;
    }
    let result = self.geometry(&mut geometry, 1).and_then(|()| self.pixels(3));
    if self.accept(result) {
      self.inner.apply_mask(geometry, mode, inverted, opacity, first, last);
    }
  }
  fn end_layer(&mut self, composite: Composite) {
    let result = self.pixels(3);
    if self.accept(result) {
      self.depth = self.depth.saturating_sub(match composite {
        Composite::Over { .. } => 1,
        Composite::Matte { .. } => 2,
      });
      self.inner.end_layer(composite);
    }
  }
  fn retains_geometry(&self, key: u128) -> bool {
    self.inner.retains_geometry(key)
  }
}

impl Budget {
  pub(crate) fn property<T>(&self, property: &crate::property::Property<T>) -> Result<()> {
    if let crate::property::Property::Animated(timeline) = property {
      let n = timeline.rest.len().saturating_add(1);
      if n > Limits::default().max_keyframes {
        return Err(Error::LimitExceeded(Limit::Keyframes));
      }
      self.work(n)?;
    }
    Ok(())
  }
  fn values<T>(&self, property: &crate::property::Property<T>, mut check: impl FnMut(&T) -> Result<()>) -> Result<()> {
    self.property(property)?;
    match property {
      crate::property::Property::Static(value) => check(value),
      crate::property::Property::Animated(timeline) => {
        for key in core::iter::once(&timeline.first).chain(&timeline.rest) {
          check(&key.value)?;
          if let Some(end) = &key.end {
            check(end)?;
          }
        }
        Ok(())
      }
    }
  }
  pub(crate) fn path_property(&self, property: &crate::property::Property<crate::model::PathData>) -> Result<()> {
    self.values(property, |p| {
      let n = p.vertices.len().max(p.in_tangents.len()).max(p.out_tangents.len());
      if n > Limits::default().max_path_points {
        return Err(Error::LimitExceeded(Limit::PathPoints));
      }
      self.work(n)
    })
  }
  fn stops(&self, property: &crate::property::Property<crate::model::FloatList>) -> Result<()> {
    self.values(property, |p| {
      if p.0.len() > Limits::default().max_gradient_stop_values {
        return Err(Error::LimitExceeded(Limit::GradientStopValues));
      }
      self.work(p.0.len())
    })
  }
  pub(crate) fn transform(&self, transform: &crate::model::Transform) -> Result<()> {
    self.property(&transform.anchor)?;
    self.property(&transform.scale)?;
    self.property(&transform.rotation)?;
    self.property(&transform.opacity)?;
    self.property(&transform.skew)?;
    self.property(&transform.skew_axis)?;
    match &transform.position {
      crate::model::Position::Combined(p) => self.property(p),
      crate::model::Position::Split { x, y } => {
        self.property(x)?;
        self.property(y)
      }
    }
  }
  fn dashes(&self, dashes: &[crate::model::DashElement]) -> Result<()> {
    if dashes.len() > Limits::default().max_dash_elements {
      return Err(Error::LimitExceeded(Limit::DashElements));
    }
    for dash in dashes {
      self.property(&dash.value)?;
    }
    Ok(())
  }
  pub(crate) fn shape_tree(&self, shapes: &[crate::model::Shape], depth: usize) -> Result<()> {
    if depth > 40 {
      return Err(Error::LimitExceeded(Limit::NestingDepth));
    }
    self.work(shapes.len())?;
    for shape in shapes {
      self.shape(shape)?;
      if let crate::model::Shape::Group(group) = shape {
        self.shape_tree(&group.shapes, depth + 1)?;
      }
    }
    Ok(())
  }
  pub(crate) fn shape(&self, shape: &crate::model::Shape) -> Result<()> {
    use crate::model::Shape;
    macro_rules! props { ($s:ident, $($field:ident),+) => { $(self.property(&$s.$field)?;)+ }; }
    match shape {
      Shape::Group(g) => self.transform(&g.transform)?,
      Shape::Path(p) => self.path_property(&p.path)?,
      Shape::Rect(r) => {
        props!(r, position, size, radius);
      }
      Shape::Ellipse(e) => {
        props!(e, position, size);
      }
      Shape::Polystar(p) => {
        props!(p, points, position, rotation, inner_radius, outer_radius, inner_roundness, outer_roundness);
      }
      Shape::RoundCorners(r) => {
        props!(r, radius);
      }
      Shape::Trim(t) => {
        props!(t, start, end, offset);
      }
      Shape::Repeater(r) => {
        props!(r, copies, offset, start_opacity, end_opacity);
        self.transform(&r.transform)?;
      }
      Shape::Fill(f) => {
        props!(f, color, opacity);
      }
      Shape::Stroke(s) => {
        props!(s, color, opacity, width);
        self.dashes(&s.dashes)?;
      }
      Shape::GradientFill(g) => {
        props!(g, start, end, opacity, highlight_len, highlight_angle);
        self.stops(&g.stops)?;
      }
      Shape::GradientStroke(g) => {
        props!(g, start, end, opacity, highlight_len, highlight_angle, width);
        self.stops(&g.stops)?;
        self.dashes(&g.dashes)?;
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[derive(Default)]
  struct Sink {
    saves: usize,
    draws: usize,
  }
  impl FrameRenderer for Sink {
    fn save_layer(&mut self) {
      self.saves += 1;
    }
    fn draw(&mut self, _: Geometry<'_>, _: Paint<'_>) {
      self.draws += 1;
    }
    fn apply_mask(&mut self, _: Geometry<'_>, _: u8, _: bool, _: u8, _: bool, _: bool) {}
    fn end_layer(&mut self, _: Composite) {}
  }
  #[test]
  fn nested_surfaces_stop_before_backend_allocation() {
    let mut sink = Sink::default();
    let mut guarded = GuardedRenderer::new(&mut sink, 1024, 1024, Default::default());
    for _ in 0..100 {
      guarded.save_layer();
    }
    assert_eq!(guarded.status(), Err(Error::LimitExceeded(Limit::RenderMemory)));
    assert!(sink.saves < 100);
  }
  #[test]
  fn cached_geometry_still_consumes_pixel_budget() {
    let mut sink = Sink::default();
    let mut guarded = GuardedRenderer::new(&mut sink, 128, 128, Default::default());
    let paint = Paint::Solid(SolidPaint {
      rule: Rule::NonZero,
      rgba: u32::MAX,
      color: crate::math::Color::BLACK,
      opacity: 1.0,
    });
    for _ in 0..16385 {
      guarded.draw(Geometry::new(&[], 1), paint);
    }
    assert_eq!(guarded.status(), Err(Error::LimitExceeded(Limit::RenderWork)));
    assert_eq!(sink.draws, 16384);
  }

  #[test]
  fn cached_coverage_is_charged_for_the_current_gradient_kind() {
    use crate::geometry::Contour;
    use crate::math::Vec2;
    // The first frame caches coverage with a cheap paint. A later frame can
    // reuse those points with a different (or newly focal) animated gradient.
    let contours = [Contour {
      points: alloc::vec![Vec2::new(0.0, 0.0), Vec2::new(128.0, 0.0), Vec2::new(128.0, 128.0), Vec2::new(0.0, 128.0)],
      ..Contour::default()
    }];
    let mut sink = Sink::default();
    let mut guarded = GuardedRenderer::new(&mut sink, 128, 128, Default::default());
    guarded.draw(
      Geometry::new(&contours, 1),
      Paint::Solid(SolidPaint {
        rule: Rule::NonZero,
        rgba: u32::MAX,
        color: crate::math::Color::BLACK,
        opacity: 1.0,
      }),
    );
    assert_eq!(guarded.status(), Ok(()));
    let costs = guarded.pixel_costs;
    for (kind, expected_draws) in [
      (
        GradientKind::Linear {
          sx: 0.0,
          sy: 0.0,
          dx: 1.0,
          dy: 1.0,
          inv_len_sq: 0.5,
        },
        2048,
      ),
      (GradientKind::Radial { sx: 0.0, sy: 0.0, inv_r: 1.0 }, 1024),
      (
        GradientKind::Focal {
          fx: 0.0,
          fy: 0.0,
          dx: 1.0,
          dy: 0.0,
          a: 1.0,
          r: 1.0,
        },
        512,
      ),
    ] {
      let gradient = GradientPaint {
        rule: Rule::NonZero,
        lut: GradientLut::new([u32::MAX; GRADIENT_LUT_SIZE], Some(255)),
        transform: GradientTransform {
          a: 1.0,
          b: 0.0,
          c: 0.0,
          d: 1.0,
          tx: 0.0,
          ty: 0.0,
        },
        kind,
        source_key: 0,
        alpha: 255,
      };
      let mut sink = Sink::default();
      let mut guarded = GuardedRenderer::new(&mut sink, 128, 128, costs.clone());
      for _ in 0..=expected_draws {
        guarded.draw(Geometry::new(&[], 1), Paint::Gradient(&gradient));
      }
      assert_eq!(guarded.status(), Err(Error::LimitExceeded(Limit::RenderWork)));
      assert_eq!(sink.draws, expected_draws);
    }
  }
}

#[cfg(feature = "cpu")]
pub(crate) fn parent_matrix(layers: &[crate::model::Layer], layer: &crate::model::Layer, frame: f32, budget: &Budget) -> Result<crate::math::Mat2x3> {
  use crate::math::Mat2x3;
  let mut chain = [Mat2x3::IDENTITY; 128];
  let mut depth = 0usize;
  let mut parent = layer.parent;
  while let Some(index) = parent {
    budget.work(layers.len())?;
    let Some(layer) = layers.iter().find(|layer| layer.index == index) else { break };
    let Some(slot) = chain.get_mut(depth) else {
      return Err(Error::LimitExceeded(Limit::ParentChainDepth));
    };
    budget.transform(&layer.transform)?;
    *slot = crate::renderer::cpu::executor::transform_at(&layer.transform, frame).0;
    depth += 1;
    parent = layer.parent;
  }
  let mut matrix = Mat2x3::IDENTITY;
  for transform in chain.iter().take(depth).rev() {
    matrix = matrix.concat(*transform);
  }
  Ok(matrix)
}

/// Inspect immutable property sizes and expanded model work once, independently
/// of parser validation (tests and embedders may bypass parser limits).
#[cfg(feature = "cpu")]
pub(crate) fn validate_model(comp: &crate::Composition) -> Result<()> {
  let limits = Limits::default();
  if comp.assets.len() > limits.max_assets {
    return Err(Error::LimitExceeded(Limit::Assets));
  }
  fn layers(comp: &crate::Composition, items: &[crate::model::Layer], resources: &Budget, depth: usize) -> Result<()> {
    if depth > 16 {
      return Err(Error::LimitExceeded(Limit::NestingDepth));
    }
    resources.layers(items.len())?;
    for layer in items {
      resources.transform(&layer.transform)?;
      if let Some(tm) = &layer.time_remap {
        resources.property(tm)?;
      }
      parent_matrix(items, layer, 0.0, resources)?;
      resources.shape_tree(&layer.shapes, 0)?;
      if layer.masks.len() > Limits::default().max_masks_per_layer {
        return Err(Error::LimitExceeded(Limit::MasksPerLayer));
      }
      for mask in &layer.masks {
        resources.path_property(&mask.path)?;
        resources.property(&mask.opacity)?;
      }
      if layer.kind == crate::model::LayerKind::Precomp {
        resources.work(comp.assets.len())?;
        if let Some(asset) = layer.ref_id.as_deref().and_then(|id| comp.assets.iter().find(|asset| asset.id == id)) {
          layers(comp, &asset.layers, resources, depth + 1)?;
        }
      }
    }
    Ok(())
  }
  layers(comp, &comp.layers, &Budget::default(), 0)
}
