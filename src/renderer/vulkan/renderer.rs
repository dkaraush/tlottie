//! Experimental Vulkan drawing helpers for tlottie.
//!
//! This crate does not create or own a Vulkan instance/device. Hosts such as
//! `tlottie-cli` set up Vulkan and pass command-buffer targets here. Enable
//! this module with the `vulkan` feature.

#![allow(unsafe_code)]

use crate::{internal as tlottie_internal, Composition, RenderOptions};
use ash::vk;

#[derive(Clone, Debug, Default)]
struct RecordedContour {
  points: Vec<tlottie_internal::Point>,
  closed: bool,
}

#[derive(Clone, Debug)]
enum RecordedOp {
  Solid(tlottie_internal::SolidPaint),
  Gradient(tlottie_internal::GradientPaint),
  BeginLayer,
  EndLayer { opacity: u8 },
  BeginMatte,
  BeginMatteTarget,
  EndMatte { kind: u8, opacity: u8, source_opacity: u8 },
  Mask { mode: u8, inverted: bool, opacity: u8, first: bool, last: bool },
}

#[derive(Clone, Debug)]
struct RecordedCommand {
  op: RecordedOp,
  start: usize,
  end: usize,
}

#[derive(Clone, Debug, Default)]
struct EvaluatedFrame {
  contours: Vec<RecordedContour>,
  commands: Vec<RecordedCommand>,
}

impl tlottie_internal::FrameRenderer for VulkanRenderer<'_> {
  fn save_layer(&mut self) {
    let at = self.frame.contours.len();
    let index = self.frame.commands.len();
    self.frame.commands.push(RecordedCommand {
      op: RecordedOp::BeginLayer,
      start: at,
      end: at,
    });
    self.saves.push(index);
  }

  fn draw(&mut self, geometry: tlottie_internal::Geometry<'_>, paint: tlottie_internal::Paint<'_>) {
    let start = self.frame.contours.len();
    for contour in geometry.contours() {
      self.frame.contours.push(RecordedContour {
        points: contour.points().collect(),
        closed: contour.closed(),
      });
    }
    let op = match paint {
      tlottie_internal::Paint::Solid(solid) => RecordedOp::Solid(solid),
      tlottie_internal::Paint::Gradient(gradient) => RecordedOp::Gradient(gradient.clone()),
    };
    self.frame.commands.push(RecordedCommand {
      op,
      start,
      end: self.frame.contours.len(),
    });
  }

  fn apply_mask(&mut self, geometry: tlottie_internal::Geometry<'_>, mode: u8, inverted: bool, opacity: u8, first: bool, last: bool) {
    let start = self.frame.contours.len();
    for contour in geometry.contours() {
      self.frame.contours.push(RecordedContour {
        points: contour.points().collect(),
        closed: contour.closed(),
      });
    }
    self.frame.commands.push(RecordedCommand {
      op: RecordedOp::Mask { mode, inverted, opacity, first, last },
      start,
      end: self.frame.contours.len(),
    });
  }

  fn end_layer(&mut self, composite: tlottie_internal::Composite) {
    match composite {
      tlottie_internal::Composite::Over { opacity } => {
        let _ = self.saves.pop();
        let at = self.frame.contours.len();
        self.frame.commands.push(RecordedCommand {
          op: RecordedOp::EndLayer { opacity },
          start: at,
          end: at,
        });
      }
      tlottie_internal::Composite::Matte { kind, opacity, source_opacity } => {
        let target = self.saves.pop();
        let source = self.saves.pop();
        if let Some(index) = source {
          if let Some(saved) = self.frame.commands.get_mut(index) {
            saved.op = RecordedOp::BeginMatte;
          }
        }
        if let Some(index) = target {
          if let Some(saved) = self.frame.commands.get_mut(index) {
            saved.op = RecordedOp::BeginMatteTarget;
          }
        }
        let at = self.frame.contours.len();
        self.frame.commands.push(RecordedCommand {
          op: RecordedOp::EndMatte { kind, opacity, source_opacity },
          start: at,
          end: at,
        });
      }
    }
  }
}

/// Vulkan helper result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from command recording helpers.
#[derive(Debug)]
pub enum Error {
  /// Invalid dimensions or rectangle bounds.
  BadTarget,
  /// Lottie frame evaluation failed before command recording.
  Lottie(crate::Error),
  /// Evaluated frame data cannot fit in the Vulkan upload path.
  FrameTooLarge,
  /// Vulkan resource creation failed.
  Vulkan(&'static str, vk::Result),
}

impl std::fmt::Display for Error {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Error::BadTarget => write!(f, "bad Vulkan draw target"),
      Error::Lottie(e) => write!(f, "Lottie evaluation failed: {e}"),
      Error::FrameTooLarge => write!(f, "evaluated frame is too large for Vulkan upload"),
      Error::Vulkan(operation, result) => write!(f, "{operation} failed: {result:?}"),
    }
  }
}

impl std::error::Error for Error {}

impl From<crate::Error> for Error {
  fn from(value: crate::Error) -> Self {
    Error::Lottie(value)
  }
}

/// Integer rectangle in target pixels.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
  /// Left edge in pixels.
  pub x: u32,
  /// Top edge in pixels.
  pub y: u32,
  /// Width in pixels.
  pub w: u32,
  /// Height in pixels.
  pub h: u32,
}

/// Host-owned premultiplied RGBA8 buffer target.
///
/// The buffer is interpreted as `width * height` tightly packed `u32` pixels in
/// tlottie's `0xAABBGGRR` convention (`[R, G, B, A]` in little-endian memory).
#[derive(Clone, Copy, Debug)]
pub struct BufferTarget {
  /// Vulkan buffer supplied and owned by the host.
  pub buffer: vk::Buffer,
  /// Target width in pixels.
  pub width: u32,
  /// Target height in pixels.
  pub height: u32,
  /// Total buffer capacity in bytes.
  pub bytes: vk::DeviceSize,
}

/// Host-owned image target for Vulkan rendering.
///
/// The image must use `VK_FORMAT_B8G8R8A8_UNORM` or
/// `VK_FORMAT_R8G8B8A8_UNORM`.
#[derive(Clone, Copy)]
pub struct ImageTarget {
  /// Vulkan image supplied and owned by the host.
  pub image: vk::Image,
  /// Pixel format of `image`.
  pub format: vk::Format,
  /// Target width in pixels.
  pub width: u32,
  /// Target height in pixels.
  pub height: u32,
  /// Layout of the image when recording begins.
  pub layout: vk::ImageLayout,
  /// Layout required when recording ends.
  pub final_layout: vk::ImageLayout,
  /// Optional S8 image used by the stencil-and-cover profiling path.
  pub stencil_image: Option<vk::Image>,
  /// Optional multisampled BGRA color attachment resolved into `image`.
  pub multisample_image: Option<vk::Image>,
  /// Optional multisampled tile-local accumulator for isolated alpha groups.
  pub group_multisample_image: Option<vk::Image>,
}

/// Timestamp query slots written while recording the compute renderer.
///
/// The caller must reset seven consecutive queries before recording. The slots
/// contain timestamps after uploads, bin count/prefix/scatter,
/// coverage/composition, and output copy; `first` is reserved for the caller's
/// top-of-pipe timestamp.
#[derive(Clone, Copy)]
pub struct ProfileQueries {
  /// Host-owned timestamp query pool.
  pub pool: vk::QueryPool,
  /// First query index reserved for this recording operation.
  pub first: u32,
}

/// Vulkan renderer state for command recording.
///
/// This object may eventually own renderer-local resources such as shader
/// modules, descriptor layouts, pipelines, upload rings, and caches. It does
/// not create or own the Vulkan instance, physical device, logical device,
/// queues, command pools, command buffers, or output targets.
pub struct VulkanRenderer<'a> {
  device: &'a ash::Device,
  walker: crate::renderer::frame::FrameWalker,
  frame: EvaluatedFrame,
  saves: Vec<usize>,
  geometry: GeometryCache,
  cache_stats: CacheStats,
  render_pass: vk::RenderPass,
  descriptor_set_layout: vk::DescriptorSetLayout,
  descriptor_pool: vk::DescriptorPool,
  descriptor_set: vk::DescriptorSet,
  pipeline_layout: vk::PipelineLayout,
  pipeline: vk::Pipeline,
  stencil_nonzero_pipeline: vk::Pipeline,
  stencil_evenodd_pipeline: vk::Pipeline,
  group_composite_pipeline: vk::Pipeline,
  group_sampler: vk::Sampler,
  raster_order_groups: bool,
  bin_pipeline: vk::Pipeline,
  compute_pipeline: vk::Pipeline,
  simple_compute_pipeline: vk::Pipeline,
  target_framebuffers: Vec<TargetFramebuffer>,
  multi_draw_indirect: bool,
  mode: RendererMode,
  scene_layout: SceneLayout,
  uploaded_scene: UploadedScene,
  retained_bins: RetainedBins,
}

/// Backward-compatible name for [`VulkanRenderer`].
#[deprecated(note = "renamed to VulkanRenderer")]
pub type Renderer<'a> = VulkanRenderer<'a>;

/// GPU rasterization path used by [`VulkanRenderer::record`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RendererMode {
  /// Portable per-pixel compute ownership with exact fill-rule ordering.
  #[default]
  Compute,
  /// Direct triangle decomposition for profiling simple convex content.
  Triangles,
  /// Hardware stencil winding followed by one paint cover draw.
  StencilCover,
}

/// Geometry reuse statistics for the most recently prepared frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
  /// Contours whose point storage was reused without an upload.
  pub reused_contours: u32,
  /// Reused contours represented by a new translation instead of point uploads.
  pub translated_contours: u32,
  /// Resident point values retained by translation-only contour updates.
  pub translated_points: u32,
  /// Reused contours represented by a verified non-translation affine map.
  pub affine_contours: u32,
  /// Resident point values retained by verified affine contour updates.
  pub affine_points: u32,
  /// Contours updated in place because their point count stayed constant.
  pub updated_contours: u32,
  /// Contours assigned a new or differently sized allocation.
  pub allocated_contours: u32,
  /// Point values that need uploading for the prepared frame.
  pub dirty_points: u32,
  /// Number of coalesced point upload ranges.
  pub dirty_ranges: u32,
  /// Total point capacity currently retained by the arena.
  pub arena_points: u32,
  /// Indirect contour draws emitted for the frame.
  pub draws: u32,
  /// Solid contour draws currently submitted to the Vulkan pipeline.
  pub solid_draws: u32,
  /// Gradient contour draws submitted to the Vulkan pipeline.
  pub gradient_draws: u32,
  /// Scene words uploaded for the most recently prepared frame.
  pub scene_upload_bytes: u64,
  /// Coalesced scene upload ranges for the most recently prepared frame.
  pub scene_upload_ranges: u32,
  /// Point and contour bytes uploaded for the most recent frame.
  pub geometry_upload_bytes: u64,
  /// Paint record and gradient LUT bytes uploaded for the most recent frame.
  pub paint_upload_bytes: u64,
  /// Tile and edge-bin bytes uploaded for the most recent frame.
  pub bin_upload_bytes: u64,
  /// Whether the most recent frame reused GPU-generated tile and edge bins.
  pub reused_bins: bool,
  /// Whether the most recent frame used the stack-free compute shader.
  pub simple_compute: bool,
  /// Whether the most recent frame used hardware stencil winding and cover draws.
  pub stencil_cover: bool,
}

impl<'a> VulkanRenderer<'a> {
  /// Initializes renderer-local Vulkan resources against a host-owned device.
  ///
  /// The renderer owns pipelines, descriptor state, CPU geometry allocation,
  /// and retained upload plans. Vulkan buffers and images remain host-owned.
  pub fn new(device: &'a ash::Device) -> Result<VulkanRenderer<'a>> {
    Self::new_with_raster_order_groups(device, false)
  }

  /// Initializes the renderer with ordered tile-local alpha-group blending.
  /// The host must enable rasterization-order color attachment access and
  /// sample-rate shading on the Vulkan device before selecting this path.
  pub fn new_with_raster_order_groups(device: &'a ash::Device, enabled: bool) -> Result<VulkanRenderer<'a>> {
    let resources = create_triangle_pipeline(device, enabled)?;
    let bin_pipeline = match create_bin_pipeline(device, resources.pipeline_layout) {
      Ok(pipeline) => pipeline,
      Err(error) => {
        // SAFETY: resources were just created and are not in use.
        unsafe { destroy_triangle_pipeline(device, &resources) };
        return Err(error);
      }
    };
    let compute_pipeline = match create_compute_pipeline(device, resources.pipeline_layout) {
      Ok(pipeline) => pipeline,
      Err(error) => {
        // SAFETY: resources were just created and are not in use.
        unsafe {
          device.destroy_pipeline(bin_pipeline, None);
          destroy_triangle_pipeline(device, &resources);
        }
        return Err(error);
      }
    };
    let simple_compute_pipeline = match create_simple_compute_pipeline(device, resources.pipeline_layout) {
      Ok(pipeline) => pipeline,
      Err(error) => {
        // SAFETY: resources were just created and are not in use.
        unsafe {
          device.destroy_pipeline(compute_pipeline, None);
          device.destroy_pipeline(bin_pipeline, None);
          destroy_triangle_pipeline(device, &resources);
        }
        return Err(error);
      }
    };
    Ok(VulkanRenderer {
      device,
      walker: Default::default(),
      frame: Default::default(),
      saves: Vec::new(),
      geometry: GeometryCache::default(),
      cache_stats: CacheStats::default(),
      render_pass: resources.render_pass,
      descriptor_set_layout: resources.descriptor_set_layout,
      descriptor_pool: resources.descriptor_pool,
      descriptor_set: resources.descriptor_set,
      pipeline_layout: resources.pipeline_layout,
      pipeline: resources.pipeline,
      stencil_nonzero_pipeline: resources.stencil_nonzero_pipeline,
      stencil_evenodd_pipeline: resources.stencil_evenodd_pipeline,
      group_composite_pipeline: resources.group_composite_pipeline,
      group_sampler: resources.group_sampler,
      raster_order_groups: enabled,
      bin_pipeline,
      compute_pipeline,
      simple_compute_pipeline,
      target_framebuffers: Vec::new(),
      multi_draw_indirect: false,
      mode: RendererMode::default(),
      scene_layout: SceneLayout::default(),
      uploaded_scene: UploadedScene::default(),
      retained_bins: RetainedBins::default(),
    })
  }

  /// Returns geometry reuse statistics for the last call to [`Self::record`].
  pub fn cache_stats(&self) -> CacheStats {
    self.cache_stats
  }

  /// Selects the GPU rasterization path for subsequent recordings.
  pub fn set_mode(&mut self, mode: RendererMode) {
    self.mode = mode;
  }

  /// Enables multi-draw indirect submission for stencil contours.
  ///
  /// The host must enable both `multiDrawIndirect` and
  /// `drawIndirectFirstInstance` when creating the Vulkan device.
  pub fn set_multi_draw_indirect(&mut self, enabled: bool) {
    self.multi_draw_indirect = enabled;
  }

  fn target_framebuffer(&mut self, target: ImageTarget) -> Result<(vk::Framebuffer, vk::Framebuffer, Option<vk::ImageView>)> {
    let multisample_image = if self.raster_order_groups {
      vk::Image::null()
    } else {
      target.multisample_image.ok_or(Error::BadTarget)?
    };
    let stencil_image = target.stencil_image.ok_or(Error::BadTarget)?;
    let group_image = target.group_multisample_image.unwrap_or(vk::Image::null());
    if self.raster_order_groups && group_image == vk::Image::null() {
      return Err(Error::BadTarget);
    }
    if let Some(cached) = self.target_framebuffers.iter().find(|cached| {
      cached.resolve_image == target.image
        && cached.multisample_image == multisample_image
        && cached.stencil_image == stencil_image
        && cached.group_image == group_image
        && cached.width == target.width
        && cached.height == target.height
    }) {
      return Ok((cached.framebuffer, cached.group_framebuffer, if self.raster_order_groups { cached.views.get(2).copied() } else { None }));
    }
    let (views, framebuffer, group_framebuffer) = if self.raster_order_groups {
      create_raster_order_group_framebuffer(self.device, self.render_pass, target.image, stencil_image, group_image, target.width, target.height)?
    } else {
      let (views, framebuffer) = create_target_framebuffer(self.device, self.render_pass, target.image, multisample_image, stencil_image, None, target.width, target.height)?;
      (views, framebuffer, vk::Framebuffer::null())
    };
    let group_view = if self.raster_order_groups { views.get(2).copied() } else { None };
    self.target_framebuffers.push(TargetFramebuffer {
      resolve_image: target.image,
      multisample_image,
      stencil_image,
      group_image,
      width: target.width,
      height: target.height,
      views,
      framebuffer,
      group_framebuffer,
    });
    Ok((framebuffer, group_framebuffer, group_view))
  }

  /// Records a simple premultiplied RGBA8 rectangle draw into a host-owned buffer.
  ///
  /// This is a phase-0 bring-up primitive, not the final vector renderer.
  ///
  /// # Safety
  /// `cmd` and `target.buffer` must belong to the device used to construct
  /// this renderer. `cmd` must be in recording state. `target.buffer` must be
  /// large enough for `target.width * target.height * 4` bytes and have
  /// `VK_BUFFER_USAGE_TRANSFER_DST_BIT`.
  pub unsafe fn record_rgba_rect(&mut self, cmd: vk::CommandBuffer, target: BufferTarget, rect: Rect, rgba: u32) -> Result<()> {
    // SAFETY: forwarded from this method's caller contract.
    unsafe { cmd_draw_rgba_rect(self.device, cmd, target.buffer, target.width, target.height, rect, rgba) }
  }

  /// Records a simple premultiplied RGBA8 rectangle draw into a host-owned image.
  ///
  /// This is a phase-0 bring-up primitive. It uses `scratch` as temporary
  /// transfer storage, copies it into `target.image`, and leaves the image in
  /// `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` for host readback.
  ///
  /// # Safety
  /// `cmd`, `scratch.buffer`, and `target.image` must belong to the device
  /// used to construct this renderer. `cmd` must be in recording state.
  /// `scratch.buffer` must be large enough for the target and have transfer
  /// source and destination usage.
  /// `target.image` must be a `VK_FORMAT_B8G8R8A8_UNORM` 2D image with
  /// transfer source/destination usage, one mip level, one array layer, and
  /// the layout declared by `target.layout`.
  pub unsafe fn record_rgba_rect_image(&mut self, cmd: vk::CommandBuffer, scratch: BufferTarget, target: ImageTarget, rect: Rect, rgba: u32) -> Result<()> {
    if scratch.width != target.width || scratch.height != target.height {
      return Err(Error::BadTarget);
    }
    // SAFETY: forwarded from this method's caller contract.
    unsafe {
      cmd_draw_rgba_rect(self.device, cmd, scratch.buffer, scratch.width, scratch.height, rect, rgba)?;
      cmd_copy_rgba_buffer_to_image(self.device, cmd, scratch.buffer, target)
    }
  }

  /// Records a Lottie frame into a host-owned image.
  ///
  /// This is the real renderer entry point: hosts pass the parsed
  /// composition, frame, output target, and [`RenderOptions`]. The current
  /// default compute path uploads compact evaluated contours and ordered
  /// paint commands, bins them into tiles, evaluates analytic nonzero or
  /// even-odd coverage, samples solids and gradients, and composites with
  /// pixel-local ownership. The optional triangle mode remains available for
  /// profiling simple convex content. No CPU-rasterized frame is uploaded.
  ///
  /// # Safety
  /// `cmd`, `scratch.buffer`, and `target.image` must belong to the device
  /// used to construct this renderer. `cmd` must be in recording state.
  /// `scratch.buffer` must be large enough for the target and have transfer
  /// destination and storage-buffer usage.
  /// `target.image` must be a `VK_FORMAT_B8G8R8A8_UNORM` 2D image with
  /// transfer source/destination usage, one mip level, one array layer, and
  /// the layout declared by `target.layout`. Triangle mode additionally
  /// requires color-attachment usage and a fresh image.
  pub unsafe fn record(&mut self, cmd: vk::CommandBuffer, scratch: BufferTarget, target: ImageTarget, composition: &Composition, frame: f32, options: RenderOptions) -> Result<()> {
    unsafe { self.record_internal(cmd, scratch, target, composition, frame, options, None) }
  }

  /// Records a frame and writes stage timestamps to caller-owned queries.
  ///
  /// # Safety
  /// This has the same requirements as [`Self::record`]. `profile.pool` must
  /// belong to the same device and contain five reset queries beginning at
  /// `profile.first`.
  pub unsafe fn record_profiled(
    &mut self,
    cmd: vk::CommandBuffer,
    scratch: BufferTarget,
    target: ImageTarget,
    composition: &Composition,
    frame: f32,
    options: RenderOptions,
    profile: ProfileQueries,
  ) -> Result<()> {
    unsafe { self.record_internal(cmd, scratch, target, composition, frame, options, Some(profile)) }
  }

  unsafe fn record_internal(
    &mut self,
    cmd: vk::CommandBuffer,
    scratch: BufferTarget,
    target: ImageTarget,
    composition: &Composition,
    frame: f32,
    options: RenderOptions,
    profile: Option<ProfileQueries>,
  ) -> Result<()> {
    if scratch.width != target.width || scratch.height != target.height {
      return Err(Error::BadTarget);
    }
    self.frame = EvaluatedFrame::default();
    self.saves.clear();
    let mut walker = core::mem::take(&mut self.walker);
    let walk_result = walker.render(composition, frame, target.width, target.height, options, self);
    self.walker = walker;
    walk_result?;
    let prepared = self.geometry.prepare(&self.frame)?;
    self.cache_stats = prepared.stats;
    let mut group_depth = 0u32;
    let stencil_compatible = prepared.paints.iter().all(|paint| match paint.paint_kind {
      0 | 1 => true,
      2 if self.raster_order_groups && group_depth == 0 => {
        group_depth = 1;
        true
      }
      3 if self.raster_order_groups && group_depth == 1 => {
        group_depth = 0;
        true
      }
      _ => false,
    }) && group_depth == 0;
    // SAFETY: forwarded from this method's caller contract.
    unsafe {
      if options.alpha_only {
        return self.record_compute(cmd, scratch, target, &prepared, options.antialias, true, profile);
      }
      match self.mode {
        RendererMode::Compute => self.record_compute(cmd, scratch, target, &prepared, options.antialias, false, profile),
        RendererMode::Triangles => {
          if stencil_compatible {
            self.record_triangles(cmd, scratch, target, &prepared, profile)
          } else {
            self.record_compute(cmd, scratch, target, &prepared, options.antialias, false, profile)
          }
        }
        RendererMode::StencilCover => {
          if stencil_compatible {
            self.cache_stats.stencil_cover = true;
            self.record_triangles(cmd, scratch, target, &prepared, profile)
          } else {
            self.record_compute(cmd, scratch, target, &prepared, options.antialias, false, profile)
          }
        }
      }
    }
  }

  unsafe fn record_compute(
    &mut self,
    cmd: vk::CommandBuffer,
    scratch: BufferTarget,
    target: ImageTarget,
    prepared: &PreparedGeometry,
    antialias: bool,
    alpha_only: bool,
    profile: Option<ProfileQueries>,
  ) -> Result<()> {
    let mut scene = build_compute_scene(
      target.width,
      target.height,
      &self.geometry.arena.points,
      &self.geometry.contours,
      prepared,
      antialias,
      alpha_only,
      &mut self.scene_layout,
    )?;
    match target.format {
      vk::Format::B8G8R8A8_UNORM => scene.push.compact_flags &= !4,
      vk::Format::R8G8B8A8_UNORM => scene.push.compact_flags |= 4,
      _ => return Err(Error::BadTarget),
    }
    let simple_compute = prepared.paints.iter().all(|paint| paint.paint_kind < 2);
    self.cache_stats.simple_compute = simple_compute;
    let output_bytes = vk::DeviceSize::from(scene.layout.output_words).checked_mul(4).ok_or(Error::FrameTooLarge)?;
    let required = vk::DeviceSize::from(scene.layout.total_words).checked_mul(4).ok_or(Error::FrameTooLarge)?;
    if required > scratch.bytes {
      return Err(Error::FrameTooLarge);
    }
    let retained_buffer = self.uploaded_scene.buffer == scratch.buffer;
    let bin_key = compute_bin_key(&scene)?;
    let reuse_bins = self.retained_bins.buffer == scratch.buffer && self.retained_bins.key == Some(bin_key) && retained_bin_layout_matches(self.retained_bins.layout, scene.layout);
    self.cache_stats.reused_bins = reuse_bins;
    let mut uploaded_bytes = 0u64;
    let mut uploaded_ranges = 0u32;
    let mut domain_bytes = [0u64; 3];
    for (section_index, words) in scene.sections.iter().enumerate() {
      let retained_section = retained_buffer && self.uploaded_scene.layout.offset(section_index)? == scene.layout.offset(section_index)?;
      let previous = retained_section.then(|| self.uploaded_scene.sections.get(section_index)).flatten().map(Vec::as_slice);
      let dirty = scene_dirty_ranges(previous, words);
      for range in &dirty {
        let range_words = words.get(range.clone()).ok_or(Error::FrameTooLarge)?;
        let section_offset = scene.layout.offsets.get(section_index).copied().ok_or(Error::FrameTooLarge)?;
        let word_offset = section_offset.checked_add(u32::try_from(range.start).map_err(|_| Error::FrameTooLarge)?).ok_or(Error::FrameTooLarge)?;
        let bytes_len = std::mem::size_of_val(range_words);
        // SAFETY: u32 scene words are initialized and byte-compatible.
        let bytes = unsafe { std::slice::from_raw_parts(range_words.as_ptr().cast::<u8>(), bytes_len) };
        // SAFETY: layout capacities and the host buffer size were
        // checked above; every range is four-byte aligned.
        unsafe {
          cmd_upload_buffer_bytes(
            self.device,
            cmd,
            scratch.buffer,
            vk::DeviceSize::from(word_offset).checked_mul(4).ok_or(Error::FrameTooLarge)?,
            bytes,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
          )?;
        }
        let bytes_len = bytes_len as u64;
        uploaded_bytes = uploaded_bytes.checked_add(bytes_len).ok_or(Error::FrameTooLarge)?;
        let domain = scene_section_domain(section_index);
        let slot = domain_bytes.get_mut(domain).ok_or(Error::FrameTooLarge)?;
        *slot = slot.checked_add(bytes_len).ok_or(Error::FrameTooLarge)?;
      }
      uploaded_ranges = uploaded_ranges.checked_add(u32::try_from(dirty.len()).map_err(|_| Error::FrameTooLarge)?).ok_or(Error::FrameTooLarge)?;
    }
    self.cache_stats.scene_upload_bytes = uploaded_bytes;
    self.cache_stats.scene_upload_ranges = uploaded_ranges;
    let [geometry_bytes, paint_bytes, bin_bytes] = domain_bytes;
    self.cache_stats.geometry_upload_bytes = geometry_bytes;
    self.cache_stats.paint_upload_bytes = paint_bytes;
    self.cache_stats.bin_upload_bytes = bin_bytes;
    self.uploaded_scene.buffer = scratch.buffer;
    self.uploaded_scene.layout = scene.layout;
    self.uploaded_scene.sections.clone_from(&scene.sections);

    let buffer_info = vk::DescriptorBufferInfo::builder().buffer(scratch.buffer).offset(0).range(vk::WHOLE_SIZE).build();
    let descriptor_write = vk::WriteDescriptorSet::builder()
      .dst_set(self.descriptor_set)
      .dst_binding(0)
      .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
      .buffer_info(std::slice::from_ref(&buffer_info))
      .build();
    // SAFETY: descriptor set and buffer belong to this device.
    unsafe { self.device.update_descriptor_sets(&[descriptor_write], &[]) };
    if let Some(profile) = profile {
      unsafe {
        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TRANSFER, profile.pool, profile.first + 1);
      }
    }
    // SAFETY: ComputePush is repr(C), fully initialized, and fits the
    // pipeline layout's push-constant range.
    let push_bytes = unsafe { std::slice::from_raw_parts((&scene.push as *const ComputePush).cast::<u8>(), std::mem::size_of::<ComputePush>()) };
    let tile_count = scene.push.tiles_x.checked_mul(scene.push.tiles_y).ok_or(Error::FrameTooLarge)?;
    let edge_bin_count = scene.push.paint_count.checked_mul(scene.push.tiles_y).ok_or(Error::FrameTooLarge)?;
    let bin_invocations = tile_count.max(edge_bin_count);
    // SAFETY: pipelines, descriptor set, buffer, and command buffer are live.
    unsafe {
      self
        .device
        .cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.descriptor_set], &[]);
      if !reuse_bins {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.bin_pipeline);
        let tile_blocks = tile_count.saturating_add(63) / 64;
        let edge_blocks = edge_bin_count.saturating_add(63) / 64;
        for phase in 0..5u32 {
          let mut bin_push = scene.push;
          bin_push.antialias = phase;
          let bin_push_bytes = std::slice::from_raw_parts((&bin_push as *const ComputePush).cast::<u8>(), std::mem::size_of::<ComputePush>());
          self.device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bin_push_bytes);
          let groups = match phase {
            1 => tile_blocks.saturating_add(edge_blocks),
            2 => 1,
            _ => bin_invocations.saturating_add(63) / 64,
          };
          self.device.cmd_dispatch(cmd, groups, 1, 1);
          let bins_ready = vk::BufferMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .buffer(scratch.buffer)
            .offset(vk::DeviceSize::from(scene.layout.offset(TILE_SECTION)?) * 4)
            .size(vk::WHOLE_SIZE)
            .build();
          self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[bins_ready],
            &[],
          );
          if let Some(query) = match phase {
            0 => Some(2),
            3 => Some(3),
            4 => Some(4),
            _ => None,
          } {
            if let Some(profile) = profile {
              self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, profile.pool, profile.first + query);
            }
          }
        }
      } else if let Some(profile) = profile {
        for query in 2..=4 {
          self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, profile.pool, profile.first + query);
        }
      }
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, if simple_compute { self.simple_compute_pipeline } else { self.compute_pipeline });
      self.device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
      self.device.cmd_dispatch(cmd, target.width.saturating_add(7) / 8, target.height.saturating_add(7) / 8, 1);
      if let Some(profile) = profile {
        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, profile.pool, profile.first + 5);
      }
    }
    if !reuse_bins {
      self.retained_bins = RetainedBins {
        buffer: scratch.buffer,
        layout: scene.layout,
        key: Some(bin_key),
      };
    }
    let barrier = vk::BufferMemoryBarrier::builder()
      .src_access_mask(vk::AccessFlags::SHADER_WRITE)
      .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
      .buffer(scratch.buffer)
      .offset(0)
      .size(output_bytes)
      .build();
    // SAFETY: compute dispatch precedes the transfer read in this command
    // buffer and writes the entire output range.
    unsafe {
      self.device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[barrier],
        &[],
      );
      cmd_copy_rgba_buffer_to_image(self.device, cmd, scratch.buffer, target)
    }?;
    if let Some(profile) = profile {
      unsafe {
        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TRANSFER, profile.pool, profile.first + 6);
      }
    }
    Ok(())
  }

  unsafe fn record_triangles(&mut self, cmd: vk::CommandBuffer, scratch: BufferTarget, target: ImageTarget, prepared: &PreparedGeometry, profile: Option<ProfileQueries>) -> Result<()> {
    if prepared.paints.iter().any(|paint| paint.paint_kind >= 4 || (paint.paint_kind >= 2 && !self.raster_order_groups)) {
      return Err(Error::BadTarget);
    }
    let points = &self.geometry.arena.points;
    let point_bytes = point_payload_bytes(points);
    let gradient_bytes = std::mem::size_of_val(prepared.gradient_luts.as_slice());
    let use_indirect = self.multi_draw_indirect && prepared.draw_data.windows(2).any(|draws| draws[0].paint_index == draws[1].paint_index);
    let draw_bytes = if use_indirect { std::mem::size_of_val(prepared.draw_data.as_slice()) } else { 0 };
    let indirect_bytes = if use_indirect { std::mem::size_of_val(prepared.fan_indirect.as_slice()) } else { 0 };
    let gradient_offset = point_bytes;
    let draw_offset = gradient_offset.checked_add(gradient_bytes).ok_or(Error::FrameTooLarge)?;
    let indirect_offset = draw_offset.checked_add(draw_bytes).ok_or(Error::FrameTooLarge)?;
    let upload_bytes = point_bytes
      .checked_add(gradient_bytes)
      .and_then(|bytes| bytes.checked_add(draw_bytes))
      .and_then(|bytes| bytes.checked_add(indirect_bytes))
      .ok_or(Error::FrameTooLarge)?;
    self.cache_stats.scene_upload_bytes = upload_bytes as u64;
    self.cache_stats.scene_upload_ranges = u32::from(point_bytes != 0)
      .saturating_add(u32::from(gradient_bytes != 0))
      .saturating_add(u32::from(draw_bytes != 0))
      .saturating_add(u32::from(indirect_bytes != 0));
    self.cache_stats.geometry_upload_bytes = point_bytes.saturating_add(draw_bytes).saturating_add(indirect_bytes) as u64;
    self.cache_stats.paint_upload_bytes = gradient_bytes as u64;
    self.cache_stats.bin_upload_bytes = 0;
    if upload_bytes as vk::DeviceSize > scratch.bytes {
      return Err(Error::FrameTooLarge);
    }
    // SAFETY: GpuPoint is repr(C), contains two initialized f32 values, and
    // has no invalid byte representation.
    let bytes = unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<u8>(), point_bytes) };
    // SAFETY: scratch belongs to this device and has transfer-dst and
    // vertex-buffer usage.
    unsafe {
      cmd_upload_buffer_bytes(self.device, cmd, scratch.buffer, 0, bytes, vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::VERTEX_SHADER)?;
    }
    // SAFETY: u32 LUT entries are initialized and have no invalid byte
    // representation.
    let gradient_luts = unsafe { std::slice::from_raw_parts(prepared.gradient_luts.as_ptr().cast::<u8>(), gradient_bytes) };
    unsafe {
      cmd_upload_buffer_bytes(
        self.device,
        cmd,
        scratch.buffer,
        gradient_offset as vk::DeviceSize,
        gradient_luts,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
      )?;
    }
    let draw_data = unsafe { std::slice::from_raw_parts(prepared.draw_data.as_ptr().cast::<u8>(), draw_bytes) };
    unsafe {
      cmd_upload_buffer_bytes(
        self.device,
        cmd,
        scratch.buffer,
        draw_offset as vk::DeviceSize,
        draw_data,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::VERTEX_SHADER,
      )?;
    }
    let indirect = unsafe { std::slice::from_raw_parts(prepared.fan_indirect.as_ptr().cast::<u8>(), indirect_bytes) };
    unsafe {
      cmd_upload_buffer_bytes(
        self.device,
        cmd,
        scratch.buffer,
        indirect_offset as vk::DeviceSize,
        indirect,
        vk::AccessFlags::INDIRECT_COMMAND_READ,
        vk::PipelineStageFlags::DRAW_INDIRECT,
      )?;
    }
    let (framebuffer, group_framebuffer, group_view) = self.target_framebuffer(target)?;

    let point_buffer = vk::DescriptorBufferInfo::builder().buffer(scratch.buffer).offset(0).range(vk::WHOLE_SIZE).build();
    let descriptor_write = vk::WriteDescriptorSet::builder()
      .dst_set(self.descriptor_set)
      .dst_binding(0)
      .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
      .buffer_info(std::slice::from_ref(&point_buffer))
      .build();
    let group_descriptor = group_view.map(|view| {
      vk::DescriptorImageInfo::builder()
        .sampler(self.group_sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build()
    });
    let group_write = group_descriptor.as_ref().map(|descriptor| {
      vk::WriteDescriptorSet::builder()
        .dst_set(self.descriptor_set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(descriptor))
        .build()
    });
    let mut descriptor_writes = vec![descriptor_write];
    if let Some(write) = group_write {
      descriptor_writes.push(write);
    }
    // SAFETY: descriptor set and buffer belong to this device.
    unsafe {
      self.device.update_descriptor_sets(&descriptor_writes, &[]);
      if let Some(profile) = profile {
        for query in 1..=4 {
          self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, profile.pool, profile.first + query);
        }
      }
    }

    if self.raster_order_groups {
      let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      };
      let (src_stage, src_access) = match target.layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::MEMORY_READ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
        _ => return Err(Error::BadTarget),
      };
      let to_clear = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(target.layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(target.image)
        .subresource_range(range)
        .build();
      unsafe {
        self
          .device
          .cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_clear]);
        self
          .device
          .cmd_clear_color_image(cmd, target.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
        let to_color = vk::ImageMemoryBarrier::builder()
          .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
          .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
          .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
          .image(target.image)
          .subresource_range(range)
          .build();
        self.device.cmd_pipeline_barrier(
          cmd,
          vk::PipelineStageFlags::TRANSFER,
          vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
          vk::DependencyFlags::empty(),
          &[],
          &[],
          &[to_color],
        );
      }
    }

    let clear = vec![
      vk::ClearValue {
        color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
      },
      vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
      },
    ];
    let clear = if self.raster_order_groups {
      clear
    } else {
      let mut clear = clear;
      clear.push(vk::ClearValue {
        color: vk::ClearColorValue { float32: [0.0; 4] },
      });
      clear
    };
    let begin = vk::RenderPassBeginInfo::builder()
      .render_pass(self.render_pass)
      .framebuffer(framebuffer)
      .render_area(vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
          width: target.width,
          height: target.height,
        },
      })
      .clear_values(&clear);
    // SAFETY: framebuffer, pipeline, and command buffer belong to this
    // device and remain live through command execution.
    unsafe {
      self.device.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
      self
        .device
        .cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.descriptor_set], &[]);
      self.device.cmd_set_viewport(
        cmd,
        0,
        &[vk::Viewport {
          x: 0.0,
          y: 0.0,
          width: target.width as f32,
          height: target.height as f32,
          min_depth: 0.0,
          max_depth: 1.0,
        }],
      );
      self.device.cmd_set_scissor(
        cmd,
        0,
        &[vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: vk::Extent2D {
            width: target.width,
            height: target.height,
          },
        }],
      );
    }

    let lut_word_base = u32::try_from(point_bytes / 4).map_err(|_| Error::FrameTooLarge)?;
    let draw_word_base = u32::try_from(draw_offset / 4).map_err(|_| Error::FrameTooLarge)?;
    let indirect_stride = u32::try_from(std::mem::size_of::<vk::DrawIndirectCommand>()).map_err(|_| Error::FrameTooLarge)?;
    let mut draw_cursor = 0usize;
    for (paint_index, paint) in prepared.paints.iter().enumerate() {
      let draw_start = draw_cursor;
      while prepared.draw_data.get(draw_cursor).is_some_and(|draw| draw.paint_index as usize == paint_index) {
        draw_cursor += 1;
      }
      let draw_count = draw_cursor.saturating_sub(draw_start);
      if paint.paint_kind == 2 {
        let group_image = target.group_multisample_image.ok_or(Error::BadTarget)?;
        let range = vk::ImageSubresourceRange {
          aspect_mask: vk::ImageAspectFlags::COLOR,
          base_mip_level: 0,
          level_count: 1,
          base_array_layer: 0,
          layer_count: 1,
        };
        unsafe {
          self.device.cmd_end_render_pass(cmd);
          let to_clear = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(group_image)
            .subresource_range(range)
            .build();
          self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_clear],
          );
          self
            .device
            .cmd_clear_color_image(cmd, group_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
          let to_color = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(group_image)
            .subresource_range(range)
            .build();
          self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_color],
          );
          let group_begin = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(group_framebuffer)
            .render_area(vk::Rect2D {
              offset: vk::Offset2D { x: 0, y: 0 },
              extent: vk::Extent2D {
                width: target.width,
                height: target.height,
              },
            })
            .clear_values(&clear);
          self.device.cmd_begin_render_pass(cmd, &group_begin, vk::SubpassContents::INLINE);
        }
        continue;
      }
      if paint.paint_kind == 3 {
        let group_image = target.group_multisample_image.ok_or(Error::BadTarget)?;
        let range = vk::ImageSubresourceRange {
          aspect_mask: vk::ImageAspectFlags::COLOR,
          base_mip_level: 0,
          level_count: 1,
          base_array_layer: 0,
          layer_count: 1,
        };
        unsafe {
          self.device.cmd_end_render_pass(cmd);
          let to_sample = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(group_image)
            .subresource_range(range)
            .build();
          self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_sample],
          );
          self.device.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
        }
        let composite = TrianglePush {
          viewport: [target.width as f32, target.height as f32],
          rgba: paint.argb,
          point_offset: 0,
          paint_kind: paint.paint_kind,
          gradient_kind: 0,
          lut_word_offset: 0,
          padding: 0,
          inverse0: [0.0; 4],
          inverse1: [0.0; 4],
          params0: [0.0; 4],
          params1: [0.0; 4],
          affine0: [0.0, 0.0, target.width as f32, target.height as f32],
          affine1: [0.0; 2],
        };
        let composite_bytes = unsafe { std::slice::from_raw_parts((&composite as *const TrianglePush).cast::<u8>(), std::mem::size_of::<TrianglePush>()) };
        unsafe {
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.group_composite_pipeline);
          self
            .device
            .cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, composite_bytes);
          self.device.cmd_draw(cmd, 6, 1, 0, 0);
        }
        continue;
      }
      let winding_pipeline = if paint.rule == 0 { self.stencil_nonzero_pipeline } else { self.stencil_evenodd_pipeline };
      if use_indirect && draw_count > 1 {
        let stencil = TrianglePush {
          viewport: [target.width as f32, target.height as f32],
          rgba: paint.argb,
          point_offset: draw_word_base,
          paint_kind: paint.paint_kind,
          gradient_kind: paint.gradient_kind,
          lut_word_offset: lut_word_base.checked_add(paint.lut_word_offset).ok_or(Error::FrameTooLarge)?,
          padding: 1,
          inverse0: paint.inverse0,
          inverse1: paint.inverse1,
          params0: paint.params0,
          params1: paint.params1,
          affine0: [0.0; 4],
          affine1: [0.0; 2],
        };
        let stencil_bytes = unsafe { std::slice::from_raw_parts((&stencil as *const TrianglePush).cast::<u8>(), std::mem::size_of::<TrianglePush>()) };
        unsafe {
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, winding_pipeline);
          self
            .device
            .cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, stencil_bytes);
          let byte_offset = indirect_offset
            .checked_add(draw_start.checked_mul(indirect_stride as usize).ok_or(Error::FrameTooLarge)?)
            .ok_or(Error::FrameTooLarge)?;
          self.device.cmd_draw_indirect(
            cmd,
            scratch.buffer,
            byte_offset as vk::DeviceSize,
            u32::try_from(draw_count).map_err(|_| Error::FrameTooLarge)?,
            indirect_stride,
          );
        }
      } else {
        unsafe {
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, winding_pipeline);
        }
        for draw in &prepared.draw_data[draw_start..draw_cursor] {
          let stencil = TrianglePush {
            viewport: [target.width as f32, target.height as f32],
            rgba: paint.argb,
            point_offset: draw.point_offset,
            paint_kind: paint.paint_kind,
            gradient_kind: paint.gradient_kind,
            lut_word_offset: lut_word_base.checked_add(paint.lut_word_offset).ok_or(Error::FrameTooLarge)?,
            padding: 0,
            inverse0: paint.inverse0,
            inverse1: paint.inverse1,
            params0: paint.params0,
            params1: paint.params1,
            affine0: [draw.transform[0], draw.transform[1], draw.transform[2], draw.transform[3]],
            affine1: [draw.transform[4], draw.transform[5]],
          };
          let stencil_bytes = unsafe { std::slice::from_raw_parts((&stencil as *const TrianglePush).cast::<u8>(), std::mem::size_of::<TrianglePush>()) };
          unsafe {
            self
              .device
              .cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, stencil_bytes);
            self.device.cmd_draw(cmd, draw.point_count, 1, 0, 0);
          }
        }
      }

      let cover = TrianglePush {
        viewport: [target.width as f32, target.height as f32],
        rgba: paint.argb,
        point_offset: 0,
        paint_kind: paint.paint_kind,
        gradient_kind: paint.gradient_kind,
        lut_word_offset: lut_word_base.checked_add(paint.lut_word_offset).ok_or(Error::FrameTooLarge)?,
        padding: 0,
        inverse0: paint.inverse0,
        inverse1: paint.inverse1,
        params0: paint.params0,
        params1: paint.params1,
        affine0: paint.bounds,
        affine1: [0.0; 2],
      };
      let cover_bytes = unsafe { std::slice::from_raw_parts((&cover as *const TrianglePush).cast::<u8>(), std::mem::size_of::<TrianglePush>()) };
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        self
          .device
          .cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, cover_bytes);
        self.device.cmd_draw(cmd, 6, 1, 0, 0);
      }
    }
    // SAFETY: a render pass is active on this command buffer.
    unsafe { self.device.cmd_end_render_pass(cmd) };
    if target.final_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
      let (dst_stage, dst_access) = match target.final_layout {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
        vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::empty()),
        _ => return Err(Error::BadTarget),
      };
      let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(dst_access)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(target.final_layout)
        .image(target.image)
        .subresource_range(vk::ImageSubresourceRange {
          aspect_mask: vk::ImageAspectFlags::COLOR,
          base_mip_level: 0,
          level_count: 1,
          base_array_layer: 0,
          layer_count: 1,
        })
        .build();
      unsafe {
        self
          .device
          .cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
      }
    }
    if let Some(profile) = profile {
      unsafe {
        for query in 5..=6 {
          self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_GRAPHICS, profile.pool, profile.first + query);
        }
      }
    }
    Ok(())
  }
}

impl Drop for VulkanRenderer<'_> {
  fn drop(&mut self) {
    // SAFETY: callers must keep renderer resources alive until recorded
    // commands complete. VulkanRenderer owns every handle destroyed here.
    unsafe {
      for target in self.target_framebuffers.drain(..) {
        self.device.destroy_framebuffer(target.framebuffer, None);
        self.device.destroy_framebuffer(target.group_framebuffer, None);
        for view in target.views {
          self.device.destroy_image_view(view, None);
        }
      }
      self.device.destroy_pipeline(self.pipeline, None);
      self.device.destroy_pipeline(self.stencil_nonzero_pipeline, None);
      self.device.destroy_pipeline(self.stencil_evenodd_pipeline, None);
      self.device.destroy_pipeline(self.group_composite_pipeline, None);
      self.device.destroy_sampler(self.group_sampler, None);
      self.device.destroy_pipeline(self.bin_pipeline, None);
      self.device.destroy_pipeline(self.compute_pipeline, None);
      self.device.destroy_pipeline(self.simple_compute_pipeline, None);
      self.device.destroy_pipeline_layout(self.pipeline_layout, None);
      self.device.destroy_descriptor_pool(self.descriptor_pool, None);
      self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
      self.device.destroy_render_pass(self.render_pass, None);
    }
  }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct TrianglePush {
  viewport: [f32; 2],
  rgba: u32,
  point_offset: u32,
  paint_kind: u32,
  gradient_kind: u32,
  lut_word_offset: u32,
  padding: u32,
  inverse0: [f32; 4],
  inverse1: [f32; 4],
  params0: [f32; 4],
  params1: [f32; 4],
  affine0: [f32; 4],
  affine1: [f32; 2],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct ComputePush {
  width: u32,
  height: u32,
  output_word: u32,
  point_word: u32,
  contour_word: u32,
  paint_word: u32,
  lut_word: u32,
  paint_count: u32,
  antialias: u32,
  tile_word: u32,
  tile_index_word: u32,
  tiles_x: u32,
  edge_bin_word: u32,
  edge_word: u32,
  tiles_y: u32,
  compact_flags: u32,
}

struct TrianglePipeline {
  render_pass: vk::RenderPass,
  descriptor_set_layout: vk::DescriptorSetLayout,
  descriptor_pool: vk::DescriptorPool,
  descriptor_set: vk::DescriptorSet,
  pipeline_layout: vk::PipelineLayout,
  pipeline: vk::Pipeline,
  stencil_nonzero_pipeline: vk::Pipeline,
  stencil_evenodd_pipeline: vk::Pipeline,
  group_composite_pipeline: vk::Pipeline,
  group_sampler: vk::Sampler,
}

/// Releases a fully constructed graphics pipeline bundle when a later compute
/// pipeline fails during renderer initialization.
///
/// # Safety
/// Every handle in `resources` must belong to `device` and must not be in use.
unsafe fn destroy_triangle_pipeline(device: &ash::Device, resources: &TrianglePipeline) {
  unsafe {
    device.destroy_pipeline(resources.pipeline, None);
    device.destroy_pipeline(resources.stencil_nonzero_pipeline, None);
    device.destroy_pipeline(resources.stencil_evenodd_pipeline, None);
    device.destroy_pipeline(resources.group_composite_pipeline, None);
    device.destroy_sampler(resources.group_sampler, None);
    device.destroy_pipeline_layout(resources.pipeline_layout, None);
    device.destroy_descriptor_pool(resources.descriptor_pool, None);
    device.destroy_descriptor_set_layout(resources.descriptor_set_layout, None);
    device.destroy_render_pass(resources.render_pass, None);
  }
}

struct TargetFramebuffer {
  resolve_image: vk::Image,
  multisample_image: vk::Image,
  stencil_image: vk::Image,
  group_image: vk::Image,
  width: u32,
  height: u32,
  views: Vec<vk::ImageView>,
  framebuffer: vk::Framebuffer,
  group_framebuffer: vk::Framebuffer,
}

fn create_raster_order_group_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
  let color = vk::AttachmentDescription2::builder()
    .format(vk::Format::B8G8R8A8_UNORM)
    .samples(vk::SampleCountFlags::TYPE_1)
    .load_op(vk::AttachmentLoadOp::LOAD)
    .store_op(vk::AttachmentStoreOp::STORE)
    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .build();
  let stencil = vk::AttachmentDescription2::builder()
    .format(vk::Format::S8_UINT)
    .samples(vk::SampleCountFlags::TYPE_1)
    .load_op(vk::AttachmentLoadOp::DONT_CARE)
    .store_op(vk::AttachmentStoreOp::DONT_CARE)
    .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
    .initial_layout(vk::ImageLayout::UNDEFINED)
    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    .build();
  let main_ref = vk::AttachmentReference2::builder()
    .attachment(0)
    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .aspect_mask(vk::ImageAspectFlags::COLOR)
    .build();
  let stencil_ref = vk::AttachmentReference2::builder()
    .attachment(1)
    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    .aspect_mask(vk::ImageAspectFlags::STENCIL)
    .build();
  let mut multisampled = vk::MultisampledRenderToSingleSampledInfoEXT::builder()
    .multisampled_render_to_single_sampled_enable(true)
    .rasterization_samples(STENCIL_SAMPLE_COUNT);
  let subpass = vk::SubpassDescription2::builder()
    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
    .color_attachments(std::slice::from_ref(&main_ref))
    .depth_stencil_attachment(&stencil_ref)
    .push_next(&mut multisampled)
    .build();
  let dependency = vk::SubpassDependency2::builder()
    .src_subpass(vk::SUBPASS_EXTERNAL)
    .dst_subpass(0)
    .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
    .build();
  let attachments = [color, stencil];
  let info = vk::RenderPassCreateInfo2::builder()
    .attachments(&attachments)
    .subpasses(std::slice::from_ref(&subpass))
    .dependencies(std::slice::from_ref(&dependency));
  unsafe { device.create_render_pass2(&info, None) }.map_err(|e| Error::Vulkan("vkCreateRenderPass2(alpha groups)", e))
}

fn create_triangle_pipeline(device: &ash::Device, raster_order_groups: bool) -> Result<TrianglePipeline> {
  let group_sampler = if raster_order_groups {
    let info = vk::SamplerCreateInfo::builder()
      .mag_filter(vk::Filter::NEAREST)
      .min_filter(vk::Filter::NEAREST)
      .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
      .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| Error::Vulkan("vkCreateSampler(alpha groups)", e))?
  } else {
    vk::Sampler::null()
  };
  let color_attachment = vk::AttachmentDescription::builder()
    .format(vk::Format::B8G8R8A8_UNORM)
    .samples(STENCIL_SAMPLE_COUNT)
    .load_op(vk::AttachmentLoadOp::CLEAR)
    .store_op(vk::AttachmentStoreOp::DONT_CARE)
    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
    .initial_layout(vk::ImageLayout::UNDEFINED)
    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .build();
  let stencil_attachment = vk::AttachmentDescription::builder()
    .format(vk::Format::S8_UINT)
    .samples(STENCIL_SAMPLE_COUNT)
    .load_op(vk::AttachmentLoadOp::DONT_CARE)
    .store_op(vk::AttachmentStoreOp::DONT_CARE)
    .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
    .initial_layout(vk::ImageLayout::UNDEFINED)
    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    .build();
  let resolve_attachment = vk::AttachmentDescription::builder()
    .format(vk::Format::B8G8R8A8_UNORM)
    .samples(vk::SampleCountFlags::TYPE_1)
    .load_op(vk::AttachmentLoadOp::DONT_CARE)
    .store_op(vk::AttachmentStoreOp::STORE)
    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
    .initial_layout(vk::ImageLayout::UNDEFINED)
    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .build();
  let group_attachment = vk::AttachmentDescription::builder()
    .format(vk::Format::B8G8R8A8_UNORM)
    .samples(STENCIL_SAMPLE_COUNT)
    .load_op(vk::AttachmentLoadOp::CLEAR)
    .store_op(vk::AttachmentStoreOp::DONT_CARE)
    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
    .initial_layout(vk::ImageLayout::UNDEFINED)
    .final_layout(vk::ImageLayout::GENERAL)
    .build();
  let color_ref = vk::AttachmentReference {
    attachment: 0,
    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
  };
  let stencil_ref = vk::AttachmentReference {
    attachment: 1,
    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };
  let resolve_ref = vk::AttachmentReference {
    attachment: 2,
    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
  };
  let group_ref = vk::AttachmentReference {
    attachment: 3,
    layout: vk::ImageLayout::GENERAL,
  };
  let unused_ref = vk::AttachmentReference {
    attachment: vk::ATTACHMENT_UNUSED,
    layout: vk::ImageLayout::UNDEFINED,
  };
  let color_refs = [color_ref, group_ref];
  let resolve_refs = [resolve_ref, unused_ref];
  let mut subpass_builder = vk::SubpassDescription::builder()
    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
    .depth_stencil_attachment(&stencil_ref);
  if raster_order_groups {
    subpass_builder = subpass_builder
      .flags(vk::SubpassDescriptionFlags::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_ARM)
      .color_attachments(&color_refs)
      .resolve_attachments(&resolve_refs)
      .input_attachments(std::slice::from_ref(&group_ref));
  } else {
    subpass_builder = subpass_builder
      .color_attachments(std::slice::from_ref(&color_ref))
      .resolve_attachments(std::slice::from_ref(&resolve_ref));
  }
  let subpass = subpass_builder.build();
  let dependency = vk::SubpassDependency::builder()
    .src_subpass(vk::SUBPASS_EXTERNAL)
    .dst_subpass(0)
    .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
    .build();
  let self_dependency = vk::SubpassDependency::builder()
    .src_subpass(0)
    .dst_subpass(0)
    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
    .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
    .dependency_flags(vk::DependencyFlags::BY_REGION)
    .build();
  let dependencies = if raster_order_groups { vec![dependency, self_dependency] } else { vec![dependency] };
  let mut attachments = vec![color_attachment, stencil_attachment, resolve_attachment];
  if raster_order_groups {
    attachments.push(group_attachment);
  }
  let render_pass_info = vk::RenderPassCreateInfo::builder()
    .attachments(&attachments)
    .subpasses(std::slice::from_ref(&subpass))
    .dependencies(&dependencies);
  // SAFETY: create info references live stack data for this call.
  let mut render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }.map_err(|e| Error::Vulkan("vkCreateRenderPass", e))?;
  if raster_order_groups {
    unsafe { device.destroy_render_pass(render_pass, None) };
    render_pass = create_raster_order_group_render_pass(device)?;
  }

  let descriptor_binding = vk::DescriptorSetLayoutBinding::builder()
    .binding(0)
    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
    .descriptor_count(1)
    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
    .build();
  let group_binding = vk::DescriptorSetLayoutBinding::builder()
    .binding(1)
    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .descriptor_count(1)
    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    .build();
  let mut descriptor_bindings = vec![descriptor_binding];
  if raster_order_groups {
    descriptor_bindings.push(group_binding);
  }
  let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_bindings);
  // SAFETY: create info references live stack data for this call.
  let descriptor_set_layout = match unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) } {
    Ok(layout) => layout,
    Err(e) => {
      // SAFETY: render pass belongs to this device and is unused.
      unsafe { device.destroy_render_pass(render_pass, None) };
      return Err(Error::Vulkan("vkCreateDescriptorSetLayout", e));
    }
  };

  let push_range = vk::PushConstantRange::builder()
    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
    .offset(0)
    .size(std::mem::size_of::<TrianglePush>() as u32)
    .build();
  let layout_info = vk::PipelineLayoutCreateInfo::builder()
    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
    .push_constant_ranges(std::slice::from_ref(&push_range));
  // SAFETY: create info is valid and device is live.
  let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
    Ok(layout) => layout,
    Err(e) => {
      // SAFETY: render pass was created above and is unused.
      unsafe {
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_render_pass(render_pass, None);
      }
      return Err(Error::Vulkan("vkCreatePipelineLayout", e));
    }
  };

  let vertex = create_shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv")))?;
  let cover_vertex = create_shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/cover.vert.spv")))?;
  let fragment = create_shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv")))?;
  let group_composite_fragment = create_shader_module(device, include_bytes!("shaders/group-composite.frag.spv"))?;
  let stencil_fragment = create_shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/stencil.frag.spv")))?;

  let cover_stages = [
    vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::VERTEX)
      .module(cover_vertex)
      .name(c"vs_cover")
      .build(),
    vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::FRAGMENT)
      .module(fragment)
      .name(c"fs_main")
      .build(),
  ];
  let stencil_stages = [
    vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::VERTEX).module(vertex).name(c"vs_main").build(),
    vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::FRAGMENT)
      .module(stencil_fragment)
      .name(c"fs_stencil")
      .build(),
  ];
  let group_composite_stages = [
    vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::VERTEX)
      .module(cover_vertex)
      .name(c"vs_cover")
      .build(),
    vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::FRAGMENT)
      .module(group_composite_fragment)
      .name(c"main")
      .build(),
  ];
  let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();
  let cover_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
  let winding_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_FAN);
  let viewport = vk::PipelineViewportStateCreateInfo::builder().viewport_count(1).scissor_count(1);
  let raster = vk::PipelineRasterizationStateCreateInfo::builder()
    .polygon_mode(vk::PolygonMode::FILL)
    .cull_mode(vk::CullModeFlags::NONE)
    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
    .line_width(1.0);
  let multisample = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(STENCIL_SAMPLE_COUNT);
  let blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
    .blend_enable(true)
    .src_color_blend_factor(vk::BlendFactor::ONE)
    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
    .color_blend_op(vk::BlendOp::ADD)
    .src_alpha_blend_factor(vk::BlendFactor::ONE)
    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
    .alpha_blend_op(vk::BlendOp::ADD)
    .color_write_mask(vk::ColorComponentFlags::RGBA)
    .build();
  let no_color_attachment = vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::empty()).build();
  let main_blends = [blend_attachment, no_color_attachment];
  let no_blends = [no_color_attachment, no_color_attachment];
  let blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&main_blends[..1]);
  let no_color = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&no_blends[..1]);
  let cover_stencil = vk::StencilOpState::builder()
    .fail_op(vk::StencilOp::KEEP)
    .pass_op(vk::StencilOp::ZERO)
    .depth_fail_op(vk::StencilOp::KEEP)
    .compare_op(vk::CompareOp::NOT_EQUAL)
    .compare_mask(0xff)
    .write_mask(0xff)
    .reference(0)
    .build();
  let cover_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder().stencil_test_enable(true).front(cover_stencil).back(cover_stencil);
  let winding_front = vk::StencilOpState::builder()
    .fail_op(vk::StencilOp::KEEP)
    .pass_op(vk::StencilOp::INCREMENT_AND_WRAP)
    .depth_fail_op(vk::StencilOp::KEEP)
    .compare_op(vk::CompareOp::ALWAYS)
    .compare_mask(0xff)
    .write_mask(0xff)
    .build();
  let winding_back = vk::StencilOpState {
    pass_op: vk::StencilOp::DECREMENT_AND_WRAP,
    ..winding_front
  };
  let winding_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder().stencil_test_enable(true).front(winding_front).back(winding_back);
  let parity = vk::StencilOpState {
    pass_op: vk::StencilOp::INVERT,
    ..winding_front
  };
  let parity_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder().stencil_test_enable(true).front(parity).back(parity);
  let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
  let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic);
  let cover_info = vk::GraphicsPipelineCreateInfo::builder()
    .stages(&cover_stages)
    .vertex_input_state(&vertex_input)
    .input_assembly_state(&cover_assembly)
    .viewport_state(&viewport)
    .rasterization_state(&raster)
    .multisample_state(&multisample)
    .depth_stencil_state(&cover_depth_stencil)
    .color_blend_state(&blend)
    .dynamic_state(&dynamic_state)
    .layout(pipeline_layout)
    .render_pass(render_pass)
    .subpass(0)
    .build();
  let winding_base = || {
    vk::GraphicsPipelineCreateInfo::builder()
      .stages(&stencil_stages)
      .vertex_input_state(&vertex_input)
      .input_assembly_state(&winding_assembly)
      .viewport_state(&viewport)
      .rasterization_state(&raster)
      .multisample_state(&multisample)
      .color_blend_state(&no_color)
      .dynamic_state(&dynamic_state)
      .layout(pipeline_layout)
      .render_pass(render_pass)
      .subpass(0)
  };
  let winding_info = winding_base().depth_stencil_state(&winding_depth_stencil).build();
  let parity_info = winding_base().depth_stencil_state(&parity_depth_stencil).build();
  let mut infos = vec![cover_info, winding_info, parity_info];
  if raster_order_groups {
    infos.push(
      vk::GraphicsPipelineCreateInfo::builder()
        .stages(&group_composite_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&cover_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build(),
    );
  }
  // SAFETY: all pipeline state and shader modules are valid for this device.
  #[cfg(target_os = "android")]
  eprintln!("TLottieVulkan: creating base raster-order pipelines");
  let pipelines = match unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &infos[..3], None) } {
    Ok(mut base) if raster_order_groups => {
      #[cfg(target_os = "android")]
      eprintln!("TLottieVulkan: creating alpha-group pipelines");
      match unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &infos[3..], None) } {
        Ok(mut groups) => {
          base.append(&mut groups);
          Ok(base)
        }
        Err(error) => {
          unsafe {
            for pipeline in base {
              device.destroy_pipeline(pipeline, None);
            }
          }
          Err(error)
        }
      }
    }
    other => other,
  };
  // SAFETY: pipeline compilation has consumed both modules.
  unsafe {
    device.destroy_shader_module(vertex, None);
    device.destroy_shader_module(cover_vertex, None);
    device.destroy_shader_module(fragment, None);
    device.destroy_shader_module(group_composite_fragment, None);
    device.destroy_shader_module(stencil_fragment, None);
  }
  let expected_pipeline_count = if raster_order_groups { 4 } else { 3 };
  let pipeline = match pipelines {
    Ok(pipelines) if pipelines.len() == expected_pipeline_count => Ok((pipelines[0], pipelines[1], pipelines[2], pipelines.get(3).copied().unwrap_or(vk::Pipeline::null()))),
    Ok(pipelines) => {
      unsafe {
        for pipeline in pipelines {
          device.destroy_pipeline(pipeline, None);
        }
      }
      Err(Error::FrameTooLarge)
    }
    Err((partial, e)) => {
      // SAFETY: destroy any partially created pipelines.
      unsafe {
        for pipeline in partial {
          device.destroy_pipeline(pipeline, None);
        }
      }
      Err(Error::Vulkan("vkCreateGraphicsPipelines", e))
    }
  };
  match pipeline {
    Ok((pipeline, stencil_nonzero_pipeline, stencil_evenodd_pipeline, group_composite_pipeline)) => {
      let mut pool_sizes = vec![vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
      }];
      if raster_order_groups {
        pool_sizes.push(vk::DescriptorPoolSize {
          ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
          descriptor_count: 1,
        });
      }
      let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
      // SAFETY: descriptor pool create info is valid.
      let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
        Ok(pool) => pool,
        Err(error) => {
          // SAFETY: resources belong to this device and are unused.
          unsafe {
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline(stencil_nonzero_pipeline, None);
            device.destroy_pipeline(stencil_evenodd_pipeline, None);
            device.destroy_pipeline(group_composite_pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            device.destroy_render_pass(render_pass, None);
          }
          return Err(Error::Vulkan("vkCreateDescriptorPool", error));
        }
      };
      let set_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));
      // SAFETY: pool and layout belong to this device.
      let descriptor_set = match unsafe { device.allocate_descriptor_sets(&set_info) } {
        Ok(sets) => match sets.first().copied() {
          Some(set) => set,
          None => {
            // SAFETY: resources belong to this device and are unused.
            unsafe {
              device.destroy_descriptor_pool(descriptor_pool, None);
              device.destroy_pipeline(pipeline, None);
              device.destroy_pipeline(stencil_nonzero_pipeline, None);
              device.destroy_pipeline(stencil_evenodd_pipeline, None);
              device.destroy_pipeline(group_composite_pipeline, None);
              device.destroy_pipeline_layout(pipeline_layout, None);
              device.destroy_descriptor_set_layout(descriptor_set_layout, None);
              device.destroy_render_pass(render_pass, None);
            }
            return Err(Error::FrameTooLarge);
          }
        },
        Err(error) => {
          // SAFETY: resources belong to this device and are unused.
          unsafe {
            device.destroy_descriptor_pool(descriptor_pool, None);
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline(stencil_nonzero_pipeline, None);
            device.destroy_pipeline(stencil_evenodd_pipeline, None);
            device.destroy_pipeline(group_composite_pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            device.destroy_render_pass(render_pass, None);
          }
          return Err(Error::Vulkan("vkAllocateDescriptorSets", error));
        }
      };
      Ok(TrianglePipeline {
        render_pass,
        descriptor_set_layout,
        descriptor_pool,
        descriptor_set,
        pipeline_layout,
        pipeline,
        stencil_nonzero_pipeline,
        stencil_evenodd_pipeline,
        group_composite_pipeline,
        group_sampler,
      })
    }
    Err(e) => {
      // SAFETY: resources belong to this device and are unused.
      unsafe {
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_render_pass(render_pass, None);
      }
      Err(e)
    }
  }
}

fn create_compute_pipeline(device: &ash::Device, pipeline_layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
  create_compute_pipeline_from_spirv(device, pipeline_layout, include_bytes!(concat!(env!("OUT_DIR"), "/coverage.comp.spv")), c"main")
}

fn create_simple_compute_pipeline(device: &ash::Device, pipeline_layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
  create_compute_pipeline_from_spirv(device, pipeline_layout, include_bytes!(concat!(env!("OUT_DIR"), "/coverage-simple.comp.spv")), c"simple_main")
}

fn create_bin_pipeline(device: &ash::Device, pipeline_layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
  create_compute_pipeline_from_spirv(device, pipeline_layout, include_bytes!(concat!(env!("OUT_DIR"), "/bins.comp.spv")), c"main")
}

fn create_compute_pipeline_from_spirv(device: &ash::Device, pipeline_layout: vk::PipelineLayout, spirv: &[u8], entry: &std::ffi::CStr) -> Result<vk::Pipeline> {
  let shader = create_shader_module(device, spirv)?;
  let stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(entry).build();
  let info = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(pipeline_layout).build();
  // SAFETY: shader and pipeline layout belong to this device.
  let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
  // SAFETY: pipeline creation has consumed the module.
  unsafe { device.destroy_shader_module(shader, None) };
  match result {
    Ok(pipelines) => pipelines.first().copied().ok_or(Error::FrameTooLarge),
    Err((partial, error)) => {
      // SAFETY: destroy any partially created pipelines.
      unsafe {
        for pipeline in partial {
          device.destroy_pipeline(pipeline, None);
        }
      }
      Err(Error::Vulkan("vkCreateComputePipelines", error))
    }
  }
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
  let mut cursor = std::io::Cursor::new(bytes);
  let words = ash::util::read_spv(&mut cursor).map_err(|_| Error::FrameTooLarge)?;
  let info = vk::ShaderModuleCreateInfo::builder().code(&words);
  // SAFETY: build-time generated SPIR-V is valid for this device call.
  unsafe { device.create_shader_module(&info, None) }.map_err(|e| Error::Vulkan("vkCreateShaderModule", e))
}

fn create_raster_order_group_framebuffer(
  device: &ash::Device,
  render_pass: vk::RenderPass,
  color_image: vk::Image,
  stencil_image: vk::Image,
  group_image: vk::Image,
  width: u32,
  height: u32,
) -> Result<(Vec<vk::ImageView>, vk::Framebuffer, vk::Framebuffer)> {
  let make_view = |image, format, aspect| {
    let info = vk::ImageViewCreateInfo::builder()
      .image(image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      });
    unsafe { device.create_image_view(&info, None) }.map_err(|e| Error::Vulkan("vkCreateImageView(alpha groups)", e))
  };
  let mut views = Vec::with_capacity(3);
  for (image, format, aspect) in [
    (color_image, vk::Format::B8G8R8A8_UNORM, vk::ImageAspectFlags::COLOR),
    (stencil_image, vk::Format::S8_UINT, vk::ImageAspectFlags::STENCIL),
    (group_image, vk::Format::B8G8R8A8_UNORM, vk::ImageAspectFlags::COLOR),
  ] {
    match make_view(image, format, aspect) {
      Ok(view) => views.push(view),
      Err(error) => {
        unsafe {
          for view in views {
            device.destroy_image_view(view, None);
          }
        }
        return Err(error);
      }
    }
  }
  let main_attachments = [views[0], views[1]];
  let info = vk::FramebufferCreateInfo::builder()
    .render_pass(render_pass)
    .attachments(&main_attachments)
    .width(width)
    .height(height)
    .layers(1);
  match unsafe { device.create_framebuffer(&info, None) } {
    Ok(framebuffer) => {
      let group_attachments = [views[2], views[1]];
      let group_info = vk::FramebufferCreateInfo::builder()
        .render_pass(render_pass)
        .attachments(&group_attachments)
        .width(width)
        .height(height)
        .layers(1);
      match unsafe { device.create_framebuffer(&group_info, None) } {
        Ok(group_framebuffer) => Ok((views, framebuffer, group_framebuffer)),
        Err(e) => {
          unsafe {
            device.destroy_framebuffer(framebuffer, None);
            for view in views {
              device.destroy_image_view(view, None);
            }
          }
          Err(Error::Vulkan("vkCreateFramebuffer(alpha group)", e))
        }
      }
    }
    Err(e) => {
      unsafe {
        for view in views {
          device.destroy_image_view(view, None);
        }
      }
      Err(Error::Vulkan("vkCreateFramebuffer(alpha groups)", e))
    }
  }
}

fn create_target_framebuffer(
  device: &ash::Device,
  render_pass: vk::RenderPass,
  resolve_image: vk::Image,
  multisample_image: vk::Image,
  stencil_image: vk::Image,
  group_image: Option<vk::Image>,
  width: u32,
  height: u32,
) -> Result<(Vec<vk::ImageView>, vk::Framebuffer)> {
  let view_info = vk::ImageViewCreateInfo::builder()
    .image(multisample_image)
    .view_type(vk::ImageViewType::TYPE_2D)
    .format(vk::Format::B8G8R8A8_UNORM)
    .subresource_range(vk::ImageSubresourceRange {
      aspect_mask: vk::ImageAspectFlags::COLOR,
      base_mip_level: 0,
      level_count: 1,
      base_array_layer: 0,
      layer_count: 1,
    });
  // SAFETY: image belongs to this device and has a compatible format.
  let multisample_view = unsafe { device.create_image_view(&view_info, None) }.map_err(|e| Error::Vulkan("vkCreateImageView", e))?;
  let stencil_view_info = vk::ImageViewCreateInfo::builder()
    .image(stencil_image)
    .view_type(vk::ImageViewType::TYPE_2D)
    .format(vk::Format::S8_UINT)
    .subresource_range(vk::ImageSubresourceRange {
      aspect_mask: vk::ImageAspectFlags::STENCIL,
      base_mip_level: 0,
      level_count: 1,
      base_array_layer: 0,
      layer_count: 1,
    });
  let stencil_view = match unsafe { device.create_image_view(&stencil_view_info, None) } {
    Ok(view) => view,
    Err(e) => {
      unsafe { device.destroy_image_view(multisample_view, None) };
      return Err(Error::Vulkan("vkCreateImageView(stencil)", e));
    }
  };
  let resolve_view_info = vk::ImageViewCreateInfo::builder()
    .image(resolve_image)
    .view_type(vk::ImageViewType::TYPE_2D)
    .format(vk::Format::B8G8R8A8_UNORM)
    .subresource_range(vk::ImageSubresourceRange {
      aspect_mask: vk::ImageAspectFlags::COLOR,
      base_mip_level: 0,
      level_count: 1,
      base_array_layer: 0,
      layer_count: 1,
    });
  let resolve_view = match unsafe { device.create_image_view(&resolve_view_info, None) } {
    Ok(view) => view,
    Err(e) => {
      unsafe {
        device.destroy_image_view(multisample_view, None);
        device.destroy_image_view(stencil_view, None);
      }
      return Err(Error::Vulkan("vkCreateImageView(resolve)", e));
    }
  };
  let mut attachments = vec![multisample_view, stencil_view, resolve_view];
  if let Some(group_image) = group_image {
    let group_view_info = vk::ImageViewCreateInfo::builder()
      .image(group_image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(vk::Format::B8G8R8A8_UNORM)
      .subresource_range(vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      });
    match unsafe { device.create_image_view(&group_view_info, None) } {
      Ok(view) => attachments.push(view),
      Err(e) => {
        unsafe {
          device.destroy_image_view(multisample_view, None);
          device.destroy_image_view(stencil_view, None);
          device.destroy_image_view(resolve_view, None);
        }
        return Err(Error::Vulkan("vkCreateImageView(group)", e));
      }
    }
  }
  let framebuffer_info = vk::FramebufferCreateInfo::builder()
    .render_pass(render_pass)
    .attachments(&attachments)
    .width(width)
    .height(height)
    .layers(1);
  // SAFETY: image view and render pass are compatible and live.
  match unsafe { device.create_framebuffer(&framebuffer_info, None) } {
    Ok(framebuffer) => Ok((attachments, framebuffer)),
    Err(e) => {
      // SAFETY: view belongs to this device and is unused.
      unsafe {
        for view in attachments {
          device.destroy_image_view(view, None);
        }
      }
      Err(Error::Vulkan("vkCreateFramebuffer", e))
    }
  }
}

/// Uploads packed point bytes and makes them visible to vertex fetch.
///
/// # Safety
/// `device`, `cmd`, and `target` belong to the same live device. The command
/// buffer is recording, and target has transfer-destination and vertex usage.
unsafe fn cmd_upload_buffer_bytes(
  device: &ash::Device,
  cmd: vk::CommandBuffer,
  target: vk::Buffer,
  target_offset: vk::DeviceSize,
  bytes: &[u8],
  dst_access: vk::AccessFlags,
  dst_stage: vk::PipelineStageFlags,
) -> Result<()> {
  const MAX_UPDATE_BYTES: usize = 65_536;
  for (chunk_index, chunk) in bytes.chunks(MAX_UPDATE_BYTES).enumerate() {
    let chunk_offset = chunk_index.checked_mul(MAX_UPDATE_BYTES).ok_or(Error::FrameTooLarge)?;
    let offset = target_offset.checked_add(chunk_offset as vk::DeviceSize).ok_or(Error::FrameTooLarge)?;
    // SAFETY: caller guarantees target capacity and usage; point chunks
    // have four-byte-aligned offsets and lengths.
    unsafe { device.cmd_update_buffer(cmd, target, offset, chunk) };
  }
  if !bytes.is_empty() {
    let barrier = vk::BufferMemoryBarrier::builder()
      .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
      .dst_access_mask(dst_access)
      .buffer(target)
      .offset(target_offset)
      .size(bytes.len() as vk::DeviceSize)
      .build();
    // SAFETY: command buffer and target are live and recording.
    unsafe {
      device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, dst_stage, vk::DependencyFlags::empty(), &[], &[barrier], &[]);
    }
  }
  Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
struct GpuPoint {
  x: f32,
  y: f32,
}

const IDENTITY_AFFINE: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const AFFINE_REUSE_EPSILON: f32 = 1.0 / 4096.0;

fn apply_affine(point: GpuPoint, transform: [f32; 6]) -> GpuPoint {
  if transform[..4] == IDENTITY_AFFINE[..4] {
    return GpuPoint {
      x: point.x + transform[4],
      y: point.y + transform[5],
    };
  }
  GpuPoint {
    x: transform[0] * point.x + transform[2] * point.y + transform[4],
    y: transform[1] * point.x + transform[3] * point.y + transform[5],
  }
}

fn point_payload_bytes(points: &[GpuPoint]) -> usize {
  std::mem::size_of_val(points)
}

fn scene_dirty_ranges(previous: Option<&[u32]>, current: &[u32]) -> Vec<std::ops::Range<usize>> {
  const MERGE_GAP_WORDS: usize = 16;
  let Some(previous) = previous else {
    return (!current.is_empty()).then_some(0..current.len()).into_iter().collect();
  };
  let mut ranges = Vec::<std::ops::Range<usize>>::new();
  let mut index = 0usize;
  while index < current.len() {
    if previous.get(index) == current.get(index) {
      index += 1;
      continue;
    }
    let start = index;
    index += 1;
    let mut last_dirty = index;
    while index < current.len() {
      if previous.get(index) != current.get(index) {
        last_dirty = index + 1;
      } else if index.saturating_sub(last_dirty) >= MERGE_GAP_WORDS {
        break;
      }
      index += 1;
    }
    ranges.push(start..last_dirty);
  }
  ranges
}

impl From<tlottie_internal::Point> for GpuPoint {
  fn from(value: tlottie_internal::Point) -> Self {
    Self { x: value.x, y: value.y }
  }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PointRange {
  first: u32,
  count: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GeometryKey {
  a: u64,
  b: u64,
}

#[derive(Clone, Copy, Debug)]
struct CachedContour {
  key: GeometryKey,
  range: PointRange,
  closed: bool,
  transform: [f32; 6],
  bounds: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DirtyRange {
  first_point: u32,
  point_count: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
struct ContourDrawData {
  point_offset: u32,
  point_count: u32,
  vertex_count: u32,
  paint_index: u32,
  flags: u32,
  argb: u32,
  transform: [f32; 6],
}

#[derive(Clone, Copy, Debug, Default)]
struct PreparedPaint {
  contour_start: u32,
  contour_count: u32,
  rule: u32,
  argb: u32,
  paint_kind: u32,
  gradient_kind: u32,
  lut_word_offset: u32,
  inverse0: [f32; 4],
  inverse1: [f32; 4],
  params0: [f32; 4],
  params1: [f32; 4],
  bounds: [f32; 4],
}

#[derive(Default)]
struct PreparedGeometry {
  dirty: Vec<DirtyRange>,
  draw_data: Vec<ContourDrawData>,
  paints: Vec<PreparedPaint>,
  gradient_luts: Vec<u32>,
  indirect: Vec<vk::DrawIndirectCommand>,
  fan_indirect: Vec<vk::DrawIndirectCommand>,
  stats: CacheStats,
}

const COMPUTE_TILE_SIZE: u32 = 16;
const SCENE_SECTION_COUNT: usize = 8;
const POINT_SECTION: usize = 0;
const CONTOUR_SECTION: usize = 1;
const PAINT_SECTION: usize = 2;
const LUT_SECTION: usize = 3;
const TILE_SECTION: usize = 4;
const TILE_INDEX_SECTION: usize = 5;
const EDGE_BIN_SECTION: usize = 6;
const EDGE_SECTION: usize = 7;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SceneLayout {
  output_words: u32,
  offsets: [u32; SCENE_SECTION_COUNT],
  capacities: [u32; SCENE_SECTION_COUNT],
  total_words: u32,
}

impl SceneLayout {
  fn ensure(&mut self, output_words: u32, lengths: [usize; SCENE_SECTION_COUNT]) -> Result<()> {
    let mut required = [0u32; SCENE_SECTION_COUNT];
    for (slot, length) in required.iter_mut().zip(lengths) {
      *slot = u32::try_from(length).map_err(|_| Error::FrameTooLarge)?;
    }
    let rebuild = self.total_words == 0 || self.output_words != output_words || required.iter().zip(self.capacities).any(|(&needed, capacity)| needed > capacity);
    if rebuild {
      let mut cursor = output_words;
      for (index, needed) in required.into_iter().enumerate() {
        let capacity = scene_section_capacity(needed)?;
        let offset = self.offsets.get_mut(index).ok_or(Error::FrameTooLarge)?;
        *offset = cursor;
        let slot = self.capacities.get_mut(index).ok_or(Error::FrameTooLarge)?;
        *slot = capacity;
        cursor = cursor.checked_add(capacity).ok_or(Error::FrameTooLarge)?;
      }
      self.output_words = output_words;
      self.total_words = cursor;
    }
    Ok(())
  }

  fn offset(&self, section: usize) -> Result<u32> {
    self.offsets.get(section).copied().ok_or(Error::FrameTooLarge)
  }
}

fn scene_section_capacity(needed: u32) -> Result<u32> {
  needed.max(1).checked_next_power_of_two().ok_or(Error::FrameTooLarge)
}

struct ComputeScene {
  sections: [Vec<u32>; SCENE_SECTION_COUNT],
  layout: SceneLayout,
  push: ComputePush,
}

struct UploadedScene {
  buffer: vk::Buffer,
  layout: SceneLayout,
  sections: [Vec<u32>; SCENE_SECTION_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BinKey {
  a: u64,
  b: u64,
}

struct RetainedBins {
  buffer: vk::Buffer,
  layout: SceneLayout,
  key: Option<BinKey>,
}

impl Default for RetainedBins {
  fn default() -> Self {
    Self {
      buffer: vk::Buffer::null(),
      layout: SceneLayout::default(),
      key: None,
    }
  }
}

impl Default for UploadedScene {
  fn default() -> Self {
    Self {
      buffer: vk::Buffer::null(),
      layout: SceneLayout::default(),
      sections: std::array::from_fn(|_| Vec::new()),
    }
  }
}

fn compute_bin_key(scene: &ComputeScene) -> Result<BinKey> {
  const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
  let mut a = 0xcbf2_9ce4_8422_2325u64;
  let mut b = 0x8422_2325_cbf2_9ce4u64;
  let mut mix = |word: u32| {
    a = (a ^ u64::from(word)).wrapping_mul(FNV_PRIME);
    b = (b ^ u64::from(word.rotate_left(13))).wrapping_mul(FNV_PRIME).rotate_left(17);
  };
  for word in [
    scene.push.width,
    scene.push.height,
    scene.push.paint_count,
    scene.push.tiles_x,
    scene.push.tiles_y,
    scene.push.compact_flags,
  ] {
    mix(word);
  }
  for section_index in [POINT_SECTION, CONTOUR_SECTION] {
    let section = scene.sections.get(section_index).ok_or(Error::FrameTooLarge)?;
    mix(u32::try_from(section.len()).map_err(|_| Error::FrameTooLarge)?);
    for &word in section {
      mix(word);
    }
  }
  let paints = scene.sections.get(PAINT_SECTION).ok_or(Error::FrameTooLarge)?;
  if paints.len() % 24 != 0 {
    return Err(Error::FrameTooLarge);
  }
  for paint in paints.chunks_exact(24) {
    for index in [0usize, 1, 20, 21, 22, 23] {
      mix(*paint.get(index).ok_or(Error::FrameTooLarge)?);
    }
  }
  Ok(BinKey { a, b })
}

fn retained_bin_layout_matches(previous: SceneLayout, current: SceneLayout) -> bool {
  (TILE_SECTION..=EDGE_SECTION).all(|index| previous.offsets.get(index) == current.offsets.get(index) && previous.capacities.get(index) == current.capacities.get(index))
}

fn scene_section_domain(section: usize) -> usize {
  match section {
    POINT_SECTION | CONTOUR_SECTION => 0,
    PAINT_SECTION | LUT_SECTION => 1,
    _ => 2,
  }
}

fn build_compute_scene(
  width: u32,
  height: u32,
  points: &[GpuPoint],
  contours: &[CachedContour],
  prepared: &PreparedGeometry,
  antialias: bool,
  alpha_only: bool,
  retained_layout: &mut SceneLayout,
) -> Result<ComputeScene> {
  let output_words = width.checked_mul(height).ok_or(Error::FrameTooLarge)?;
  let mut point_words = Vec::with_capacity(points.len().saturating_mul(2));
  for point in points {
    point_words.push(point.x.to_bits());
    point_words.push(point.y.to_bits());
  }
  let mut contour_words = Vec::with_capacity(contours.len().saturating_mul(8));
  for contour in contours {
    contour_words.push(contour.range.first);
    contour_words.push(contour.range.count);
    contour_words.extend(contour.transform.iter().map(|value| value.to_bits()));
  }
  let paint_count = u32::try_from(prepared.paints.len()).map_err(|_| Error::FrameTooLarge)?;
  let tiles_x = width.saturating_add(COMPUTE_TILE_SIZE - 1) / COMPUTE_TILE_SIZE;
  let tiles_y = height.saturating_add(COMPUTE_TILE_SIZE - 1) / COMPUTE_TILE_SIZE;
  let tile_count = tiles_x.checked_mul(tiles_y).ok_or(Error::FrameTooLarge)?;
  let compact_tile_indices = paint_count <= u16::MAX as u32 + 1;
  let compact_edge_indices = points.len() <= u16::MAX as usize + 1;
  let mut paint_words = Vec::with_capacity(prepared.paints.len().saturating_mul(24));
  let mut edge_entry_count = 0u32;
  for paint in &prepared.paints {
    let contour_end = paint.contour_start.checked_add(paint.contour_count).ok_or(Error::FrameTooLarge)?;
    for contour_index in paint.contour_start..contour_end {
      let contour = contours.get(usize::try_from(contour_index).map_err(|_| Error::FrameTooLarge)?).ok_or(Error::FrameTooLarge)?;
      if contour.range.count < 2 {
        continue;
      }
      for edge_index in 0..contour.range.count {
        let a_index = contour.range.first.checked_add(edge_index).ok_or(Error::FrameTooLarge)?;
        let b_index = contour.range.first.checked_add((edge_index + 1) % contour.range.count).ok_or(Error::FrameTooLarge)?;
        let a = points.get(usize::try_from(a_index).map_err(|_| Error::FrameTooLarge)?).ok_or(Error::FrameTooLarge)?;
        let b = points.get(usize::try_from(b_index).map_err(|_| Error::FrameTooLarge)?).ok_or(Error::FrameTooLarge)?;
        let a_y = apply_affine(*a, contour.transform).y;
        let b_y = apply_affine(*b, contour.transform).y;
        let row_start = (a_y.min(b_y) / COMPUTE_TILE_SIZE as f32).floor().clamp(0.0, tiles_y as f32) as u32;
        let row_end = (a_y.max(b_y) / COMPUTE_TILE_SIZE as f32).ceil().clamp(0.0, tiles_y as f32) as u32;
        edge_entry_count = edge_entry_count.checked_add(row_end.saturating_sub(row_start)).ok_or(Error::FrameTooLarge)?;
      }
    }
    paint_words.extend_from_slice(&[
      paint.contour_start,
      paint.contour_count,
      paint.paint_kind,
      paint.rule,
      paint.argb,
      paint.gradient_kind,
      paint.lut_word_offset,
      0,
    ]);
    paint_words.extend(paint.inverse0.iter().map(|value| value.to_bits()));
    paint_words.push(paint.inverse1[0].to_bits());
    paint_words.push(paint.inverse1[1].to_bits());
    paint_words.extend(paint.params0.iter().map(|value| value.to_bits()));
    paint_words.push(paint.params1[0].to_bits());
    paint_words.push(paint.params1[1].to_bits());
    paint_words.extend(paint.bounds.iter().map(|value| value.to_bits()));
  }
  let lut_words = prepared.gradient_luts.clone();
  let tile_record_words = tile_count.checked_mul(2).ok_or(Error::FrameTooLarge)?;
  let tile_stride_upper = if compact_tile_indices {
    paint_count.checked_add(1).ok_or(Error::FrameTooLarge)?
  } else {
    paint_count
  };
  let tile_entries = tile_count.checked_mul(tile_stride_upper).ok_or(Error::FrameTooLarge)?;
  let tile_index_words = if compact_tile_indices { tile_entries.saturating_add(1) / 2 } else { tile_entries };
  let edge_bin_count = paint_count.checked_mul(tiles_y).ok_or(Error::FrameTooLarge)?;
  let edge_bin_words = edge_bin_count.checked_mul(2).ok_or(Error::FrameTooLarge)?;
  let edge_words = edge_entry_count.checked_mul(5).ok_or(Error::FrameTooLarge)?;
  let sections = [point_words, contour_words, paint_words, lut_words, Vec::new(), Vec::new(), Vec::new(), Vec::new()];
  let lengths = [
    sections[POINT_SECTION].len(),
    sections[CONTOUR_SECTION].len(),
    sections[PAINT_SECTION].len(),
    sections[LUT_SECTION].len(),
    usize::try_from(tile_record_words).map_err(|_| Error::FrameTooLarge)?,
    usize::try_from(tile_index_words).map_err(|_| Error::FrameTooLarge)?,
    usize::try_from(edge_bin_words).map_err(|_| Error::FrameTooLarge)?,
    usize::try_from(edge_words).map_err(|_| Error::FrameTooLarge)?,
  ];
  retained_layout.ensure(output_words, lengths)?;
  let layout = *retained_layout;
  Ok(ComputeScene {
    sections,
    layout,
    push: ComputePush {
      width,
      height,
      output_word: 0,
      point_word: layout.offset(POINT_SECTION)?,
      contour_word: layout.offset(CONTOUR_SECTION)?,
      paint_word: layout.offset(PAINT_SECTION)?,
      lut_word: layout.offset(LUT_SECTION)?,
      paint_count,
      antialias: u32::from(antialias),
      tile_word: layout.offset(TILE_SECTION)?,
      tile_index_word: layout.offset(TILE_INDEX_SECTION)?,
      tiles_x,
      edge_bin_word: layout.offset(EDGE_BIN_SECTION)?,
      edge_word: layout.offset(EDGE_SECTION)?,
      tiles_y,
      compact_flags: u32::from(compact_tile_indices) | (u32::from(compact_edge_indices) << 1) | (1 << 2) | (u32::from(alpha_only) << 3),
    },
  })
}

#[derive(Debug, Default)]
struct PointArena {
  points: Vec<GpuPoint>,
  free: Vec<PointRange>,
}

impl PointArena {
  fn allocate(&mut self, count: u32) -> Result<PointRange> {
    if count == 0 {
      return Ok(PointRange::default());
    }
    if let Some(index) = self.free.iter().position(|range| range.count >= count) {
      let Some(range) = self.free.get(index).copied() else {
        return Err(Error::FrameTooLarge);
      };
      if range.count == count {
        self.free.remove(index);
      } else if let Some(free) = self.free.get_mut(index) {
        free.first = free.first.checked_add(count).ok_or(Error::FrameTooLarge)?;
        free.count -= count;
      }
      return Ok(PointRange { first: range.first, count });
    }

    let first = u32::try_from(self.points.len()).map_err(|_| Error::FrameTooLarge)?;
    let end = first.checked_add(count).ok_or(Error::FrameTooLarge)?;
    let end_usize = usize::try_from(end).map_err(|_| Error::FrameTooLarge)?;
    self.points.resize(end_usize, GpuPoint::default());
    Ok(PointRange { first, count })
  }

  fn release(&mut self, range: PointRange) {
    if range.count == 0 {
      return;
    }
    self.free.push(range);
    self.free.sort_unstable_by_key(|item| item.first);
    let mut merged: Vec<PointRange> = Vec::with_capacity(self.free.len());
    for item in self.free.drain(..) {
      if let Some(last) = merged.last_mut() {
        if last.first.checked_add(last.count) == Some(item.first) {
          last.count = last.count.saturating_add(item.count);
          continue;
        }
      }
      merged.push(item);
    }
    self.free = merged;
  }

  fn write(&mut self, range: PointRange, points: &[tlottie_internal::Point]) -> Result<()> {
    let start = usize::try_from(range.first).map_err(|_| Error::FrameTooLarge)?;
    let count = usize::try_from(range.count).map_err(|_| Error::FrameTooLarge)?;
    let end = start.checked_add(count).ok_or(Error::FrameTooLarge)?;
    let Some(destination) = self.points.get_mut(start..end) else {
      return Err(Error::FrameTooLarge);
    };
    if destination.len() != points.len() {
      return Err(Error::FrameTooLarge);
    }
    for (dst, src) in destination.iter_mut().zip(points) {
      *dst = (*src).into();
    }
    Ok(())
  }

  fn matches_affine(&self, range: PointRange, transform: [f32; 6], points: &[tlottie_internal::Point]) -> bool {
    let Ok(start) = usize::try_from(range.first) else {
      return false;
    };
    let Ok(count) = usize::try_from(range.count) else {
      return false;
    };
    let Some(end) = start.checked_add(count) else {
      return false;
    };
    let Some(stored) = self.points.get(start..end) else {
      return false;
    };
    stored.len() == points.len()
      && stored.iter().zip(points).all(|(base, current)| {
        let mapped = apply_affine(*base, transform);
        (mapped.x - current.x).abs() <= AFFINE_REUSE_EPSILON && (mapped.y - current.y).abs() <= AFFINE_REUSE_EPSILON
      })
  }

  fn exact_translation(&self, range: PointRange, points: &[tlottie_internal::Point]) -> Option<[f32; 6]> {
    let start = usize::try_from(range.first).ok()?;
    let count = usize::try_from(range.count).ok()?;
    let end = start.checked_add(count)?;
    let stored = self.points.get(start..end)?;
    if stored.len() != points.len() {
      return None;
    }
    let Some((first_stored, first_current)) = stored.first().zip(points.first()) else {
      return Some(IDENTITY_AFFINE);
    };
    let translation = [first_current.x - first_stored.x, first_current.y - first_stored.y];
    if !translation[0].is_finite() || !translation[1].is_finite() {
      return None;
    }
    stored
      .iter()
      .zip(points)
      .all(|(base, current)| (base.x + translation[0]).to_bits() == current.x.to_bits() && (base.y + translation[1]).to_bits() == current.y.to_bits())
      .then_some([1.0, 0.0, 0.0, 1.0, translation[0], translation[1]])
  }

  fn verified_affine(&self, range: PointRange, points: &[tlottie_internal::Point]) -> Option<[f32; 6]> {
    let start = usize::try_from(range.first).ok()?;
    let count = usize::try_from(range.count).ok()?;
    let end = start.checked_add(count)?;
    let stored = self.points.get(start..end)?;
    if stored.len() != points.len() || stored.len() < 3 {
      return None;
    }
    let p0 = *stored.first()?;
    let (p1_index, max_distance_sq) = stored
      .iter()
      .enumerate()
      .skip(1)
      .map(|(index, point)| {
        let x = f64::from(point.x - p0.x);
        let y = f64::from(point.y - p0.y);
        (index, x * x + y * y)
      })
      .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if max_distance_sq <= f64::EPSILON {
      return None;
    }
    let p1 = *stored.get(p1_index)?;
    let ux = f64::from(p1.x - p0.x);
    let uy = f64::from(p1.y - p0.y);
    let (p2_index, max_area) = stored
      .iter()
      .enumerate()
      .filter(|(index, _)| *index != p1_index)
      .map(|(index, point)| {
        let vx = f64::from(point.x - p0.x);
        let vy = f64::from(point.y - p0.y);
        (index, (ux * vy - uy * vx).abs())
      })
      .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if max_area <= max_distance_sq * 1.0e-6 {
      return None;
    }
    let p2 = *stored.get(p2_index)?;
    let q0 = points.first()?;
    let q1 = points.get(p1_index)?;
    let q2 = points.get(p2_index)?;
    let vx = f64::from(p2.x - p0.x);
    let vy = f64::from(p2.y - p0.y);
    let det = ux * vy - uy * vx;
    let q1x = f64::from(q1.x - q0.x);
    let q1y = f64::from(q1.y - q0.y);
    let q2x = f64::from(q2.x - q0.x);
    let q2y = f64::from(q2.y - q0.y);
    let a = (q1x * vy - q2x * uy) / det;
    let c = (ux * q2x - vx * q1x) / det;
    let b = (q1y * vy - q2y * uy) / det;
    let d = (ux * q2y - vx * q1y) / det;
    let tx = f64::from(q0.x) - a * f64::from(p0.x) - c * f64::from(p0.y);
    let ty = f64::from(q0.y) - b * f64::from(p0.x) - d * f64::from(p0.y);
    let transform = [a as f32, b as f32, c as f32, d as f32, tx as f32, ty as f32];
    transform
      .iter()
      .all(|value| value.is_finite())
      .then_some(transform)
      .filter(|transform| self.matches_affine(range, *transform, points))
  }
}

#[derive(Debug, Default)]
struct GeometryCache {
  arena: PointArena,
  contours: Vec<CachedContour>,
}

impl GeometryCache {
  fn prepare(&mut self, frame: &EvaluatedFrame) -> Result<PreparedGeometry> {
    let mut prepared = PreparedGeometry::default();

    for (index, contour) in frame.contours.iter().enumerate() {
      let point_count = u32::try_from(contour.points.len()).map_err(|_| Error::FrameTooLarge)?;
      let key = geometry_key(contour);
      let bounds = contour_bounds(&contour.points);
      let existing = self.contours.get(index).copied();
      let exact_translation = existing.and_then(|old| {
        if old.closed == contour.closed && old.range.count == point_count {
          self.arena.exact_translation(old.range, &contour.points)
        } else {
          None
        }
      });
      let verified_affine = existing.and_then(|old| {
        if old.closed == contour.closed && old.range.count == point_count {
          self.arena.verified_affine(old.range, &contour.points)
        } else {
          None
        }
      });

      let cached = if let Some(old) = existing {
        if old.key == key && old.closed == contour.closed && old.range.count == point_count && self.arena.matches_affine(old.range, old.transform, &contour.points) {
          prepared.stats.reused_contours = prepared.stats.reused_contours.saturating_add(1);
          CachedContour { key, bounds, ..old }
        } else if let Some(transform) = exact_translation {
          prepared.stats.reused_contours = prepared.stats.reused_contours.saturating_add(1);
          prepared.stats.translated_contours = prepared.stats.translated_contours.saturating_add(1);
          prepared.stats.translated_points = prepared.stats.translated_points.saturating_add(old.range.count);
          CachedContour {
            key,
            range: old.range,
            closed: contour.closed,
            transform,
            bounds,
          }
        } else if let Some(transform) = verified_affine {
          prepared.stats.reused_contours = prepared.stats.reused_contours.saturating_add(1);
          prepared.stats.affine_contours = prepared.stats.affine_contours.saturating_add(1);
          prepared.stats.affine_points = prepared.stats.affine_points.saturating_add(old.range.count);
          CachedContour {
            key,
            range: old.range,
            closed: contour.closed,
            transform,
            bounds,
          }
        } else if old.range.count == point_count {
          self.arena.write(old.range, &contour.points)?;
          push_dirty(&mut prepared.dirty, old.range);
          prepared.stats.updated_contours = prepared.stats.updated_contours.saturating_add(1);
          CachedContour {
            key,
            range: old.range,
            closed: contour.closed,
            transform: IDENTITY_AFFINE,
            bounds,
          }
        } else {
          self.arena.release(old.range);
          let range = self.arena.allocate(point_count)?;
          self.arena.write(range, &contour.points)?;
          push_dirty(&mut prepared.dirty, range);
          prepared.stats.allocated_contours = prepared.stats.allocated_contours.saturating_add(1);
          CachedContour {
            key,
            range,
            closed: contour.closed,
            transform: IDENTITY_AFFINE,
            bounds,
          }
        }
      } else {
        let range = self.arena.allocate(point_count)?;
        self.arena.write(range, &contour.points)?;
        push_dirty(&mut prepared.dirty, range);
        prepared.stats.allocated_contours = prepared.stats.allocated_contours.saturating_add(1);
        CachedContour {
          key,
          range,
          closed: contour.closed,
          transform: IDENTITY_AFFINE,
          bounds,
        }
      };

      if let Some(slot) = self.contours.get_mut(index) {
        *slot = cached;
      } else {
        self.contours.push(cached);
      }
    }

    while self.contours.len() > frame.contours.len() {
      if let Some(retired) = self.contours.pop() {
        self.arena.release(retired.range);
      }
    }

    self.build_draws(frame, &mut prepared)?;
    prepared.stats.dirty_points = prepared.dirty.iter().fold(0u32, |sum, range| sum.saturating_add(range.point_count));
    prepared.stats.dirty_ranges = u32::try_from(prepared.dirty.len()).map_err(|_| Error::FrameTooLarge)?;
    prepared.stats.arena_points = u32::try_from(self.arena.points.len()).map_err(|_| Error::FrameTooLarge)?;
    prepared.stats.draws = u32::try_from(prepared.indirect.len()).map_err(|_| Error::FrameTooLarge)?;
    Ok(prepared)
  }

  fn build_draws(&self, frame: &EvaluatedFrame, prepared: &mut PreparedGeometry) -> Result<()> {
    let mut lut_offsets = std::collections::HashMap::<u64, Vec<u32>>::new();
    for command in &frame.commands {
      let paint_index = u32::try_from(prepared.paints.len()).map_err(|_| Error::FrameTooLarge)?;
      let (rule, mut prepared_paint) = match &command.op {
        RecordedOp::Solid(solid) => (
          solid.rule,
          PreparedPaint {
            argb: solid.rgba,
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::Gradient(gradient) => {
          let lut_word_offset = intern_gradient_lut(&mut prepared.gradient_luts, &mut lut_offsets, gradient.lut.as_ref())?;
          let transform = gradient.transform;
          let mut gpu = PreparedPaint {
            paint_kind: 1,
            lut_word_offset,
            inverse0: [transform.a, transform.b, transform.c, transform.d],
            inverse1: [transform.tx, transform.ty, 0.0, 0.0],
            ..PreparedPaint::default()
          };
          match gradient.kind {
            tlottie_internal::GradientKind::Linear { sx, sy, dx, dy, inv_len_sq } => {
              gpu.params0 = [sx, sy, dx, dy];
              gpu.params1[0] = inv_len_sq;
            }
            tlottie_internal::GradientKind::Radial { sx, sy, inv_r } => {
              gpu.gradient_kind = 1;
              gpu.params0 = [sx, sy, inv_r, 0.0];
            }
            tlottie_internal::GradientKind::Focal { fx, fy, dx, dy, a, r } => {
              gpu.gradient_kind = 2;
              gpu.params0 = [fx, fy, dx, dy];
              gpu.params1 = [a, r, 0.0, 0.0];
            }
          }
          (gradient.rule, gpu)
        }
        RecordedOp::BeginLayer => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 2,
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::EndLayer { opacity } => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 3,
            argb: u32::from(*opacity) << 24,
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::BeginMatte => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 4,
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::BeginMatteTarget => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 5,
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::EndMatte { kind, opacity, source_opacity } => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 6,
            argb: (u32::from(*opacity) << 24) | (u32::from(*source_opacity) << 8) | u32::from(*kind),
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
        RecordedOp::Mask { mode, inverted, opacity, first, last } => (
          tlottie_internal::Rule::NonZero,
          PreparedPaint {
            paint_kind: 7,
            argb: u32::from(*mode) | (u32::from(*inverted) << 8) | (u32::from(*first) << 9) | (u32::from(*last) << 10) | (u32::from(*opacity) << 24),
            bounds: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            ..PreparedPaint::default()
          },
        ),
      };
      prepared_paint.contour_start = u32::try_from(command.start).map_err(|_| Error::FrameTooLarge)?;
      prepared_paint.contour_count = u32::try_from(command.end.saturating_sub(command.start)).map_err(|_| Error::FrameTooLarge)?;
      prepared_paint.rule = u32::from(rule == tlottie_internal::Rule::EvenOdd);
      if prepared_paint.paint_kind < 2 {
        prepared_paint.bounds = self.paint_bounds(command.start, command.end);
      }
      let gradient = prepared_paint.paint_kind == 1;
      prepared.paints.push(prepared_paint);
      for contour_index in command.start..command.end {
        let Some(contour) = self.contours.get(contour_index) else {
          continue;
        };
        if contour.range.count < 3 {
          continue;
        }
        let draw_index = u32::try_from(prepared.draw_data.len()).map_err(|_| Error::FrameTooLarge)?;
        let mut flags = 0u32;
        if contour.closed {
          flags |= 1;
        }
        if rule == tlottie_internal::Rule::EvenOdd {
          flags |= 1 << 1;
        }
        if gradient {
          flags |= 1 << 2;
          prepared.stats.gradient_draws = prepared.stats.gradient_draws.saturating_add(1);
        } else {
          prepared.stats.solid_draws = prepared.stats.solid_draws.saturating_add(1);
        }
        let vertex_count = contour.range.count.checked_sub(2).and_then(|triangles| triangles.checked_mul(3)).ok_or(Error::FrameTooLarge)?;
        prepared.draw_data.push(ContourDrawData {
          point_offset: contour.range.first,
          point_count: contour.range.count,
          vertex_count,
          paint_index,
          flags,
          argb: prepared_paint.argb,
          transform: contour.transform,
        });
        prepared.indirect.push(vk::DrawIndirectCommand {
          vertex_count,
          instance_count: 1,
          first_vertex: 0,
          first_instance: draw_index,
        });
        prepared.fan_indirect.push(vk::DrawIndirectCommand {
          vertex_count: contour.range.count,
          instance_count: 1,
          first_vertex: 0,
          first_instance: draw_index,
        });
      }
    }
    Ok(())
  }

  fn paint_bounds(&self, start: usize, end: usize) -> [f32; 4] {
    let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    let Some(contours) = self.contours.get(start..end.min(self.contours.len())) else {
      return [0.0; 4];
    };
    for contour in contours {
      bounds[0] = bounds[0].min(contour.bounds[0]);
      bounds[1] = bounds[1].min(contour.bounds[1]);
      bounds[2] = bounds[2].max(contour.bounds[2]);
      bounds[3] = bounds[3].max(contour.bounds[3]);
    }
    if bounds[2] < bounds[0] || bounds[3] < bounds[1] {
      [0.0; 4]
    } else {
      bounds
    }
  }
}

fn intern_gradient_lut(words: &mut Vec<u32>, offsets: &mut std::collections::HashMap<u64, Vec<u32>>, lut: &[u32; 1024]) -> Result<u32> {
  const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
  const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
  let hash = lut.iter().fold(FNV_OFFSET, |state, word| (state ^ u64::from(*word)).wrapping_mul(FNV_PRIME));
  if let Some(candidates) = offsets.get(&hash) {
    for &offset in candidates {
      let start = usize::try_from(offset).map_err(|_| Error::FrameTooLarge)?;
      let end = start.checked_add(lut.len()).ok_or(Error::FrameTooLarge)?;
      if words.get(start..end) == Some(lut.as_slice()) {
        return Ok(offset);
      }
    }
  }
  let offset = u32::try_from(words.len()).map_err(|_| Error::FrameTooLarge)?;
  words.extend_from_slice(lut);
  offsets.entry(hash).or_default().push(offset);
  Ok(offset)
}

fn push_dirty(dirty: &mut Vec<DirtyRange>, range: PointRange) {
  if range.count == 0 {
    return;
  }
  if let Some(last) = dirty.last_mut() {
    if last.first_point.checked_add(last.point_count) == Some(range.first) {
      last.point_count = last.point_count.saturating_add(range.count);
      return;
    }
  }
  dirty.push(DirtyRange {
    first_point: range.first,
    point_count: range.count,
  });
}

fn geometry_key(contour: &RecordedContour) -> GeometryKey {
  const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
  let mut a = 0xcbf2_9ce4_8422_2325u64;
  let mut b = 0x8422_2325_cbf2_9ce4u64;
  let mut mix = |word: u32| {
    a ^= u64::from(word);
    a = a.wrapping_mul(FNV_PRIME);
    b ^= u64::from(word.rotate_left(13));
    b = b.wrapping_mul(FNV_PRIME).rotate_left(17);
  };
  mix(u32::from(contour.closed));
  for point in &contour.points {
    mix(point.x.to_bits());
    mix(point.y.to_bits());
  }
  GeometryKey { a, b }
}

fn contour_bounds(points: &[tlottie_internal::Point]) -> [f32; 4] {
  let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
  for point in points {
    if point.x.is_finite() && point.y.is_finite() {
      bounds[0] = bounds[0].min(point.x);
      bounds[1] = bounds[1].min(point.y);
      bounds[2] = bounds[2].max(point.x);
      bounds[3] = bounds[3].max(point.y);
    }
  }
  if bounds[2] < bounds[0] || bounds[3] < bounds[1] {
    [0.0; 4]
  } else {
    bounds
  }
}

/// Records a simple premultiplied RGBA8 rectangle draw into a linear pixel buffer.
///
/// The target buffer is interpreted as `width * height` tightly packed `u32`
/// pixels. The command stream first clears the full target to transparent,
/// then fills each rectangle row with `rgba` (`0xAABBGGRR`).
///
/// # Safety
/// `device`, `cmd`, and `target` must belong to the same live Vulkan device.
/// `cmd` must be in recording state. `target` must be large enough for
/// `width * height * 4` bytes and have `VK_BUFFER_USAGE_TRANSFER_DST_BIT`.
pub unsafe fn cmd_draw_rgba_rect(device: &ash::Device, cmd: vk::CommandBuffer, target: vk::Buffer, width: u32, height: u32, rect: Rect, rgba: u32) -> Result<()> {
  // SAFETY: forwarded from this function's caller contract.
  unsafe {
    cmd_clear_rgba_buffer(device, cmd, target, width, height)?;
    cmd_fill_rgba_rect(device, cmd, target, width, height, rect, rgba)
  }
}

/// Records a full transparent clear into a premultiplied RGBA8 linear pixel buffer.
///
/// # Safety
/// `device`, `cmd`, and `target` must belong to the same live Vulkan device.
/// `cmd` must be in recording state. `target` must be large enough for
/// `width * height * 4` bytes and have `VK_BUFFER_USAGE_TRANSFER_DST_BIT`.
pub unsafe fn cmd_clear_rgba_buffer(device: &ash::Device, cmd: vk::CommandBuffer, target: vk::Buffer, width: u32, height: u32) -> Result<()> {
  let pixels = width.checked_mul(height).and_then(|n| n.checked_mul(4)).ok_or(Error::BadTarget)?;
  if width == 0 || height == 0 {
    return Err(Error::BadTarget);
  }
  // SAFETY: guaranteed by this function's caller contract.
  unsafe {
    device.cmd_fill_buffer(cmd, target, 0, pixels as vk::DeviceSize, 0);
  }
  Ok(())
}

/// Records a rectangle fill into a premultiplied RGBA8 linear pixel buffer.
///
/// Existing pixels outside the rectangle are left unchanged.
///
/// # Safety
/// `device`, `cmd`, and `target` must belong to the same live Vulkan device.
/// `cmd` must be in recording state. `target` must be large enough for
/// `width * height * 4` bytes and have `VK_BUFFER_USAGE_TRANSFER_DST_BIT`.
pub unsafe fn cmd_fill_rgba_rect(device: &ash::Device, cmd: vk::CommandBuffer, target: vk::Buffer, width: u32, height: u32, rect: Rect, rgba: u32) -> Result<()> {
  let x1 = rect.x.checked_add(rect.w).ok_or(Error::BadTarget)?;
  let y1 = rect.y.checked_add(rect.h).ok_or(Error::BadTarget)?;
  if width == 0 || height == 0 || x1 > width || y1 > height {
    return Err(Error::BadTarget);
  }

  for row in rect.y..y1 {
    let start_px = row.checked_mul(width).and_then(|v| v.checked_add(rect.x)).ok_or(Error::BadTarget)?;
    let offset = start_px.checked_mul(4).ok_or(Error::BadTarget)?;
    let bytes = rect.w.checked_mul(4).ok_or(Error::BadTarget)?;
    // SAFETY: guaranteed by this function's caller contract; offsets and
    // sizes are 4-byte aligned and bounded by the validated target rect.
    unsafe {
      device.cmd_fill_buffer(cmd, target, offset as vk::DeviceSize, bytes as vk::DeviceSize, rgba);
    }
  }
  Ok(())
}

/// Copies a tightly packed premultiplied RGBA8 buffer into an image and
/// leaves the image ready for transfer readback.
///
/// # Safety
/// `device`, `cmd`, `buffer`, and `target.image` must belong to the same live
/// Vulkan device. `cmd` must be in recording state. `target.image` must be in
/// `target.layout` on entry and have transfer source/destination usage.
/// `buffer` must have transfer source usage and contain at least
/// `target.width * target.height * 4` bytes.
pub unsafe fn cmd_copy_rgba_buffer_to_image(device: &ash::Device, cmd: vk::CommandBuffer, buffer: vk::Buffer, target: ImageTarget) -> Result<()> {
  if target.width == 0 || target.height == 0 {
    return Err(Error::BadTarget);
  }
  let _bytes = target.width.checked_mul(target.height).and_then(|n| n.checked_mul(4)).ok_or(Error::BadTarget)?;
  let subresource = vk::ImageSubresourceRange::builder()
    .aspect_mask(vk::ImageAspectFlags::COLOR)
    .base_mip_level(0)
    .level_count(1)
    .base_array_layer(0)
    .layer_count(1)
    .build();
  let (src_access, src_stage) = match target.layout {
    vk::ImageLayout::UNDEFINED => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
    vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::AccessFlags::TRANSFER_READ, vk::PipelineStageFlags::TRANSFER),
    vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
    vk::ImageLayout::PRESENT_SRC_KHR => (vk::AccessFlags::MEMORY_READ, vk::PipelineStageFlags::BOTTOM_OF_PIPE),
    _ => return Err(Error::BadTarget),
  };
  let to_dst = vk::ImageMemoryBarrier::builder()
    .old_layout(target.layout)
    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    .src_access_mask(src_access)
    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
    .image(target.image)
    .subresource_range(subresource)
    .build();
  // SAFETY: guaranteed by this function's caller contract.
  unsafe {
    device.cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
  }

  let layers = vk::ImageSubresourceLayers::builder()
    .aspect_mask(vk::ImageAspectFlags::COLOR)
    .mip_level(0)
    .base_array_layer(0)
    .layer_count(1)
    .build();
  let copy = vk::BufferImageCopy::builder()
    .buffer_offset(0)
    .buffer_row_length(0)
    .buffer_image_height(0)
    .image_subresource(layers)
    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
    .image_extent(vk::Extent3D {
      width: target.width,
      height: target.height,
      depth: 1,
    })
    .build();
  // SAFETY: buffer and image have transfer usages and validated extents.
  unsafe {
    device.cmd_copy_buffer_to_image(cmd, buffer, target.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
  }

  let final_layout = match target.final_layout {
    vk::ImageLayout::TRANSFER_SRC_OPTIMAL | vk::ImageLayout::PRESENT_SRC_KHR => target.final_layout,
    _ => return Err(Error::BadTarget),
  };
  let to_final = vk::ImageMemoryBarrier::builder()
    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    .new_layout(final_layout)
    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
    .dst_access_mask(if final_layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
      vk::AccessFlags::TRANSFER_READ
    } else {
      vk::AccessFlags::MEMORY_READ
    })
    .image(target.image)
    .subresource_range(subresource)
    .build();
  // SAFETY: image was written by the preceding transfer copy.
  unsafe {
    device.cmd_pipeline_barrier(
      cmd,
      if final_layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        vk::PipelineStageFlags::TRANSFER
      } else {
        vk::PipelineStageFlags::BOTTOM_OF_PIPE
      },
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::empty(),
      &[],
      &[],
      &[to_final],
    );
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn contour(points: &[(f32, f32)]) -> RecordedContour {
    RecordedContour {
      points: points.iter().map(|&(x, y)| tlottie_internal::Point { x, y }).collect(),
      closed: true,
    }
  }

  fn frame(contours: Vec<RecordedContour>, rule: tlottie_internal::Rule, argb: u32) -> EvaluatedFrame {
    let end = contours.len();
    EvaluatedFrame {
      contours,
      commands: vec![RecordedCommand {
        op: RecordedOp::Solid(tlottie_internal::SolidPaint {
          rule,
          rgba: argb,
          color: crate::math::Color::BLACK,
          opacity: 1.0,
        }),
        start: 0,
        end,
      }],
    }
  }

  #[test]
  fn geometry_cache_uploads_once_for_unchanged_contours() -> Result<()> {
    let walked = frame(vec![contour(&[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])], tlottie_internal::Rule::NonZero, 0xffff_0000);
    let mut cache = GeometryCache::default();

    let first = cache.prepare(&walked)?;
    assert_eq!(first.stats.allocated_contours, 1);
    assert_eq!(first.stats.dirty_points, 4);
    assert_eq!(first.stats.dirty_ranges, 1);
    assert_eq!(first.stats.draws, 1);
    assert_eq!(first.indirect.first().map(|draw| draw.vertex_count), Some(6));
    assert_eq!(first.indirect.first().map(|draw| draw.first_instance), Some(0));

    let second = cache.prepare(&walked)?;
    assert_eq!(second.stats.reused_contours, 1);
    assert_eq!(second.stats.dirty_points, 0);
    assert!(second.dirty.is_empty());
    Ok(())
  }

  #[test]
  fn translation_only_animation_reuses_resident_points() -> Result<()> {
    let original = [(1.0, 2.0), (5.0, 2.0), (5.0, 7.0), (1.0, 7.0)];
    let mut walked = frame(vec![contour(&original)], tlottie_internal::Rule::NonZero, 0xffff_0000);
    let mut cache = GeometryCache::default();
    cache.prepare(&walked)?;

    for point in &mut walked.contours[0].points {
      point.x += 10.0;
      point.y -= 3.0;
    }
    let translated = cache.prepare(&walked)?;
    assert_eq!(translated.stats.reused_contours, 1);
    assert_eq!(translated.stats.translated_contours, 1);
    assert_eq!(translated.stats.translated_points, 4);
    assert_eq!(translated.stats.updated_contours, 0);
    assert_eq!(translated.stats.dirty_points, 0);
    assert_eq!(cache.contours[0].transform, [1.0, 0.0, 0.0, 1.0, 10.0, -3.0]);
    assert_eq!(cache.contours[0].bounds, [11.0, -1.0, 15.0, 4.0]);
    assert_eq!(translated.draw_data[0].transform, [1.0, 0.0, 0.0, 1.0, 10.0, -3.0]);
    for (stored, &(x, y)) in cache.arena.points.iter().zip(&original) {
      assert_eq!([stored.x, stored.y], [x, y]);
    }

    let repeated = cache.prepare(&walked)?;
    assert_eq!(repeated.stats.reused_contours, 1);
    assert_eq!(repeated.stats.translated_contours, 0);
    assert_eq!(repeated.stats.dirty_points, 0);

    let mut layout = SceneLayout::default();
    let scene = build_compute_scene(32, 32, &cache.arena.points, &cache.contours, &repeated, true, false, &mut layout)?;
    assert_eq!(
      scene.sections[CONTOUR_SECTION],
      vec![0, 4, 1.0f32.to_bits(), 0.0f32.to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits(), 10.0f32.to_bits(), (-3.0f32).to_bits(),]
    );
    Ok(())
  }

  #[test]
  fn affine_animation_reuses_resident_points() -> Result<()> {
    let original = [(1.0, 2.0), (5.0, 2.0), (5.0, 7.0), (1.0, 7.0)];
    let mut walked = frame(vec![contour(&original)], tlottie_internal::Rule::NonZero, 0xff00_ff00);
    let mut cache = GeometryCache::default();
    cache.prepare(&walked)?;
    let expected = [0.5, 0.25, -0.75, 1.5, 10.0, -4.0];
    for point in &mut walked.contours[0].points {
      let mapped = apply_affine(GpuPoint::from(*point), expected);
      point.x = mapped.x;
      point.y = mapped.y;
    }

    let affine = cache.prepare(&walked)?;
    assert_eq!(affine.stats.affine_contours, 1);
    assert_eq!(affine.stats.affine_points, 4);
    assert_eq!(affine.stats.dirty_points, 0);
    assert!(cache.arena.matches_affine(cache.contours[0].range, cache.contours[0].transform, &walked.contours[0].points));
    for (stored, &(x, y)) in cache.arena.points.iter().zip(&original) {
      assert_eq!([stored.x, stored.y], [x, y]);
    }
    Ok(())
  }

  #[test]
  fn nonuniform_deformation_replaces_translated_resident_points() -> Result<()> {
    let mut walked = frame(vec![contour(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])], tlottie_internal::Rule::NonZero, 0xffff_ffff);
    let mut cache = GeometryCache::default();
    cache.prepare(&walked)?;
    for point in &mut walked.contours[0].points {
      point.x += 6.0;
      point.y += 2.0;
    }
    cache.prepare(&walked)?;
    walked.contours[0].points[2].x += 1.0;

    let deformed = cache.prepare(&walked)?;
    assert_eq!(deformed.stats.translated_contours, 0);
    assert_eq!(deformed.stats.updated_contours, 1);
    assert_eq!(deformed.stats.dirty_points, 4);
    assert_eq!(cache.contours[0].transform, IDENTITY_AFFINE);
    assert!(cache.arena.matches_affine(cache.contours[0].range, IDENTITY_AFFINE, &walked.contours[0].points));
    Ok(())
  }

  #[test]
  fn point_upload_counts_the_slice_payload() {
    let points = [GpuPoint::default(); 4];
    assert_eq!(point_payload_bytes(&points), 32);
  }

  #[test]
  fn prepares_gradient_lut_and_shader_parameters() -> Result<()> {
    let walked = EvaluatedFrame {
      contours: vec![contour(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])],
      commands: vec![RecordedCommand {
        op: RecordedOp::Gradient(tlottie_internal::GradientPaint {
          rule: tlottie_internal::Rule::NonZero,
          lut: std::sync::Arc::new([0xff12_3456; 1024]),
          transform: tlottie_internal::GradientTransform {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 2.0,
            ty: 3.0,
          },
          kind: tlottie_internal::GradientKind::Linear {
            sx: 0.0,
            sy: 0.0,
            dx: 4.0,
            dy: 0.0,
            inv_len_sq: 0.0625,
          },
          source_key: 0,
        }),
        start: 0,
        end: 1,
      }],
    };
    let prepared = GeometryCache::default().prepare(&walked)?;
    let paint = prepared.paints.first().ok_or(Error::FrameTooLarge)?;
    assert_eq!(prepared.gradient_luts.len(), 1024);
    assert_eq!(prepared.gradient_luts.first(), Some(&0xff12_3456));
    assert_eq!(paint.paint_kind, 1);
    assert_eq!(paint.params0, [0.0, 0.0, 4.0, 0.0]);
    assert_eq!(paint.params1[0], 0.0625);
    assert_eq!(prepared.stats.gradient_draws, 1);
    Ok(())
  }

  #[test]
  fn identical_gradient_paints_share_one_lut_payload() -> Result<()> {
    let gradient = tlottie_internal::GradientPaint {
      rule: tlottie_internal::Rule::NonZero,
      lut: std::sync::Arc::new(std::array::from_fn(|index| index as u32)),
      transform: tlottie_internal::GradientTransform {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
      },
      kind: tlottie_internal::GradientKind::Radial { sx: 0.0, sy: 0.0, inv_r: 0.25 },
      source_key: 0,
    };
    let walked = EvaluatedFrame {
      contours: vec![contour(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)])],
      commands: vec![
        RecordedCommand {
          op: RecordedOp::Gradient(gradient.clone()),
          start: 0,
          end: 1,
        },
        RecordedCommand {
          op: RecordedOp::Gradient(gradient),
          start: 0,
          end: 1,
        },
      ],
    };

    let prepared = GeometryCache::default().prepare(&walked)?;
    assert_eq!(prepared.gradient_luts.len(), 1024);
    assert_eq!(prepared.paints.len(), 2);
    assert_eq!(prepared.paints[0].lut_word_offset, 0);
    assert_eq!(prepared.paints[1].lut_word_offset, 0);
    Ok(())
  }

  #[test]
  fn compute_scene_preserves_even_odd_rule_and_sample_mode() -> Result<()> {
    let walked = frame(
      vec![contour(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]), contour(&[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])],
      tlottie_internal::Rule::EvenOdd,
      0xff00_ff00,
    );
    let mut cache = GeometryCache::default();
    let prepared = cache.prepare(&walked)?;
    let mut layout = SceneLayout::default();
    let scene = build_compute_scene(4, 4, &cache.arena.points, &cache.contours, &prepared, true, false, &mut layout)?;
    let paint_words = scene.sections.get(PAINT_SECTION).ok_or(Error::FrameTooLarge)?;
    assert_eq!(scene.push.antialias, 1);
    assert_eq!(paint_words.first(), Some(&0));
    assert_eq!(paint_words.get(1), Some(&2));
    assert_eq!(paint_words.get(3), Some(&1));
    for section in TILE_SECTION..SCENE_SECTION_COUNT {
      assert!(scene.sections[section].is_empty());
      assert!(scene.layout.capacities[section] > 0);
    }
    assert!(scene.layout.capacities[TILE_INDEX_SECTION] >= 1);
    assert!(scene.layout.capacities[EDGE_SECTION] >= 4);
    Ok(())
  }

  #[test]
  fn bin_key_tracks_only_spatial_scene_inputs() -> Result<()> {
    let walked = frame(vec![contour(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])], tlottie_internal::Rule::NonZero, 0xff12_3456);
    let mut cache = GeometryCache::default();
    let prepared = cache.prepare(&walked)?;
    let mut layout = SceneLayout::default();
    let mut scene = build_compute_scene(32, 32, &cache.arena.points, &cache.contours, &prepared, true, false, &mut layout)?;
    let original = compute_bin_key(&scene)?;

    scene.sections[PAINT_SECTION][4] = 0xffab_cdef;
    assert_eq!(compute_bin_key(&scene)?, original);
    scene.sections[PAINT_SECTION][2] = 1;
    assert_eq!(compute_bin_key(&scene)?, original);
    scene.sections[PAINT_SECTION][20] = 1.0f32.to_bits();
    assert_ne!(compute_bin_key(&scene)?, original);
    scene.sections[PAINT_SECTION][20] = 0.0f32.to_bits();
    assert_eq!(compute_bin_key(&scene)?, original);
    scene.sections[CONTOUR_SECTION][2] = 2.0f32.to_bits();
    assert_ne!(compute_bin_key(&scene)?, original);
    Ok(())
  }

  #[test]
  fn retained_bin_layout_ignores_unrelated_domain_relocation() {
    let previous = SceneLayout {
      offsets: [10, 20, 30, 40, 50, 60, 70, 80],
      capacities: [1, 2, 3, 4, 5, 6, 7, 8],
      ..SceneLayout::default()
    };
    let mut current = previous;
    current.offsets[LUT_SECTION] = 100;
    current.capacities[LUT_SECTION] = 16;
    assert!(retained_bin_layout_matches(previous, current));
    current.offsets[EDGE_SECTION] += 8;
    assert!(!retained_bin_layout_matches(previous, current));
  }

  #[test]
  fn scene_layout_compacts_when_capacity_is_exceeded() -> Result<()> {
    let mut layout = SceneLayout::default();
    layout.ensure(64, [3, 2, 24, 0, 8, 5, 4, 9])?;
    let first = layout;
    layout.ensure(64, [4, 1, 20, 0, 8, 6, 3, 16])?;
    assert_eq!(layout, first);
    layout.ensure(64, [5, 1, 20, 0, 8, 6, 3, 16])?;
    assert_ne!(layout, first);
    assert!(layout.capacities.get(POINT_SECTION).copied().unwrap_or(0) >= 5);
    assert_eq!(layout.offset(POINT_SECTION)?, first.offset(POINT_SECTION)?);
    assert!(layout.total_words < first.total_words + 8);
    assert_eq!(layout.offset(CONTOUR_SECTION)?, first.offset(CONTOUR_SECTION)? + 4);
    Ok(())
  }

  #[test]
  fn scene_upload_diff_skips_unchanged_words_and_coalesces_nearby_changes() {
    let previous: Vec<u32> = (0..64).collect();
    let mut current = previous.clone();
    current[3] = 100;
    current[10] = 101;
    current[40] = 102;
    assert_eq!(scene_dirty_ranges(Some(&previous), &current), vec![3..11, 40..41]);
    assert!(scene_dirty_ranges(Some(&current), &current).is_empty());
    assert_eq!(scene_dirty_ranges(None, &current), vec![0..64]);
  }

  #[test]
  fn prepares_pixel_local_layer_commands_without_geometry() -> Result<()> {
    let walked = EvaluatedFrame {
      contours: Vec::new(),
      commands: vec![
        RecordedCommand {
          op: RecordedOp::BeginLayer,
          start: 0,
          end: 0,
        },
        RecordedCommand {
          op: RecordedOp::EndLayer { opacity: 128 },
          start: 0,
          end: 0,
        },
        RecordedCommand {
          op: RecordedOp::BeginMatte,
          start: 0,
          end: 0,
        },
        RecordedCommand {
          op: RecordedOp::BeginMatteTarget,
          start: 0,
          end: 0,
        },
        RecordedCommand {
          op: RecordedOp::EndMatte {
            kind: 4,
            opacity: 127,
            source_opacity: 255,
          },
          start: 0,
          end: 0,
        },
      ],
    };
    let prepared = GeometryCache::default().prepare(&walked)?;
    assert_eq!(prepared.paints.len(), 5);
    assert_eq!(prepared.paints[0].paint_kind, 2);
    assert_eq!(prepared.paints[1].paint_kind, 3);
    assert_eq!(prepared.paints[1].argb, 0x8000_0000);
    assert_eq!(prepared.paints[2].paint_kind, 4);
    assert_eq!(prepared.paints[3].paint_kind, 5);
    assert_eq!(prepared.paints[4].paint_kind, 6);
    assert_eq!(prepared.paints[4].argb, 0x7f00_ff04);
    assert!(prepared.indirect.is_empty());
    Ok(())
  }

  #[test]
  fn prepares_mask_as_pixel_local_operation_with_geometry() -> Result<()> {
    let walked = EvaluatedFrame {
      contours: vec![contour(&[(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0)])],
      commands: vec![RecordedCommand {
        op: RecordedOp::Mask {
          mode: b's',
          inverted: true,
          opacity: 128,
          first: true,
          last: true,
        },
        start: 0,
        end: 1,
      }],
    };
    let prepared = GeometryCache::default().prepare(&walked)?;
    assert_eq!(prepared.paints.len(), 1);
    assert_eq!(prepared.paints[0].paint_kind, 7);
    assert_eq!(prepared.paints[0].argb, 0x8000_0773);
    assert_eq!(prepared.paints[0].contour_count, 1);
    assert_eq!(prepared.indirect.len(), 1);
    Ok(())
  }

  #[test]
  fn matte_factors_match_cpu_integer_semantics() {
    fn factor(rgba: u32, kind: u8) -> u32 {
      let alpha = (rgba >> 24) & 255;
      if kind == 1 {
        return alpha;
      }
      if kind == 2 {
        return 255 - alpha;
      }
      let luma = if alpha == 0 {
        0
      } else {
        let mut red = rgba & 255;
        let mut green = (rgba >> 8) & 255;
        let mut blue = (rgba >> 16) & 255;
        if alpha != 255 {
          red = red * 255 / alpha;
          green = green * 255 / alpha;
          blue = blue * 255 / alpha;
        }
        ((red * 299 + green * 587 + blue * 114) / 1000).min(255)
      };
      if kind == 3 {
        luma
      } else {
        255 - luma
      }
    }

    let matte = 0x8010_2040;
    assert_eq!(factor(matte, 1), 128);
    assert_eq!(factor(matte, 2), 127);
    assert_eq!(factor(matte, 3), 78);
    assert_eq!(factor(matte, 4), 177);
    assert_eq!(factor(0, 3), 0);
    assert_eq!(factor(0, 4), 255);
  }

  #[test]
  fn same_topology_deformation_updates_only_its_range() -> Result<()> {
    let mut walked = frame(
      vec![contour(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]), contour(&[(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])],
      tlottie_internal::Rule::NonZero,
      0xffff_ffff,
    );
    let mut cache = GeometryCache::default();
    let first = cache.prepare(&walked)?;
    let first_offsets: Vec<u32> = first.draw_data.iter().map(|draw| draw.point_offset).collect();

    let Some(changed) = walked.contours.get_mut(1).and_then(|contour| contour.points.get_mut(2)) else {
      return Err(Error::FrameTooLarge);
    };
    changed.x = 7.0;
    let second = cache.prepare(&walked)?;
    let second_offsets: Vec<u32> = second.draw_data.iter().map(|draw| draw.point_offset).collect();

    assert_eq!(second.stats.reused_contours, 1);
    assert_eq!(second.stats.updated_contours, 1);
    assert_eq!(second.stats.allocated_contours, 0);
    assert_eq!(second.stats.dirty_points, 4);
    assert_eq!(first_offsets, second_offsets);
    assert_eq!(second.dirty.first(), Some(&DirtyRange { first_point: 4, point_count: 4 }));
    Ok(())
  }

  #[test]
  fn topology_change_keeps_other_contour_allocations_stable() -> Result<()> {
    let mut walked = frame(
      vec![contour(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]), contour(&[(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])],
      tlottie_internal::Rule::EvenOdd,
      0xff00_ffff,
    );
    let mut cache = GeometryCache::default();
    let first = cache.prepare(&walked)?;
    let other_offset = first.draw_data.get(1).map(|draw| draw.point_offset);

    let Some(changed) = walked.contours.get_mut(0) else {
      return Err(Error::FrameTooLarge);
    };
    changed.points.push(tlottie_internal::Point { x: 1.0, y: 3.0 });
    let second = cache.prepare(&walked)?;

    assert_eq!(second.stats.allocated_contours, 1);
    assert_eq!(second.stats.reused_contours, 1);
    assert_eq!(second.draw_data.get(1).map(|draw| draw.point_offset), other_offset);
    assert_eq!(second.indirect.first().map(|draw| draw.vertex_count), Some(9));
    Ok(())
  }
}
const STENCIL_SAMPLE_COUNT: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_4;
