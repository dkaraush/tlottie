#[cfg(feature = "vulkan")]
mod imp {
  use std::ffi::CStr;
  use std::process::ExitCode;
  use std::time::Instant;

  use ash::{vk, Entry};
  use tlottie::{Composition, RenderOptions};

  struct VkCtx {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    timestamp_period: f32,
    multi_draw_indirect: bool,
    raster_order_groups: bool,
  }

  impl VkCtx {
    fn new() -> Result<VkCtx, String> {
      // SAFETY: dynamic loader import only.
      let entry = unsafe { Entry::load() }.map_err(|e| format!("load Vulkan: {e}"))?;
      let app_name = CStr::from_bytes_with_nul(b"tlottie-cli\0").map_err(|e| format!("app name: {e}"))?;
      let engine_name = CStr::from_bytes_with_nul(b"tlottie\0").map_err(|e| format!("engine name: {e}"))?;
      let app = vk::ApplicationInfo::builder()
        .application_name(app_name)
        .application_version(1)
        .engine_name(engine_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);
      let info = vk::InstanceCreateInfo::builder().application_info(&app);
      // SAFETY: create info references live stack data during the call.
      let instance = unsafe { entry.create_instance(&info, None) }.map_err(|e| format!("vkCreateInstance: {e:?}"))?;
      let picked = pick_device(&instance)?;
      let supported_features = unsafe { instance.get_physical_device_features(picked.physical_device) };
      let multi_draw_indirect = supported_features.multi_draw_indirect != 0 && supported_features.draw_indirect_first_instance != 0;
      let extension_properties = unsafe { instance.enumerate_device_extension_properties(picked.physical_device) }.map_err(|e| format!("vkEnumerateDeviceExtensionProperties: {e:?}"))?;
      let raster_order_groups = std::env::var_os("TLOTTIE_VK_GROUPS").is_some()
        && extension_properties
          .iter()
          .any(|extension| (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }) == vk::ExtRasterizationOrderAttachmentAccessFn::name());
      let mut raster_features = vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::default();
      let mut features2 = vk::PhysicalDeviceFeatures2::builder().push_next(&mut raster_features);
      unsafe { instance.get_physical_device_features2(picked.physical_device, &mut features2) };
      let raster_order_groups = raster_order_groups && raster_features.rasterization_order_color_attachment_access != 0;
      let enabled_features = vk::PhysicalDeviceFeatures::builder()
        .multi_draw_indirect(multi_draw_indirect)
        .draw_indirect_first_instance(multi_draw_indirect);
      let priorities = [1.0f32];
      let queues = [vk::DeviceQueueCreateInfo::builder().queue_family_index(picked.queue_family_index).queue_priorities(&priorities).build()];
      let mut device_extensions = Vec::new();
      if raster_order_groups {
        device_extensions.push(vk::ExtRasterizationOrderAttachmentAccessFn::name().as_ptr());
      }
      let mut enabled_raster = vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::builder().rasterization_order_color_attachment_access(raster_order_groups);
      let mut dinfo = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queues)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&enabled_features);
      if raster_order_groups {
        dinfo = dinfo.push_next(&mut enabled_raster);
      }
      // SAFETY: picked device/queue family came from this instance.
      let device = unsafe { instance.create_device(picked.physical_device, &dinfo, None) }.map_err(|e| format!("vkCreateDevice: {e:?}"))?;
      // SAFETY: queue 0 was requested above.
      let queue = unsafe { device.get_device_queue(picked.queue_family_index, 0) };
      let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(picked.queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
      // SAFETY: device is live and queue family belongs to it.
      let command_pool = unsafe { device.create_command_pool(&pool_info, None) }.map_err(|e| format!("vkCreateCommandPool: {e:?}"))?;
      // SAFETY: physical device belongs to this instance.
      let props = unsafe { instance.get_physical_device_properties(picked.physical_device) };
      Ok(VkCtx {
        _entry: entry,
        instance,
        physical_device: picked.physical_device,
        device,
        queue_family_index: picked.queue_family_index,
        queue,
        command_pool,
        timestamp_period: props.limits.timestamp_period,
        multi_draw_indirect,
        raster_order_groups,
      })
    }
  }

  impl Drop for VkCtx {
    fn drop(&mut self) {
      // SAFETY: context owns these resources.
      unsafe {
        let _ = self.device.device_wait_idle();
        self.device.destroy_command_pool(self.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_instance(None);
      }
    }
  }

  struct PickedDevice {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
  }

  fn pick_device(instance: &ash::Instance) -> Result<PickedDevice, String> {
    // SAFETY: instance is live.
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| format!("vkEnumeratePhysicalDevices: {e:?}"))?;
    for physical_device in devices {
      // SAFETY: physical device belongs to this instance.
      let queues = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
      for (i, q) in queues.iter().enumerate() {
        let flags = q.queue_flags;
        if flags.contains(vk::QueueFlags::GRAPHICS) || flags.contains(vk::QueueFlags::COMPUTE) {
          return Ok(PickedDevice {
            physical_device,
            queue_family_index: i as u32,
          });
        }
      }
    }
    Err("no Vulkan device with graphics/compute queue".to_string())
  }

  struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: vk::DeviceSize,
  }

  impl GpuBuffer {
    fn new(ctx: &VkCtx, bytes: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags, name: &str) -> Result<GpuBuffer, String> {
      let info = vk::BufferCreateInfo::builder().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
      // SAFETY: device is live.
      let buffer = unsafe { ctx.device.create_buffer(&info, None) }.map_err(|e| format!("vkCreateBuffer({name}): {e:?}"))?;
      // SAFETY: buffer belongs to this device.
      let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
      let mem_type = memory_type(ctx, req.memory_type_bits, properties, name)?;
      let alloc = vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(mem_type);
      // SAFETY: allocation request uses a valid memory type.
      let memory = unsafe { ctx.device.allocate_memory(&alloc, None) }.map_err(|e| format!("vkAllocateMemory({name}): {e:?}"))?;
      // SAFETY: buffer/memory belong to same device and offset is aligned.
      unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| format!("vkBindBufferMemory({name}): {e:?}"))?;
      Ok(GpuBuffer { buffer, memory, bytes })
    }

    fn read_rgba(&self, ctx: &VkCtx, out: &mut [u32]) -> Result<(), String> {
      let byte_len = out.len().checked_mul(4).ok_or_else(|| "readback size overflow".to_string())?;
      if byte_len as vk::DeviceSize > self.bytes {
        return Err("readback target too small".to_string());
      }
      // SAFETY: memory is HOST_VISIBLE|HOST_COHERENT and idle after fence.
      let ptr = unsafe { ctx.device.map_memory(self.memory, 0, byte_len as vk::DeviceSize, vk::MemoryMapFlags::empty()) }.map_err(|e| format!("vkMapMemory: {e:?}"))?;
      // SAFETY: mapped range has at least byte_len bytes; out is valid.
      unsafe {
        let src = std::slice::from_raw_parts(ptr.cast::<u32>(), out.len());
        out.copy_from_slice(src);
        ctx.device.unmap_memory(self.memory);
      }
      Ok(())
    }
  }

  impl GpuBuffer {
    fn destroy(&self, ctx: &VkCtx) {
      // SAFETY: caller waits for GPU idle before destroy.
      unsafe {
        ctx.device.destroy_buffer(self.buffer, None);
        ctx.device.free_memory(self.memory, None);
      }
    }
  }

  struct RenderImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
  }

  impl RenderImage {
    fn new(ctx: &VkCtx, width: u32, height: u32) -> Result<RenderImage, String> {
      Self::new_with(
        ctx,
        width,
        height,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::SampleCountFlags::TYPE_1,
        "render",
      )
    }

    fn new_multisample(ctx: &VkCtx, width: u32, height: u32) -> Result<RenderImage, String> {
      Self::new_with(
        ctx,
        width,
        height,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::SampleCountFlags::TYPE_4,
        "multisample color",
      )
    }

    fn new_group(ctx: &VkCtx, width: u32, height: u32) -> Result<RenderImage, String> {
      Self::new_with(
        ctx,
        width,
        height,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::SampleCountFlags::TYPE_4,
        "alpha group color",
      )
    }

    fn new_stencil(ctx: &VkCtx, width: u32, height: u32) -> Result<RenderImage, String> {
      Self::new_with(
        ctx,
        width,
        height,
        vk::Format::S8_UINT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::SampleCountFlags::TYPE_4,
        "stencil",
      )
    }

    fn new_with(ctx: &VkCtx, width: u32, height: u32, format: vk::Format, usage: vk::ImageUsageFlags, samples: vk::SampleCountFlags, label: &str) -> Result<RenderImage, String> {
      let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
      // SAFETY: device is live.
      let image = unsafe { ctx.device.create_image(&info, None) }.map_err(|e| format!("vkCreateImage({label}): {e:?}"))?;
      // SAFETY: image belongs to this device.
      let req = unsafe { ctx.device.get_image_memory_requirements(image) };
      let mem_type = if usage.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT) {
        memory_type(ctx, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED, label)
          .or_else(|_| memory_type(ctx, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, label))?
      } else {
        memory_type(ctx, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, label)?
      };
      let alloc = vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(mem_type);
      // SAFETY: allocation request uses a valid memory type.
      let memory = unsafe { ctx.device.allocate_memory(&alloc, None) }.map_err(|e| format!("vkAllocateMemory({label}): {e:?}"))?;
      // SAFETY: image/memory belong to same device and offset is aligned.
      unsafe { ctx.device.bind_image_memory(image, memory, 0) }.map_err(|e| format!("vkBindImageMemory({label}): {e:?}"))?;
      Ok(RenderImage { image, memory, width, height })
    }

    fn destroy(&self, ctx: &VkCtx) {
      // SAFETY: caller waits for GPU idle before destroy.
      unsafe {
        ctx.device.destroy_image(self.image, None);
        ctx.device.free_memory(self.memory, None);
      }
    }
  }

  fn memory_type(ctx: &VkCtx, bits: u32, properties: vk::MemoryPropertyFlags, name: &str) -> Result<u32, String> {
    // SAFETY: physical device belongs to this instance.
    let props = unsafe { ctx.instance.get_physical_device_memory_properties(ctx.physical_device) };
    for i in 0..props.memory_type_count {
      let mask = 1u32.checked_shl(i).unwrap_or(0);
      let Some(mt) = props.memory_types.get(i as usize) else {
        continue;
      };
      if bits & mask != 0 && mt.property_flags.contains(properties) {
        return Ok(i);
      }
    }
    Err(format!("no memory type with flags 0x{:x} for {name}", properties.as_raw()))
  }

  /// Runs a line-oriented benchmark worker while retaining the Vulkan device,
  /// pipelines, and fixed-size targets across animations.
  pub(crate) fn batch_with_options(width: u32, height: u32, options: RenderOptions) -> ExitCode {
    use std::io::{BufRead, Write};

    let Some(bytes) = (width as vk::DeviceSize).checked_mul(height as vk::DeviceSize).and_then(|n| n.checked_mul(4)) else {
      eprintln!("vulkan target too large");
      return ExitCode::FAILURE;
    };
    let ctx = match VkCtx::new() {
      Ok(ctx) => ctx,
      Err(error) => {
        eprintln!("vulkan init error: {error}");
        return ExitCode::FAILURE;
      }
    };
    // Complex stickers can have thousands of paints and conservative tile
    // lists much larger than the output image. Retain one generously sized
    // scene buffer in batch mode instead of failing later files or reallocating
    // between animations.
    let scratch_bytes = bytes.saturating_mul(4).max(64 * 1024 * 1024);
    let scratch = match GpuBuffer::new(
      &ctx,
      scratch_bytes,
      vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
      vk::MemoryPropertyFlags::DEVICE_LOCAL,
      "scratch pixels",
    ) {
      Ok(buffer) => buffer,
      Err(error) => {
        eprintln!("vulkan scratch buffer error: {error}");
        return ExitCode::FAILURE;
      }
    };
    let image = match RenderImage::new(&ctx, width, height) {
      Ok(image) => image,
      Err(error) => {
        scratch.destroy(&ctx);
        eprintln!("vulkan render image error: {error}");
        return ExitCode::FAILURE;
      }
    };
    let stencil = match RenderImage::new_stencil(&ctx, width, height) {
      Ok(image) => image,
      Err(error) => {
        image.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan stencil image error: {error}");
        return ExitCode::FAILURE;
      }
    };
    let multisample = match RenderImage::new_multisample(&ctx, width, height) {
      Ok(image) => image,
      Err(error) => {
        stencil.destroy(&ctx);
        image.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan multisample image error: {error}");
        return ExitCode::FAILURE;
      }
    };
    let group = if ctx.raster_order_groups {
      match RenderImage::new_group(&ctx, width, height) {
        Ok(image) => Some(image),
        Err(error) => {
          multisample.destroy(&ctx);
          stencil.destroy(&ctx);
          image.destroy(&ctx);
          scratch.destroy(&ctx);
          eprintln!("vulkan alpha-group image error: {error}");
          return ExitCode::FAILURE;
        }
      }
    } else {
      None
    };
    let staging = match GpuBuffer::new(
      &ctx,
      bytes,
      vk::BufferUsageFlags::TRANSFER_DST,
      vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
      "readback pixels",
    ) {
      Ok(buffer) => buffer,
      Err(error) => {
        if let Some(group) = &group {
          group.destroy(&ctx);
        }
        multisample.destroy(&ctx);
        stencil.destroy(&ctx);
        image.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan staging buffer error: {error}");
        return ExitCode::FAILURE;
      }
    };
    let mut renderer = match tlottie::vulkan::VulkanRenderer::new_with_raster_order_groups(&ctx.device, ctx.raster_order_groups) {
      Ok(renderer) => renderer,
      Err(error) => {
        staging.destroy(&ctx);
        if let Some(group) = &group {
          group.destroy(&ctx);
        }
        multisample.destroy(&ctx);
        stencil.destroy(&ctx);
        image.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("tlottie-vulkan init error: {error}");
        return ExitCode::FAILURE;
      }
    };
    if std::env::var("TLOTTIE_VK_MODE").as_deref() == Ok("stencil") {
      renderer.set_mode(tlottie::vulkan::RendererMode::StencilCover);
    }
    renderer.set_multi_draw_indirect(ctx.multi_draw_indirect);

    let pixel_count = width as usize * height as usize;
    let mut pixels = vec![0u32; pixel_count];
    let mut raw_bytes = Vec::with_capacity(pixel_count.saturating_mul(4));
    let mut image_layout = vk::ImageLayout::UNDEFINED;
    let stdin = std::io::stdin();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
      let line = match line {
        Ok(line) => line,
        Err(error) => {
          eprintln!("vulkan batch input error: {error}");
          break;
        }
      };
      let fields = line.split('\t').collect::<Vec<_>>();
      let response = if let [file, start, frames, raw_path] = fields.as_slice() {
        let request = start.parse::<f32>().ok().zip(frames.parse::<u32>().ok()).filter(|(_, frames)| *frames > 0);
        match request {
          Some((start, frames)) if start.is_finite() => match std::fs::read(file)
            .map_err(|error| error.to_string())
            .and_then(|bytes| Composition::parse(&bytes, &tlottie::Limits::default()).map_err(|error| error.to_string()))
          {
            Ok(comp) => {
              let mut raw_file = if raw_path == &"-" {
                None
              } else {
                std::path::Path::new(raw_path)
                  .parent()
                  .map(std::fs::create_dir_all)
                  .transpose()
                  .and_then(|_| std::fs::File::create(raw_path))
                  .ok()
              };
              let raw_ready = raw_path == &"-" || raw_file.is_some();
              let frame_count = comp.frame_count().max(1);
              let mut ok = raw_ready;
              for index in 0..frames {
                let sequence_frame = (start + index as f32) % frame_count as f32;
                let code = record_submit_read(
                  &ctx,
                  &scratch,
                  &image,
                  &stencil,
                  &multisample,
                  group.as_ref(),
                  &staging,
                  &mut renderer,
                  &comp,
                  sequence_frame,
                  &mut pixels,
                  width,
                  height,
                  options,
                  image_layout,
                );
                if code != ExitCode::SUCCESS {
                  ok = false;
                  break;
                }
                image_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                if let Some(file) = raw_file.as_mut() {
                  raw_bytes.clear();
                  raw_bytes.extend(pixels.iter().flat_map(|pixel| pixel.to_le_bytes()));
                  if file.write_all(&raw_bytes).is_err() {
                    ok = false;
                    break;
                  }
                }
              }
              if ok {
                format!("OK\t{frame_count}\t{frames}")
              } else {
                "ERR\trender".to_string()
              }
            }
            Err(error) => format!("ERR\t{}", error.replace(['\t', '\n', '\r'], " ")),
          },
          _ => "ERR\tbad request".to_string(),
        }
      } else {
        "ERR\tbad request".to_string()
      };
      if writeln!(stdout, "{response}").and_then(|_| stdout.flush()).is_err() {
        break;
      }
    }

    // Every submitted frame is fence-complete before the next request.
    drop(renderer);
    staging.destroy(&ctx);
    if let Some(group) = &group {
      group.destroy(&ctx);
    }
    multisample.destroy(&ctx);
    stencil.destroy(&ctx);
    image.destroy(&ctx);
    scratch.destroy(&ctx);
    ExitCode::SUCCESS
  }

  pub(crate) fn render_with_options(comp: &Composition, frame: f32, pixels: &mut [u32], width: u32, height: u32, options: RenderOptions) -> ExitCode {
    let Some(bytes) = (width as vk::DeviceSize).checked_mul(height as vk::DeviceSize).and_then(|n| n.checked_mul(4)) else {
      eprintln!("vulkan target too large");
      return ExitCode::FAILURE;
    };
    let ctx = match VkCtx::new() {
      Ok(ctx) => ctx,
      Err(e) => {
        eprintln!("vulkan init error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let scratch_bytes = bytes.saturating_mul(4).max(64 * 1024 * 1024);
    let scratch = match GpuBuffer::new(
      &ctx,
      scratch_bytes,
      vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
      vk::MemoryPropertyFlags::DEVICE_LOCAL,
      "scratch pixels",
    ) {
      Ok(b) => b,
      Err(e) => {
        eprintln!("vulkan scratch buffer error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let image = match RenderImage::new(&ctx, width, height) {
      Ok(image) => image,
      Err(e) => {
        scratch.destroy(&ctx);
        eprintln!("vulkan render image error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let stencil = match RenderImage::new_stencil(&ctx, width, height) {
      Ok(image) => image,
      Err(e) => {
        image.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan stencil image error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let multisample = match RenderImage::new_multisample(&ctx, width, height) {
      Ok(image) => image,
      Err(e) => {
        image.destroy(&ctx);
        stencil.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan multisample image error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let group = if ctx.raster_order_groups {
      match RenderImage::new_group(&ctx, width, height) {
        Ok(image) => Some(image),
        Err(e) => {
          image.destroy(&ctx);
          stencil.destroy(&ctx);
          multisample.destroy(&ctx);
          scratch.destroy(&ctx);
          eprintln!("vulkan alpha-group image error: {e}");
          return ExitCode::FAILURE;
        }
      }
    } else {
      None
    };
    let staging = match GpuBuffer::new(
      &ctx,
      bytes,
      vk::BufferUsageFlags::TRANSFER_DST,
      vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
      "readback pixels",
    ) {
      Ok(b) => b,
      Err(e) => {
        image.destroy(&ctx);
        stencil.destroy(&ctx);
        multisample.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("vulkan staging buffer error: {e}");
        return ExitCode::FAILURE;
      }
    };
    let mut renderer = match tlottie::vulkan::VulkanRenderer::new_with_raster_order_groups(&ctx.device, ctx.raster_order_groups) {
      Ok(renderer) => renderer,
      Err(e) => {
        staging.destroy(&ctx);
        image.destroy(&ctx);
        stencil.destroy(&ctx);
        multisample.destroy(&ctx);
        scratch.destroy(&ctx);
        eprintln!("tlottie-vulkan init error: {e}");
        return ExitCode::FAILURE;
      }
    };
    if std::env::var("TLOTTIE_VK_MODE").as_deref() == Ok("stencil") {
      renderer.set_mode(tlottie::vulkan::RendererMode::StencilCover);
    }
    renderer.set_multi_draw_indirect(ctx.multi_draw_indirect);
    let sequence_frames = std::env::var("TLOTTIE_VK_FRAMES").ok().and_then(|value| value.parse::<u32>().ok()).unwrap_or(1).max(1);
    let sequence_step = std::env::var("TLOTTIE_VK_FRAME_STEP")
      .ok()
      .and_then(|value| value.parse::<f32>().ok())
      .filter(|value| value.is_finite())
      .unwrap_or(1.0);
    let frame_count = comp.frame_count().max(1);
    let raw_dir = std::env::var_os("TLOTTIE_VK_RAW_DIR").map(std::path::PathBuf::from);
    if let Some(dir) = &raw_dir {
      if let Err(error) = std::fs::create_dir_all(dir) {
        eprintln!("vulkan raw output directory error: {error}");
        drop(renderer);
        staging.destroy(&ctx);
        image.destroy(&ctx);
        stencil.destroy(&ctx);
        multisample.destroy(&ctx);
        scratch.destroy(&ctx);
        return ExitCode::FAILURE;
      }
    }
    let mut code = ExitCode::SUCCESS;
    for index in 0..sequence_frames {
      let sequence_frame = (frame + index as f32 * sequence_step) % frame_count as f32;
      code = record_submit_read(
        &ctx,
        &scratch,
        &image,
        &stencil,
        &multisample,
        group.as_ref(),
        &staging,
        &mut renderer,
        comp,
        sequence_frame,
        pixels,
        width,
        height,
        options,
        if index == 0 { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL },
      );
      if code != ExitCode::SUCCESS {
        break;
      }
      if let Some(dir) = &raw_dir {
        let path = dir.join(format!("frame-{index:06}.rgba"));
        if let Err(error) = write_rgba_raw(&path, pixels) {
          eprintln!("vulkan raw output error: {error}");
          code = ExitCode::FAILURE;
          break;
        }
      }
    }
    // SAFETY: `record_submit_read` waits for completion before returning.
    drop(renderer);
    staging.destroy(&ctx);
    image.destroy(&ctx);
    stencil.destroy(&ctx);
    multisample.destroy(&ctx);
    if let Some(group) = &group {
      group.destroy(&ctx);
    }
    scratch.destroy(&ctx);
    code
  }

  fn write_rgba_raw(path: &std::path::Path, pixels: &[u32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(pixels.len().saturating_mul(4));
    for pixel in pixels {
      bytes.extend_from_slice(&pixel.to_le_bytes());
    }
    std::fs::write(path, bytes)
  }

  fn record_submit_read(
    ctx: &VkCtx,
    scratch: &GpuBuffer,
    image: &RenderImage,
    stencil: &RenderImage,
    multisample: &RenderImage,
    group: Option<&RenderImage>,
    staging: &GpuBuffer,
    renderer: &mut tlottie::vulkan::VulkanRenderer<'_>,
    comp: &Composition,
    frame: f32,
    pixels: &mut [u32],
    width: u32,
    height: u32,
    options: RenderOptions,
    image_layout: vk::ImageLayout,
  ) -> ExitCode {
    let alloc = vk::CommandBufferAllocateInfo::builder()
      .command_pool(ctx.command_pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);
    // SAFETY: command pool is live.
    let bufs = match unsafe { ctx.device.allocate_command_buffers(&alloc) } {
      Ok(b) => b,
      Err(e) => {
        eprintln!("vkAllocateCommandBuffers: {e:?}");
        return ExitCode::FAILURE;
      }
    };
    let Some(&cmd) = bufs.first() else {
      eprintln!("vkAllocateCommandBuffers returned no command buffers");
      return ExitCode::FAILURE;
    };
    let query_pool = create_timestamp_pool(ctx);
    let begin = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: command buffer belongs to resettable pool and is initial.
    if let Err(e) = unsafe { ctx.device.begin_command_buffer(cmd, &begin) } {
      eprintln!("vkBeginCommandBuffer: {e:?}");
      return ExitCode::FAILURE;
    }
    if let Some(qp) = query_pool {
      // SAFETY: query pool has two timestamp queries.
      unsafe {
        ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
        ctx.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
      }
    }
    let scratch_target = tlottie::vulkan::BufferTarget {
      buffer: scratch.buffer,
      width,
      height,
      bytes: scratch.bytes,
    };
    let image_target = tlottie::vulkan::ImageTarget {
      image: image.image,
      format: vk::Format::B8G8R8A8_UNORM,
      width: image.width,
      height: image.height,
      layout: image_layout,
      final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
      stencil_image: Some(stencil.image),
      multisample_image: Some(multisample.image),
      group_multisample_image: group.map(|image| image.image),
    };
    // SAFETY: command buffer is recording; scratch buffer and image are
    // transfer resources sized for the full target.
    let record_t0 = Instant::now();
    if let Err(e) = unsafe { renderer.record(cmd, scratch_target, image_target, comp, frame, options) } {
      eprintln!("tlottie-vulkan draw error: {e}");
      return ExitCode::FAILURE;
    }
    let record_ns = record_t0.elapsed().as_nanos() as u64;
    let cache_stats = renderer.cache_stats();
    if let Some(qp) = query_pool {
      // SAFETY: query pool has two timestamp queries.
      unsafe {
        ctx.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
      }
    }
    record_image_readback(ctx, cmd, image, staging);
    // SAFETY: command buffer is recording.
    if let Err(e) = unsafe { ctx.device.end_command_buffer(cmd) } {
      eprintln!("vkEndCommandBuffer: {e:?}");
      return ExitCode::FAILURE;
    }
    let cmds = [cmd];
    let submits = [vk::SubmitInfo::builder().command_buffers(&cmds).build()];
    let fence_info = vk::FenceCreateInfo::builder();
    // SAFETY: device is live.
    let fence = match unsafe { ctx.device.create_fence(&fence_info, None) } {
      Ok(f) => f,
      Err(e) => {
        eprintln!("vkCreateFence: {e:?}");
        return ExitCode::FAILURE;
      }
    };
    let submit_t0 = Instant::now();
    // SAFETY: queue/cmd/fence belong to this device.
    if let Err(e) = unsafe { ctx.device.queue_submit(ctx.queue, &submits, fence) } {
      eprintln!("vkQueueSubmit: {e:?}");
      return ExitCode::FAILURE;
    }
    // SAFETY: fence is live.
    if let Err(e) = unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX) } {
      eprintln!("vkWaitForFences: {e:?}");
      return ExitCode::FAILURE;
    }
    let submit_wait_ns = submit_t0.elapsed().as_nanos() as u64;
    let gpu_ns = query_pool.and_then(|qp| read_timestamps_ns(ctx, qp));
    if let Err(e) = staging.read_rgba(ctx, pixels) {
      eprintln!("vulkan readback error: {e}");
      return ExitCode::FAILURE;
    }
    // SAFETY: fence completed; resources are idle.
    unsafe {
      ctx.device.destroy_fence(fence, None);
      if let Some(qp) = query_pool {
        ctx.device.destroy_query_pool(qp, None);
      }
      ctx.device.free_command_buffers(ctx.command_pool, &cmds);
    }
    eprintln!(
            "VK queue_family={} record_ns={} submit_wait_ns={} gpu_elapsed_ns={} solids={} gradients={} reused={} translated={} translated_points={} affine={} affine_points={} updated={} dirty_points={} reused_bins={} upload_bytes={} upload_ranges={} geometry_bytes={} paint_bytes={} bin_bytes={}",
            ctx.queue_family_index,
            record_ns,
            submit_wait_ns,
            gpu_ns
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            cache_stats.solid_draws,
            cache_stats.gradient_draws,
            cache_stats.reused_contours,
            cache_stats.translated_contours,
            cache_stats.translated_points,
            cache_stats.affine_contours,
            cache_stats.affine_points,
            cache_stats.updated_contours,
            cache_stats.dirty_points,
            u32::from(cache_stats.reused_bins),
            cache_stats.scene_upload_bytes,
            cache_stats.scene_upload_ranges,
            cache_stats.geometry_upload_bytes,
            cache_stats.paint_upload_bytes,
            cache_stats.bin_upload_bytes,
        );
    ExitCode::SUCCESS
  }

  fn record_image_readback(ctx: &VkCtx, cmd: vk::CommandBuffer, image: &RenderImage, staging: &GpuBuffer) {
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
        width: image.width,
        height: image.height,
        depth: 1,
      })
      .build();
    // SAFETY: renderer leaves image in TRANSFER_SRC_OPTIMAL; staging has
    // transfer-dst usage and is large enough for the image.
    unsafe {
      ctx.device.cmd_copy_image_to_buffer(cmd, image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, staging.buffer, &[copy]);
    }
  }

  fn create_timestamp_pool(ctx: &VkCtx) -> Option<vk::QueryPool> {
    // SAFETY: physical device belongs to this instance.
    let props = unsafe { ctx.instance.get_physical_device_queue_family_properties(ctx.physical_device) };
    let supports_timestamps = props.get(ctx.queue_family_index as usize).map(|q| q.timestamp_valid_bits > 0).unwrap_or(false);
    if !supports_timestamps {
      return None;
    }
    let info = vk::QueryPoolCreateInfo::builder().query_type(vk::QueryType::TIMESTAMP).query_count(2);
    // SAFETY: device is live; create info is valid.
    unsafe { ctx.device.create_query_pool(&info, None) }.ok()
  }

  fn read_timestamps_ns(ctx: &VkCtx, qp: vk::QueryPool) -> Option<u64> {
    let mut ticks = [0u64; 2];
    // SAFETY: fence has completed and query pool contains two timestamp
    // queries. WAIT flag is harmless after completion.
    let res = unsafe { ctx.device.get_query_pool_results(qp, 0, 2, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) };
    if res.is_err() {
      return None;
    }
    let end = *ticks.get(1)?;
    let start = *ticks.first()?;
    Some(((end.saturating_sub(start) as f64) * ctx.timestamp_period as f64) as u64)
  }
}

#[cfg(feature = "vulkan")]
pub(crate) use imp::batch_with_options as batch;
#[cfg(feature = "vulkan")]
pub(crate) use imp::render_with_options as render;

#[cfg(not(feature = "vulkan"))]
pub(crate) fn render(_comp: &tlottie::Composition, _frame: f32, _pixels: &mut [u32], _width: u32, _height: u32, _options: tlottie::RenderOptions) -> std::process::ExitCode {
  eprintln!("tlottie-cli was built without --features vulkan");
  std::process::ExitCode::FAILURE
}

#[cfg(not(feature = "vulkan"))]
pub(crate) fn batch(_width: u32, _height: u32, _options: tlottie::RenderOptions) -> std::process::ExitCode {
  eprintln!("tlottie-cli was built without --features vulkan");
  std::process::ExitCode::FAILURE
}
