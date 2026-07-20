use std::ffi::{c_void, CStr};
use std::time::Instant;

use ash::extensions::khr::{AndroidSurface, Surface, Swapchain};
use ash::{vk, Entry};
use jni::objects::JObject;
use jni::sys::jobject;
use jni::JNIEnv;
use tlottie::{Composition, RenderOptions};

#[repr(C)]
struct ANativeWindow {
    _private: [u8; 0],
}

#[link(name = "android")]
extern "C" {
    fn ANativeWindow_fromSurface(env: *mut c_void, surface: jobject) -> *mut ANativeWindow;
    fn ANativeWindow_release(window: *mut ANativeWindow);
}

struct Context {
    window: *mut ANativeWindow,
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    timestamp_period: f32,
    timestamp_valid_bits: u32,
    multi_draw_indirect: bool,
    raster_order_groups: bool,
    command_pool: vk::CommandPool,
    swapchain_loader: Swapchain,
}

impl Context {
    fn new(env: &JNIEnv<'_>, java_surface: &JObject<'_>) -> Result<Self, String> {
        // SAFETY: Android owns both JNI pointers for the duration of this JNI
        // call. ANativeWindow_fromSurface acquires a reference for this host.
        let window = unsafe {
            ANativeWindow_fromSurface(
                env.get_native_interface().cast::<c_void>(),
                java_surface.as_raw(),
            )
        };
        if window.is_null() {
            return Err("ANativeWindow_fromSurface returned null".to_string());
        }

        let result = Self::new_with_window(window);
        if result.is_err() {
            // SAFETY: fromSurface acquired exactly one reference above.
            unsafe { ANativeWindow_release(window) };
        }
        result
    }

    fn new_with_window(window: *mut ANativeWindow) -> Result<Self, String> {
        // SAFETY: ash only loads the Android Vulkan loader entry points here.
        let entry = unsafe { Entry::load() }.map_err(|error| format!("load Vulkan: {error}"))?;
        let name = CStr::from_bytes_with_nul(b"tlottie-android\0")
            .map_err(|error| format!("Vulkan app name: {error}"))?;
        let app = vk::ApplicationInfo::builder()
            .application_name(name)
            .application_version(1)
            .engine_name(name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_2);
        let extensions = [Surface::name().as_ptr(), AndroidSurface::name().as_ptr()];
        let create = vk::InstanceCreateInfo::builder()
            .application_info(&app)
            .enabled_extension_names(&extensions);
        // SAFETY: create info only borrows live stack data during this call.
        let instance = unsafe { entry.create_instance(&create, None) }
            .map_err(|error| format!("vkCreateInstance: {error:?}"))?;
        let surface_loader = Surface::new(&entry, &instance);
        let android_surface = AndroidSurface::new(&entry, &instance);
        let surface_info = vk::AndroidSurfaceCreateInfoKHR::builder().window(window.cast());
        // SAFETY: window is a live acquired ANativeWindow.
        let surface = unsafe { android_surface.create_android_surface(&surface_info, None) }
            .map_err(|error| format!("vkCreateAndroidSurfaceKHR: {error:?}"))?;

        let (physical_device, queue_family, timestamp_valid_bits) =
            pick_device(&instance, &surface_loader, surface)?;
        // SAFETY: physical_device belongs to instance.
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let supported_features =
            unsafe { instance.get_physical_device_features(physical_device) };
        let multi_draw_indirect = supported_features.multi_draw_indirect != 0
            && supported_features.draw_indirect_first_instance != 0;
        let device_extension_properties = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
        }
        .map_err(|error| format!("vkEnumerateDeviceExtensionProperties: {error:?}"))?;
        let arm_raster_order = device_extension_properties.iter().any(|extension| {
            (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
                == ash::vk::ArmRasterizationOrderAttachmentAccessFn::name()
        });
        let multisampled_to_single = device_extension_properties.iter().any(|extension| {
            (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
                == vk::ExtMultisampledRenderToSingleSampledFn::name()
        });
        let mut raster_order_support =
            vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM::default();
        let mut multisampled_support =
            vk::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut raster_order_support)
            .push_next(&mut multisampled_support);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let raster_order_groups = arm_raster_order
            && multisampled_to_single
            && properties.api_version >= vk::API_VERSION_1_2
            && raster_order_support.rasterization_order_color_attachment_access != 0
            && multisampled_support.multisampled_render_to_single_sampled != 0;
        // Disabled after the Mali driver crashed while compiling the feedback
        // pipeline. Keep the proven stencil+cover/compute routing installed.
        let raster_order_groups = false;
        let enabled_features = vk::PhysicalDeviceFeatures::builder()
            .multi_draw_indirect(multi_draw_indirect)
            .draw_indirect_first_instance(multi_draw_indirect);
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)
            .build()];
        let mut device_extensions = vec![Swapchain::name().as_ptr()];
        if raster_order_groups {
            device_extensions.push(
                ash::vk::ArmRasterizationOrderAttachmentAccessFn::name().as_ptr(),
            );
            device_extensions.push(
                vk::ExtMultisampledRenderToSingleSampledFn::name().as_ptr(),
            );
        }
        let mut enabled_raster_order =
            vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM::builder()
                .rasterization_order_color_attachment_access(raster_order_groups);
        let mut enabled_multisampled =
            vk::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT::builder()
                .multisampled_render_to_single_sampled(raster_order_groups);
        let mut device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&enabled_features);
        if raster_order_groups {
            device_info = device_info
                .push_next(&mut enabled_raster_order)
                .push_next(&mut enabled_multisampled);
        }
        // SAFETY: physical device and queue family came from this instance and
        // support graphics plus presentation to this surface.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|error| format!("vkCreateDevice: {error:?}"))?;
        // SAFETY: queue zero was requested above.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: device and queue family are live and compatible.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|error| format!("vkCreateCommandPool: {error:?}"))?;
        let swapchain_loader = Swapchain::new(&instance, &device);

        Ok(Self {
            window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            timestamp_period: properties.limits.timestamp_period,
            timestamp_valid_bits,
            multi_draw_indirect,
            raster_order_groups,
            command_pool,
            swapchain_loader,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: GpuRenderer drops its swapchain and device objects before the
        // Context field itself is dropped.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
            ANativeWindow_release(self.window);
        }
    }
}

fn pick_device(
    instance: &ash::Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), String> {
    // SAFETY: instance is live.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| format!("vkEnumeratePhysicalDevices: {error:?}"))?;
    for physical_device in devices {
        // SAFETY: physical device belongs to instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in families.iter().enumerate() {
            if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            // SAFETY: physical device, family, and surface belong to instance.
            let present = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )
            }
            .map_err(|error| format!("vkGetPhysicalDeviceSurfaceSupportKHR: {error:?}"))?;
            if present {
                return Ok((physical_device, index as u32, family.timestamp_valid_bits));
            }
        }
    }
    Err("no Vulkan queue supports graphics and this TextureView surface".to_string())
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: vk::DeviceSize,
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Image {
    fn new(context: &Context, extent: vk::Extent2D) -> Result<Self, String> {
        Self::new_with(
            context,
            extent,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::SampleCountFlags::TYPE_1,
            "offscreen",
        )
    }

    fn new_multisample(context: &Context, extent: vk::Extent2D) -> Result<Self, String> {
        Self::new_with(
            context,
            extent,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::SampleCountFlags::TYPE_4,
            "multisample color",
        )
    }

    fn new_group(context: &Context, extent: vk::Extent2D) -> Result<Self, String> {
        Self::new_with(
            context,
            extent,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::SampleCountFlags::TYPE_1,
            "alpha group color",
        )
    }

    fn new_stencil(context: &Context, extent: vk::Extent2D) -> Result<Self, String> {
        Self::new_with(
            context,
            extent,
            vk::Format::S8_UINT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | if context.raster_order_groups {
                    vk::ImageUsageFlags::empty()
                } else {
                    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                },
            if context.raster_order_groups {
                vk::SampleCountFlags::TYPE_1
            } else {
                vk::SampleCountFlags::TYPE_4
            },
            "stencil",
        )
    }

    fn new_with(
        context: &Context,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
        label: &str,
    ) -> Result<Self, String> {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: device is live and create info is self-contained.
        let image = unsafe { context.device.create_image(&info, None) }
            .map_err(|error| format!("vkCreateImage({label}): {error:?}"))?;
        // SAFETY: image belongs to device.
        let requirements = unsafe { context.device.get_image_memory_requirements(image) };
        let memory_type = if usage.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT) {
            memory_type(
                context,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
                label,
            )
            .or_else(|_| {
                memory_type(
                    context,
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    label,
                )
            })?
        } else {
            memory_type(
                context,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                label,
            )?
        };
        let allocation = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: allocation uses a compatible device-local memory type.
        let memory = unsafe { context.device.allocate_memory(&allocation, None) }
            .map_err(|error| format!("vkAllocateMemory({label}): {error:?}"))?;
        // SAFETY: image and memory belong to one device at a valid offset.
        unsafe { context.device.bind_image_memory(image, memory, 0) }
            .map_err(|error| format!("vkBindImageMemory({label}): {error:?}"))?;
        Ok(Self { image, memory })
    }

    fn destroy(&self, context: &Context) {
        // SAFETY: device is idle and this wrapper owns both handles.
        unsafe {
            context.device.destroy_image(self.image, None);
            context.device.free_memory(self.memory, None);
        }
    }
}

impl Buffer {
    fn new(
        context: &Context,
        bytes: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        label: &str,
    ) -> Result<Self, String> {
        let info = vk::BufferCreateInfo::builder()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: device is live.
        let buffer = unsafe { context.device.create_buffer(&info, None) }
            .map_err(|error| format!("vkCreateBuffer({label}): {error:?}"))?;
        // SAFETY: buffer belongs to device.
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = memory_type(context, requirements.memory_type_bits, properties, label)?;
        let allocation = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: allocation uses a compatible memory type.
        let memory = unsafe { context.device.allocate_memory(&allocation, None) }
            .map_err(|error| format!("vkAllocateMemory({label}): {error:?}"))?;
        // SAFETY: buffer and memory belong to one device and offset zero is
        // aligned to the allocation requirements.
        unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|error| format!("vkBindBufferMemory({label}): {error:?}"))?;
        Ok(Self {
            buffer,
            memory,
            bytes,
        })
    }

    fn destroy(&self, context: &Context) {
        // SAFETY: device is idle and this wrapper owns both handles.
        unsafe {
            context.device.destroy_buffer(self.buffer, None);
            context.device.free_memory(self.memory, None);
        }
    }
}

fn memory_type(
    context: &Context,
    bits: u32,
    properties: vk::MemoryPropertyFlags,
    label: &str,
) -> Result<u32, String> {
    // SAFETY: physical device belongs to instance.
    let memory = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical_device)
    };
    for index in 0..memory.memory_type_count {
        let mask = 1u32.checked_shl(index).unwrap_or(0);
        let Some(memory_type) = memory.memory_types.get(index as usize) else {
            continue;
        };
        if bits & mask != 0 && memory_type.property_flags.contains(properties) {
            return Ok(index);
        }
    }
    Err(format!("no compatible Vulkan memory type for {label}"))
}

struct SwapchainState {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    initialized: Vec<bool>,
    extent: vk::Extent2D,
    format: vk::Format,
    direct_resolve: bool,
}

fn create_swapchain(
    context: &Context,
    requested_width: u32,
    requested_height: u32,
) -> Result<SwapchainState, String> {
    // SAFETY: physical device and surface are live and associated.
    let capabilities = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_capabilities(context.physical_device, context.surface)
    }
    .map_err(|error| format!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR: {error:?}"))?;
    // SAFETY: physical device and surface are live and associated.
    let formats = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)
    }
    .map_err(|error| format!("vkGetPhysicalDeviceSurfaceFormatsKHR: {error:?}"))?;
    let format = formats
        .iter()
        .copied()
        .find(|candidate| candidate.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| {
            formats
                .iter()
                .copied()
                .find(|candidate| candidate.format == vk::Format::R8G8B8A8_UNORM)
        })
        .ok_or_else(|| "TextureView supports neither BGRA8 nor RGBA8 UNORM".to_string())?;
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        return Err(
            "TextureView swapchain lacks Vulkan transfer source/destination usage".to_string(),
        );
    }
    let direct_resolve = format.format == vk::Format::B8G8R8A8_UNORM
        && capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    let image_usage = vk::ImageUsageFlags::TRANSFER_DST
        | if direct_resolve {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::empty()
        };
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: requested_width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: requested_height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };
    if extent.width == 0 || extent.height == 0 {
        return Err("TextureView surface has zero extent".to_string());
    }
    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count != 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    let composite_alpha = [
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ]
    .into_iter()
    .find(|mode| capabilities.supported_composite_alpha.contains(*mode))
    .ok_or_else(|| "TextureView exposes no supported composite alpha mode".to_string())?;
    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(context.surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(image_usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(composite_alpha)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true);
    // SAFETY: surface/device are live and the requested values came from its
    // queried capabilities.
    let swapchain = unsafe { context.swapchain_loader.create_swapchain(&info, None) }
        .map_err(|error| format!("vkCreateSwapchainKHR: {error:?}"))?;
    // SAFETY: swapchain belongs to this loader/device.
    let images = unsafe { context.swapchain_loader.get_swapchain_images(swapchain) }
        .map_err(|error| format!("vkGetSwapchainImagesKHR: {error:?}"))?;
    let initialized = vec![false; images.len()];
    Ok(SwapchainState {
        swapchain,
        images,
        initialized,
        extent,
        format: format.format,
        direct_resolve,
    })
}

struct FrameSlot {
    renderer: Option<tlottie::vulkan::VulkanRenderer<'static>>,
    scratch: Buffer,
    offscreen: Image,
    multisample: Option<Image>,
    group: Option<Image>,
    stencil: Image,
    offscreen_initialized: bool,
    command: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    frame_fence: vk::Fence,
    query_ready: bool,
}

/// Direct TextureView Vulkan presenter. No image is copied back to the CPU.
pub struct GpuRenderer {
    slots: Vec<FrameSlot>,
    next_slot: usize,
    swapchain: SwapchainState,
    timestamp_queries: vk::QueryPool,
    latest_gpu_ns: u64,
    context: Box<Context>,
}

impl GpuRenderer {
    pub fn new(
        env: &JNIEnv<'_>,
        surface: &JObject<'_>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let context = Box::new(Context::new(env, surface)?);
        let swapchain = create_swapchain(&context, width, height)?;
        let pixels = vk::DeviceSize::from(swapchain.extent.width)
            .checked_mul(vk::DeviceSize::from(swapchain.extent.height))
            .ok_or_else(|| "Vulkan surface dimensions overflow".to_string())?;
        let output_bytes = pixels
            .checked_mul(4)
            .ok_or_else(|| "Vulkan surface byte size overflow".to_string())?;
        // A third in-flight frame improves pacing for large targets on the
        // tested Mali device; smaller targets retain lower latency and memory.
        let frame_slot_count = if swapchain.extent.width.max(swapchain.extent.height) >= 640 {
            3
        } else {
            2
        };
        let allocation = vk::CommandBufferAllocateInfo::builder()
            .command_pool(context.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frame_slot_count as u32);
        // SAFETY: command pool belongs to device.
        let commands = unsafe { context.device.allocate_command_buffers(&allocation) }
            .map_err(|error| format!("vkAllocateCommandBuffers: {error:?}"))?;
        if commands.len() != frame_slot_count {
            return Err("Vulkan returned too few command buffers".to_string());
        }
        if context.timestamp_valid_bits == 0 {
            return Err("selected Vulkan queue does not support timestamp queries".to_string());
        }
        let query_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count((frame_slot_count * 7) as u32);
        // SAFETY: device is live and timestamp queries are supported by the queue.
        let timestamp_queries = unsafe { context.device.create_query_pool(&query_info, None) }
            .map_err(|error| format!("vkCreateQueryPool(timestamp): {error:?}"))?;

        // Context is boxed, so its ash::Device has a stable address. Drop
        // destroys VulkanRenderer before Context, making this extended borrow sound.
        let device: &'static ash::Device = unsafe { &*(&context.device as *const ash::Device) };
        let mut slots = Vec::with_capacity(frame_slot_count);
        for command in commands {
            let scratch = Buffer::new(
                &context,
                output_bytes.saturating_mul(4).max(1024 * 1024),
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "scene scratch",
            )?;
            let offscreen = Image::new(&context, swapchain.extent)?;
            let multisample = if context.raster_order_groups {
                None
            } else {
                Some(Image::new_multisample(&context, swapchain.extent)?)
            };
            let group = if context.raster_order_groups {
                Some(Image::new_group(&context, swapchain.extent)?)
            } else {
                None
            };
            let stencil = Image::new_stencil(&context, swapchain.extent)?;
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            let image_available = unsafe { context.device.create_semaphore(&semaphore_info, None) }
                .map_err(|error| format!("vkCreateSemaphore(acquire): {error:?}"))?;
            let render_finished = unsafe { context.device.create_semaphore(&semaphore_info, None) }
                .map_err(|error| format!("vkCreateSemaphore(render): {error:?}"))?;
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let frame_fence = unsafe { context.device.create_fence(&fence_info, None) }
                .map_err(|error| format!("vkCreateFence: {error:?}"))?;
            let mut renderer = tlottie::vulkan::VulkanRenderer::new_with_raster_order_groups(
                device,
                context.raster_order_groups,
            )
                .map_err(|error| format!("initialize tlottie Vulkan renderer: {error}"))?;
            renderer.set_multi_draw_indirect(context.multi_draw_indirect);
            slots.push(FrameSlot {
                renderer: Some(renderer),
                scratch,
                offscreen,
                multisample,
                group,
                stencil,
                offscreen_initialized: false,
                command,
                image_available,
                render_finished,
                frame_fence,
                query_ready: false,
            });
        }
        Ok(Self {
            slots,
            next_slot: 0,
            swapchain,
            timestamp_queries,
            latest_gpu_ns: 0,
            context,
        })
    }

    pub fn latest_gpu_ns(&self) -> u64 {
        self.latest_gpu_ns
    }

    pub fn render(
        &mut self,
        composition: &Composition,
        frame: f32,
        options: RenderOptions,
    ) -> Result<(), String> {
        let slot_index = self.next_slot;
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| "Vulkan frame slot is missing".to_string())?;
        // Reusing this slot waits only for work from two frames ago, allowing
        // CPU recording to overlap the other slot's GPU execution.
        unsafe {
            self.context
                .device
                .wait_for_fences(&[slot.frame_fence], true, u64::MAX)
        }
        .map_err(|error| format!("vkWaitForFences: {error:?}"))?;
        let query_first = (slot_index * 7) as u32;
        if slot.query_ready {
            let mut timestamps = [0u64; 7];
            unsafe {
                self.context.device.get_query_pool_results(
                    self.timestamp_queries,
                    query_first,
                    7,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .map_err(|error| format!("live vkGetQueryPoolResults: {error:?}"))?;
            let mask = if self.context.timestamp_valid_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << self.context.timestamp_valid_bits) - 1
            };
            let ticks = timestamps[6].wrapping_sub(timestamps[0]) & mask;
            self.latest_gpu_ns =
                (ticks as f64 * f64::from(self.context.timestamp_period)) as u64;
        }
        // SAFETY: swapchain and semaphore are live.
        let (image_index, _) = unsafe {
            self.context.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                slot.image_available,
                vk::Fence::null(),
            )
        }
        .map_err(|error| format!("vkAcquireNextImageKHR: {error:?}"))?;
        // SAFETY: prior fence completed; command buffer and fence are reusable.
        unsafe { self.context.device.reset_fences(&[slot.frame_fence]) }
            .map_err(|error| format!("vkResetFences: {error:?}"))?;
        // SAFETY: the prior frame fence completed, so this buffer is idle.
        unsafe {
            self.context
                .device
                .reset_command_buffer(slot.command, vk::CommandBufferResetFlags::empty())
        }
        .map_err(|error| format!("vkResetCommandBuffer: {error:?}"))?;
        let begin = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: command buffer was reset above.
        unsafe {
            self.context
                .device
                .begin_command_buffer(slot.command, &begin)
        }
        .map_err(|error| format!("vkBeginCommandBuffer: {error:?}"))?;
        unsafe {
            self.context.device.cmd_reset_query_pool(
                slot.command,
                self.timestamp_queries,
                query_first,
                7,
            );
            self.context.device.cmd_write_timestamp(
                slot.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.timestamp_queries,
                query_first,
            );
        }

        let index = image_index as usize;
        let swapchain_image = *self
            .swapchain
            .images
            .get(index)
            .ok_or_else(|| "acquired swapchain image index is invalid".to_string())?;
        let initialized = *self
            .swapchain
            .initialized
            .get(index)
            .ok_or_else(|| "swapchain layout state is missing".to_string())?;
        let extent = self.swapchain.extent;
        let scratch = tlottie::vulkan::BufferTarget {
            buffer: slot.scratch.buffer,
            width: extent.width,
            height: extent.height,
            bytes: slot.scratch.bytes,
        };
        let direct_present = self.swapchain.direct_resolve;
        let target = tlottie::vulkan::ImageTarget {
            image: if direct_present {
                swapchain_image
            } else {
                slot.offscreen.image
            },
            format: if direct_present {
                self.swapchain.format
            } else {
                vk::Format::B8G8R8A8_UNORM
            },
            width: extent.width,
            height: extent.height,
            layout: if direct_present && initialized {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else if direct_present {
                vk::ImageLayout::UNDEFINED
            } else if slot.offscreen_initialized {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            },
            final_layout: if direct_present {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            },
            stencil_image: Some(slot.stencil.image),
            multisample_image: slot.multisample.as_ref().map(|image| image.image),
            group_multisample_image: slot.group.as_ref().map(|image| image.image),
        };
        let renderer = slot
            .renderer
            .as_mut()
            .ok_or_else(|| "Vulkan renderer was already destroyed".to_string())?;
        renderer.set_mode(tlottie::vulkan::RendererMode::StencilCover);
        // SAFETY: command, scratch, and acquired swapchain image all belong to
        // the same device and match tlottie's Vulkan target contract.
        unsafe {
            renderer.record_profiled(
                slot.command,
                scratch,
                target,
                composition,
                frame,
                options,
                tlottie::vulkan::ProfileQueries {
                    pool: self.timestamp_queries,
                    first: query_first,
                },
            )
        }
        .map_err(|error| format!("record Vulkan frame: {error}"))?;
        if !direct_present {
            copy_offscreen_to_swapchain(
                &self.context,
                slot.command,
                slot.offscreen.image,
                swapchain_image,
                extent,
                self.swapchain.format,
                initialized,
            )?;
        }
        // SAFETY: command buffer is recording.
        unsafe { self.context.device.end_command_buffer(slot.command) }
            .map_err(|error| format!("vkEndCommandBuffer: {error:?}"))?;

        let wait_semaphores = [slot.image_available];
        let signal_semaphores = [slot.render_finished];
        let wait_stages = [if direct_present {
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        } else {
            vk::PipelineStageFlags::TRANSFER
        }];
        let commands = [slot.command];
        let submits = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&commands)
            .signal_semaphores(&signal_semaphores)
            .build()];
        // SAFETY: queue, synchronization objects, and command belong to device.
        unsafe {
            self.context
                .device
                .queue_submit(self.context.queue, &submits, slot.frame_fence)
        }
        .map_err(|error| format!("vkQueueSubmit: {error:?}"))?;
        slot.query_ready = true;
        let swapchains = [self.swapchain.swapchain];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: acquired image is transitioned to PRESENT_SRC_KHR and its
        // submission signals render_finished before presentation.
        unsafe {
            self.context
                .swapchain_loader
                .queue_present(self.context.queue, &present)
        }
        .map_err(|error| format!("vkQueuePresentKHR: {error:?}"))?;
        if let Some(slot) = self.swapchain.initialized.get_mut(index) {
            *slot = true;
        }
        slot.offscreen_initialized |= !direct_present;
        self.next_slot = (slot_index + 1) % self.slots.len();
        Ok(())
    }

    pub fn benchmark(
        &mut self,
        composition: &Composition,
        warmup_frames: usize,
        measured_frames: usize,
        options: RenderOptions,
    ) -> Result<GpuBenchmark, String> {
        let fences = self
            .slots
            .iter()
            .map(|slot| slot.frame_fence)
            .collect::<Vec<_>>();
        unsafe {
            self.context
                .device
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .map_err(|error| format!("benchmark wait for frame slots: {error:?}"))?;
        let slot = self
            .slots
            .first_mut()
            .ok_or_else(|| "Vulkan benchmark frame slot is missing".to_string())?;
        let extent = self.swapchain.extent;
        let scratch = tlottie::vulkan::BufferTarget {
            buffer: slot.scratch.buffer,
            width: extent.width,
            height: extent.height,
            bytes: slot.scratch.bytes,
        };
        let frame_count = composition.frame_count().max(1) as usize;
        let total_frames = warmup_frames.saturating_add(measured_frames);
        let mut result = GpuBenchmark::new(extent.width, measured_frames);

        for index in 0..total_frames {
            let started = Instant::now();
            // One frame is deliberately serialized here so wall time represents
            // submit-to-completion without swapchain acquire, present, or vsync.
            unsafe {
                self.context
                    .device
                    .wait_for_fences(&[slot.frame_fence], true, u64::MAX)
            }
            .map_err(|error| format!("benchmark vkWaitForFences: {error:?}"))?;
            unsafe { self.context.device.reset_fences(&[slot.frame_fence]) }
                .map_err(|error| format!("benchmark vkResetFences: {error:?}"))?;
            unsafe {
                self.context
                    .device
                    .reset_command_buffer(slot.command, vk::CommandBufferResetFlags::empty())
            }
            .map_err(|error| format!("benchmark vkResetCommandBuffer: {error:?}"))?;
            let begin = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.context
                    .device
                    .begin_command_buffer(slot.command, &begin)
            }
            .map_err(|error| format!("benchmark vkBeginCommandBuffer: {error:?}"))?;
            unsafe {
                self.context.device.cmd_reset_query_pool(
                    slot.command,
                    self.timestamp_queries,
                    0,
                    7,
                );
                self.context.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.timestamp_queries,
                    0,
                );
            }

            let target = tlottie::vulkan::ImageTarget {
                image: slot.offscreen.image,
                format: vk::Format::B8G8R8A8_UNORM,
                width: extent.width,
                height: extent.height,
                layout: if slot.offscreen_initialized {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stencil_image: Some(slot.stencil.image),
                multisample_image: slot.multisample.as_ref().map(|image| image.image),
                group_multisample_image: slot.group.as_ref().map(|image| image.image),
            };
            let record_started = Instant::now();
            let renderer = slot
                .renderer
                .as_mut()
                .ok_or_else(|| "Vulkan renderer was already destroyed".to_string())?;
            renderer.set_mode(tlottie::vulkan::RendererMode::StencilCover);
            unsafe {
                renderer.record_profiled(
                    slot.command,
                    scratch,
                    target,
                    composition,
                    (index % frame_count) as f32,
                    options,
                    tlottie::vulkan::ProfileQueries {
                        pool: self.timestamp_queries,
                        first: 0,
                    },
                )
            }
            .map_err(|error| format!("benchmark record Vulkan frame: {error}"))?;
            let record_ns = record_started.elapsed().as_nanos() as u64;
            let upload_bytes = renderer.cache_stats().scene_upload_bytes;
            let simple_compute = renderer.cache_stats().simple_compute;
            let stencil_cover = renderer.cache_stats().stencil_cover;

            unsafe { self.context.device.end_command_buffer(slot.command) }
                .map_err(|error| format!("benchmark vkEndCommandBuffer: {error:?}"))?;
            let commands = [slot.command];
            let submits = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
            unsafe {
                self.context
                    .device
                    .queue_submit(self.context.queue, &submits, slot.frame_fence)
            }
            .map_err(|error| format!("benchmark vkQueueSubmit: {error:?}"))?;
            unsafe {
                self.context
                    .device
                    .wait_for_fences(&[slot.frame_fence], true, u64::MAX)
            }
            .map_err(|error| format!("benchmark completion wait: {error:?}"))?;
            let mut timestamps = [0u64; 7];
            unsafe {
                self.context.device.get_query_pool_results(
                    self.timestamp_queries,
                    0,
                    7,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .map_err(|error| format!("vkGetQueryPoolResults: {error:?}"))?;
            slot.offscreen_initialized = true;

            if index >= warmup_frames {
                let mask = if self.context.timestamp_valid_bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << self.context.timestamp_valid_bits) - 1
                };
                let elapsed = |start: usize, end: usize| {
                    let ticks = timestamps[end].wrapping_sub(timestamps[start]) & mask;
                    (ticks as f64 * f64::from(self.context.timestamp_period)) as u64
                };
                result.total_ns.push(started.elapsed().as_nanos() as u64);
                result.record_ns.push(record_ns);
                result.gpu_ns.push(elapsed(0, 6));
                result.upload_gpu_ns.push(elapsed(0, 1));
                result.bin_gpu_ns.push(elapsed(1, 4));
                result.bin_count_gpu_ns.push(elapsed(1, 2));
                result.bin_prefix_gpu_ns.push(elapsed(2, 3));
                result.bin_scatter_gpu_ns.push(elapsed(3, 4));
                result.coverage_gpu_ns.push(elapsed(4, 5));
                result.copy_gpu_ns.push(elapsed(5, 6));
                result.upload_bytes.push(upload_bytes);
                result.simple_compute &= simple_compute;
                result.stencil_cover &= stencil_cover;
            }
        }
        Ok(result)
    }
}

pub struct GpuBenchmark {
    size: u32,
    total_ns: Vec<u64>,
    record_ns: Vec<u64>,
    gpu_ns: Vec<u64>,
    upload_gpu_ns: Vec<u64>,
    bin_gpu_ns: Vec<u64>,
    bin_count_gpu_ns: Vec<u64>,
    bin_prefix_gpu_ns: Vec<u64>,
    bin_scatter_gpu_ns: Vec<u64>,
    coverage_gpu_ns: Vec<u64>,
    copy_gpu_ns: Vec<u64>,
    upload_bytes: Vec<u64>,
    simple_compute: bool,
    stencil_cover: bool,
}

impl GpuBenchmark {
    fn new(size: u32, frames: usize) -> Self {
        Self {
            size,
            total_ns: Vec::with_capacity(frames),
            record_ns: Vec::with_capacity(frames),
            gpu_ns: Vec::with_capacity(frames),
            upload_gpu_ns: Vec::with_capacity(frames),
            bin_gpu_ns: Vec::with_capacity(frames),
            bin_count_gpu_ns: Vec::with_capacity(frames),
            bin_prefix_gpu_ns: Vec::with_capacity(frames),
            bin_scatter_gpu_ns: Vec::with_capacity(frames),
            coverage_gpu_ns: Vec::with_capacity(frames),
            copy_gpu_ns: Vec::with_capacity(frames),
            upload_bytes: Vec::with_capacity(frames),
            simple_compute: true,
            stencil_cover: true,
        }
    }

    pub fn summary(&self, antialias: bool, curve_tolerance: f32) -> String {
        fn percentile(values: &[u64], percent: usize) -> u64 {
            if values.is_empty() {
                return 0;
            }
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[(sorted.len() - 1) * percent / 100]
        }
        format!(
            "backend=vulkan size={} aa={} curve_tolerance={:.3} simple={} stencil={} frames={} total_median_ns={} total_p90_ns={} total_p99_ns={} record_median_ns={} gpu_median_ns={} gpu_p90_ns={} gpu_p99_ns={} upload_gpu_median_ns={} bin_gpu_median_ns={} bin_count_gpu_median_ns={} bin_prefix_gpu_median_ns={} bin_scatter_gpu_median_ns={} coverage_gpu_median_ns={} copy_gpu_median_ns={} upload_median_bytes={}",
            self.size,
            u8::from(antialias),
            curve_tolerance,
            u8::from(self.simple_compute),
            u8::from(self.stencil_cover),
            self.total_ns.len(),
            percentile(&self.total_ns, 50),
            percentile(&self.total_ns, 90),
            percentile(&self.total_ns, 99),
            percentile(&self.record_ns, 50),
            percentile(&self.gpu_ns, 50),
            percentile(&self.gpu_ns, 90),
            percentile(&self.gpu_ns, 99),
            percentile(&self.upload_gpu_ns, 50),
            percentile(&self.bin_gpu_ns, 50),
            percentile(&self.bin_count_gpu_ns, 50),
            percentile(&self.bin_prefix_gpu_ns, 50),
            percentile(&self.bin_scatter_gpu_ns, 50),
            percentile(&self.coverage_gpu_ns, 50),
            percentile(&self.copy_gpu_ns, 50),
            percentile(&self.upload_bytes, 50),
        )
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build()
}

fn copy_offscreen_to_swapchain(
    context: &Context,
    command: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    extent: vk::Extent2D,
    destination_format: vk::Format,
    initialized: bool,
) -> Result<(), String> {
    let to_destination = vk::ImageMemoryBarrier::builder()
        .old_layout(if initialized {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        })
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(if initialized {
            vk::AccessFlags::MEMORY_READ
        } else {
            vk::AccessFlags::empty()
        })
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(destination)
        .subresource_range(color_range())
        .build();
    // SAFETY: destination is the acquired swapchain image and the acquire
    // semaphore gates this command buffer's execution.
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            if initialized {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                vk::PipelineStageFlags::TOP_OF_PIPE
            },
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_destination],
        );
    }

    let layers = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    if destination_format == vk::Format::B8G8R8A8_UNORM {
        let copy = vk::ImageCopy::builder()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .build();
        // SAFETY: images have matching BGRA8 formats, transfer usages, and
        // extents; both are in their declared transfer layouts.
        unsafe {
            context.device.cmd_copy_image(
                command,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }
    } else {
        // BGRA-to-RGBA blit performs the required component format conversion
        // entirely on the GPU.
        let source_features = unsafe {
            context.instance.get_physical_device_format_properties(
                context.physical_device,
                vk::Format::B8G8R8A8_UNORM,
            )
        };
        let destination_features = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.physical_device, destination_format)
        };
        if !source_features
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_SRC)
            || !destination_features
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::BLIT_DST)
        {
            return Err("GPU cannot blit BGRA tlottie output into RGBA TextureView".to_string());
        }
        let blit = vk::ImageBlit::builder()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: 1,
                },
            ])
            .build();
        // SAFETY: queried format features support this same-size blit.
        unsafe {
            context.device.cmd_blit_image(
                command,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
        }
    }

    let to_present = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .image(destination)
        .subresource_range(color_range())
        .build();
    // SAFETY: the copy/blit above wrote the acquired swapchain image.
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
    }
    Ok(())
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        // SAFETY: Java serializes rendering and surface destruction.
        unsafe {
            let _ = self.context.device.device_wait_idle();
        }
        for slot in &mut self.slots {
            drop(slot.renderer.take());
        }
        // SAFETY: all queued work is idle and this host owns these objects.
        unsafe {
            self.context
                .device
                .destroy_query_pool(self.timestamp_queries, None);
            self.context
                .swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);
        }
        for slot in &self.slots {
            unsafe {
                self.context.device.destroy_fence(slot.frame_fence, None);
                self.context
                    .device
                    .destroy_semaphore(slot.render_finished, None);
                self.context
                    .device
                    .destroy_semaphore(slot.image_available, None);
                self.context
                    .device
                    .free_command_buffers(self.context.command_pool, &[slot.command]);
            }
            slot.offscreen.destroy(&self.context);
            if let Some(multisample) = &slot.multisample {
                multisample.destroy(&self.context);
            }
            if let Some(group) = &slot.group {
                group.destroy(&self.context);
            }
            slot.stencil.destroy(&self.context);
            slot.scratch.destroy(&self.context);
        }
    }
}

