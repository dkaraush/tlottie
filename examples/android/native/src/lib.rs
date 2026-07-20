mod rlottie;
mod thorvg;
mod thorvg_gl;
mod vulkan;

use std::sync::Arc;

use jni::objects::{JByteArray, JClass, JIntArray, JObject};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring};
use jni::JNIEnv;
use tlottie::{CPURenderer, Composition, Limits, RenderOptions};

struct State {
    json: Vec<u8>,
    composition: Arc<Composition>,
    cpu: CPURenderer,
    gpu: Option<vulkan::GpuRenderer>,
    pixels: Vec<u32>,
    rlottie: [Option<rlottie::Rlottie>; 3],
    thorvg_cpu: Option<thorvg::ThorvgCpu>,
    thorvg_gpu: Option<thorvg_gl::ThorvgGl>,
}

fn state_mut(handle: jlong) -> Result<&'static mut State, String> {
    if handle == 0 {
        return Err("animation handle is null".to_string());
    }
    // SAFETY: handles are created from Box<State> below and Java serializes
    // access on its UI thread. destroy consumes the handle exactly once.
    unsafe {
        (handle as *mut State)
            .as_mut()
            .ok_or_else(|| "animation handle is invalid".to_string())
    }
}

fn java_error(env: &mut JNIEnv<'_>, result: Result<(), String>) -> jstring {
    match result {
        Ok(()) => std::ptr::null_mut(),
        Err(error) => env
            .new_string(error)
            .map(|value| value.into_raw())
            .unwrap_or(std::ptr::null_mut()),
    }
}

fn java_string(env: &mut JNIEnv<'_>, result: Result<String, String>) -> jstring {
    let value = result.unwrap_or_else(|error| format!("error={error}"));
    env.new_string(value)
        .map(|value| value.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

fn percentile(samples: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let index = samples.len().saturating_sub(1).saturating_mul(numerator) / denominator.max(1);
    samples.get(index).copied().unwrap_or(0)
}

fn summarize(label: &str, size: u32, antialias: bool, mut samples: Vec<u64>) -> String {
    let frames = samples.len();
    let total = samples.iter().copied().sum::<u64>();
    let mean = if frames == 0 {
        0
    } else {
        total / frames as u64
    };
    let median = percentile(&mut samples.clone(), 50, 100);
    let p90 = percentile(&mut samples.clone(), 90, 100);
    let p99 = percentile(&mut samples, 99, 100);
    format!(
        "backend={label} size={size} aa={} frames={frames} mean_ns={mean} median_ns={median} p90_ns={p90} p99_ns={p99}",
        u8::from(antialias)
    )
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_create(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    json: JByteArray<'_>,
) -> jlong {
    let bytes = match env.convert_byte_array(&json) {
        Ok(bytes) => bytes,
        Err(_) => return 0,
    };
    let composition = match Composition::parse(&bytes, &Limits::default()) {
        Ok(composition) => Arc::new(composition),
        Err(_) => return 0,
    };
    let cpu = CPURenderer::from_shared(Arc::clone(&composition));
    Box::into_raw(Box::new(State {
        json: bytes,
        composition,
        cpu,
        gpu: None,
        pixels: Vec::new(),
        rlottie: [None, None, None],
        thorvg_cpu: None,
        thorvg_gpu: None,
    })) as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_destroy(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    if handle != 0 {
        // SAFETY: handle came from create and Java calls destroy once.
        drop(unsafe { Box::from_raw(handle as *mut State) });
    }
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_frameCount(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jint {
    state_mut(handle)
        .map(|state| state.composition.frame_count().min(jint::MAX as u32) as jint)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_frameRate(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jfloat {
    state_mut(handle)
        .map(|state| state.composition.frame_rate)
        .unwrap_or(0.0)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_renderCpu(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    frame: jfloat,
    width: jint,
    height: jint,
    output: JIntArray<'_>,
) -> jstring {
    let result = render_cpu(&mut env, handle, frame, width, height, &output);
    java_error(&mut env, result)
}

fn render_cpu(
    env: &mut JNIEnv<'_>,
    handle: jlong,
    frame: f32,
    width: jint,
    height: jint,
    output: &JIntArray<'_>,
) -> Result<(), String> {
    let width = u32::try_from(width).map_err(|_| "negative render width".to_string())?;
    let height = u32::try_from(height).map_err(|_| "negative render height".to_string())?;
    let len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "render dimensions overflow".to_string())?;
    let array_len = env
        .get_array_length(output)
        .map_err(|error| format!("get output length: {error}"))?;
    if usize::try_from(array_len).unwrap_or(0) < len {
        return Err("Java pixel buffer is too small".to_string());
    }

    let state = state_mut(handle)?;
    state.pixels.resize(len, 0);
    state
        .cpu
        .render(frame, &mut state.pixels, width, height)
        .map_err(|error| format!("CPU render: {error}"))?;

    // SAFETY: u32 and Java jint are both aligned four-byte integer types, and
    // the JNI call only reads the slice for the duration of this call.
    let signed = unsafe {
        std::slice::from_raw_parts(state.pixels.as_ptr().cast::<i32>(), state.pixels.len())
    };
    env.set_int_array_region(output, 0, signed)
        .map_err(|error| format!("copy pixels to Java: {error}"))
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_renderRlottie(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    variant: jint,
    frame: jfloat,
    width: jint,
    height: jint,
    output: JIntArray<'_>,
) -> jstring {
    let result = render_rlottie(&mut env, handle, variant, frame, width, height, &output);
    java_error(&mut env, result)
}

fn render_rlottie(
    env: &mut JNIEnv<'_>,
    handle: jlong,
    variant: jint,
    frame: f32,
    width: jint,
    height: jint,
    output: &JIntArray<'_>,
) -> Result<(), String> {
    let variant = usize::try_from(variant).map_err(|_| "negative rlottie variant".to_string())?;
    if variant >= 3 {
        return Err(format!("unknown rlottie variant {variant}"));
    }
    let width = u32::try_from(width).map_err(|_| "negative render width".to_string())?;
    let height = u32::try_from(height).map_err(|_| "negative render height".to_string())?;
    let len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "render dimensions overflow".to_string())?;
    if usize::try_from(env.get_array_length(output).unwrap_or(0)).unwrap_or(0) < len {
        return Err("Java pixel buffer is too small".to_string());
    }
    let state = state_mut(handle)?;
    state.pixels.resize(len, 0);
    if state.rlottie[variant].is_none() {
        state.rlottie[variant] = Some(rlottie::Rlottie::new(variant, &state.json)?);
    }
    state.rlottie[variant]
        .as_mut()
        .expect("rlottie backend initialized above")
        .render(frame, &mut state.pixels, width, height);
    if variant == 2 {
        for pixel in &mut state.pixels {
            *pixel = (*pixel & 0xff00_ff00)
                | ((*pixel & 0x00ff_0000) >> 16)
                | ((*pixel & 0x0000_00ff) << 16);
        }
    }
    let signed = unsafe {
        std::slice::from_raw_parts(state.pixels.as_ptr().cast::<i32>(), state.pixels.len())
    };
    env.set_int_array_region(output, 0, signed)
        .map_err(|error| format!("copy rlottie pixels to Java: {error}"))
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_renderThorvgCpu(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    frame: jfloat,
    width: jint,
    height: jint,
    output: JIntArray<'_>,
) -> jstring {
    let result = (|| {
        let width = u32::try_from(width).map_err(|_| "negative render width".to_string())?;
        let height = u32::try_from(height).map_err(|_| "negative render height".to_string())?;
        let len = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "render dimensions overflow".to_string())?;
        if usize::try_from(env.get_array_length(&output).unwrap_or(0)).unwrap_or(0) < len {
            return Err("Java pixel buffer is too small".to_string());
        }
        let state = state_mut(handle)?;
        state.pixels.resize(len, 0);
        if state.thorvg_cpu.is_none() {
            state.thorvg_cpu = Some(thorvg::ThorvgCpu::new(
                &state.json,
                &mut state.pixels,
                width,
                height,
            )?);
        }
        state
            .thorvg_cpu
            .as_mut()
            .expect("ThorVG CPU initialized above")
            .render(frame, &mut state.pixels, width, height)?;
        let signed = unsafe {
            std::slice::from_raw_parts(state.pixels.as_ptr().cast::<i32>(), state.pixels.len())
        };
        env.set_int_array_region(&output, 0, signed)
            .map_err(|error| format!("copy ThorVG pixels to Java: {error}"))
    })();
    java_error(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_setSurface(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    surface: JObject<'_>,
    width: jint,
    height: jint,
) -> jstring {
    let result = (|| {
        let width = u32::try_from(width).map_err(|_| "negative surface width".to_string())?;
        let height = u32::try_from(height).map_err(|_| "negative surface height".to_string())?;
        let state = state_mut(handle)?;
        state.gpu = None;
        state.gpu = Some(vulkan::GpuRenderer::new(&env, &surface, width, height)?);
        Ok(())
    })();
    java_error(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_clearSurface(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    if let Ok(state) = state_mut(handle) {
        state.gpu = None;
        state.thorvg_gpu = None;
    }
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_setThorvgSurface(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    surface: JObject<'_>,
    width: jint,
    height: jint,
) -> jstring {
    let result = (|| {
        let width = u32::try_from(width).map_err(|_| "negative surface width".to_string())?;
        let height = u32::try_from(height).map_err(|_| "negative surface height".to_string())?;
        let state = state_mut(handle)?;
        state.gpu = None;
        state.thorvg_gpu = None;
        state.thorvg_gpu = Some(thorvg_gl::ThorvgGl::new(
            &env,
            &surface,
            width,
            height,
            &state.json,
        )?);
        Ok(())
    })();
    java_error(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_renderThorvgGpu(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    frame: jfloat,
) -> jstring {
    let result = (|| {
        state_mut(handle)?
            .thorvg_gpu
            .as_mut()
            .ok_or_else(|| "ThorVG GPU surface is not ready".to_string())?
            .render(frame)
    })();
    java_error(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_renderVulkan(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    frame: jfloat,
    antialias: jboolean,
    curve_tolerance: jfloat,
) -> jstring {
    let result = (|| {
        let state = state_mut(handle)?;
        let gpu = state
            .gpu
            .as_mut()
            .ok_or_else(|| "TextureView Vulkan surface is not ready".to_string())?;
        gpu.render(
            &state.composition,
            frame,
            RenderOptions {
                antialias: antialias != 0,
                curve_tolerance,
                ..RenderOptions::default()
            },
        )
    })();
    java_error(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_lastVulkanGpuNs(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jlong {
    state_mut(handle)
        .ok()
        .and_then(|state| state.gpu.as_ref())
        .map(|gpu| gpu.latest_gpu_ns().min(jlong::MAX as u64) as jlong)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_benchmarkCpu(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    warmup_frames: jint,
    measured_frames: jint,
    size: jint,
    antialias: jboolean,
) -> jstring {
    let result = benchmark_cpu(handle, warmup_frames, measured_frames, size, antialias != 0);
    java_string(&mut env, result)
}

fn benchmark_cpu(
    handle: jlong,
    warmup_frames: jint,
    measured_frames: jint,
    size: jint,
    antialias: bool,
) -> Result<String, String> {
    let size = u32::try_from(size).map_err(|_| "negative benchmark size".to_string())?;
    let warmup = usize::try_from(warmup_frames.max(0)).unwrap_or(0);
    let measured = usize::try_from(measured_frames.max(1)).unwrap_or(1);
    let state = state_mut(handle)?;
    let pixels = usize::try_from(size)
        .ok()
        .and_then(|side| side.checked_mul(side))
        .ok_or_else(|| "benchmark dimensions overflow".to_string())?;
    state.pixels.resize(pixels, 0);
    let frame_count = state.composition.frame_count().max(1) as usize;
    let options = RenderOptions {
        antialias,
        ..RenderOptions::default()
    };
    let mut samples = Vec::with_capacity(measured);
    for index in 0..warmup.saturating_add(measured) {
        let frame = (index % frame_count) as f32;
        let started = std::time::Instant::now();
        state
            .cpu
            .render_with_options(frame, &mut state.pixels, size, size, options)
            .map_err(|error| format!("CPU benchmark render: {error}"))?;
        if index >= warmup {
            samples.push(started.elapsed().as_nanos() as u64);
        }
    }
    Ok(summarize("cpu", size, antialias, samples))
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_benchmarkVulkan(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    warmup_frames: jint,
    measured_frames: jint,
    antialias: jboolean,
    curve_tolerance: jfloat,
) -> jstring {
    let result = (|| {
        let warmup = usize::try_from(warmup_frames.max(0)).unwrap_or(0);
        let measured = usize::try_from(measured_frames.max(1)).unwrap_or(1);
        let state = state_mut(handle)?;
        let gpu = state
            .gpu
            .as_mut()
            .ok_or_else(|| "Vulkan benchmark surface is not ready".to_string())?;
        let benchmark = gpu.benchmark(
            &state.composition,
            warmup,
            measured,
            RenderOptions {
                antialias: antialias != 0,
                curve_tolerance,
                ..RenderOptions::default()
            },
        )?;
        Ok(benchmark.summary(antialias != 0, curve_tolerance))
    })();
    java_string(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_example_tlottie_NativeBridge_benchmarkThorvgGpu(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    warmup_frames: jint,
    measured_frames: jint,
) -> jstring {
    let result = (|| {
        let warmup = usize::try_from(warmup_frames.max(0)).unwrap_or(0);
        let measured = usize::try_from(measured_frames.max(1)).unwrap_or(1);
        let state = state_mut(handle)?;
        let (size, samples, disjoint) = state
            .thorvg_gpu
            .as_mut()
            .ok_or_else(|| "ThorVG GPU surface is not ready".to_string())?
            .benchmark(warmup, measured)?;
        Ok(format!(
            "{} disjoint_samples={disjoint}",
            summarize("thorvg-gl-gpu", size, true, samples)
        ))
    })();
    java_string(&mut env, result)
}

