use std::ffi::{c_char, c_void};

use jni::objects::JObject;
use jni::sys::jobject;
use jni::JNIEnv;
use libloading::Library;

#[repr(C)]
struct ANativeWindow {
    _private: [u8; 0],
}

type EglDisplay = *mut c_void;
type EglSurface = *mut c_void;
type EglContext = *mut c_void;
type EglConfig = *mut c_void;
type Handle = *mut c_void;
type ResultCode = u32;

const EGL_NONE: i32 = 0x3038;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_WINDOW_BIT: i32 = 0x0004;
const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_OPENGL_ES_API: u32 = 0x30a0;
const GL_TIME_ELAPSED_EXT: u32 = 0x88bf;
const GL_QUERY_RESULT: u32 = 0x8866;
const GL_GPU_DISJOINT_EXT: u32 = 0x8fbb;

#[link(name = "android")]
extern "C" {
    fn ANativeWindow_fromSurface(env: *mut c_void, surface: jobject) -> *mut ANativeWindow;
    fn ANativeWindow_release(window: *mut ANativeWindow);
}

#[link(name = "EGL")]
extern "C" {
    fn eglGetDisplay(display_id: *mut c_void) -> EglDisplay;
    fn eglInitialize(display: EglDisplay, major: *mut i32, minor: *mut i32) -> u32;
    fn eglChooseConfig(
        display: EglDisplay,
        attributes: *const i32,
        config: *mut EglConfig,
        config_size: i32,
        count: *mut i32,
    ) -> u32;
    fn eglBindAPI(api: u32) -> u32;
    fn eglCreateWindowSurface(
        display: EglDisplay,
        config: EglConfig,
        window: *mut ANativeWindow,
        attributes: *const i32,
    ) -> EglSurface;
    fn eglCreateContext(
        display: EglDisplay,
        config: EglConfig,
        share: EglContext,
        attributes: *const i32,
    ) -> EglContext;
    fn eglMakeCurrent(
        display: EglDisplay,
        draw: EglSurface,
        read: EglSurface,
        context: EglContext,
    ) -> u32;
    fn eglSwapBuffers(display: EglDisplay, surface: EglSurface) -> u32;
    fn eglDestroyContext(display: EglDisplay, context: EglContext) -> u32;
    fn eglDestroySurface(display: EglDisplay, surface: EglSurface) -> u32;
    fn eglTerminate(display: EglDisplay) -> u32;
}

#[link(name = "GLESv3")]
extern "C" {
    fn glGenQueries(count: i32, queries: *mut u32);
    fn glDeleteQueries(count: i32, queries: *const u32);
    fn glBeginQuery(target: u32, query: u32);
    fn glEndQuery(target: u32);
    fn glGetQueryObjectuiv(query: u32, parameter: u32, value: *mut u32);
    fn glGetIntegerv(parameter: u32, value: *mut i32);
    fn glGetError() -> u32;
}

type EngineInit = unsafe extern "C" fn(u32) -> ResultCode;
type EngineTerm = unsafe extern "C" fn() -> ResultCode;
type AnimationNew = unsafe extern "C" fn() -> Handle;
type AnimationDel = unsafe extern "C" fn(Handle) -> ResultCode;
type AnimationPicture = unsafe extern "C" fn(Handle) -> Handle;
type AnimationFrame = unsafe extern "C" fn(Handle, f32) -> ResultCode;
type PictureLoadData = unsafe extern "C" fn(
    Handle,
    *const c_char,
    u32,
    *const c_char,
    *const c_char,
    bool,
) -> ResultCode;
type PictureSize = unsafe extern "C" fn(Handle, f32, f32) -> ResultCode;
type CanvasCreate = unsafe extern "C" fn(u32) -> Handle;
type GlTarget = unsafe extern "C" fn(
    Handle,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    u32,
    u32,
    u32,
) -> ResultCode;
type CanvasPaint = unsafe extern "C" fn(Handle, Handle) -> ResultCode;
type CanvasCall = unsafe extern "C" fn(Handle) -> ResultCode;
type CanvasDraw = unsafe extern "C" fn(Handle, bool) -> ResultCode;

pub struct ThorvgGl {
    window: *mut ANativeWindow,
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
    width: u32,
    animation: Handle,
    canvas: Handle,
    animation_del: AnimationDel,
    animation_frame: AnimationFrame,
    canvas_update: CanvasCall,
    canvas_draw: CanvasDraw,
    canvas_sync: CanvasCall,
    canvas_destroy: CanvasCall,
    engine_term: EngineTerm,
    _library: Library,
}

impl ThorvgGl {
    pub fn new(
        env: &JNIEnv<'_>,
        java_surface: &JObject<'_>,
        width: u32,
        height: u32,
        json: &[u8],
    ) -> Result<Self, String> {
        let window = unsafe {
            ANativeWindow_fromSurface(
                env.get_native_interface().cast::<c_void>(),
                java_surface.as_raw(),
            )
        };
        if window.is_null() {
            return Err("ThorVG ANativeWindow_fromSurface returned null".to_string());
        }
        let result = Self::new_with_window(window, width, height, json);
        if result.is_err() {
            unsafe { ANativeWindow_release(window) };
        }
        result
    }

    fn new_with_window(
        window: *mut ANativeWindow,
        width: u32,
        height: u32,
        json: &[u8],
    ) -> Result<Self, String> {
        let display = unsafe { eglGetDisplay(std::ptr::null_mut()) };
        if display.is_null()
            || unsafe { eglInitialize(display, std::ptr::null_mut(), std::ptr::null_mut()) } == 0
        {
            return Err("ThorVG eglInitialize failed".to_string());
        }
        let attributes = [
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_SURFACE_TYPE,
            EGL_WINDOW_BIT,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_ES3_BIT,
            EGL_NONE,
        ];
        let mut config = std::ptr::null_mut();
        let mut count = 0;
        if unsafe { eglChooseConfig(display, attributes.as_ptr(), &mut config, 1, &mut count) } == 0
            || count == 0
        {
            unsafe { eglTerminate(display) };
            return Err("ThorVG eglChooseConfig failed".to_string());
        }
        unsafe { eglBindAPI(EGL_OPENGL_ES_API) };
        let surface = unsafe { eglCreateWindowSurface(display, config, window, std::ptr::null()) };
        let context_attributes = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
        let context = unsafe {
            eglCreateContext(
                display,
                config,
                std::ptr::null_mut(),
                context_attributes.as_ptr(),
            )
        };
        if surface.is_null()
            || context.is_null()
            || unsafe { eglMakeCurrent(display, surface, surface, context) } == 0
        {
            unsafe {
                if !context.is_null() {
                    eglDestroyContext(display, context);
                }
                if !surface.is_null() {
                    eglDestroySurface(display, surface);
                }
                eglTerminate(display);
            }
            return Err("ThorVG could not create an EGL/GLES 3 surface".to_string());
        }

        let library = unsafe { Library::new("libthorvg.so") }
            .map_err(|error| format!("load libthorvg.so: {error}"))?;
        macro_rules! symbol {
            ($ty:ty, $name:literal) => {{
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|error| format!("resolve {}: {error}", $name))?
            }};
        }
        let engine_init = symbol!(EngineInit, "tvg_engine_init");
        let engine_term = symbol!(EngineTerm, "tvg_engine_term");
        let animation_new = symbol!(AnimationNew, "tvg_animation_new");
        let animation_del = symbol!(AnimationDel, "tvg_animation_del");
        let animation_picture = symbol!(AnimationPicture, "tvg_animation_get_picture");
        let animation_frame = symbol!(AnimationFrame, "tvg_animation_set_frame");
        let picture_load = symbol!(PictureLoadData, "tvg_picture_load_data");
        let picture_size = symbol!(PictureSize, "tvg_picture_set_size");
        let canvas_create = symbol!(CanvasCreate, "tvg_glcanvas_create");
        let canvas_target = symbol!(GlTarget, "tvg_glcanvas_set_target");
        let canvas_add = symbol!(CanvasPaint, "tvg_canvas_add");
        let canvas_update = symbol!(CanvasCall, "tvg_canvas_update");
        let canvas_draw = symbol!(CanvasDraw, "tvg_canvas_draw");
        let canvas_sync = symbol!(CanvasCall, "tvg_canvas_sync");
        let canvas_destroy = symbol!(CanvasCall, "tvg_canvas_destroy");
        check(unsafe { engine_init(0) }, "tvg_engine_init")?;
        let animation = unsafe { animation_new() };
        let picture = unsafe { animation_picture(animation) };
        check(
            unsafe {
                picture_load(
                    picture,
                    json.as_ptr().cast(),
                    json.len().min(u32::MAX as usize) as u32,
                    c"json".as_ptr(),
                    c"".as_ptr(),
                    true,
                )
            },
            "tvg_picture_load_data",
        )?;
        check(
            unsafe { picture_size(picture, width as f32, height as f32) },
            "tvg_picture_set_size",
        )?;
        let canvas = unsafe { canvas_create(0) };
        if animation.is_null() || picture.is_null() || canvas.is_null() {
            return Err("ThorVG GL object allocation failed".to_string());
        }
        check(
            unsafe { canvas_target(canvas, display, surface, context, 0, width, height, 2) },
            "tvg_glcanvas_set_target",
        )?;
        check(unsafe { canvas_add(canvas, picture) }, "tvg_canvas_add")?;
        Ok(Self {
            window,
            display,
            surface,
            context,
            width,
            animation,
            canvas,
            animation_del,
            animation_frame,
            canvas_update,
            canvas_draw,
            canvas_sync,
            canvas_destroy,
            engine_term,
            _library: library,
        })
    }

    pub fn render(&mut self, frame: f32) -> Result<(), String> {
        if unsafe { eglMakeCurrent(self.display, self.surface, self.surface, self.context) } == 0 {
            return Err("ThorVG eglMakeCurrent failed".to_string());
        }
        check_frame(
            unsafe { (self.animation_frame)(self.animation, frame.max(0.0)) },
        )?;
        check(
            unsafe { (self.canvas_update)(self.canvas) },
            "tvg_canvas_update",
        )?;
        check(
            unsafe { (self.canvas_draw)(self.canvas, true) },
            "tvg_canvas_draw",
        )?;
        check(
            unsafe { (self.canvas_sync)(self.canvas) },
            "tvg_canvas_sync",
        )?;
        if unsafe { eglSwapBuffers(self.display, self.surface) } == 0 {
            return Err("ThorVG eglSwapBuffers failed".to_string());
        }
        Ok(())
    }

    pub fn benchmark(
        &mut self,
        warmup: usize,
        measured: usize,
    ) -> Result<(u32, Vec<u64>, u32), String> {
        if unsafe { eglMakeCurrent(self.display, self.surface, self.surface, self.context) } == 0 {
            return Err("ThorVG eglMakeCurrent failed".to_string());
        }
        let mut query = 0u32;
        unsafe { glGenQueries(1, &mut query) };
        if query == 0 {
            return Err("GL_EXT_disjoint_timer_query did not create a query".to_string());
        }
        let result = (|| {
            let mut samples = Vec::with_capacity(measured);
            let mut disjoint_samples = 0u32;
            for index in 0..warmup.saturating_add(measured) {
                check_frame(
                    unsafe { (self.animation_frame)(self.animation, index as f32) },
                )?;
                check(unsafe { (self.canvas_update)(self.canvas) }, "tvg_canvas_update")?;
                let measuring = index >= warmup;
                if measuring {
                    unsafe { glBeginQuery(GL_TIME_ELAPSED_EXT, query) };
                }
                check(
                    unsafe { (self.canvas_draw)(self.canvas, true) },
                    "tvg_canvas_draw",
                )?;
                check(unsafe { (self.canvas_sync)(self.canvas) }, "tvg_canvas_sync")?;
                if measuring {
                    unsafe { glEndQuery(GL_TIME_ELAPSED_EXT) };
                }
                if unsafe { eglSwapBuffers(self.display, self.surface) } == 0 {
                    return Err("ThorVG eglSwapBuffers failed".to_string());
                }
                if measuring {
                    let mut elapsed = 0u32;
                    let mut disjoint = 0i32;
                    unsafe {
                        glGetQueryObjectuiv(query, GL_QUERY_RESULT, &mut elapsed);
                        glGetIntegerv(GL_GPU_DISJOINT_EXT, &mut disjoint);
                    }
                    let error = unsafe { glGetError() };
                    if error != 0 {
                        return Err(format!("ThorVG GL timer query failed: 0x{error:04x}"));
                    }
                    disjoint_samples = disjoint_samples.saturating_add(u32::from(disjoint != 0));
                    samples.push(u64::from(elapsed));
                }
            }
            Ok((self.width, samples, disjoint_samples))
        })();
        unsafe { glDeleteQueries(1, &query) };
        result
    }
}

fn check(result: ResultCode, operation: &str) -> Result<(), String> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed with ThorVG result {result}"))
    }
}

fn check_frame(result: ResultCode) -> Result<(), String> {
    // ThorVG reports an already-current frame as INSUFFICIENT_CONDITION.
    if result == 0 || result == 2 {
        Ok(())
    } else {
        Err(format!(
            "tvg_animation_set_frame failed with ThorVG result {result}"
        ))
    }
}

impl Drop for ThorvgGl {
    fn drop(&mut self) {
        unsafe {
            eglMakeCurrent(self.display, self.surface, self.surface, self.context);
            (self.canvas_destroy)(self.canvas);
            (self.animation_del)(self.animation);
            (self.engine_term)();
            eglDestroyContext(self.display, self.context);
            eglDestroySurface(self.display, self.surface);
            eglTerminate(self.display);
            ANativeWindow_release(self.window);
        }
    }
}

