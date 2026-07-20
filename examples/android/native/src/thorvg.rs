use std::ffi::{c_char, c_void};

use libloading::Library;

type Handle = *mut c_void;
type ResultCode = u32;
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
type CanvasTarget = unsafe extern "C" fn(Handle, *mut u32, u32, u32, u32, u32) -> ResultCode;
type CanvasPaint = unsafe extern "C" fn(Handle, Handle) -> ResultCode;
type CanvasCall = unsafe extern "C" fn(Handle) -> ResultCode;
type CanvasDraw = unsafe extern "C" fn(Handle, bool) -> ResultCode;

pub struct ThorvgCpu {
    animation: Handle,
    picture: Handle,
    canvas: Handle,
    animation_del: AnimationDel,
    animation_frame: AnimationFrame,
    picture_size: PictureSize,
    canvas_target: CanvasTarget,
    canvas_update: CanvasCall,
    canvas_draw: CanvasDraw,
    canvas_sync: CanvasCall,
    canvas_destroy: CanvasCall,
    engine_term: EngineTerm,
    target: *mut u32,
    target_width: u32,
    target_height: u32,
    _library: Library,
}

impl ThorvgCpu {
    pub fn new(json: &[u8], pixels: &mut [u32], width: u32, height: u32) -> Result<Self, String> {
        // SAFETY: packaged ThorVG remains loaded for this object's lifetime.
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
        let canvas_create = symbol!(CanvasCreate, "tvg_swcanvas_create");
        let canvas_target = symbol!(CanvasTarget, "tvg_swcanvas_set_target");
        let canvas_add = symbol!(CanvasPaint, "tvg_canvas_add");
        let canvas_update = symbol!(CanvasCall, "tvg_canvas_update");
        let canvas_draw = symbol!(CanvasDraw, "tvg_canvas_draw");
        let canvas_sync = symbol!(CanvasCall, "tvg_canvas_sync");
        let canvas_destroy = symbol!(CanvasCall, "tvg_canvas_destroy");

        check(unsafe { engine_init(0) }, "tvg_engine_init")?;
        let animation = unsafe { animation_new() };
        if animation.is_null() {
            unsafe { engine_term() };
            return Err("tvg_animation_new failed".to_string());
        }
        let picture = unsafe { animation_picture(animation) };
        let load = unsafe {
            picture_load(
                picture,
                json.as_ptr().cast(),
                json.len().min(u32::MAX as usize) as u32,
                c"json".as_ptr(),
                c"".as_ptr(),
                true,
            )
        };
        if let Err(error) = check(load, "tvg_picture_load_data") {
            unsafe {
                animation_del(animation);
                engine_term();
            }
            return Err(error);
        }
        let canvas = unsafe { canvas_create(0) };
        if canvas.is_null() {
            unsafe {
                animation_del(animation);
                engine_term();
            }
            return Err("tvg_swcanvas_create failed".to_string());
        }
        check(
            unsafe { picture_size(picture, width as f32, height as f32) },
            "tvg_picture_set_size",
        )?;
        check(
            unsafe { canvas_target(canvas, pixels.as_mut_ptr(), width, width, height, 1) },
            "tvg_swcanvas_set_target",
        )?;
        check(unsafe { canvas_add(canvas, picture) }, "tvg_canvas_add")?;
        Ok(Self {
            animation,
            picture,
            canvas,
            animation_del,
            animation_frame,
            picture_size,
            canvas_target,
            canvas_update,
            canvas_draw,
            canvas_sync,
            canvas_destroy,
            engine_term,
            target: pixels.as_mut_ptr(),
            target_width: width,
            target_height: height,
            _library: library,
        })
    }

    pub fn render(
        &mut self,
        frame: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        check(
            unsafe { (self.picture_size)(self.picture, width as f32, height as f32) },
            "tvg_picture_set_size",
        )?;
        if self.target != pixels.as_mut_ptr()
            || self.target_width != width
            || self.target_height != height
        {
            check(
                unsafe {
                    (self.canvas_target)(self.canvas, pixels.as_mut_ptr(), width, width, height, 1)
                },
                "tvg_swcanvas_set_target",
            )?;
            self.target = pixels.as_mut_ptr();
            self.target_width = width;
            self.target_height = height;
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
        )
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

impl Drop for ThorvgCpu {
    fn drop(&mut self) {
        unsafe {
            (self.canvas_destroy)(self.canvas);
            (self.animation_del)(self.animation);
            (self.engine_term)();
        }
    }
}

