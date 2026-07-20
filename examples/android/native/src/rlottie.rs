use std::ffi::{c_char, c_void, CString};

use libloading::Library;

type Animation = c_void;
type FromData = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut Animation;
type Destroy = unsafe extern "C" fn(*mut Animation);
type Render = unsafe extern "C" fn(*mut Animation, usize, *mut u32, usize, usize, usize);
type Init = unsafe extern "C" fn();
type Shutdown = unsafe extern "C" fn();

pub struct Rlottie {
    animation: *mut Animation,
    destroy: Destroy,
    render: Render,
    shutdown: Option<Shutdown>,
    // Kept last so function pointers and animation are released first.
    _library: Library,
}

impl Rlottie {
    pub fn new(variant: usize, json: &[u8]) -> Result<Self, String> {
        let library_name = match variant {
            0 => "librlottie_current.so",
            1 => "librlottie_2019.so",
            2 => "librlottie_2019_patched.so",
            _ => return Err(format!("unknown rlottie variant {variant}")),
        };
        // SAFETY: the packaged libraries are held by this object until after
        // all resolved functions and their animation have been released.
        let library = unsafe { Library::new(library_name) }
            .map_err(|error| format!("load {library_name}: {error}"))?;
        let from_data = *unsafe { library.get::<FromData>(b"lottie_animation_from_data\0") }
            .map_err(|error| format!("resolve rlottie from_data: {error}"))?;
        let destroy = *unsafe { library.get::<Destroy>(b"lottie_animation_destroy\0") }
            .map_err(|error| format!("resolve rlottie destroy: {error}"))?;
        let render = *unsafe { library.get::<Render>(b"lottie_animation_render\0") }
            .map_err(|error| format!("resolve rlottie render: {error}"))?;
        let init = unsafe { library.get::<Init>(b"lottie_init\0") }
            .ok()
            .map(|symbol| *symbol);
        let shutdown = unsafe { library.get::<Shutdown>(b"lottie_shutdown\0") }
            .ok()
            .map(|symbol| *symbol);
        if let Some(init) = init {
            unsafe { init() };
        }
        let data = CString::new(json).map_err(|_| "Lottie JSON contains a NUL byte".to_string())?;
        let key = CString::new(format!("tlottie-android-{variant}"))
            .map_err(|error| format!("rlottie cache key: {error}"))?;
        let resource = c"";
        let animation = unsafe { from_data(data.as_ptr(), key.as_ptr(), resource.as_ptr()) };
        if animation.is_null() {
            if let Some(shutdown) = shutdown {
                unsafe { shutdown() };
            }
            return Err(format!("{library_name} rejected the Lottie JSON"));
        }
        Ok(Self {
            animation,
            destroy,
            render,
            shutdown,
            _library: library,
        })
    }

    pub fn render(&mut self, frame: f32, pixels: &mut [u32], width: u32, height: u32) {
        unsafe {
            (self.render)(
                self.animation,
                frame.round().max(0.0) as usize,
                pixels.as_mut_ptr(),
                width as usize,
                height as usize,
                width as usize * 4,
            );
        }
    }
}

impl Drop for Rlottie {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.animation);
            if let Some(shutdown) = self.shutdown {
                shutdown();
            }
        }
    }
}

