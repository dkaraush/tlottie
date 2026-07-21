//! OpenGL 3.3, OpenGL ES 3.0, and WebGL2 presentation backend.
//!
//! Enable this module with the `opengl` Cargo feature. The same renderer code
//! runs on desktop OpenGL/OpenGL ES and WebGL2 through [`glow`]. Frame geometry
//! is currently rasterized by tlottie's CPU backend and uploaded as one
//! premultiplied RGBA texture before a fullscreen GPU draw. This is a useful
//! canvas/window integration backend, but it is not yet a GPU geometry
//! rasterizer like the Vulkan backend.
//!
//! On native targets the host creates and owns the GL context and passes a
//! loaded [`glow::Context`] to [`OpenGlRenderer::new`]. With the additional
//! `webgl` feature on `wasm32`, `WebGlRenderer` creates WebGL2 directly from
//! an `HtmlCanvasElement`.

#![allow(unsafe_code)]

pub use glow;
use glow::HasContext;

use crate::{CPURenderer, Composition, RenderOptions};

/// OpenGL backend result type.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors returned by the OpenGL and WebGL2 presentation backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
  /// The requested target dimensions are zero, overflow addressable memory,
  /// or exceed the GL signed-dimension range.
  BadTarget,
  /// Lottie frame evaluation or CPU rasterization failed.
  Lottie(crate::Error),
  /// A shader or GL object could not be created.
  Gl(String),
  /// A WebGL2 context could not be obtained from the canvas.
  #[cfg(target_arch = "wasm32")]
  WebGlContext,
}

impl core::fmt::Display for Error {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Error::BadTarget => write!(f, "bad OpenGL draw target"),
      Error::Lottie(error) => write!(f, "Lottie evaluation failed: {error}"),
      Error::Gl(error) => write!(f, "OpenGL error: {error}"),
      #[cfg(target_arch = "wasm32")]
      Error::WebGlContext => write!(f, "canvas did not provide a WebGL2 context"),
    }
  }
}

impl std::error::Error for Error {}

impl From<crate::Error> for Error {
  fn from(value: crate::Error) -> Self {
    Self::Lottie(value)
  }
}

/// CPU rasterizer plus a shared OpenGL/OpenGL ES/WebGL2 presentation path.
///
/// Rendering targets whichever framebuffer is bound when [`Self::render`] is
/// called. The backend sets viewport, program, vertex-array, texture-unit 0,
/// unpack alignment, and common fixed-function state; hosts that mix tlottie
/// with other GL drawing should restore their own state afterwards.
pub struct OpenGlRenderer {
  gl: glow::Context,
  cpu: CPURenderer,
  pixels: Vec<u32>,
  rgba: Vec<u8>,
  program: glow::Program,
  vao: glow::VertexArray,
  texture: glow::Texture,
  texture_size: (u32, u32),
}

impl OpenGlRenderer {
  /// Creates a renderer from a loaded GL context and a composition.
  ///
  /// The context must support OpenGL 3.3, OpenGL ES 3.0, or WebGL2.
  ///
  /// # Safety
  ///
  /// `gl` must refer to a live context which is current on this thread. The
  /// same context must be current for every call to [`Self::render`] and when
  /// this renderer is dropped.
  pub unsafe fn new(gl: glow::Context, composition: Composition) -> Result<Self> {
    // SAFETY: the caller supplies a current, compatible context.
    let (program, vao, texture) = unsafe { create_resources(&gl)? };
    Ok(Self {
      gl,
      cpu: CPURenderer::new(composition),
      pixels: Vec::new(),
      rgba: Vec::new(),
      program,
      vao,
      texture,
      texture_size: (0, 0),
    })
  }

  /// Returns the parsed composition rendered by this instance.
  pub fn composition(&self) -> &Composition {
    self.cpu.composition()
  }

  /// Rasterizes and presents one frame into the currently bound framebuffer.
  ///
  /// The framebuffer must have at least `width` by `height` writable pixels.
  pub fn render(&mut self, frame: f32, width: u32, height: u32, options: RenderOptions) -> Result<()> {
    let (pixel_count, byte_count) = target_lengths(width, height)?;
    if self.pixels.len() != pixel_count {
      self.pixels.resize(pixel_count, 0);
    }
    self.cpu.render(frame, &mut self.pixels, width, height, options)?;
    if self.rgba.len() != byte_count {
      self.rgba.resize(byte_count, 0);
    }
    copy_premultiplied_rgba(&self.pixels, &mut self.rgba);

    // SAFETY: resources belong to the current context by the constructor
    // contract, dimensions fit GL's i32 API, and `rgba` has width*height*4
    // initialized bytes.
    unsafe { self.present(width, height) };
    Ok(())
  }

  unsafe fn present(&mut self, width: u32, height: u32) {
    let gl = &self.gl;
    // SAFETY: upheld by `render` and the type's context-current contract.
    unsafe {
      gl.viewport(0, 0, width as i32, height as i32);
      gl.disable(glow::BLEND);
      gl.disable(glow::DEPTH_TEST);
      gl.disable(glow::STENCIL_TEST);
      gl.disable(glow::SCISSOR_TEST);
      gl.color_mask(true, true, true, true);
      gl.active_texture(glow::TEXTURE0);
      gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
      gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

      if self.texture_size == (width, height) {
        gl.tex_sub_image_2d(
          glow::TEXTURE_2D,
          0,
          0,
          0,
          width as i32,
          height as i32,
          glow::RGBA,
          glow::UNSIGNED_BYTE,
          glow::PixelUnpackData::Slice(Some(&self.rgba)),
        );
      } else {
        gl.tex_image_2d(
          glow::TEXTURE_2D,
          0,
          glow::RGBA as i32,
          width as i32,
          height as i32,
          0,
          glow::RGBA,
          glow::UNSIGNED_BYTE,
          glow::PixelUnpackData::Slice(Some(&self.rgba)),
        );
        self.texture_size = (width, height);
      }

      gl.use_program(Some(self.program));
      gl.bind_vertex_array(Some(self.vao));
      gl.draw_arrays(glow::TRIANGLES, 0, 3);
    }
  }
}

impl Drop for OpenGlRenderer {
  fn drop(&mut self) {
    // SAFETY: the public constructor contract requires this renderer's GL
    // context to be current when it is dropped.
    unsafe {
      self.gl.delete_texture(self.texture);
      self.gl.delete_vertex_array(self.vao);
      self.gl.delete_program(self.program);
    }
  }
}

fn target_lengths(width: u32, height: u32) -> Result<(usize, usize)> {
  if width == 0 || height == 0 || width > i32::MAX as u32 || height > i32::MAX as u32 {
    return Err(Error::BadTarget);
  }
  let pixels = (width as usize).checked_mul(height as usize).ok_or(Error::BadTarget)?;
  let bytes = pixels.checked_mul(4).ok_or(Error::BadTarget)?;
  Ok((pixels, bytes))
}

fn copy_premultiplied_rgba(src: &[u32], dst: &mut [u8]) {
  for (&pixel, rgba) in src.iter().zip(dst.chunks_exact_mut(4)) {
    if let [red, green, blue, alpha] = rgba {
      *red = pixel as u8;
      *green = (pixel >> 8) as u8;
      *blue = (pixel >> 16) as u8;
      *alpha = (pixel >> 24) as u8;
    }
  }
}

unsafe fn create_resources(gl: &glow::Context) -> Result<(glow::Program, glow::VertexArray, glow::Texture)> {
  let vertex_source = if cfg!(target_arch = "wasm32") {
    r#"#version 300 es
precision highp float;
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}"#
  } else {
    r#"#version 330 core
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}"#
  };
  let fragment_source = if cfg!(target_arch = "wasm32") {
    r#"#version 300 es
precision highp float;
uniform sampler2D frame_texture;
in vec2 uv;
out vec4 color;
void main() { color = texture(frame_texture, uv); }"#
  } else {
    r#"#version 330 core
uniform sampler2D frame_texture;
in vec2 uv;
out vec4 color;
void main() { color = texture(frame_texture, uv); }"#
  };

  // SAFETY: called only with a live, current context.
  unsafe {
    let vertex = compile_shader(gl, glow::VERTEX_SHADER, vertex_source)?;
    let fragment = match compile_shader(gl, glow::FRAGMENT_SHADER, fragment_source) {
      Ok(shader) => shader,
      Err(error) => {
        gl.delete_shader(vertex);
        return Err(error);
      }
    };
    let program = match link_program(gl, vertex, fragment) {
      Ok(program) => program,
      Err(error) => {
        gl.delete_shader(vertex);
        gl.delete_shader(fragment);
        return Err(error);
      }
    };
    gl.delete_shader(vertex);
    gl.delete_shader(fragment);

    let vao = match gl.create_vertex_array() {
      Ok(vao) => vao,
      Err(error) => {
        gl.delete_program(program);
        return Err(Error::Gl(error));
      }
    };
    let texture = match gl.create_texture() {
      Ok(texture) => texture,
      Err(error) => {
        gl.delete_vertex_array(vao);
        gl.delete_program(program);
        return Err(Error::Gl(error));
      }
    };

    gl.use_program(Some(program));
    if let Some(location) = gl.get_uniform_location(program, "frame_texture") {
      gl.uniform_1_i32(Some(&location), 0);
    }
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    Ok((program, vao, texture))
  }
}

unsafe fn compile_shader(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::Shader> {
  // SAFETY: called only with a live, current context and a valid shader kind.
  unsafe {
    let shader = gl.create_shader(kind).map_err(Error::Gl)?;
    gl.shader_source(shader, source);
    gl.compile_shader(shader);
    if gl.get_shader_compile_status(shader) {
      Ok(shader)
    } else {
      let log = gl.get_shader_info_log(shader);
      gl.delete_shader(shader);
      Err(Error::Gl(log))
    }
  }
}

unsafe fn link_program(gl: &glow::Context, vertex: glow::Shader, fragment: glow::Shader) -> Result<glow::Program> {
  // SAFETY: called only with a live, current context and compiled shaders.
  unsafe {
    let program = gl.create_program().map_err(Error::Gl)?;
    gl.attach_shader(program, vertex);
    gl.attach_shader(program, fragment);
    gl.link_program(program);
    if gl.get_program_link_status(program) {
      Ok(program)
    } else {
      let log = gl.get_program_info_log(program);
      gl.delete_program(program);
      Err(Error::Gl(log))
    }
  }
}

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
mod web;

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
pub use web::WebGlRenderer;

#[cfg(test)]
mod tests {
  use super::{copy_premultiplied_rgba, target_lengths, Error};

  #[test]
  fn copies_premultiplied_rgba_without_unpremultiplying() {
    let mut output = [0; 8];
    copy_premultiplied_rgba(&[0x8000_2040, 0xff56_3412], &mut output);
    assert_eq!(output, [0x40, 0x20, 0x00, 0x80, 0x12, 0x34, 0x56, 0xff]);
  }

  #[test]
  fn rejects_empty_targets() {
    assert!(matches!(target_lengths(0, 10), Err(Error::BadTarget)));
    assert!(matches!(target_lengths(10, 0), Err(Error::BadTarget)));
  }
}
