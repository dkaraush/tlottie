#[cfg(feature = "vulkan")]
use std::path::{Path, PathBuf};

#[cfg(not(feature = "vulkan"))]
fn main() {}

#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
  let shader = Path::new("src/renderer/vulkan/shaders/triangle.wgsl");
  println!("cargo:rerun-if-changed={}", shader.display());
  let source = std::fs::read_to_string(shader)?;
  let module = naga::front::wgsl::parse_str(&source)?;
  let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module)?;
  let mut options = naga::back::spv::Options::default();
  options.flags.remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
  let out = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);

  for (entry, stage, file) in [
    ("vs_main", naga::ShaderStage::Vertex, "triangle.vert.spv"),
    ("vs_cover", naga::ShaderStage::Vertex, "cover.vert.spv"),
    ("fs_main", naga::ShaderStage::Fragment, "triangle.frag.spv"),
    ("fs_stencil", naga::ShaderStage::Fragment, "stencil.frag.spv"),
  ] {
    let pipeline = naga::back::spv::PipelineOptions {
      shader_stage: stage,
      entry_point: entry.to_string(),
    };
    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))?;
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    std::fs::write(out.join(file), bytes)?;
  }

  let coverage_shader = Path::new("src/renderer/vulkan/shaders/coverage.wgsl");
  println!("cargo:rerun-if-changed={}", coverage_shader.display());
  let coverage_source = std::fs::read_to_string(coverage_shader)?;
  let coverage_module = naga::front::wgsl::parse_str(&coverage_source)?;
  let coverage_info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&coverage_module)?;
  for (entry, file) in [("main", "coverage.comp.spv"), ("simple_main", "coverage-simple.comp.spv")] {
    let coverage_pipeline = naga::back::spv::PipelineOptions {
      shader_stage: naga::ShaderStage::Compute,
      entry_point: entry.to_string(),
    };
    let coverage_words = naga::back::spv::write_vec(&coverage_module, &coverage_info, &options, Some(&coverage_pipeline))?;
    let coverage_bytes = coverage_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    std::fs::write(out.join(file), coverage_bytes)?;
  }

  let bins_shader = Path::new("src/renderer/vulkan/shaders/bins.wgsl");
  println!("cargo:rerun-if-changed={}", bins_shader.display());
  let bins_source = std::fs::read_to_string(bins_shader)?;
  let bins_module = naga::front::wgsl::parse_str(&bins_source)?;
  let bins_info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&bins_module)?;
  let bins_pipeline = naga::back::spv::PipelineOptions {
    shader_stage: naga::ShaderStage::Compute,
    entry_point: "main".to_string(),
  };
  let bins_words = naga::back::spv::write_vec(&bins_module, &bins_info, &options, Some(&bins_pipeline))?;
  let bins_bytes = bins_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
  std::fs::write(out.join("bins.comp.spv"), bins_bytes)?;

  Ok(())
}
