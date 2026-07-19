//! `tlottie-cli info <file.json>...` — parse and print header metadata.
//! `tlottie-cli render [options] <file.json> <frame> <size> <out.png>` — render frame to png file
//! `tlottie-cli bench <file.json> <size> <frames>` — native CPU timing harness

use std::process::ExitCode;

use tlottie::{CPURenderer, Composition, Limits, RenderOptions};

#[path = "../pixel.rs"]
mod pixel;
mod png;
mod vulkan_host;

struct RenderArgs<'a> {
  backend: &'a str,
  file: &'a str,
  frame: &'a str,
  size: &'a str,
  out: &'a str,
  options: RenderOptions,
}

fn parse_render_args(args: &[String]) -> Result<RenderArgs<'_>, String> {
  let mut backend = "cpu";
  let mut options = RenderOptions::default();
  let mut positional = Vec::with_capacity(4);
  let mut args = args.iter();

  while let Some(arg) = args.next() {
    match arg.as_str() {
      "--backend" => backend = args.next().ok_or("--backend requires cpu or vulkan")?,
      "--antialias" | "--aa" | "-a" => options.antialias = true,
      "--no-antialias" | "--no-aa" => options.antialias = false,
      "--curve-tolerance" => {
        let value = args.next().ok_or("--curve-tolerance requires a positive number")?;
        options.curve_tolerance = value.parse().map_err(|_| "--curve-tolerance requires a positive number")?;
        if !options.curve_tolerance.is_finite() || options.curve_tolerance <= 0.0 {
          return Err("--curve-tolerance requires a positive number".into());
        }
      }
      "--single-color" => options.single_color = true,
      option if option.starts_with("--aa=") || option.starts_with("--antialias=") => {
        let Some((_, value)) = option.split_once('=') else {
          return Err(format!("invalid antialias option: {option}"));
        };
        options.antialias = match value {
          "true" => true,
          "false" => false,
          _ => return Err(format!("{option} requires true or false")),
        };
      }
      option if option.starts_with("--") => return Err(format!("unknown render option: {option}")),
      value => positional.push(value),
    }
  }

  let [file, frame, size, out] = positional.as_slice() else {
    return Err("render requires a file, frame, size, and output path".into());
  };
  Ok(RenderArgs {
    backend,
    file,
    frame,
    size,
    out,
    options,
  })
}

fn main() -> ExitCode {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.split_first() {
    Some((cmd, files)) if cmd == "info" && !files.is_empty() => info(files),
    Some((cmd, rest)) if cmd == "render" => render_cmd(rest),
    Some((cmd, rest)) if cmd == "bench" => bench_cmd(rest),
    _ => usage(),
  }
}

fn usage() -> ExitCode {
  eprintln!("usage: tlottie-cli info <file.json>...");
  eprintln!("       tlottie-cli render [--backend cpu|vulkan] [--curve-tolerance <pixels>] <file.json> <frame> <size> <out.png>");
  eprintln!("       tlottie-cli bench <file.json> <size> <frames>");
  ExitCode::from(2)
}

fn bench_cmd(args: &[String]) -> ExitCode {
  let [file, size, frames] = args else {
    return usage();
  };
  let (Ok(size), Ok(frames)) = (size.parse::<u32>(), frames.parse::<usize>()) else {
    eprintln!("bench: size and frames must be positive integers");
    return ExitCode::from(2);
  };
  if size == 0 || frames == 0 {
    eprintln!("bench: size and frames must be positive integers");
    return ExitCode::from(2);
  }

  let fms_started = std::time::Instant::now();
  let bytes = match std::fs::read(file) {
    Ok(bytes) => bytes,
    Err(error) => {
      eprintln!("bench: {file}: {error}");
      return ExitCode::FAILURE;
    }
  };
  let composition = match Composition::parse(&bytes, &Limits::default()) {
    Ok(composition) => composition,
    Err(error) => {
      eprintln!("bench: {file}: parse error: {error}");
      return ExitCode::FAILURE;
    }
  };
  let frame_count = composition.frame_count().max(1) as usize;
  let mut renderer = CPURenderer::new(composition);
  let Some(pixel_count) = (size as usize).checked_mul(size as usize) else {
    eprintln!("bench: dimensions overflow");
    return ExitCode::FAILURE;
  };
  let mut pixels = vec![0u32; pixel_count];
  let options = RenderOptions::default();
  for index in 0..frames {
    let started = std::time::Instant::now();
    if let Err(error) = renderer.render((index % frame_count) as f32, &mut pixels, size, size, options) {
      eprintln!("bench: render {index}: {error}");
      return ExitCode::FAILURE;
    }
    println!("F {index} {}", started.elapsed().as_nanos());
    if index == 0 {
      println!("FMS {:.6}", fms_started.elapsed().as_secs_f64() * 1000.0);
    }
  }
  println!("T {frame_count}");
  ExitCode::SUCCESS
}

fn render_cmd(args: &[String]) -> ExitCode {
  let args = match parse_render_args(args) {
    Ok(args) => args,
    Err(error) => {
      eprintln!("render: {error}");
      return usage();
    }
  };
  render(args.file, args.frame, args.size, args.out, args.backend, args.options)
}

fn info(files: &[String]) -> ExitCode {
  let limits = Limits::default();
  let mut failures: u64 = 0;
  for path in files {
    let bytes = match std::fs::read(path) {
      Ok(bytes) => bytes,
      Err(e) => {
        println!("{path}\t{e}");
        failures += 1;
        continue;
      }
    };
    match Composition::parse(&bytes, &limits) {
      Ok(comp) => {
        println!(
          "{path}\t{}x{}\t{}fps\t{} frames\t{:.2}s",
          comp.width,
          comp.height,
          comp.frame_rate,
          comp.frame_count(),
          comp.duration_secs()
        );
      }
      Err(e) => {
        println!("{path}\tERROR\t{e}");
        failures += 1;
      }
    }
  }
  if files.len() > 1 {
    eprintln!("{} ok, {} failed", files.len() as u64 - failures, failures);
  }
  if failures == 0 {
    ExitCode::SUCCESS
  } else {
    ExitCode::FAILURE
  }
}

fn render(file: &str, frame: &str, size: &str, out: &str, backend: &str, options: RenderOptions) -> ExitCode {
  let (Ok(frame), Ok(size)) = (frame.parse::<f32>(), size.parse::<u32>()) else {
    eprintln!("bad frame/size");
    return ExitCode::from(2);
  };
  if size == 0 {
    eprintln!("bad frame/size");
    return ExitCode::from(2);
  }
  let bytes = match std::fs::read(file) {
    Ok(b) => b,
    Err(e) => {
      eprintln!("{file}: {e}");
      return ExitCode::FAILURE;
    }
  };
  let comp = match Composition::parse(&bytes, &Limits::default()) {
    Ok(c) => c,
    Err(e) => {
      eprintln!("{file}: parse error: {e}");
      return ExitCode::FAILURE;
    }
  };
  let n = (size as usize).saturating_mul(size as usize);
  let mut pixels = vec![0u32; n];
  match backend {
    "cpu" => {
      let mut renderer = CPURenderer::new(comp);
      if let Err(e) = renderer.render(frame, &mut pixels, size, size, options) {
        eprintln!("{file}: render error: {e}");
        return ExitCode::FAILURE;
      }
    }
    "vulkan" => {
      let code = vulkan_host::render(&comp, frame, &mut pixels, size, size, options);
      if code != ExitCode::SUCCESS {
        return code;
      }
    }
    _ => {
      eprintln!("unknown backend: {backend}");
      return ExitCode::from(2);
    }
  }

  let covered = pixels.iter().filter(|px| (*px >> 24) & 0xff > 8).count();
  if let Err(e) = png::write_png(out, size, &pixels) {
    eprintln!("{out}: {e}");
    return ExitCode::FAILURE;
  }
  println!("{out}: frame {frame} at {size}x{size}, {covered}/{n} pixels covered ({:.1}%)", 100.0 * covered as f64 / n as f64);
  return ExitCode::SUCCESS;
}

#[cfg(test)]
mod tests {
  use super::*;

  fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
  }

  #[test]
  fn parses_render_options_and_positionals_in_any_order() {
    let values = args(&["animation.json", "--no-antialias", "12", "--backend", "vulkan", "512", "out.png", "--curve-tolerance", "0.125"]);
    let parsed = parse_render_args(&values).unwrap();

    assert_eq!(parsed.backend, "vulkan");
    assert_eq!((parsed.file, parsed.frame, parsed.size, parsed.out), ("animation.json", "12", "512", "out.png"));
    assert!(!parsed.options.antialias);
    assert_eq!(parsed.options.curve_tolerance, 0.125);
  }

  #[test]
  fn render_options_use_accurate_defaults() {
    let values = args(&["animation.json", "0", "256", "out.png"]);
    let parsed = parse_render_args(&values).unwrap();

    assert_eq!(parsed.backend, "cpu");
    assert!(parsed.options.antialias);
    assert_eq!(parsed.options.curve_tolerance, 0.05);
  }

  #[test]
  fn rejects_invalid_curve_tolerance() {
    let values = args(&["--curve-tolerance", "0", "animation.json", "0", "256", "out.png"]);
    assert!(parse_render_args(&values).is_err());
  }

  #[test]
  fn parses_antialias_aliases() {
    for enabled in ["--aa", "--aa=true", "--antialias", "--antialias=true", "-a"] {
      let values = args(&[enabled, "animation.json", "0", "256", "out.png"]);
      assert!(parse_render_args(&values).unwrap().options.antialias, "{enabled}");
    }
    for disabled in ["--no-aa", "--aa=false", "--no-antialias", "--antialias=false"] {
      let values = args(&[disabled, "animation.json", "0", "256", "out.png"]);
      assert!(!parse_render_args(&values).unwrap().options.antialias, "{disabled}");
    }
  }

  #[test]
  fn rejects_invalid_antialias_value() {
    let values = args(&["--aa=maybe", "animation.json", "0", "256", "out.png"]);
    assert!(parse_render_args(&values).is_err());
  }
}
