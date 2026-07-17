//! Dev tool.
//! `tlottie-cli info <file.json>...` — parse and print header metadata.
//! `tlottie-cli render <file.json> <frame> <size> <out.ppm>` — render one
//! frame to a binary PPM (P6, white background) for eyeballing.

use std::process::ExitCode;
use std::time::Instant;

use tlottie::{Composition, Limits, RenderOptions};

mod vulkan;

fn render_options() -> RenderOptions {
    let antialias = std::env::var("TLOTTIE_AA")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
        .unwrap_or(true);
    let curve_tolerance = std::env::var("TLOTTIE_CURVE_TOLERANCE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.05);
    RenderOptions {
        antialias,
        curve_tolerance,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, files)) if cmd == "info" && !files.is_empty() => info(files),
        Some((cmd, rest)) if cmd == "render" => render_cmd(rest),
        Some((cmd, rest)) if cmd == "dump" => match rest {
            [file, size, frames, outdir] => dump(file, size, frames, outdir),
            _ => usage(),
        },
        _ => usage(),
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: tlottie-cli info <file.json>...");
    eprintln!("       tlottie-cli render [--backend cpu|vulkan] <file.json> <frame> <size> <out.ppm|out.png>");
    eprintln!("       tlottie-cli dump <file.json> <size> <f0,f1,...> <outdir>");
    ExitCode::from(2)
}

fn render_cmd(args: &[String]) -> ExitCode {
    let (backend, rest) = match args {
        [flag, backend, rest @ ..] if flag == "--backend" => (backend.as_str(), rest),
        rest => ("cpu", rest),
    };
    match rest {
        [file, frame, size, out] => render(file, frame, size, out, backend),
        _ => usage(),
    }
}

/// Same protocol as tools/rlottie_dump.cpp: renders each frame, writes
/// premultiplied ARGB32 raws, prints "F <frame> <ns_per_render> <reps>"
/// per frame and "T <total_renders>" at the end.
fn dump(file: &str, size: &str, frames: &str, outdir: &str) -> ExitCode {
    let Ok(size) = size.parse::<u32>() else {
        eprintln!("bad size");
        return ExitCode::from(2);
    };
    let frame_list: Vec<f32> = frames
        .split(',')
        .filter_map(|s| s.parse::<f32>().ok())
        .collect();
    let fms_t0 = Instant::now();
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
    let mut anim = tlottie::Animation::new(comp);
    let options = render_options();
    let mut total_renders: u64 = 0;
    let mut first_frame_ms: Option<f64> = None;
    for &frame in &frame_list {
        let frame = frame.min(anim.composition().frame_count().saturating_sub(1) as f32);
        let t0 = Instant::now();
        if let Err(e) = anim.render_with_options(frame, &mut pixels, size, size, options) {
            eprintln!("{file}: render error: {e}");
            return ExitCode::FAILURE;
        }
        if first_frame_ms.is_none() {
            first_frame_ms = Some(fms_t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut dt = t0.elapsed().as_nanos() as u64;
        total_renders += 1;
        let mut reps: u64 = 1;
        // TLOTTIE_ONCE=1: render each listed frame exactly once (steady-state
        // measurement across explicit loop passes; adaptive reps would warm
        // the coverage cache within a single frame otherwise).
        if std::env::var("TLOTTIE_ONCE").is_err()
            && std::env::var("BENCH_ONCE").is_err()
            && dt < 2_000_000
        {
            reps = 2_000_000 / dt.max(1000) + 1;
            reps = reps.min(500);
            let t0 = Instant::now();
            for _ in 0..reps {
                if anim
                    .render_with_options(frame, &mut pixels, size, size, options)
                    .is_err()
                {
                    return ExitCode::FAILURE;
                }
            }
            dt = (t0.elapsed().as_nanos() as u64) / reps;
            total_renders += reps;
        }
        println!("F {frame} {dt} {reps}");
        // DUMP_NO_WRITE=1: benchmark-only mode — skip the .raw output
        // (on-device runs would otherwise write GBs to flash per pass).
        if std::env::var("DUMP_NO_WRITE").is_err() {
            let mut raw = Vec::with_capacity(n * 4);
            for px in &pixels {
                raw.extend_from_slice(&px.to_le_bytes());
            }
            if let Err(e) = std::fs::write(format!("{outdir}/f{frame}.raw"), &raw) {
                eprintln!("write error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(ms) = first_frame_ms {
        println!("FMS {ms:.3}");
    }
    // Peak RSS in bytes (battery/memory proxy for the harness; macOS
    // reports bytes, Linux kilobytes).
    let mut ru = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
        let scale = if cfg!(target_os = "macos") { 1 } else { 1024 };
        println!("M {}", ru.ru_maxrss as i64 * scale);
        // CPU time (user+sys) in ns — the uniform battery proxy where the
        // harness can't read child rusage across adb (device runs).
        let cpu_ns = (ru.ru_utime.tv_sec as i64 + ru.ru_stime.tv_sec as i64) * 1_000_000_000
            + (ru.ru_utime.tv_usec as i64 + ru.ru_stime.tv_usec as i64) * 1_000;
        println!("C {cpu_ns}");
    }
    // Energy consumed by this process in nanojoules (Apple Silicon,
    // TASK_POWER_INFO_V2.task_energy — the Activity Monitor source).
    // ri_billed_energy was tried first and reads 0 for CLI processes.
    #[cfg(target_os = "macos")]
    {
        // mach task_info FFI (libc doesn't expose task_power_info_v2).
        // Layout per <mach/task_info.h>: task_power_info (6×u64),
        // gpu_energy_data (4×u64: utilisation + 3 reserved), then on
        // arm64: task_energy, task_ptime, task_pset_switches (u64 each).
        #[repr(C)]
        #[derive(Default)]
        struct TaskPowerInfoV2 {
            total_user: u64,
            total_system: u64,
            task_interrupt_wakeups: u64,
            task_platform_idle_wakeups: u64,
            task_timer_wakeups_bin_1: u64,
            task_timer_wakeups_bin_2: u64,
            task_gpu_utilisation: u64,
            task_gpu_stat_reserved0: u64,
            task_gpu_stat_reserved1: u64,
            task_gpu_stat_reserved2: u64,
            task_energy: u64,
            task_ptime: u64,
            task_pset_switches: u64,
        }
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: u32, out: *mut TaskPowerInfoV2, cnt: *mut u32) -> i32;
        }
        const TASK_POWER_INFO_V2: u32 = 26;
        let mut info = TaskPowerInfoV2::default();
        let mut count = (core::mem::size_of::<TaskPowerInfoV2>() / 4) as u32;
        let kr = unsafe { task_info(mach_task_self(), TASK_POWER_INFO_V2, &mut info, &mut count) };
        if kr == 0 {
            println!("E {}", info.task_energy);
        }
    }
    println!("T {total_renders}");
    // Dev instrumentation: gradient pixels per kind + batched coverage.
    if std::env::var("TLOTTIE_GRAD_STATS").is_ok() {
        let m = tlottie::mode_stats();
        println!("MS s {} d_extent {} d_density {}", m[0], m[1], m[2]);
        let g = tlottie::gradient_stats();
        println!(
            "G lin {} rad {} foc {} linrad_batched {} foc_batched {}",
            g[0], g[1], g[2], g[3], g[4]
        );
        let p = tlottie::px_stats();
        println!(
            "PX replay_cov {} replay_span {} fresh_s {} fresh_d {} mask_cov {} mask_walk {} offclear {} composite {} spans_fresh {} spans_replay {} modulate {} offtakes {}",
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]
        );
        let s = tlottie::stroke_stats();
        println!(
            "SK outline_open {} outline_closed {} pieces_open {} pieces_closed {} pieces_total {}",
            s[0], s[1], s[2], s[3], s[4]
        );
    }
    ExitCode::SUCCESS
}

fn info(files: &[String]) -> ExitCode {
    let limits = Limits::default();
    let mut failures: u64 = 0;
    for path in files {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("{path}\tREAD-ERROR\t{e}");
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

/// Append one PNG chunk (length + type + data + CRC32).
fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = 0xffff_ffffu32;
    for &b in kind.iter().chain(data.iter()) {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    out.extend_from_slice(&(!crc).to_be_bytes());
}

/// Minimal PNG writer: 8-bit straight-alpha RGBA, stored (uncompressed)
/// deflate blocks. Dev-tool output only — keeps the zero-dependency rule.
fn write_png(path: &str, size: u32, pixels: &[u32]) -> std::io::Result<()> {
    let w = size as usize;
    // scanlines: filter byte 0 + un-premultiplied RGBA
    let mut raw = Vec::with_capacity((w * 4 + 1) * w);
    for row in pixels.chunks(w.max(1)) {
        raw.push(0u8);
        for px in row {
            let a = (px >> 24) & 0xff;
            let unmul = |c: u32| {
                if a == 0 {
                    0
                } else {
                    ((c * 255 + a / 2) / a).min(255) as u8
                }
            };
            raw.push(unmul((px >> 16) & 0xff));
            raw.push(unmul((px >> 8) & 0xff));
            raw.push(unmul(px & 0xff));
            raw.push(a as u8);
        }
    }
    // zlib: header, stored blocks (<=65535 bytes), adler32
    let mut idat = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    while let Some(block) = blocks.next() {
        idat.push(u8::from(blocks.peek().is_none()));
        let len = block.len() as u16;
        idat.extend_from_slice(&len.to_le_bytes());
        idat.extend_from_slice(&(!len).to_le_bytes());
        idat.extend_from_slice(block);
    }
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    idat.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

    let mut out = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&size.to_be_bytes());
    ihdr.extend_from_slice(&size.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth 8, color type RGBA
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &idat);
    png_chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out)
}

fn render(file: &str, frame: &str, size: &str, out: &str, backend: &str) -> ExitCode {
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
            if let Err(e) =
                comp.render_with_options(frame, &mut pixels, size, size, render_options())
            {
                eprintln!("{file}: render error: {e}");
                return ExitCode::FAILURE;
            }
        }
        "vulkan" => {
            let code = vulkan::render_with_options(
                &comp,
                frame,
                &mut pixels,
                size,
                size,
                render_options(),
            );
            if code != ExitCode::SUCCESS {
                return code;
            }
        }
        _ => {
            eprintln!("unknown backend: {backend}");
            return ExitCode::from(2);
        }
    }

    // .png → transparent PNG (straight alpha); anything else → P6 PPM
    // composited over white (PPM has no alpha channel).
    if out.ends_with(".png") {
        let covered = pixels.iter().filter(|px| (*px >> 24) & 0xff > 8).count();
        if let Err(e) = write_png(out, size, &pixels) {
            eprintln!("{out}: {e}");
            return ExitCode::FAILURE;
        }
        println!(
            "{out}: frame {frame} at {size}x{size}, {covered}/{n} pixels covered ({:.1}%)",
            100.0 * covered as f64 / n as f64
        );
        return ExitCode::SUCCESS;
    }
    let mut ppm = Vec::with_capacity(n * 3 + 32);
    ppm.extend_from_slice(format!("P6\n{size} {size}\n255\n").as_bytes());
    let mut colored = 0u64;
    for px in &pixels {
        let a = (px >> 24) & 0xff;
        let r = (px >> 16) & 0xff;
        let g = (px >> 8) & 0xff;
        let b = px & 0xff;
        let inv = 255 - a;
        // dst is opaque white; src is premultiplied.
        ppm.push((r + inv).min(255) as u8);
        ppm.push((g + inv).min(255) as u8);
        ppm.push((b + inv).min(255) as u8);
        if a > 8 {
            colored += 1;
        }
    }
    if let Err(e) = std::fs::write(out, &ppm) {
        eprintln!("{out}: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "{out}: frame {frame} at {size}x{size}, {colored}/{n} pixels covered ({:.1}%)",
        100.0 * colored as f64 / n as f64
    );
    ExitCode::SUCCESS
}
