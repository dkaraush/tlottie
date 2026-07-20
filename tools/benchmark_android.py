#!/usr/bin/env python3
"""Run the native renderer fleet on an attached arm64 Android device."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tlottie_host_benchmark", ROOT / "tools" / "benchmark.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load tools/benchmark.py")
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)

DEFAULT_DEVICE_ROOT = "/data/local/tmp/tgs_dump"
DEFAULT_REMOTE = "/data/local/tmp/tlottie-android-benchmark"
ANDROID_RENDERERS = {
    "tlottie": f"{DEFAULT_REMOTE}/tlottie-cli",
    "rlottie": "/data/local/tmp/rlottie_dump_ref",
    "rlottie_2019": "/data/local/tmp/rlottie_dump_rl19",
    "rlottie_2019_patched": "/data/local/tmp/rlottie_dump_rlp",
    "thorvg": "/data/local/tmp/thorvg_dump",
}
FRAME_RE = re.compile(r"^F\s+\d+\s+(\d+)(?:\s+\d+)?$", re.MULTILINE)
FMS_RE = re.compile(r"^FMS\s+([0-9.]+)$", re.MULTILINE)
RSS_RE = re.compile(r"^Max RSS \(KiB\):\s+(\d+)$", re.MULTILINE)


def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, **kwargs)


def adb(serial: str, *args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    executable = os.environ.get("ADB", "adb")
    return subprocess.run([executable, "-s", serial, *args], check=check, text=True, capture_output=capture)


def android_toolchain() -> Path:
    sdk = Path(os.environ.get("ANDROID_SDK_ROOT", os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk")))
    configured = os.environ.get("ANDROID_NDK_HOME")
    if configured:
        ndk = Path(configured)
    else:
        candidates = sorted((sdk / "ndk").glob("*"), reverse=True)
        if not candidates:
            raise SystemExit(f"Android NDK not found under {sdk / 'ndk'}")
        ndk = candidates[0]
    roots = list((ndk / "toolchains/llvm/prebuilt").glob("*/bin"))
    if not roots:
        raise SystemExit(f"Android NDK toolchain not found under {ndk}")
    return roots[0]


def connected_serial(requested: str | None) -> str:
    if requested:
        return requested
    executable = os.environ.get("ADB", "adb")
    result = command([executable, "devices"], capture_output=True)
    serials = [line.split("\t", 1)[0] for line in result.stdout.splitlines() if line.endswith("\tdevice")]
    if len(serials) != 1:
        raise SystemExit(f"expected one connected Android device, found {len(serials)}; pass --serial")
    return serials[0]


def build_tlottie() -> Path:
    toolchain = android_toolchain()
    linker = toolchain / "aarch64-linux-android28-clang"
    env = os.environ.copy()
    env["CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"] = str(linker)
    env["RUSTFLAGS"] = env.get("RUSTFLAGS", "-C target-cpu=cortex-a76")
    command(
        ["cargo", "build", "--target", "aarch64-linux-android", "--release", "--bin", "tlottie-cli", "--features", "cli"],
        cwd=ROOT,
        env=env,
    )
    return ROOT / "target/aarch64-linux-android/release/tlottie-cli"


def device_info(serial: str) -> str:
    values = adb(
        serial,
        "shell",
        "printf '%s|%s|Android %s (API %s)' \"$(getprop ro.product.model)\" \"$(getprop ro.product.cpu.abi)\" \"$(getprop ro.build.version.release)\" \"$(getprop ro.build.version.sdk)\"",
        capture=True,
    ).stdout.strip()
    return values.replace("|", " / ")


def device_machine_details(serial: str) -> str:
    values = adb(
        serial,
        "shell",
        "cpu=\"$(getprop ro.soc.model)\"; [ -n \"$cpu\" ] || cpu=\"$(getprop ro.hardware)\"; "
        "printf '%s|%s|%s|%s|%s' \"$cpu\" \"$(getprop ro.build.version.release)\" "
        "\"$(getprop ro.build.version.sdk)\" \"$(getprop ro.product.cpu.abi)\" "
        "\"$(getprop ro.product.model)\"",
        capture=True,
    ).stdout.strip().split("|", 4)
    if len(values) != 5:
        return device_info(serial)
    cpu, release, api, arch, model = values
    return f"CPU {cpu or 'unknown'}; OS Android {release} (API {api}); arch {arch}; device {model}"


def benchmark_invocation(args: argparse.Namespace) -> str:
    command_line = [
        "python3",
        "tools/benchmark_android.py",
        str(args.input),
        "--out",
        str(args.out),
        "--serial",
        args.serial,
        "--device-root",
        args.device_root,
        "--sizes",
        args.sizes,
        "--frames",
        str(args.frames),
        "--reps",
        str(args.reps),
        "--packs",
        args.packs,
        "--renderers",
        args.renderers,
        "--core-mask",
        args.core_mask,
    ]
    for option, value in (("--limit", args.limit), ("--curve-tolerance", args.curve_tolerance)):
        if value is not None:
            command_line.extend((option, str(value)))
    for option, enabled in (
        ("--skip-build", args.skip_build),
        ("--no-open", args.no_open),
        ("--write-raw", args.write_raw),
    ):
        if enabled:
            command_line.append(option)
    return shlex.join(command_line)


def frame_count(file: Path) -> int:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        return max(1, int(float(data.get("op", 1)) - float(data.get("ip", 0)) + 0.999999))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1


def renderer_command(renderer: str, device_file: str, size: int, frames: int) -> str:
    binary = ANDROID_RENDERERS[renderer]
    if renderer == "tlottie":
        args = [binary, "bench"]
        if renderer_command.curve_tolerance is not None:
            args.extend(("--curve-tolerance", renderer_command.curve_tolerance))
        args.extend((device_file, str(size), str(frames)))
    else:
        count = frame_count(Path(renderer_command.host_file))
        sequence = ",".join(str(index % count) for index in range(frames))
        args = [binary, device_file, str(size), sequence, f"{DEFAULT_REMOTE}/out"]
    return " ".join(shlex.quote(arg) for arg in args)


renderer_command.host_file = Path()  # type: ignore[attr-defined]
renderer_command.curve_tolerance = None  # type: ignore[attr-defined]


def make_script(
    files: list[Path],
    root: Path,
    device_root: str,
    renderers: tuple[str, ...],
    sizes: tuple[int, ...],
    frames: int,
    reps: int,
    core_mask: str,
) -> str:
    lines = ["#!/system/bin/sh", "set -u", "export DUMP_NO_WRITE=1 BENCH_ONCE=1", f"mkdir -p {shlex.quote(DEFAULT_REMOTE + '/out')}"]
    for size in sizes:
        for rep in range(reps):
            for index, file in enumerate(files):
                relative = file.relative_to(root).as_posix()
                device_file = f"{device_root.rstrip('/')}/{relative}"
                order = renderers if (rep + index) % 2 == 0 else tuple(reversed(renderers))
                for renderer in order:
                    marker = json.dumps({"renderer": renderer, "size": size, "rep": rep + 1, "file": relative}, separators=(",", ":"))
                    lines.append(f"echo {shlex.quote('### ' + marker)}")
                    renderer_command.host_file = file  # type: ignore[attr-defined]
                    run = renderer_command(renderer, device_file, size, frames)
                    lines.append(f"toybox time -v taskset {shlex.quote(core_mask)} {run} 2>&1")
                    lines.append("echo '### END'")
    return "\n".join(lines) + "\n"


def parse_log(text: str, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sections = re.findall(r"^### (\{[^\n]*\})\r?\n(.*?)^### END\r?$", text, re.MULTILINE | re.DOTALL)
    for marker, body in sections:
        meta = json.loads(marker)
        timings = [int(value) for value in FRAME_RE.findall(body)]
        fms = FMS_RE.search(body)
        rss = RSS_RE.search(body)
        error = "" if timings and fms else body.strip()[-500:]
        steady = timings[1:]
        rows.append(
            {
                "pack": HOST.pack_of(root, root / str(meta["file"])),
                "file": meta["file"],
                "size": int(meta["size"]),
                "rep": int(meta["rep"]),
                "renderer": meta["renderer"],
                "ok": not error,
                "first_frame_ms": float(fms.group(1)) if fms else 0.0,
                "frame_ms": statistics.mean(steady) / 1_000_000.0 if steady else None,
                "other_frames": len(steady),
                "measured_ms": sum(timings) / 1_000_000.0,
                "memory_avg_mb": int(rss.group(1)) / 1024.0 if rss else 0.0,
                "memory_max_mb": int(rss.group(1)) / 1024.0 if rss else 0.0,
                "energy_j": None,
                "error": error,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=HOST.DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=ROOT / "target/benchmark-android")
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument("--device-root", default=DEFAULT_DEVICE_ROOT)
    parser.add_argument("--sizes", default="64,320,720")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--packs", default="5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--renderers", default=",".join(ANDROID_RENDERERS))
    parser.add_argument("--core-mask", default="80", help="taskset mask; 80 pins CPU 7 on the reference device")
    parser.add_argument("--curve-tolerance", type=float, help="override tlottie's device-space curve tolerance")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--write-raw", action="store_true")
    args = parser.parse_args()
    if args.curve_tolerance is not None and (not math.isfinite(args.curve_tolerance) or args.curve_tolerance <= 0):
        raise SystemExit("--curve-tolerance must be a positive finite number")
    renderer_command.curve_tolerance = None if args.curve_tolerance is None else str(args.curve_tolerance)
    args.serial = connected_serial(args.serial)
    if args.frames < 2 or args.reps < 1:
        raise SystemExit("--frames must be >=2 and --reps must be positive")
    renderers = tuple(item.strip() for item in args.renderers.split(",") if item.strip())
    unknown = [item for item in renderers if item not in ANDROID_RENDERERS]
    if unknown:
        raise SystemExit(f"unknown Android renderers: {','.join(unknown)}")
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    files = HOST.discover(args.input, args.limit)
    packs = sorted({HOST.pack_of(args.input, file) for file in files})
    selected = HOST.select_packs(packs, args.packs)
    files = [file for file in files if HOST.pack_of(args.input, file) in set(selected)]
    print(f"== Android packs {len(selected)}/{len(packs)}: {', '.join(selected)} ({len(files)} files)")
    print(f"== device {device_info(args.serial)}, core mask {args.core_mask}")

    adb(args.serial, "shell", "mkdir", "-p", DEFAULT_REMOTE)
    if not args.skip_build:
        binary = build_tlottie()
        adb(args.serial, "push", str(binary), f"{DEFAULT_REMOTE}/tlottie-cli")
        adb(args.serial, "shell", "chmod", "755", f"{DEFAULT_REMOTE}/tlottie-cli")
    for renderer in renderers:
        result = adb(args.serial, "shell", "test", "-x", ANDROID_RENDERERS[renderer], capture=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"missing Android renderer executable: {ANDROID_RENDERERS[renderer]}")

    script = make_script(files, args.input, args.device_root, renderers, sizes, args.frames, args.reps, args.core_mask)
    local_script = args.out / "benchmark-device.sh"
    args.out.mkdir(parents=True, exist_ok=True)
    local_script.write_text(script, encoding="utf-8")
    adb(args.serial, "push", str(local_script), f"{DEFAULT_REMOTE}/benchmark-device.sh")
    print(f"== running {len(files) * len(sizes) * len(renderers) * args.reps} isolated renderer cases on device", flush=True)
    run_result = adb(args.serial, "shell", "sh", f"{DEFAULT_REMOTE}/benchmark-device.sh", capture=True)
    log_path = args.out / "benchmark-device.log"
    log_path.write_text(run_result.stdout, encoding="utf-8")
    rows = parse_log(run_result.stdout, args.input)
    expected = len(files) * len(sizes) * len(renderers) * args.reps
    if len(rows) != expected:
        raise SystemExit(f"parsed {len(rows)}/{expected} Android rows; inspect {log_path}")
    failures = [row for row in rows if not row["ok"]]
    if failures:
        raise SystemExit(f"{len(failures)} Android rows failed; inspect {log_path}")

    file_rows = HOST.aggregate_file_rows(rows)
    pack_rows = HOST.aggregate_pack_rows(file_rows)
    pivot = HOST.pivot_aggregate(pack_rows, ("pack", "size"))
    tgv = args.out / "benchmark.tgv"
    html = args.out / "benchmark.html"
    HOST.write_tgv(tgv, pivot, renderers, ("pack", "size"))
    HOST.write_html(
        html,
        pack_rows,
        file_rows,
        renderers,
        False,
        args.reps,
        None,
        min(sizes),
        16,
        1.0,
        benchmark_invocation(args),
        device_machine_details(args.serial),
    )
    if args.write_raw:
        (args.out / "benchmark.raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {tgv}")
    print(f"wrote {html}")
    print(f"wrote {log_path}")
    if args.write_raw:
        print(f"wrote {args.out / 'benchmark.raw.json'}")
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
