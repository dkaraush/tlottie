#!/usr/bin/env python3
"""Run repeatable on-device playback benchmarks for every Android backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
APK = HERE / "app/build/outputs/apk/debug/app-debug.apk"
PACKAGE = "com.example.tlottie"
ACTIVITY = f"{PACKAGE}/.MainActivity"
DEFAULT_FIXTURE = ("/storage/emulated/0/Android/data/com.example.tlottie/files/tgs_dump/"
                   "HarryGorilla/effects/5282751684647339553.json")
BACKENDS = {
    "tlottie-cpu": 0,
    "tlottie-vulkan": 1,
    "rlottie": 2,
    "rlottie-2019": 3,
    "rlottie-2019-patched": 4,
    "thorvg-cpu": 5,
    "thorvg-gpu": 6,
}
ACTUAL_BACKENDS = {
    "tlottie-cpu": "tlottie_CPU",
    "tlottie-vulkan": "tlottie_Vulkan",
    "rlottie": "rlottie",
    "rlottie-2019": "rlottie_2019",
    "rlottie-2019-patched": "rlottie_2019_patched",
    "thorvg-cpu": "ThorVG_CPU",
    "thorvg-gpu": "ThorVG_GPU_(OpenGL_ES)",
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def adb(serial: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    executable = os.environ.get(
        "ADB", str(Path.home() / "Android/Sdk/platform-tools/adb")
    )
    return run([executable, "-s", serial, *arguments], check=check)


def build() -> None:
    environment = os.environ.copy()
    environment.setdefault("ANDROID_SDK_ROOT", str(Path.home() / "Android/Sdk"))
    environment.setdefault("JAVA_HOME", str(Path.home() / "Applications/Android Studio/jbr"))
    subprocess.run(
        ["./gradlew", ":app:assembleDebug"], cwd=HERE, env=environment, check=True
    )


def wait_for_result(serial: str, seconds: int, expect_gpu: bool) -> tuple[str, str, str]:
    deadline = time.monotonic() + seconds + 30
    result_seen_at: float | None = None
    presentation_started = False
    presentation = ""
    while time.monotonic() < deadline:
        output = adb(serial, "logcat", "-d", "-v", "brief").stdout
        if not presentation_started and any(
            "TLottieBench" in line and "MEASURE_START" in line
            for line in output.splitlines()
        ):
            reset_surface_stats(serial)
            presentation_started = True
        result = next(
            (line.split("RESULT ", 1)[1] for line in output.splitlines() if "TLottieBench" in line and "RESULT " in line),
            "",
        )
        error = next(
            (line for line in output.splitlines() if "TLottieBench" in line and "ERROR " in line),
            "",
        )
        if result:
            if result_seen_at is None:
                result_seen_at = time.monotonic()
                result_fields = fields(result)
                presentation = surface_presentation_summary(
                    serial,
                    expect_gpu,
                    float(result_fields.get("refresh_hz", "60")),
                )
            if not expect_gpu:
                return result, "", presentation
            gpu = next(
                (line.split("GPU ", 1)[1] for line in output.splitlines() if "TLottieBench" in line and "GPU " in line),
                "",
            )
            if gpu or time.monotonic() - result_seen_at >= 10:
                return result, gpu, presentation
        if error:
            raise RuntimeError(error)
        time.sleep(0.25)
    backend_errors = adb(serial, "logcat", "-d", "-v", "brief").stdout
    relevant = "\n".join(
        line for line in backend_errors.splitlines()
        if "TLottieBench" in line or "TLottieBackend" in line
    )
    raise TimeoutError(f"benchmark timed out\n{relevant}")


def fields(result: str) -> dict[str, str]:
    return dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", result))


def reset_surface_stats(serial: str) -> None:
    adb(serial, "shell", "dumpsys", "SurfaceFlinger", "--timestats", "-clear", check=False)
    adb(serial, "shell", "dumpsys", "SurfaceFlinger", "--timestats", "-enable", check=False)


def surface_presentation_summary(serial: str, _gpu_surface: bool, refresh_hz: float) -> str:
    dump = adb(
        serial, "shell", "dumpsys", "SurfaceFlinger", "--timestats", "-dump",
        check=False,
    ).stdout
    sections = re.split(r"(?=displayRefreshRate =)", dump)
    candidates: list[tuple[int, str]] = []
    for section in sections:
        name_match = re.search(r"^layerName = (.+)$", section, re.MULTILINE)
        frames_match = re.search(r"^totalFrames = (\d+)$", section, re.MULTILINE)
        if not name_match or not frames_match or "com.example.tlottie" not in name_match.group(1):
            continue
        name = name_match.group(1)
        if "MainActivity" not in name or "Splash Screen" in name:
            continue
        candidates.append((int(frames_match.group(1)), section))
    if not candidates:
        return "backend=surfaceflinger supported=0 presented=0"
    presented, section = max(candidates, key=lambda candidate: candidate[0])
    histogram_match = re.search(
        r"present2present histogram is as below:\s*\n([^\n]+)", section
    )
    if not histogram_match:
        return f"backend=surfaceflinger supported=0 presented={presented}"
    buckets: list[tuple[int, int]] = []
    for milliseconds, count in re.findall(r"(\d+)ms=(\d+)", histogram_match.group(1)):
        if int(count):
            buckets.append((int(milliseconds), int(count)))
    total_intervals = sum(count for _, count in buckets)

    def percentile(percent: int) -> int:
        if total_intervals == 0:
            return 0
        target = max(1, (total_intervals * percent + 99) // 100)
        cumulative = 0
        for milliseconds, count in buckets:
            cumulative += count
            if cumulative >= target:
                return milliseconds * 1_000_000
        return buckets[-1][0] * 1_000_000

    missed = sum(
        count * max(0, round(milliseconds * refresh_hz / 1000.0) - 1)
        for milliseconds, count in buckets
    )
    missed_percent = 100.0 * missed / max(1, total_intervals + missed)
    dropped = re.search(r"^droppedFrames = (\d+)$", section, re.MULTILINE)
    janky = re.search(r"^jankyFrames = (\d+)$", section, re.MULTILINE)
    return (
        f"backend=surfaceflinger supported=1 presented={presented} "
        f"sf_dropped={dropped.group(1) if dropped else '0'} "
        f"sf_janky={janky.group(1) if janky else '0'} "
        f"present_missed_pct={missed_percent:.2f} "
        f"present_interval_p50_ns={percentile(50)} "
        f"present_interval_p90_ns={percentile(90)} "
        f"present_interval_p99_ns={percentile(99)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="RZCX41P9KTM")
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--size", type=int, action="append", dest="sizes")
    parser.add_argument("--backend", action="append", choices=BACKENDS, dest="backends")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--aa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curve-tolerance", type=float, default=0.5)
    args = parser.parse_args()
    sizes = args.sizes or [512]
    backends = args.backends or list(BACKENDS)

    if not args.no_build:
        build()
    if not args.no_install:
        print(adb(args.serial, "install", "-r", str(APK)).stdout.strip())
    original_timeout = adb(
        args.serial, "shell", "settings", "get", "system", "screen_off_timeout"
    ).stdout.strip()
    adb(args.serial, "shell", "settings", "put", "system", "screen_off_timeout", "1800000")
    if adb(args.serial, "shell", "test", "-r", args.fixture).returncode != 0:
        raise FileNotFoundError(f"device fixture is not readable: {args.fixture}")

    rows: list[dict[str, str]] = []
    try:
        for size in sizes:
            for name in backends:
                print(f"running {name} at {size}px...", flush=True)
                adb(args.serial, "shell", "input", "keyevent", "224", check=False)
                adb(args.serial, "logcat", "-c")
                adb(args.serial, "shell", "am", "force-stop", PACKAGE)
                adb(
                    args.serial,
                    "shell", "am", "start", "-W", "-n", ACTIVITY,
                    "--es", "benchmark_fixture", args.fixture,
                    "--ei", "benchmark_backend", str(BACKENDS[name]),
                    "--ei", "benchmark_size", str(size),
                    "--ei", "benchmark_seconds", str(args.seconds),
                    "--ez", "benchmark_aa", str(args.aa).lower(),
                    "--ef", "benchmark_curve_tolerance", str(args.curve_tolerance),
                )
                result, gpu, presentation = wait_for_result(
                    args.serial, args.seconds, name in ("tlottie-vulkan", "thorvg-gpu")
                )
                row = fields(result)
                if row.get("backend") != ACTUAL_BACKENDS[name]:
                    raise RuntimeError(
                        f"requested {name}, but app measured {row.get('backend', 'unknown')}"
                    )
                row["requested"] = name
                rows.append(row)
                print("  " + result)
                if gpu:
                    print("  GPU " + gpu)
                if presentation:
                    row.update(fields(presentation))
                    print("  PRESENT " + presentation)
                adb(args.serial, "shell", "am", "force-stop", PACKAGE, check=False)
    finally:
        adb(args.serial, "shell", "am", "force-stop", PACKAGE, check=False)
        adb(
            args.serial, "shell", "dumpsys", "SurfaceFlinger",
            "--timestats", "-disable", check=False,
        )
        if original_timeout and original_timeout != "null":
            adb(
                args.serial, "shell", "settings", "put", "system",
                "screen_off_timeout", original_timeout, check=False,
            )

    print("\nbackend                 size  callback p50/p90/p99 ms   cb miss%  actual present p50/p90/p99 ms  present miss%  call p50/p90/p99 ms")
    for row in rows:
        frames = [int(row.get(f"interval_p{p}_ns", "0")) / 1e6 for p in (50, 90, 99)]
        timings = [int(row.get(f"render_p{p}_ns", "0")) / 1e6 for p in (50, 90, 99)]
        presents = [int(row.get(f"present_interval_p{p}_ns", "0")) / 1e6 for p in (50, 90, 99)]
        present_text = (
            f"{presents[0]:6.2f}/{presents[1]:6.2f}/{presents[2]:6.2f}"
            if any(presents) else "       n/a       "
        )
        print(
            f"{row['requested']:<23} {row.get('size', '?'):>4}  "
            f"{frames[0]:6.2f}/{frames[1]:6.2f}/{frames[2]:6.2f}  "
            f"{float(row.get('missed_pct', '0')):7.2f}  "
            f"{present_text}  "
            f"{float(row.get('present_missed_pct', '0')):12.2f}  "
            f"{timings[0]:6.2f}/{timings[1]:6.2f}/{timings[2]:6.2f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
