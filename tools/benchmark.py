#!/usr/bin/env python3
"""Benchmark tlottie, rlottie variants, and ThorVG via native libraries.

The runner processes one renderer+canvas-size batch at a time so memory and
package energy counters, when available, can be attributed to that renderer.
Within each batch it uses many worker processes, and each worker loads only the
native library being measured.

Energy: on Linux, RAPL package energy is read around the batch and prorated per
row by measured time. On macOS (Apple Silicon), per-task consumed energy
(task_info TASK_POWER_INFO_V2.task_energy, the Activity Monitor source) is
sampled around each renderer run inside the worker, giving direct per-row
attribution. Memory on macOS is sampled via proc_pid_rusage physical footprint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager, ExitStack
import ctypes as C
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import quote
import webbrowser
import zlib

try:
    import numpy as np
except ImportError:  # Keep the benchmark usable with the Python standard library only.
    np = None


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
DEFAULT_INPUT = Path.home() / "Documents" / "fixtures-full"
DEFAULT_OUT = ROOT / "target" / "benchmark"
DEFAULT_SIZES = (64, 320, 720)
DEFAULT_RENDERERS = ("tlottie", "rlottie", "rlottie_2019", "rlottie_2019_patched", "thorvg")
RENDERERS = DEFAULT_RENDERERS + ("tlottie-vulkan",)
RLOTTIE_RENDERERS = ("rlottie", "rlottie_2019", "rlottie_2019_patched")
RENDERER_URLS = {
    "tlottie": "https://github.com/dkaraush/tlottie",
    "tlottie-vulkan": "https://github.com/dkaraush/tlottie",
    "rlottie": "https://github.com/Samsung/rlottie",
    "rlottie_2019": "https://github.com/TelegramMessenger/rlottie",
    "rlottie_2019_patched": "https://github.com/dkaraush/rlottie",
    "thorvg": "https://github.com/thorvg/thorvg",
}
PROJECT_DIRS = {
    "rlottie": PROJECTS / "rlottie",
    "rlottie_2019": PROJECTS / "rlottie_2019",
    "rlottie_2019_patched": PROJECTS / "rlottie_2019_patched",
    "thorvg": PROJECTS / "thorvg",
}

LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"

LIBS = {
    "tlottie": ROOT / "target" / "release" / f"libtlottie{LIB_SUFFIX}",
    "rlottie": PROJECT_DIRS["rlottie"] / "build-release" / "src" / f"librlottie{LIB_SUFFIX}",
    "rlottie_2019": PROJECT_DIRS["rlottie_2019"]
    / "build-release"
    / "src"
    / f"librlottie{LIB_SUFFIX}",
    "rlottie_2019_patched": PROJECT_DIRS["rlottie_2019_patched"]
    / "build-release"
    / "src"
    / f"librlottie{LIB_SUFFIX}",
    "thorvg": PROJECT_DIRS["thorvg"] / "build-release" / "src" / f"libthorvg-1{LIB_SUFFIX}",
    "tlottie-vulkan": ROOT / "target" / "release" / "tlottie-cli",
}


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_builds(skip: bool) -> None:
    validate_project_dirs()
    if skip:
        validate_libs()
        return
    env = os.environ.copy()
    env["RUSTFLAGS"] = env.get("RUSTFLAGS", "-C target-cpu=native")
    run(["cargo", "build", "--release", "--lib", "--features", "c-api"], ROOT, env)
    run(
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "tlottie-cli",
            "--features",
            "cli,vulkan",
        ],
        ROOT,
        env,
    )

    meson = shutil_which("meson")
    if not meson:
        meson = "/tmp/tlottie-build-tools/bin/meson"
    if not Path(meson).exists() and "/" in meson:
        raise SystemExit("meson not found; install meson/ninja or use --skip-build")

    meson_env = os.environ.copy()
    meson_env["PATH"] = f"/tmp/tlottie-build-tools/bin:{meson_env.get('PATH', '')}"
    def setup_cmd(project: Path, extra_cpp_args: str = "") -> list[str]:
        cmd = [meson, "setup"]
        if (project / "build-release").exists():
            cmd.append("--wipe")
        cpp_args = ("-march=native " + extra_cpp_args).strip()
        return cmd + [
            "build-release",
            ".",
            "-Dbuildtype=release",
            "-Db_lto=true",
            f"-Dcpp_args={cpp_args}",
            "-Dc_args=-march=native",
        ]

    # rlottie's NEON fast path calls pixman's 32-bit ARM assembly, which its
    # meson only assembles for cpu_family 'arm' — on arm64 the symbols don't
    # exist and linking fails. Undefine the guards to use the C fallback.
    rlottie_cpp_args = ""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        rlottie_cpp_args = "-U__ARM_NEON__ -U__ARM64_NEON__"

    thorvg_args = [
        "-Dengines=cpu",
        "-Dloaders=lottie",
        "-Dbindings=capi",
        "-Dsimd=true",
        "-Dtools=",
        "-Dtests=false",
    ]
    if platform.system() == "Darwin":
        # Apple clang ships without OpenMP; thorvg's default extra=[...,'openmp']
        # defines THORVG_OPENMP_SUPPORT even when meson can't find the runtime.
        thorvg_args.append("-Dextra=lottie_exp")
    run(
        setup_cmd(PROJECTS / "thorvg") + thorvg_args,
        PROJECT_DIRS["thorvg"],
        meson_env,
    )
    run([meson, "compile", "-C", "build-release"], PROJECT_DIRS["thorvg"], meson_env)

    run(
        setup_cmd(PROJECTS / "rlottie", rlottie_cpp_args)
        + ["-Dexample=false", "-Dtest=false"],
        PROJECT_DIRS["rlottie"],
        meson_env,
    )
    run([meson, "compile", "-C", "build-release"], PROJECT_DIRS["rlottie"], meson_env)

    for project_name in ("rlottie_2019", "rlottie_2019_patched"):
        project = PROJECT_DIRS[project_name]
        run(
            setup_cmd(project, rlottie_cpp_args)
            + [
                "-Dexample=false",
                "-Dtest=false",
                "-Dmodule=false",
                "-Dwerror=false",
            ],
            project,
            meson_env,
        )
        run([meson, "compile", "-C", "build-release"], project, meson_env)
    validate_libs()


def validate_project_dirs() -> None:
    resolved: dict[Path, str] = {}
    for renderer, project in PROJECT_DIRS.items():
        if not project.exists():
            raise SystemExit(f"{renderer} project not found: {project}")
        real = project.resolve()
        if project.is_symlink():
            raise SystemExit(f"{renderer} project must be a real checkout, not symlink: {project} -> {real}")
        prev = resolved.get(real)
        if prev:
            raise SystemExit(f"{renderer} and {prev} resolve to the same project directory: {real}")
        resolved[real] = renderer


def validate_libs() -> None:
    resolved: dict[Path, str] = {}
    for renderer, lib in LIBS.items():
        if not lib.exists():
            raise SystemExit(f"{renderer} library not found: {lib}")
        real = lib.resolve()
        prev = resolved.get(real)
        if prev:
            raise SystemExit(f"{renderer} and {prev} resolve to the same library: {real}")
        resolved[real] = renderer


def shutil_which(name: str) -> str | None:
    for part in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(part) / name
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def discover(root: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    return files[:limit] if limit else files


def pack_of(root: Path, file: Path) -> str:
    try:
        rel = file.relative_to(root)
    except ValueError:
        return "."
    return rel.parts[0] if len(rel.parts) > 1 else "."


def select_packs(all_packs: list[str], selector: str) -> list[str]:
    """Select packs by exact name, count, or inclusive 1-based range."""
    selector = selector.strip()
    if not selector:
        raise SystemExit("--packs must not be empty")

    if "," in selector:
        parts = [part.strip() for part in selector.split(",")]
        if len(parts) != 2:
            raise SystemExit("--packs range must be START,END")
        try:
            start, end = (int(part) for part in parts)
        except ValueError:
            raise SystemExit("--packs range must use integer positions: START,END") from None
        if start <= 0 or end <= 0:
            raise SystemExit("--packs range positions must be positive")
        if start > end:
            raise SystemExit("--packs range START must not exceed END")
        return all_packs[start - 1 : end]

    try:
        count = int(selector)
    except ValueError:
        if selector not in all_packs:
            raise SystemExit(f"unknown pack: {selector}") from None
        return [selector]

    if count == 0:
        raise SystemExit("--packs count must not be zero")
    if count > 0:
        return all_packs[:count]
    return all_packs[count:]


_RUSAGE_INFO_V2 = 2


class _RusageInfoV2(C.Structure):
    # sys/resource.h struct rusage_info_v2 (all fields uint64 after the uuid).
    _fields_ = [
        ("ri_uuid", C.c_uint8 * 16),
        ("ri_user_time", C.c_uint64),
        ("ri_system_time", C.c_uint64),
        ("ri_pkg_idle_wkups", C.c_uint64),
        ("ri_interrupt_wkups", C.c_uint64),
        ("ri_pageins", C.c_uint64),
        ("ri_wired_size", C.c_uint64),
        ("ri_resident_size", C.c_uint64),
        ("ri_phys_footprint", C.c_uint64),
        ("ri_proc_start_abstime", C.c_uint64),
        ("ri_proc_exit_abstime", C.c_uint64),
        ("ri_child_user_time", C.c_uint64),
        ("ri_child_system_time", C.c_uint64),
        ("ri_child_pkg_idle_wkups", C.c_uint64),
        ("ri_child_interrupt_wkups", C.c_uint64),
        ("ri_child_pageins", C.c_uint64),
        ("ri_child_elapsed_abstime", C.c_uint64),
        ("ri_diskio_bytesread", C.c_uint64),
        ("ri_diskio_byteswritten", C.c_uint64),
    ]


_TASK_POWER_INFO_V2 = 26


class _TaskPowerInfoV2(C.Structure):
    # mach/task_info.h struct task_power_info_v2; task_energy only exists in
    # the arm/arm64 layout, so this struct must not be used on Intel.
    _fields_ = [
        ("total_user", C.c_uint64),
        ("total_system", C.c_uint64),
        ("task_interrupt_wakeups", C.c_uint64),
        ("task_platform_idle_wakeups", C.c_uint64),
        ("task_timer_wakeups_bin_1", C.c_uint64),
        ("task_timer_wakeups_bin_2", C.c_uint64),
        ("task_gpu_utilisation", C.c_uint64),
        ("task_gpu_stat_reserved0", C.c_uint64),
        ("task_gpu_stat_reserved1", C.c_uint64),
        ("task_gpu_stat_reserved2", C.c_uint64),
        ("task_energy", C.c_uint64),
        ("task_ptime", C.c_uint64),
        ("task_pset_switches", C.c_uint64),
    ]


_LIBPROC = None
_LIBSYSTEM = None
if platform.system() == "Darwin":
    try:
        _LIBPROC = C.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _LIBPROC.proc_pid_rusage.argtypes = [C.c_int, C.c_int, C.c_void_p]
        _LIBPROC.proc_pid_rusage.restype = C.c_int
    except (OSError, AttributeError):
        _LIBPROC = None
    if platform.machine() == "arm64":
        try:
            _LIBSYSTEM = C.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            _LIBSYSTEM.task_info.argtypes = [
                C.c_uint,
                C.c_uint,
                C.c_void_p,
                C.POINTER(C.c_uint),
            ]
            _LIBSYSTEM.task_info.restype = C.c_int
        except (OSError, AttributeError):
            _LIBSYSTEM = None


def process_memory_mb(pid: int) -> float:
    """Return the process memory metric used by the benchmark, in MiB.

    macOS's resident size includes shared/mapped pages and is a misleading
    comparison between native libraries.  ``ri_phys_footprint`` is the kernel's
    accounting of the process's attributable physical memory and is the value
    Activity Monitor uses for its Memory column.  Linux has no direct
    equivalent available here, so retain current RSS there.

    The historical ``rss_mb`` helper and raw-output ``*_rss_*`` aliases are
    retained for compatibility with existing benchmark consumers.
    """
    if _LIBPROC is not None:
        info = _RusageInfoV2()
        if _LIBPROC.proc_pid_rusage(pid, _RUSAGE_INFO_V2, C.byref(info)) == 0:
            return info.ri_phys_footprint / 1048576.0
        return 0.0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def rss_mb() -> float:
    return process_memory_mb(os.getpid())


def task_energy_nj() -> int | None:
    """Consumed energy of this task in nJ (macOS Apple Silicon only)."""
    if _LIBSYSTEM is None:
        return None
    info = _TaskPowerInfoV2()
    count = C.c_uint(C.sizeof(info) // 4)
    try:
        task = C.c_uint.in_dll(_LIBSYSTEM, "mach_task_self_").value
    except ValueError:
        return None
    if _LIBSYSTEM.task_info(task, _TASK_POWER_INFO_V2, C.byref(info), C.byref(count)) != 0:
        return None
    return int(info.task_energy) or None


class EnergySampler:
    def __init__(self) -> None:
        self.sources = sorted(Path("/sys/class/powercap").glob("**/energy_uj"))
        self.start_values: dict[Path, int] = {}

    def available(self) -> bool:
        return bool(self.sources)

    def start(self) -> None:
        self.start_values = {p: self._read(p) for p in self.sources}

    def stop_j(self) -> float | None:
        if not self.start_values:
            return None
        total_uj = 0
        for path, before in self.start_values.items():
            after = self._read(path)
            # RAPL counters wrap; common width is at least 32 bits in uJ.
            if after < before:
                after += 1 << 32
            total_uj += max(0, after - before)
        return total_uj / 1_000_000.0

    @staticmethod
    def _read(path: Path) -> int:
        try:
            return int(path.read_text().strip())
        except OSError:
            return 0


class FrameStream:
    """A random-access facade over one live renderer animation."""

    def __init__(self, count: int, render: Any) -> None:
        self.count = count
        self.render = render

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, frame: int) -> Any:
        if frame < 0 or frame >= self.count:
            raise IndexError(frame)
        return self.render(frame)


class Tlottie:
    def __init__(self, path: Path, curve_tolerance: float = 0.125) -> None:
        self.lib = C.CDLL(str(path))
        self.curve_tolerance = curve_tolerance
        self.lib.tlottie_new.argtypes = [C.c_void_p, C.c_size_t]
        self.lib.tlottie_new.restype = C.c_void_p
        self.lib.tlottie_drop.argtypes = [C.c_void_p]
        self.lib.tlottie_frame_count.argtypes = [C.c_void_p]
        self.lib.tlottie_frame_count.restype = C.c_uint32
        self.lib.tlottie_render.argtypes = [
            C.c_void_p,
            C.c_float,
            C.c_uint32,
            C.c_uint32,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_uint32,
        ]
        self.lib.tlottie_render.restype = C.c_int
        self.lib.tlottie_render_with_options.argtypes = [
            C.c_void_p,
            C.c_float,
            C.c_uint32,
            C.c_uint32,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_uint32,
            C.c_float,
        ]
        self.lib.tlottie_render_with_options.restype = C.c_int

    def _render(
        self,
        anim: Any,
        frame: float,
        width: int,
        height: int,
        pixels: Any,
        out_len: int,
        antialias: int,
    ) -> int:
        return int(
            self.lib.tlottie_render_with_options(
                anim,
                frame,
                width,
                height,
                pixels,
                out_len,
                antialias,
                self.curve_tolerance,
            )
        )

    @contextmanager
    def frame_stream(self, file: Path, size: int) -> Any:
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_new(buf, len(data))
        if not anim:
            raise RuntimeError("parse")
        pixels = (C.c_uint32 * (size * size))()
        count = max(1, int(self.lib.tlottie_frame_count(anim)))

        def render(frame: int) -> Any:
            rc = self._render(
                anim, float(frame % count), size, size, pixels, size * size, 1
            )
            if rc != 0:
                raise RuntimeError(f"render:{rc}@{frame}")
            return memoryview(pixels)

        try:
            yield FrameStream(count, render)
        finally:
            self.lib.tlottie_drop(anim)

    def measure(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str]:
        t0 = time.perf_counter_ns()
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_new(buf, len(data))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_frame_count(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            rc = self._render(
                anim, 0.0, size, size, pixels, size * size, 1
            )
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if rc != 0:
                return False, first_ms, None, 0, rss_mb(), rss_mb(), f"render:{rc}"
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                frame = float(i % count)
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    self.lib.tlottie_drop(anim)
                    anim = self.lib.tlottie_new(buf, len(data))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse"
                rc = self._render(
                    anim, frame, size, size, pixels, size * size, 1
                )
                render_ns += time.perf_counter_ns() - t1
                if rc != 0:
                    return False, first_ms, None, i - 1, rss_mb(), rss_mb(), f"render:{rc}"
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return True, first_ms, other_ms, other_frames, avg(rss_samples), max(rss_samples), ""
        finally:
            if anim:
                self.lib.tlottie_drop(anim)

    def render_argb(self, file: Path, size: int, frame: int) -> tuple[bool, list[int], str]:
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_new(buf, len(data))
        if not anim:
            return False, [], "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_frame_count(anim)))
            rc = self._render(
                anim, float(frame % count), size, size, pixels, size * size, 1
            )
            if rc != 0:
                return False, [], f"render:{rc}"
            return True, list(pixels), ""
        finally:
            if anim:
                self.lib.tlottie_drop(anim)

    def render_frames_argb(
        self, file: Path, size: int, max_frames: int = 0
    ) -> tuple[bool, list[bytes], int, str]:
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_new(buf, len(data))
        if not anim:
            return False, [], 0, "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_frame_count(anim)))
            frames = []
            render_count = min(count, max_frames) if max_frames > 0 else count
            for frame in range(render_count):
                rc = self._render(
                    anim, float(frame), size, size, pixels, size * size, 1
                )
                if rc != 0:
                    return False, [], count, f"render:{rc}@{frame}"
                frames.append(bytes(pixels))
            return True, frames, count, ""
        finally:
            self.lib.tlottie_drop(anim)

    def measure_frames_argb(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str, list[list[int]], int]:
        t0 = time.perf_counter_ns()
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_new(buf, len(data))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse", [], 0
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_frame_count(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            out_frames: list[list[int]] = []
            rc = self._render(
                anim, 0.0, size, size, pixels, size * size, 1
            )
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if rc != 0:
                return False, first_ms, None, 0, rss_mb(), rss_mb(), f"render:{rc}", [], count
            out_frames.append(list(pixels))
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    self.lib.tlottie_drop(anim)
                    anim = self.lib.tlottie_new(buf, len(data))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse", [], count
                rc = self._render(
                    anim, float(i % count), size, size, pixels, size * size, 1
                )
                render_ns += time.perf_counter_ns() - t1
                if rc != 0:
                    return False, first_ms, None, i - 1, rss_mb(), rss_mb(), f"render:{rc}", [], count
                out_frames.append(list(pixels))
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return (
                True,
                first_ms,
                other_ms,
                other_frames,
                avg(rss_samples),
                max(rss_samples),
                "",
                out_frames,
                count,
            )
        finally:
            self.lib.tlottie_drop(anim)


class TlottieVulkan:
    """Headless Vulkan sequence runner.

    Times CPU frame evaluation/command recording plus queue submit/fence wait.
    Process startup, Vulkan initialization, readback, and PNG encoding are
    excluded from the parsed per-frame measurements.
    """

    FRAME_RE = re.compile(
        r"VK .*?record_ns=(\d+) submit_wait_ns=(\d+) gpu_elapsed_ns=(\d+)"
    )

    def __init__(self, path: Path, curve_tolerance: float = 0.125) -> None:
        self.cli = path
        self.curve_tolerance = curve_tolerance

    @staticmethod
    def frame_count(file: Path) -> int:
        try:
            data = json.loads(file.read_text())
            return max(1, int(math.ceil(float(data.get("op", 1)) - float(data.get("ip", 0)))))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 1

    def measure(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str]:
        count = self.frame_count(file)
        frames = count if frames <= 0 else max(1, frames)
        ok, samples, _pixels, error, mem_avg, mem_max = self._run_sequence(
            file, size, 0, frames, False
        )
        if not ok:
            return False, 0.0, None, 0, mem_avg, mem_max, error
        return (
            True,
            samples[0],
            avg(samples[1:]) if len(samples) > 1 else None,
            len(samples) - 1,
            mem_avg,
            mem_max,
            "",
        )

    def render_argb(self, file: Path, size: int, frame: int) -> tuple[bool, list[int], str]:
        count = self.frame_count(file)
        ok, _samples, frames, error, _mem_avg, _mem_max = self._run_sequence(
            file, size, frame % count, 1, True
        )
        return (True, frames[0], "") if ok and frames else (False, [], error)

    def render_frames_argb(
        self, file: Path, size: int, max_frames: int = 0
    ) -> tuple[bool, list[bytes], int, str]:
        count = self.frame_count(file)
        render_count = min(count, max_frames) if max_frames > 0 else count
        ok, _samples, frames, error, _mem_avg, _mem_max = self._run_sequence(
            file, size, 0, render_count, True
        )
        return ok, frames, count, error

    def measure_frames_argb(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str, list[list[int]], int]:
        count = self.frame_count(file)
        frames = count if frames <= 0 else max(1, frames)
        ok, samples, pixels, error, mem_avg, mem_max = self._run_sequence(
            file, size, 0, frames, True
        )
        if not ok:
            return False, 0.0, None, 0, mem_avg, mem_max, error, [], count
        return (
            True,
            samples[0],
            avg(samples[1:]) if len(samples) > 1 else None,
            len(samples) - 1,
            mem_avg,
            mem_max,
            "",
            pixels,
            count,
        )

    def _run_sequence(
        self, file: Path, size: int, start: int, frames: int, capture: bool
    ) -> tuple[bool, list[float], list[list[int]], str, float, float]:
        env = os.environ.copy()
        env["TLOTTIE_VK_FRAMES"] = str(frames)
        with tempfile.TemporaryDirectory(prefix="tlottie-vulkan-") as temp:
            directory = Path(temp)
            out = directory / "last.png"
            raw_dir = directory / "raw"
            if capture:
                env["TLOTTIE_VK_RAW_DIR"] = str(raw_dir)
            memory_samples: list[float] = []
            with tempfile.TemporaryFile() as stderr_file:
                proc = subprocess.Popen(
                    [
                        str(self.cli),
                        "render",
                        "--backend",
                        "vulkan",
                        "--antialias",
                        "--curve-tolerance",
                        str(self.curve_tolerance),
                        str(file),
                        str(start),
                        str(size),
                        str(out),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
                while proc.poll() is None:
                    memory = process_memory_mb(proc.pid)
                    if memory > 0.0:
                        memory_samples.append(memory)
                    time.sleep(0.001)
                memory = process_memory_mb(proc.pid)
                if memory > 0.0:
                    memory_samples.append(memory)
                stderr_file.seek(0)
                stderr = stderr_file.read().decode(errors="replace")
            mem_avg = avg(memory_samples)
            mem_max = max(memory_samples, default=0.0)
            if proc.returncode != 0:
                return False, [], [], stderr.strip()[-500:], mem_avg, mem_max
            samples = [
                (int(record_ns) + int(submit_ns)) / 1_000_000.0
                for record_ns, submit_ns, _gpu_ns in self.FRAME_RE.findall(stderr)
            ]
            if len(samples) != frames:
                error = f"expected {frames} Vulkan samples, got {len(samples)}"
                return False, [], [], error, mem_avg, mem_max
            pixels = []
            if capture:
                pixel_count = size * size
                expected_bytes = pixel_count * 4
                for index in range(frames):
                    path = raw_dir / f"frame-{index:06}.argb"
                    try:
                        data = path.read_bytes()
                    except OSError as error:
                        return False, [], [], f"read {path.name}: {error}", mem_avg, mem_max
                    if len(data) != expected_bytes:
                        error = f"{path.name}: {len(data)} != {expected_bytes} bytes"
                        return False, [], [], error, mem_avg, mem_max
                    pixels.append(data)
            return True, samples, pixels, "", mem_avg, mem_max


class Rlottie:
    def __init__(self, path: Path) -> None:
        self.lib = C.CDLL(str(path))
        if hasattr(self.lib, "lottie_init"):
            self.lib.lottie_init()
        self.lib.lottie_animation_from_file.argtypes = [C.c_char_p]
        self.lib.lottie_animation_from_file.restype = C.c_void_p
        self.lib.lottie_animation_destroy.argtypes = [C.c_void_p]
        self.lib.lottie_animation_get_totalframe.argtypes = [C.c_void_p]
        self.lib.lottie_animation_get_totalframe.restype = C.c_size_t
        self.lib.lottie_animation_render.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
        ]

    @contextmanager
    def frame_stream(self, file: Path, size: int) -> Any:
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            raise RuntimeError("parse")
        pixels = (C.c_uint32 * (size * size))()
        count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))

        def render(frame: int) -> Any:
            self.lib.lottie_animation_render(
                anim, frame % count, pixels, size, size, size * 4
            )
            return memoryview(pixels)

        try:
            yield FrameStream(count, render)
        finally:
            self.lib.lottie_animation_destroy(anim)

    def measure(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str]:
        t0 = time.perf_counter_ns()
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            self.lib.lottie_animation_render(anim, 0, pixels, size, size, size * 4)
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    self.lib.lottie_animation_destroy(anim)
                    anim = self.lib.lottie_animation_from_file(os.fsencode(file))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse"
                self.lib.lottie_animation_render(
                    anim, i % count, pixels, size, size, size * 4
                )
                render_ns += time.perf_counter_ns() - t1
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return True, first_ms, other_ms, other_frames, avg(rss_samples), max(rss_samples), ""
        finally:
            if anim:
                self.lib.lottie_animation_destroy(anim)

    def render_argb(self, file: Path, size: int, frame: int) -> tuple[bool, list[int], str]:
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            return False, [], "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))
            self.lib.lottie_animation_render(anim, frame % count, pixels, size, size, size * 4)
            return True, list(pixels), ""
        finally:
            if anim:
                self.lib.lottie_animation_destroy(anim)

    def render_frames_argb(
        self, file: Path, size: int, max_frames: int = 0
    ) -> tuple[bool, list[bytes], int, str]:
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            return False, [], 0, "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))
            frames = []
            render_count = min(count, max_frames) if max_frames > 0 else count
            for frame in range(render_count):
                self.lib.lottie_animation_render(anim, frame, pixels, size, size, size * 4)
                frames.append(bytes(pixels))
            return True, frames, count, ""
        finally:
            self.lib.lottie_animation_destroy(anim)

    def measure_frames_argb(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str, list[list[int]], int]:
        t0 = time.perf_counter_ns()
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse", [], 0
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            out_frames: list[list[int]] = []
            self.lib.lottie_animation_render(anim, 0, pixels, size, size, size * 4)
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            out_frames.append(list(pixels))
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    self.lib.lottie_animation_destroy(anim)
                    anim = self.lib.lottie_animation_from_file(os.fsencode(file))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse", [], count
                self.lib.lottie_animation_render(anim, i % count, pixels, size, size, size * 4)
                render_ns += time.perf_counter_ns() - t1
                out_frames.append(list(pixels))
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return (
                True,
                first_ms,
                other_ms,
                other_frames,
                avg(rss_samples),
                max(rss_samples),
                "",
                out_frames,
                count,
            )
        finally:
            self.lib.lottie_animation_destroy(anim)


class Thorvg:
    SUCCESS = 0
    ENGINE_NONE = 0
    ARGB8888 = 1

    def __init__(self, path: Path) -> None:
        self.lib = C.CDLL(str(path))
        self.lib.tvg_engine_init.argtypes = [C.c_uint]
        self.lib.tvg_engine_init.restype = C.c_int
        self.lib.tvg_engine_init(0)
        self.lib.tvg_animation_new.restype = C.c_void_p
        self.lib.tvg_animation_del.argtypes = [C.c_void_p]
        self.lib.tvg_animation_get_picture.argtypes = [C.c_void_p]
        self.lib.tvg_animation_get_picture.restype = C.c_void_p
        self.lib.tvg_animation_get_total_frame.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        self.lib.tvg_animation_set_frame.argtypes = [C.c_void_p, C.c_float]
        self.lib.tvg_picture_load.argtypes = [C.c_void_p, C.c_char_p]
        self.lib.tvg_picture_set_size.argtypes = [C.c_void_p, C.c_float, C.c_float]
        self.lib.tvg_swcanvas_create.argtypes = [C.c_int]
        self.lib.tvg_swcanvas_create.restype = C.c_void_p
        self.lib.tvg_canvas_destroy.argtypes = [C.c_void_p]
        self.lib.tvg_swcanvas_set_target.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
            C.c_int,
        ]
        self.lib.tvg_canvas_add.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.tvg_canvas_update.argtypes = [C.c_void_p]
        self.lib.tvg_canvas_draw.argtypes = [C.c_void_p, C.c_bool]
        self.lib.tvg_canvas_sync.argtypes = [C.c_void_p]

    @contextmanager
    def frame_stream(self, file: Path, size: int) -> Any:
        anim = self.lib.tvg_animation_new()
        if not anim:
            raise RuntimeError("new")
        canvas = None
        try:
            pic = self.lib.tvg_animation_get_picture(anim)
            if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                raise RuntimeError("parse")
            self.lib.tvg_picture_set_size(pic, float(size), float(size))
            total = C.c_float(0.0)
            self.lib.tvg_animation_get_total_frame(anim, C.byref(total))
            count = max(1, int(total.value))
            pixels = (C.c_uint32 * (size * size))()
            canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
            if not canvas:
                raise RuntimeError("canvas")
            if (
                self.lib.tvg_swcanvas_set_target(
                    canvas, pixels, size, size, size, self.ARGB8888
                )
                != self.SUCCESS
            ):
                raise RuntimeError("target")
            if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                raise RuntimeError("add")

            def render(frame: int) -> Any:
                self.lib.tvg_animation_set_frame(anim, float(frame % count))
                self.lib.tvg_canvas_update(canvas)
                self.lib.tvg_canvas_draw(canvas, True)
                self.lib.tvg_canvas_sync(canvas)
                return memoryview(pixels)

            yield FrameStream(count, render)
        finally:
            if canvas:
                self.lib.tvg_canvas_destroy(canvas)
            self.lib.tvg_animation_del(anim)

    def measure(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str]:
        t0 = time.perf_counter_ns()
        anim = self.lib.tvg_animation_new()
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "new"
        canvas = None
        try:
            pic = self.lib.tvg_animation_get_picture(anim)
            if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse"
            self.lib.tvg_picture_set_size(pic, float(size), float(size))
            total = C.c_float(0.0)
            self.lib.tvg_animation_get_total_frame(anim, C.byref(total))
            count = max(1, int(total.value))
            frames = count if frames <= 0 else frames
            pixels = (C.c_uint32 * (size * size))()
            canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
            if not canvas:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "canvas"
            if (
                self.lib.tvg_swcanvas_set_target(
                    canvas, pixels, size, size, size, self.ARGB8888
                )
                != self.SUCCESS
            ):
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "target"
            if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "add"
            rss_samples: list[float] = []
            self.lib.tvg_animation_set_frame(anim, 0.0)
            self.lib.tvg_canvas_update(canvas)
            self.lib.tvg_canvas_draw(canvas, True)
            self.lib.tvg_canvas_sync(canvas)
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    if canvas:
                        self.lib.tvg_canvas_destroy(canvas)
                        canvas = None
                    self.lib.tvg_animation_del(anim)
                    anim = self.lib.tvg_animation_new()
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "new"
                    pic = self.lib.tvg_animation_get_picture(anim)
                    if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse"
                    self.lib.tvg_picture_set_size(pic, float(size), float(size))
                    canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
                    if not canvas:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "canvas"
                    if (
                        self.lib.tvg_swcanvas_set_target(
                            canvas, pixels, size, size, size, self.ARGB8888
                        )
                        != self.SUCCESS
                    ):
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "target"
                    if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "add"
                self.lib.tvg_animation_set_frame(anim, float(i % count))
                self.lib.tvg_canvas_update(canvas)
                self.lib.tvg_canvas_draw(canvas, True)
                self.lib.tvg_canvas_sync(canvas)
                render_ns += time.perf_counter_ns() - t1
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return True, first_ms, other_ms, other_frames, avg(rss_samples), max(rss_samples), ""
        finally:
            if canvas:
                self.lib.tvg_canvas_destroy(canvas)
            if anim:
                self.lib.tvg_animation_del(anim)

    def render_argb(self, file: Path, size: int, frame: int) -> tuple[bool, list[int], str]:
        anim = self.lib.tvg_animation_new()
        if not anim:
            return False, [], "new"
        canvas = None
        try:
            pic = self.lib.tvg_animation_get_picture(anim)
            if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                return False, [], "parse"
            self.lib.tvg_picture_set_size(pic, float(size), float(size))
            total = C.c_float(0.0)
            self.lib.tvg_animation_get_total_frame(anim, C.byref(total))
            count = max(1, int(total.value))
            pixels = (C.c_uint32 * (size * size))()
            canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
            if not canvas:
                return False, [], "canvas"
            if (
                self.lib.tvg_swcanvas_set_target(canvas, pixels, size, size, size, self.ARGB8888)
                != self.SUCCESS
            ):
                return False, [], "target"
            if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                return False, [], "add"
            self.lib.tvg_animation_set_frame(anim, float(frame % count))
            self.lib.tvg_canvas_update(canvas)
            self.lib.tvg_canvas_draw(canvas, True)
            self.lib.tvg_canvas_sync(canvas)
            return True, list(pixels), ""
        finally:
            if canvas:
                self.lib.tvg_canvas_destroy(canvas)
            if anim:
                self.lib.tvg_animation_del(anim)

    def render_frames_argb(
        self, file: Path, size: int, max_frames: int = 0
    ) -> tuple[bool, list[bytes], int, str]:
        anim = self.lib.tvg_animation_new()
        if not anim:
            return False, [], 0, "new"
        canvas = None
        try:
            pic = self.lib.tvg_animation_get_picture(anim)
            if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                return False, [], 0, "parse"
            self.lib.tvg_picture_set_size(pic, float(size), float(size))
            total = C.c_float(0.0)
            self.lib.tvg_animation_get_total_frame(anim, C.byref(total))
            count = max(1, int(total.value))
            pixels = (C.c_uint32 * (size * size))()
            canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
            if not canvas:
                return False, [], count, "canvas"
            if (
                self.lib.tvg_swcanvas_set_target(canvas, pixels, size, size, size, self.ARGB8888)
                != self.SUCCESS
            ):
                return False, [], count, "target"
            if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                return False, [], count, "add"
            frames = []
            render_count = min(count, max_frames) if max_frames > 0 else count
            for frame in range(render_count):
                self.lib.tvg_animation_set_frame(anim, float(frame))
                self.lib.tvg_canvas_update(canvas)
                self.lib.tvg_canvas_draw(canvas, True)
                self.lib.tvg_canvas_sync(canvas)
                frames.append(bytes(pixels))
            return True, frames, count, ""
        finally:
            if canvas:
                self.lib.tvg_canvas_destroy(canvas)
            self.lib.tvg_animation_del(anim)

    def measure_frames_argb(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str, list[list[int]], int]:
        t0 = time.perf_counter_ns()
        anim = self.lib.tvg_animation_new()
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "new", [], 0
        canvas = None
        try:
            pic = self.lib.tvg_animation_get_picture(anim)
            if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse", [], 0
            self.lib.tvg_picture_set_size(pic, float(size), float(size))
            total = C.c_float(0.0)
            self.lib.tvg_animation_get_total_frame(anim, C.byref(total))
            count = max(1, int(total.value))
            frames = count if frames <= 0 else frames
            pixels = (C.c_uint32 * (size * size))()
            canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
            if not canvas:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "canvas", [], count
            if (
                self.lib.tvg_swcanvas_set_target(canvas, pixels, size, size, size, self.ARGB8888)
                != self.SUCCESS
            ):
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "target", [], count
            if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                return False, 0.0, None, 0, rss_mb(), rss_mb(), "add", [], count
            rss_samples: list[float] = []
            out_frames: list[list[int]] = []
            self.lib.tvg_animation_set_frame(anim, 0.0)
            self.lib.tvg_canvas_update(canvas)
            self.lib.tvg_canvas_draw(canvas, True)
            self.lib.tvg_canvas_sync(canvas)
            first_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            out_frames.append(list(pixels))
            rss_samples.append(rss_mb())
            render_ns = 0
            for i in range(frames):
                if i == 0:
                    continue
                t1 = time.perf_counter_ns()
                if i % count == 0:
                    if canvas:
                        self.lib.tvg_canvas_destroy(canvas)
                        canvas = None
                    self.lib.tvg_animation_del(anim)
                    anim = self.lib.tvg_animation_new()
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "new", [], count
                    pic = self.lib.tvg_animation_get_picture(anim)
                    if not pic or self.lib.tvg_picture_load(pic, os.fsencode(file)) != self.SUCCESS:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse", [], count
                    self.lib.tvg_picture_set_size(pic, float(size), float(size))
                    canvas = self.lib.tvg_swcanvas_create(self.ENGINE_NONE)
                    if not canvas:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "canvas", [], count
                    if (
                        self.lib.tvg_swcanvas_set_target(
                            canvas, pixels, size, size, size, self.ARGB8888
                        )
                        != self.SUCCESS
                    ):
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "target", [], count
                    if self.lib.tvg_canvas_add(canvas, pic) != self.SUCCESS:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "add", [], count
                self.lib.tvg_animation_set_frame(anim, float(i % count))
                self.lib.tvg_canvas_update(canvas)
                self.lib.tvg_canvas_draw(canvas, True)
                self.lib.tvg_canvas_sync(canvas)
                render_ns += time.perf_counter_ns() - t1
                out_frames.append(list(pixels))
                rss_samples.append(rss_mb())
            other_frames = max(0, frames - 1)
            other_ms = (render_ns / 1_000_000.0) / other_frames if other_frames else None
            return (
                True,
                first_ms,
                other_ms,
                other_frames,
                avg(rss_samples),
                max(rss_samples),
                "",
                out_frames,
                count,
            )
        finally:
            if canvas:
                self.lib.tvg_canvas_destroy(canvas)
            self.lib.tvg_animation_del(anim)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def avg_optional(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return avg(present) if present else None


_WORKER_RENDERERS: dict[str, Any] = {}
_WORKER_RENDERER_ORDER: tuple[str, ...] = ()
_WORKER_SIZE = 0
_WORKER_FRAMES = 0
_WORKER_ROOT = DEFAULT_INPUT
_WORKER_REPS = 1
_WORKER_ACCURACY_ENABLED = False
_WORKER_ACCURACY_SIZE = 64
_WORKER_ACCURACY_TOLERANCE = 8
_WORKER_ACCURACY_DIFF_THRESHOLD = 1.0
_WORKER_CURVE_TOLERANCE = 0.125


def init_worker(
    renderers: tuple[str, ...],
    libs: dict[str, str],
    size: int,
    frames: int,
    root: str,
    reps: int,
    accuracy_enabled: bool,
    accuracy_size: int,
    accuracy_tolerance: int,
    accuracy_diff_threshold: float,
    curve_tolerance: float,
) -> None:
    global _WORKER_RENDERERS, _WORKER_RENDERER_ORDER, _WORKER_SIZE, _WORKER_FRAMES, _WORKER_ROOT, _WORKER_REPS
    global _WORKER_ACCURACY_ENABLED, _WORKER_ACCURACY_SIZE, _WORKER_ACCURACY_TOLERANCE, _WORKER_ACCURACY_DIFF_THRESHOLD, _WORKER_CURVE_TOLERANCE
    _WORKER_RENDERERS = {}
    _WORKER_RENDERER_ORDER = renderers
    _WORKER_SIZE = size
    _WORKER_FRAMES = frames
    _WORKER_ROOT = Path(root)
    _WORKER_REPS = reps
    _WORKER_ACCURACY_ENABLED = accuracy_enabled
    _WORKER_ACCURACY_SIZE = accuracy_size
    _WORKER_ACCURACY_TOLERANCE = accuracy_tolerance
    _WORKER_ACCURACY_DIFF_THRESHOLD = accuracy_diff_threshold
    _WORKER_CURVE_TOLERANCE = curve_tolerance
    for renderer in renderers:
        lib = Path(libs[renderer])
        if renderer == "tlottie":
            _WORKER_RENDERERS[renderer] = Tlottie(lib, curve_tolerance)
        elif renderer == "tlottie-vulkan":
            _WORKER_RENDERERS[renderer] = TlottieVulkan(lib, curve_tolerance)
        elif renderer in RLOTTIE_RENDERERS:
            _WORKER_RENDERERS[renderer] = Rlottie(lib)
        elif renderer == "thorvg":
            _WORKER_RENDERERS[renderer] = Thorvg(lib)
        else:
            raise RuntimeError(renderer)


def worker_measure(file_s: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    file = Path(file_s)
    rows = []
    accuracy_renderers = ("tlottie", "rlottie", "thorvg")
    capture_renderers = accuracy_renderers + (
        ("tlottie-vulkan",) if "tlottie-vulkan" in _WORKER_RENDERER_ORDER else ()
    )
    capture_accuracy = (
        _WORKER_ACCURACY_ENABLED
        and _WORKER_SIZE == _WORKER_ACCURACY_SIZE
        and all(r in _WORKER_RENDERER_ORDER for r in accuracy_renderers)
    )
    captured: dict[str, list[list[int]]] = {}
    counts: dict[str, int] = {}
    accuracy_errors: list[str] = []
    for rep in range(_WORKER_REPS):
        for renderer in _WORKER_RENDERER_ORDER:
            # tlottie-vulkan renders in a child process, whose energy is not
            # visible in this task's counter — leave its energy_j unset.
            energy_before = None if renderer == "tlottie-vulkan" else task_energy_nj()
            (
                ok,
                first_frame_ms,
                frame_ms,
                other_frames,
                mem_avg,
                mem_max,
                err,
            ) = _WORKER_RENDERERS[renderer].measure(file, _WORKER_SIZE, _WORKER_FRAMES)
            energy_j = None
            if energy_before is not None:
                energy_after = task_energy_nj()
                if energy_after is not None:
                    energy_j = max(0, energy_after - energy_before) / 1e9
            measured_ms = first_frame_ms + ((frame_ms or 0.0) * other_frames) if ok else 0.0
            rows.append(
                {
                    "pack": pack_of(_WORKER_ROOT, file),
                    "file": str(file.relative_to(_WORKER_ROOT)),
                    "size": _WORKER_SIZE,
                    "rep": rep + 1,
                    "renderer": renderer,
                    "ok": ok,
                    "first_frame_ms": first_frame_ms,
                    "frame_ms": frame_ms,
                    "other_frames": other_frames,
                    "measured_ms": measured_ms,
                    "memory_avg_mb": mem_avg,
                    "memory_max_mb": mem_max,
                    "energy_j": energy_j,
                    "error": err,
                }
            )
    # Accuracy capture is deliberately separate from timed repetitions.
    # Materializing Python pixel arrays creates substantial memory pressure at
    # 720px and must not distort or erase the subsequent-frame measurements.
    if capture_accuracy:
        for renderer in capture_renderers:
            (
                ok,
                _first_frame_ms,
                _frame_ms,
                _other_frames,
                _mem_avg,
                _mem_max,
                err,
                frames,
                count,
            ) = _WORKER_RENDERERS[renderer].measure_frames_argb(
                file, _WORKER_SIZE, _WORKER_FRAMES
            )
            captured[renderer] = frames
            counts[renderer] = count
            if not ok:
                accuracy_errors.append(f"{renderer}:{err}")
    accuracy_row = None
    if capture_accuracy:
        accuracy_row = make_accuracy_row(
            file,
            _WORKER_ROOT,
            _WORKER_SIZE,
            captured,
            counts,
            accuracy_errors,
            _WORKER_ACCURACY_TOLERANCE,
            _WORKER_ACCURACY_DIFF_THRESHOLD,
        )
    return rows, accuracy_row


def run_size_batch(
    renderers: tuple[str, ...],
    size: int,
    files: list[Path],
    root: Path,
    frames: int,
    jobs: int,
    reps: int,
    accuracy_enabled: bool,
    accuracy_size: int,
    accuracy_tolerance: int,
    accuracy_diff_threshold: float,
    curve_tolerance: float,
    progress: ProgressDisplay | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampler = EnergySampler()
    sampler.start()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    total = len(files)
    progress_every = progress_interval(total)
    owns_progress = progress is None
    progress = progress or ProgressDisplay(f"measure {renderers[0]} {size}px", total)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=init_worker,
        initargs=(
            renderers,
            {r: str(LIBS[r]) for r in renderers},
            size,
            frames,
            str(root),
            reps,
            accuracy_enabled,
            accuracy_size,
            accuracy_tolerance,
            accuracy_diff_threshold,
            curve_tolerance,
        ),
    ) as pool:
        for done, (file_rows, accuracy_row) in enumerate(
            pool.map(worker_measure, [str(p) for p in files], chunksize=1), 1
        ):
            rows.extend(file_rows)
            if accuracy_row:
                accuracy_rows.append(accuracy_row)
            if progress.interactive:
                progress.advance(
                    f"measure {renderers[0]} {size}px",
                    display_file(files[done - 1], root),
                )
            elif should_report_progress(done, total, progress_every):
                print(f"   measured {done}/{total} files", flush=True)
    if owns_progress:
        progress.finish()
    elapsed = time.perf_counter() - t0
    energy_j = sampler.stop_j()
    total_ms = sum(r["measured_ms"] for r in rows if r["ok"])
    for row in rows:
        row["batch_elapsed_s"] = elapsed
        row["batch_energy_j"] = energy_j
        if row.get("energy_j") is not None:
            continue  # per-task energy measured in the worker (macOS)
        if energy_j is not None and total_ms > 0 and row["ok"]:
            row["energy_j"] = energy_j * row["measured_ms"] / total_ms
        else:
            row["energy_j"] = None
    return rows, accuracy_rows


_ACCURACY_RENDERERS: dict[str, Any] = {}
_ACCURACY_ROOT = DEFAULT_INPUT
_ACCURACY_SIZE = 64
_ACCURACY_FRAMES = 0
_ACCURACY_TOLERANCE = 8
_ACCURACY_DIFF_THRESHOLD = 1.0
_ACCURACY_INCLUDE_VULKAN = False
_ACCURACY_CURVE_TOLERANCE = 0.125


def init_accuracy_worker(
    root: str,
    size: int,
    frames: int,
    tolerance: int,
    diff_threshold: float,
    include_vulkan: bool,
    curve_tolerance: float,
) -> None:
    global _ACCURACY_RENDERERS, _ACCURACY_ROOT, _ACCURACY_SIZE, _ACCURACY_FRAMES, _ACCURACY_TOLERANCE, _ACCURACY_DIFF_THRESHOLD, _ACCURACY_INCLUDE_VULKAN, _ACCURACY_CURVE_TOLERANCE
    _ACCURACY_ROOT = Path(root)
    _ACCURACY_SIZE = size
    _ACCURACY_FRAMES = frames
    _ACCURACY_TOLERANCE = tolerance
    _ACCURACY_DIFF_THRESHOLD = diff_threshold
    _ACCURACY_INCLUDE_VULKAN = include_vulkan
    _ACCURACY_CURVE_TOLERANCE = curve_tolerance
    _ACCURACY_RENDERERS = {
        "tlottie": Tlottie(LIBS["tlottie"], curve_tolerance),
        "rlottie": Rlottie(LIBS["rlottie"]),
        "thorvg": Thorvg(LIBS["thorvg"]),
    }
    if include_vulkan:
        _ACCURACY_RENDERERS["tlottie-vulkan"] = TlottieVulkan(
            LIBS["tlottie-vulkan"], curve_tolerance
        )


def worker_accuracy(file_s: str) -> dict[str, Any]:
    file = Path(file_s)
    rendered: dict[str, Any] = {}
    counts: dict[str, int] = {}
    errors = []
    with ExitStack() as stack:
        for renderer in ("tlottie", "rlottie", "thorvg"):
            try:
                stream = stack.enter_context(
                    _ACCURACY_RENDERERS[renderer].frame_stream(file, _ACCURACY_SIZE)
                )
            except RuntimeError as error:
                errors.append(f"{renderer}:{error}")
                continue
            counts[renderer] = stream.count
            frame_count = (
                min(stream.count, _ACCURACY_FRAMES)
                if _ACCURACY_FRAMES > 0
                else stream.count
            )
            rendered[renderer] = FrameStream(frame_count, stream.render)

        if _ACCURACY_INCLUDE_VULKAN:
            ok, frames, count, err = _ACCURACY_RENDERERS["tlottie-vulkan"].render_frames_argb(
                file, _ACCURACY_SIZE, _ACCURACY_FRAMES
            )
            if not ok:
                errors.append(f"tlottie-vulkan:{err}")
            rendered["tlottie-vulkan"] = frames
            counts["tlottie-vulkan"] = count

        try:
            return make_accuracy_row(
                file,
                _ACCURACY_ROOT,
                _ACCURACY_SIZE,
                rendered,
                counts,
                errors,
                _ACCURACY_TOLERANCE,
                _ACCURACY_DIFF_THRESHOLD,
            )
        except RuntimeError as error:
            return make_accuracy_row(
                file,
                _ACCURACY_ROOT,
                _ACCURACY_SIZE,
                {},
                counts,
                errors + [f"stream:{error}"],
                _ACCURACY_TOLERANCE,
                _ACCURACY_DIFF_THRESHOLD,
            )


def make_accuracy_row(
    file: Path,
    root: Path,
    size: int,
    rendered: dict[str, list[Any]],
    counts: dict[str, int],
    errors: list[str],
    tolerance: int,
    diff_threshold: float,
) -> dict[str, Any]:
    row = {
        "pack": pack_of(root, file),
        "file": str(file.relative_to(root)),
        "size": size,
        "ok": False,
        "frames_tested": 0,
        "max_diff_percent": None,
        "avg_diff_percent": None,
        "worst_frame": None,
        "min_consensus_percent": None,
        "frame_counts": counts,
        "frame_count_note": "",
        "vulkan_frames_tested": 0,
        "vulkan_max_diff_percent": None,
        "vulkan_avg_diff_percent": None,
        "vulkan_max_changed_percent": None,
        "vulkan_mean_distance": None,
        "vulkan_max_channel_error": None,
        "vulkan_worst_frame": None,
        "vulkan_ok": None,
        "error": "; ".join(errors),
    }
    if errors:
        return row
    if len(set(counts.values())) != 1:
        row["frame_count_note"] = "mismatch:" + ",".join(
            f"{name}={counts[name]}" for name in ("tlottie", "rlottie", "thorvg")
        )
    available_frames = min(len(rendered.get(name, [])) for name in ("tlottie", "rlottie", "thorvg"))
    if available_frames <= 0:
        row["error"] = "no_frames"
        return row
    frame_count = available_frames

    diffs: list[float] = []
    consensus_percentages: list[float] = []
    total = size * size
    worst_frame = 0
    for frame in range(frame_count):
        candidate = rendered["tlottie"][frame]
        a = rendered["rlottie"][frame]
        b = rendered["thorvg"][frame]
        bad, consensus = diff_from_consensus(candidate, a, b, tolerance)
        if consensus == 0:
            row["error"] = f"missing_consensus@{frame}"
            return row
        diff_percent = 100.0 * bad / consensus
        diffs.append(diff_percent)
        consensus_percentages.append(100.0 * consensus / total)
        if diff_percent > diffs[worst_frame]:
            worst_frame = frame

    max_diff = max(diffs, default=100.0)
    row["frames_tested"] = len(diffs)
    row["max_diff_percent"] = max_diff
    row["avg_diff_percent"] = avg(diffs)
    row["worst_frame"] = worst_frame
    row["min_consensus_percent"] = min(consensus_percentages, default=None)
    row["ok"] = max_diff <= diff_threshold
    if not row["ok"]:
        row["error"] = f"diff>{diff_threshold:.3f}%@{worst_frame}"
    vulkan = rendered.get("tlottie-vulkan", [])
    cpu = rendered.get("tlottie", [])
    vulkan_frame_count = min(len(cpu), len(vulkan))
    if vulkan_frame_count:
        bad_percentages = []
        changed_percentages = []
        distance_sum = 0.0
        compared_pixels = 0
        max_channel_error = 0
        max_channel_error_frame = 0
        max_channel_error_pixel = 0
        max_channel_error_cpu = 0
        max_channel_error_vulkan = 0
        vulkan_worst_frame = 0
        for frame in range(vulkan_frame_count):
            distances = [px_distance(a, b) for a, b in zip(cpu[frame], vulkan[frame])]
            bad_percent = 100.0 * sum(distance > tolerance for distance in distances) / total
            changed_percent = 100.0 * sum(a != b for a, b in zip(cpu[frame], vulkan[frame])) / total
            bad_percentages.append(bad_percent)
            changed_percentages.append(changed_percent)
            distance_sum += sum(distances)
            compared_pixels += len(distances)
            frame_error, frame_error_pixel, frame_cpu, frame_vulkan = max(
                (
                    (px_channel_error(a, b), index, a, b)
                    for index, (a, b) in enumerate(zip(cpu[frame], vulkan[frame]))
                ),
                default=(0, 0, 0, 0),
            )
            if frame_error > max_channel_error:
                max_channel_error = frame_error
                max_channel_error_frame = frame
                max_channel_error_pixel = frame_error_pixel
                max_channel_error_cpu = frame_cpu
                max_channel_error_vulkan = frame_vulkan
            if bad_percent > bad_percentages[vulkan_worst_frame]:
                vulkan_worst_frame = frame
        row["vulkan_frames_tested"] = vulkan_frame_count
        row["vulkan_max_diff_percent"] = max(bad_percentages)
        row["vulkan_avg_diff_percent"] = avg(bad_percentages)
        row["vulkan_max_changed_percent"] = max(changed_percentages)
        row["vulkan_mean_distance"] = distance_sum / compared_pixels if compared_pixels else None
        row["vulkan_max_channel_error"] = max_channel_error
        row["vulkan_max_channel_error_frame"] = max_channel_error_frame
        row["vulkan_max_channel_error_x"] = max_channel_error_pixel % size
        row["vulkan_max_channel_error_y"] = max_channel_error_pixel // size
        row["vulkan_max_channel_error_cpu"] = f"0x{max_channel_error_cpu:08x}"
        row["vulkan_max_channel_error_vulkan"] = f"0x{max_channel_error_vulkan:08x}"
        row["vulkan_worst_frame"] = vulkan_worst_frame
        row["vulkan_ok"] = row["vulkan_max_diff_percent"] <= diff_threshold
    return row


def diff_from_consensus(
    candidate: Any, a: Any, b: Any, tolerance: int
) -> tuple[int, int]:
    if np is not None:
        return diff_from_consensus_numpy(candidate, a, b, tolerance)
    bad = 0
    consensus = 0
    for cp, ap, bp in zip(frame_pixels(candidate), frame_pixels(a), frame_pixels(b)):
        if not px_close(ap, bp, tolerance):
            continue
        consensus += 1
        if not px_close_to_avg(cp, ap, bp, tolerance):
            bad += 1
    return bad, consensus


def frame_pixels(frame: Any) -> Any:
    """Expose a compact raw ARGB frame as uint32 pixels for the Python fallback."""
    if isinstance(frame, (bytes, memoryview)):
        return memoryview(frame).cast("B").cast("I")
    return frame


def diff_from_consensus_numpy(
    candidate: Any, a: Any, b: Any, tolerance: int
) -> tuple[int, int]:
    candidate_channels = numpy_channels(candidate)
    a_channels = numpy_channels(a)
    b_channels = numpy_channels(b)
    consensus_mask = numpy_close_mask(a_channels, b_channels, tolerance)
    consensus = int(np.count_nonzero(consensus_mask))
    if consensus == 0:
        return 0, 0

    # This identity computes floor((a + b) / 2) without widening uint8.
    avg_channels = (a_channels & b_channels) + ((a_channels ^ b_channels) >> 1)
    candidate_close = numpy_close_mask(candidate_channels, avg_channels, tolerance)
    bad = int(np.count_nonzero(consensus_mask & ~candidate_close))
    return bad, consensus


def numpy_channels(frame: Any) -> Any:
    if isinstance(frame, (bytes, memoryview)):
        return np.frombuffer(frame, dtype=np.uint8).reshape(-1, 4)
    pixels = np.asarray(frame, dtype=np.uint32)
    return pixels.view(np.uint8).reshape(-1, 4)


def numpy_close_mask(a: Any, b: Any, tolerance: int) -> Any:
    # Native render buffers are little-endian ARGB words, hence byte columns BGRA.
    alpha_delta = np.maximum(a[:, 3], b[:, 3]) - np.minimum(a[:, 3], b[:, 3])
    red_delta = np.maximum(a[:, 2], b[:, 2]) - np.minimum(a[:, 2], b[:, 2])
    green_delta = np.maximum(a[:, 1], b[:, 1]) - np.minimum(a[:, 1], b[:, 1])
    blue_delta = np.maximum(a[:, 0], b[:, 0]) - np.minimum(a[:, 0], b[:, 0])
    rgb_delta = np.maximum(np.maximum(red_delta, green_delta), blue_delta)
    weighted_rgb_delta = np.multiply(
        rgb_delta, np.maximum(a[:, 3], b[:, 3]), dtype=np.uint16
    )
    return (alpha_delta <= tolerance) & (weighted_rgb_delta <= tolerance * 255)


def px_close(a: int, b: int, tolerance: int) -> bool:
    return px_distance(a, b) <= tolerance


def px_close_to_avg(candidate: int, a: int, b: int, tolerance: int) -> bool:
    return px_close(candidate, avg_px(a, b), tolerance)


def px_distance(a: int, b: int) -> float:
    aa, ar, ag, ab = channels(a)
    ba, br, bg, bb = channels(b)
    alpha_delta = abs(aa - ba)
    if aa == 0 and ba == 0:
        return float(alpha_delta)
    opacity = max(aa, ba) / 255.0
    rgb_delta = max(abs(ar - br), abs(ag - bg), abs(ab - bb)) * opacity
    return max(float(alpha_delta), rgb_delta)


def px_channel_error(a: int, b: int) -> int:
    return max(abs(ca - cb) for ca, cb in zip(channels(a), channels(b)))


def avg_px(a: int, b: int) -> int:
    return sum(((ca + cb) // 2) << shift for ca, cb, shift in zip(channels(a), channels(b), (24, 16, 8, 0)))


def channels(px: int) -> tuple[int, int, int, int]:
    return ((px >> 24) & 0xFF, (px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)


def run_accuracy(
    files: list[Path],
    root: Path,
    size: int,
    frames: int,
    tolerance: int,
    diff_threshold: float,
    jobs: int,
    include_vulkan: bool,
    curve_tolerance: float,
    progress: ProgressDisplay | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(files)
    progress_every = progress_interval(total)
    owns_progress = progress is None
    progress = progress or ProgressDisplay(f"accuracy {size}px", total)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=init_accuracy_worker,
        initargs=(
            str(root), size, frames, tolerance, diff_threshold,
            include_vulkan, curve_tolerance,
        ),
    ) as pool:
        for done, row in enumerate(
            pool.map(worker_accuracy, [str(p) for p in files], chunksize=1), 1
        ):
            rows.append(row)
            if progress.interactive:
                progress.advance(f"accuracy {size}px", display_file(files[done - 1], root))
            elif should_report_progress(done, total, progress_every):
                print(f"   accuracy {done}/{total} files", flush=True)
    if owns_progress:
        progress.finish()
    return rows


class ProgressDisplay:
    SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.completed = 0
        self.started = time.perf_counter()
        self.last_draw = 0.0
        self.interactive = sys.stdout.isatty()
        self.unicode = (sys.stdout.encoding or "").lower().startswith("utf")
        if self.interactive:
            self.update(0, "starting", force=True)

    def advance(self, label: str, status: str) -> None:
        self.label = label
        self.completed = min(self.total, self.completed + 1)
        self.update(self.completed, status)

    def update(self, done: int, status: str, force: bool = False) -> None:
        if not self.interactive:
            return
        now = time.perf_counter()
        if not force and done < self.total and now - self.last_draw < 0.05:
            return
        self.last_draw = now
        elapsed = max(0.0, now - self.started)
        ratio = done / self.total if self.total else 1.0
        rate = done / elapsed if elapsed > 0.0 else 0.0
        eta = (self.total - done) / rate if rate > 0.0 else None
        try:
            columns = os.get_terminal_size().columns
        except OSError:
            columns = 120
        bar_width = 12 if columns >= 80 else 8
        filled = min(bar_width, int(ratio * bar_width))
        if self.unicode:
            bar = "█" * filled + "░" * (bar_width - filled)
            spinner = self.SPINNER[done % len(self.SPINNER)]
            separator = "•"
        else:
            bar = "#" * filled + "-" * (bar_width - filled)
            spinner = "*"
            separator = "|"
        metrics = (
            f"{100.0 * ratio:5.1f}% {done}/{self.total} {separator} "
            f"{rate:4.1f}/s {separator} ETA {format_duration(eta) if eta is not None else '--:--'}"
        )
        if columns >= 140:
            metrics += f" {separator} {format_duration(elapsed)} elapsed"
        prefix = f" {spinner} [{bar}] {metrics} "
        phase_width = max(0, min(28, columns - len(prefix) - 12))
        phase_text = truncate_middle(self.label, phase_width)
        phase = f"{phase_text} " if phase_text else ""
        available = max(0, columns - len(prefix) - len(phase) - 1)
        short_status = truncate_middle(status, available)
        print(f"\r\033[2K{prefix}{phase}{short_status}", end="", flush=True)

    def finish(self) -> None:
        if not self.interactive:
            return
        self.update(self.total, "complete", force=True)
        print(flush=True)


def display_file(file: Path, root: Path) -> str:
    try:
        return str(file.relative_to(root))
    except ValueError:
        return file.name


def truncate_middle(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    left = (width - 1) // 2
    return value[:left] + "…" + value[-(width - left - 1) :]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def progress_interval(total: int) -> int:
    if total <= 20:
        return 1
    return max(1, min(250, total // 100))


def should_report_progress(done: int, total: int, every: int) -> bool:
    return done == total or done % every == 0


def aggregate_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["pack"], []).append(row)
    out = {}
    for pack, items in groups.items():
        good = sum(1 for r in items if r["ok"])
        vulkan_items = [r for r in items if r.get("vulkan_ok") is not None]
        vulkan_good = sum(1 for r in vulkan_items if r["vulkan_ok"])
        out[pack] = {
            "good": good,
            "total": len(items),
            "ratio": (good / len(items)) if items else None,
            "vulkan_good": vulkan_good,
            "vulkan_total": len(vulkan_items),
            "vulkan_ratio": (vulkan_good / len(vulkan_items)) if vulkan_items else None,
            "vulkan_max_diff_percent": max(
                (float(r["vulkan_max_diff_percent"]) for r in vulkan_items),
                default=None,
            ),
        }
    return out


def save_diff_grids(
    rows: list[dict[str, Any]],
    root: Path,
    out_dir: Path,
    limit: int,
    size: int,
    tolerance: int,
    curve_tolerance: float,
) -> list[Path]:
    selected = select_diff_rows(rows, limit)
    if not selected:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_diff_dir(out_dir)
    renderers = {
        "tlottie": Tlottie(LIBS["tlottie"], curve_tolerance),
        "rlottie": Rlottie(LIBS["rlottie"]),
        "thorvg": Thorvg(LIBS["thorvg"]),
    }
    written: list[Path] = []
    used_names: set[str] = set()
    for row in selected:
        rel = Path(row["file"])
        file = root / rel
        frame = int(row.get("worst_frame") or 0)
        images: dict[str, list[int]] = {}
        errors = []
        for name, renderer in renderers.items():
            ok, pixels, err = renderer.render_argb(file, size, frame)
            if not ok:
                errors.append(f"{name}:{err}")
            images[name] = pixels
        if errors:
            print(f"   skipped diff grid for {rel}: {'; '.join(errors)}", flush=True)
            continue
        grid = make_diff_grid(
            images["tlottie"],
            images["rlottie"],
            images["thorvg"],
            size,
            tolerance,
        )
        base = (
            f"{sanitize_name(rel.stem)}"
            f"__frame{frame}__diff{float(row.get('max_diff_percent') or 0.0):.2f}.png"
        )
        name = unique_name(base, used_names)
        path = out_dir / name
        write_png_rgb(path, 3 * size, 2 * size, grid)
        written.append(path)
    return written


def select_diff_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    by_pack: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("ok") or row.get("max_diff_percent") is None:
            continue
        by_pack.setdefault(row["pack"], []).append(row)
    for items in by_pack.values():
        items.sort(key=lambda r: float(r.get("max_diff_percent") or 0.0), reverse=True)
    packs = sorted(
        by_pack,
        key=lambda p: float(by_pack[p][0].get("max_diff_percent") or 0.0),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for pack in packs:
            if index < len(by_pack[pack]):
                selected.append(by_pack[pack][index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def clear_diff_dir(out_dir: Path) -> None:
    for path in out_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def make_diff_grid(
    tlottie: list[int], rlottie: list[int], thorvg: list[int], size: int, tolerance: int
) -> bytes:
    width = 3 * size
    height = 2 * size
    out = bytearray(width * height * 3)
    paste_image(out, width, tlottie, size, 0, 0)
    paste_image(out, width, rlottie, size, size, 0)
    paste_image(out, width, thorvg, size, size * 2, 0)
    paste_diff(out, width, tlottie, rlottie, size, 0, size, tolerance)
    paste_diff(out, width, rlottie, thorvg, size, size, size, tolerance)
    paste_diff(out, width, tlottie, thorvg, size, size * 2, size, tolerance)
    return bytes(out)


def paste_image(dst: bytearray, dst_width: int, pixels: list[int], size: int, x0: int, y0: int) -> None:
    for y in range(size):
        for x in range(size):
            r, g, b = rgb_from_argb(pixels[y * size + x])
            write_rgb(dst, dst_width, x0 + x, y0 + y, r, g, b)


def paste_diff(
    dst: bytearray,
    dst_width: int,
    a: list[int],
    b: list[int],
    size: int,
    x0: int,
    y0: int,
    tolerance: int,
) -> None:
    for y in range(size):
        for x in range(size):
            ap = a[y * size + x]
            bp = b[y * size + x]
            if px_close(ap, bp, tolerance):
                write_rgb(dst, dst_width, x0 + x, y0 + y, 0, 0, 0)
            else:
                delta = px_distance(ap, bp)
                intensity = max(80, min(255, delta * 4))
                write_rgb(dst, dst_width, x0 + x, y0 + y, int(intensity), 32, 32)


def rgb_from_argb(px: int) -> tuple[int, int, int]:
    return ((px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)


def write_rgb(dst: bytearray, width: int, x: int, y: int, r: int, g: int, b: int) -> None:
    i = (y * width + x) * 3
    dst[i] = r
    dst[i + 1] = g
    dst[i + 2] = b


def write_png_rgb(path: Path, width: int, height: int, rgb: bytes) -> None:
    rows = [b"\x00" + rgb[y * width * 3 : (y + 1) * width * 3] for y in range(height)]
    raw = b"".join(rows)
    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        write_png_chunk(f, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        write_png_chunk(f, b"IDAT", zlib.compress(raw, 6))
        write_png_chunk(f, b"IEND", b"")


def write_png_chunk(f: Any, kind: bytes, data: bytes) -> None:
    f.write(struct.pack(">I", len(data)))
    f.write(kind)
    f.write(data)
    f.write(struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 2
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def aggregate_file_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["pack"], row["file"], row["size"], row["renderer"]), []).append(row)
    out = []
    for (pack, file, size, renderer), items in sorted(groups.items()):
        ok = [r for r in items if r["ok"]]
        out.append(
            {
                "pack": pack,
                "file": file,
                "size": size,
                "renderer": renderer,
                "samples": len(items),
                "ok": len(ok),
                "first_frame_ms": avg([r["first_frame_ms"] for r in ok]),
                "frame_ms": avg_optional([r["frame_ms"] for r in ok]),
                "other_frames": sum(r["other_frames"] for r in ok),
                "measured_ms": sum(r["measured_ms"] for r in ok),
                "memory_avg_mb": avg([r["memory_avg_mb"] for r in ok]),
                "memory_max_mb": max([r["memory_max_mb"] for r in ok], default=0.0),
                "energy_j": sum((r["energy_j"] or 0.0) for r in ok)
                if any(r["energy_j"] is not None for r in ok)
                else None,
                "error": "; ".join(sorted({r["error"] for r in items if r["error"]})),
            }
        )
    return out


def aggregate_pack_rows(file_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in file_rows:
        groups.setdefault((row["pack"], row["size"], row["renderer"]), []).append(row)
    out = []
    for (pack, size, renderer), items in sorted(groups.items()):
        ok = [r for r in items if r["ok"] > 0]
        out.append(
            {
                "pack": pack,
                "size": size,
                "renderer": renderer,
                "files": len(items),
                "ok": sum(1 for r in items if r["ok"] > 0),
                "first_frame_ms": avg([r["first_frame_ms"] for r in ok]),
                "frame_ms": avg_optional([r["frame_ms"] for r in ok]),
                "other_frames": sum(r["other_frames"] for r in ok),
                "measured_ms": sum(r["measured_ms"] for r in ok),
                "memory_avg_mb": avg([r["memory_avg_mb"] for r in ok]),
                "memory_max_mb": max([r["memory_max_mb"] for r in ok], default=0.0),
                "energy_j": sum((r["energy_j"] or 0.0) for r in ok)
                if any(r["energy_j"] is not None for r in ok)
                else None,
            }
        )
    return out


def pivot_aggregate(rows: list[dict[str, Any]], key_cols: tuple[str, ...]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[c] for c in key_cols)
        out = by_key.setdefault(key, {c: row[c] for c in key_cols})
        r = row["renderer"]
        out[f"{r}_ok"] = row["ok"]
        out[f"{r}_samples"] = row.get("samples", row.get("files", 0))
        out[f"{r}_files"] = row.get("files")
        out[f"{r}_first_frame_ms"] = row["first_frame_ms"] if row["ok"] else None
        out[f"{r}_frame_ms"] = row["frame_ms"] if row["ok"] else None
        out[f"{r}_other_frames"] = row.get("other_frames")
        out[f"{r}_measured_ms"] = row.get("measured_ms")
        out[f"{r}_memory_avg_mb"] = row["memory_avg_mb"] if row["ok"] else None
        out[f"{r}_memory_max_mb"] = row["memory_max_mb"] if row["ok"] else None
        out[f"{r}_rss_avg_mb"] = row["memory_avg_mb"] if row["ok"] else None
        out[f"{r}_rss_max_mb"] = row["memory_max_mb"] if row["ok"] else None
        out[f"{r}_energy_j"] = row["energy_j"] if row["ok"] else None
        out[f"{r}_error"] = row.get("error", "")
    return [by_key[k] for k in sorted(by_key)]


def write_tgv(path: Path, rows: list[dict[str, Any]], renderers: tuple[str, ...], key_cols: tuple[str, ...]) -> None:
    cols = list(key_cols)
    comparison_col = "tl vs rl19"
    include_comparison = has_tl_rl19_comparison(renderers)
    if include_comparison:
        cols.insert(cols.index("pack") + 1, comparison_col)
    for r in renderers:
        cols += [
            f"{r}_first_frame_ms",
            f"{r}_frame_ms",
            f"{r}_memory_avg_mb",
            f"{r}_memory_max_mb",
            f"{r}_energy_j",
            f"{r}_error",
        ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            values = {
                **row,
                comparison_col: format_tl_vs_rl19_percent(row) if include_comparison else None,
            }
            f.write("\t".join(format_cell(values.get(c)) for c in cols) + "\n")


def has_tl_rl19_comparison(renderers: tuple[str, ...]) -> bool:
    return "tlottie" in renderers and "rlottie_2019" in renderers


def tl_vs_rl19_percent(row: dict[str, Any]) -> float | None:
    tlottie_ms = row.get("tlottie_frame_ms")
    rlottie_2019_ms = row.get("rlottie_2019_frame_ms")
    if tlottie_ms is None or rlottie_2019_ms is None or rlottie_2019_ms <= 0:
        return None
    percent = (tlottie_ms / rlottie_2019_ms - 1.0) * 100.0
    return percent if math.isfinite(percent) else None


def format_tl_vs_rl19_percent(row: dict[str, Any]) -> str:
    percent = tl_vs_rl19_percent(row)
    return "n/a" if percent is None else f"{percent:+.1f}%"


def tl_vs_rl19_cell(row: dict[str, Any]) -> str:
    percent = tl_vs_rl19_percent(row)
    if percent is None:
        return "<td class='comparison muted'>n/a</td>"
    direction = "faster" if percent < 0 else "slower" if percent > 0 else "equal"
    fill = min(abs(percent), 100.0)
    return (
        f"<td class='comparison {direction}' style='--fill:{fill:.1f}%'>"
        f"{percent:+.1f}%</td>"
    )


def format_cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.6f}"
    return str(v).replace("\t", " ")


def metric_class(row: dict[str, Any], renderer: str, metric: str, renderers: tuple[str, ...]) -> str:
    values = [
        row.get(f"{r}_{metric}") for r in renderers if row.get(f"{r}_{metric}") is not None
    ]
    value = row.get(f"{renderer}_{metric}")
    if value is None or not values:
        return ""
    winner = min(values)
    if value == winner:
        return "winner"
    if winner > 0 and value >= winner * 2.0:
        return "loser"
    return ""


def write_html(
    path: Path,
    pack_rows: list[dict[str, Any]],
    file_rows: list[dict[str, Any]],
    renderers: tuple[str, ...],
    energy_available: bool,
    reps: int,
    accuracy_by_pack: dict[str, dict[str, Any]] | None,
    accuracy_size: int,
    accuracy_tolerance: int,
    accuracy_diff_threshold: float,
    benchmark_command: str | None = None,
    machine_details: str | None = None,
) -> None:
    benchmark_command = benchmark_command or "python3 tools/benchmark.py"
    machine_details = machine_details or current_machine_details()
    css = """
:root{--bg:#f4f6f7;--surface:#fff;--ink:#1d2429;--muted:#5c6b76;--line:#dde4e8;
--tl:#0d7f8c;--good:#1e7d46;--bad:#bb4436;--head:#f0f3f5;
--goodsoft:#e0f2e7;--warnsoft:#f5ecd2;--warn:#8a6d1a;--badsoft:#f7e2df}
@media(prefers-color-scheme:dark){:root{--bg:#14181b;--surface:#1b2126;--ink:#dde5ea;
--muted:#8b99a3;--line:#2c353c;--tl:#46becb;--good:#4cba7a;--bad:#e0705f;--head:#20272d;
--goodsoft:#163f27;--warnsoft:#4a3d17;--warn:#ffe7a3;--badsoft:#4a2320}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font:14px/1.5 system-ui,sans-serif;
padding:28px 18px 60px}
main{width:min-content;max-width:100%;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 8px}
a{color:var(--tl)}
.note{color:var(--muted);font-size:13px;margin:2px 0;}
.tablewrap{overflow-x:auto;max-width:100%;width:fit-content;background:var(--surface);
border:1px solid var(--line);border-radius:6px;margin:0 0 8px}
table{border-collapse:collapse;width:auto;font-size:12px}
th{background:var(--head);color:var(--muted);font-weight:600;text-align:right;padding:6px 8px;
white-space:nowrap;border-bottom:1px solid var(--line);font-size:10.5px;
position:sticky;top:0;z-index:1}
th.left{text-align:left}
th.renderer{text-align:center;text-transform:none;font-size:11.5px;color:var(--ink)}
td{padding:4.5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;
font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;font-size:11.5px}
td.left{font-family:system-ui;font-weight:550;text-align:left}
th.metric,td.metric{border-left:1px solid var(--line)}
th.metric-last,td.metric-last{border-right:1px solid var(--line)}
tr:last-child td{border-bottom:0}
.winner{color:var(--tl);font-weight:700}
.loser{color:var(--bad);font-weight:700}
.comparison{font-weight:700;min-width:74px}
.comparison.faster{color:var(--good);background:linear-gradient(to right,
var(--goodsoft) var(--fill),transparent var(--fill))}
.comparison.slower{color:var(--bad);background:linear-gradient(to left,
var(--badsoft) var(--fill),transparent var(--fill))}
.comparison.equal{color:var(--muted)}
.acc-badge{margin-left:8px;border-radius:4px;padding:1px 6px;font-weight:600;font-size:10.5px;
font-family:system-ui}
.acc-ok{background:var(--goodsoft);color:var(--good)}
.acc-warn{background:var(--warnsoft);color:var(--warn)}
.acc-bad{background:var(--badsoft);color:var(--bad)}
.muted{color:var(--muted)}
"""
    with path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Lottie Benchmark</title>")
        f.write(f"<style>{css}</style>")
        f.write("<main>")
        f.write(f"<p class='note'><code>{esc(benchmark_command)}</code></p>")
        f.write(f"<p class='note'>{esc(machine_details)}</p>")
        for size in sorted({r["size"] for r in pack_rows}):
            f.write(f"<h2>{size}px</h2>")
            rows = pivot_aggregate([r for r in pack_rows if r["size"] == size], ("pack", "size"))
            write_grouped_table(f, rows, renderers, ("pack",), include_size=False, accuracy_by_pack=accuracy_by_pack)
        effect_file_rows = [
            r
            for r in file_rows
            if r["size"] == 720 and len(Path(r["file"]).parts) > 1 and Path(r["file"]).parts[1] == "effects"
        ]
        if effect_file_rows:
            f.write("<h2>720px effects</h2>")
            effect_pack_rows = aggregate_pack_rows(effect_file_rows)
            rows = pivot_aggregate(effect_pack_rows, ("pack", "size"))
            write_grouped_table(f, rows, renderers, ("pack",), include_size=False, accuracy_by_pack=accuracy_by_pack)
        f.write("</main>")


def write_grouped_table(
    f: Any,
    rows: list[dict[str, Any]],
    renderers: tuple[str, ...],
    labels: tuple[str, ...],
    include_size: bool,
    accuracy_by_pack: dict[str, dict[str, Any]] | None = None,
) -> None:
    include_comparison = has_tl_rl19_comparison(renderers)
    f.write("<div class='tablewrap'><table><tr>")
    for label in labels:
        f.write(f"<th rowspan='2' class='left'>{esc(label)}</th>")
        if label == "pack" and include_comparison:
            f.write("<th rowspan='2'>tl vs rl19</th>")
    if include_size:
        f.write("<th rowspan='2'>size</th>")
    for r in renderers:
        url = RENDERER_URLS.get(r)
        name = f"<a href='{esc(url)}'>{esc(r)}</a>" if url else esc(r)
        f.write(f"<th colspan='4' class='renderer'>{name}</th>")
    f.write("</tr><tr>")
    for _ in renderers:
        f.write(
            "<th class='metric'>fms</th><th>ms</th>"
            f"<th>MiB (avg/max)</th>"
            "<th class='metric-last'>J</th>"
        )
    f.write("</tr>")
    for row in rows:
        f.write("<tr>")
        for label in labels:
            value = row[label]
            if label == "pack":
                accuracy = accuracy_by_pack.get(value) if accuracy_by_pack else None
                value = pack_label(value, accuracy)
                f.write(f"<td class='left'>{value}</td>")
            else:
                f.write(f"<td class='left'>{esc(value)}</td>")
            if label == "pack" and include_comparison:
                f.write(tl_vs_rl19_cell(row))
        if include_size:
            f.write(f"<td>{row['size']}</td>")
        for r in renderers:
            err = row.get(f"{r}_error")
            if err:
                f.write(f"<td colspan='4' class='loser left'>{esc(err)}</td>")
                continue
            first_cls = metric_class(row, r, "first_frame_ms", renderers)
            frame_cls = metric_class(row, r, "frame_ms", renderers)
            mem_cls = metric_class(row, r, "memory_avg_mb", renderers)
            energy_cls = metric_class(row, r, "energy_j", renderers)
            f.write(
                f"<td class='metric {first_cls}'>{num(row.get(f'{r}_first_frame_ms'))}</td>"
            )
            f.write(f"<td class='{frame_cls}'>{num(row.get(f'{r}_frame_ms'))}</td>")
            f.write(
                f"<td class='{mem_cls}'>{num(row.get(f'{r}_memory_avg_mb'))} / "
                f"{num(row.get(f'{r}_memory_max_mb'))}</td>"
            )
            f.write(f"<td class='metric-last {energy_cls}'>{num(row.get(f'{r}_energy_j'))}</td>")
        f.write("</tr>")
    f.write("</table></div>")


def pack_label(pack: str, accuracy: dict[str, Any] | None) -> str:
    linked_pack = (
        f"<a href='https://t.me/addstickers/{quote(pack, safe='')}'>{esc(pack)}</a>"
    )
    if not accuracy:
        return linked_pack
    good = int(accuracy.get("good", 0))
    total = int(accuracy.get("total", 0))
    ratio = accuracy.get("ratio")
    if not total or ratio is None:
        return linked_pack
    cls = "acc-ok"
    if ratio < 0.5:
        cls = "acc-bad"
    elif good < total:
        cls = "acc-warn"
    label = (
        f"{linked_pack} "
        f"<span class='acc-badge {cls}'>{good}/{total}</span>"
    )
    vulkan_total = int(accuracy.get("vulkan_total", 0))
    vulkan_ratio = accuracy.get("vulkan_ratio")
    if vulkan_total and vulkan_ratio is not None:
        vulkan_good = int(accuracy.get("vulkan_good", 0))
        vulkan_cls = "acc-ok" if vulkan_good == vulkan_total else "acc-warn"
        if vulkan_ratio < 0.5:
            vulkan_cls = "acc-bad"
        vulkan_max = float(accuracy.get("vulkan_max_diff_percent") or 0.0)
        label += (
            f" <span class='acc-badge {vulkan_cls}'>"
            f"VK {vulkan_good}/{vulkan_total}, max {vulkan_max:.2f}%</span>"
        )
    return label


def current_machine_details() -> str:
    cpu = platform.processor().strip()
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        cpu = result.stdout.strip() or cpu
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except (OSError, IndexError):
            pass
    cpu = cpu or "unknown"
    return (
        f"CPU {cpu}; OS {platform.system()} {platform.release()}; "
        f"arch {platform.machine()}"
    )


def benchmark_invocation(args: argparse.Namespace, accuracy_size: int) -> str:
    command = [
        "tools/benchmark.py",
        "--sizes",
        args.sizes,
        "--frames",
        str(args.frames),
        "--reps",
        str(args.reps),
        "--jobs",
        str(args.jobs),
        "--curve-tolerance",
        str(args.curve_tolerance),
    ]
    if not args.no_accuracy:
        command.extend([
            "--accuracy-size",
            str(accuracy_size),
            "--accuracy-tolerance",
            str(args.accuracy_tolerance),
            "--accuracy-diff-threshold",
            str(args.accuracy_diff_threshold),
        ])
    for option, value in (
        ("--limit", args.limit),
        ("--packs", args.packs),
    ):
        if value is not None:
            command.extend((option, str(value)))
    for option, enabled in (
        ("--no-accuracy", args.no_accuracy),
    ):
        if enabled:
            command.append(option)
    return shlex.join(command)


def esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def num(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="frames to render per animation per rep; 0 renders every frame",
    )
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--curve-tolerance",
        type=float,
        help=(
            "tlottie maximum device-space curve-flattening error in pixels; "
            "defaults to 0.125 with accuracy enabled and 0.3 with --no-accuracy"
        ),
    )
    ap.add_argument("--accuracy-size", type=int)
    ap.add_argument(
        "--accuracy-tolerance",
        type=int,
        default=16,
        help="max opacity-weighted ARGB distance for pixels to be considered equal",
    )
    ap.add_argument(
        "--accuracy-diff-threshold",
        type=float,
        default=1.0,
        help="max percent of consensus pixels that may differ on any frame",
    )
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument(
        "--save-diffs",
        type=int,
        default=0,
        help="save up to N worst failing diff PNG grids, balanced across packs",
    )
    ap.add_argument("--diff-dir", type=Path, help="directory for --save-diffs PNG grids")
    ap.add_argument("--write-raw", action="store_true", help="write benchmark raw JSON files")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--packs",
        help=(
            "select packs by exact name, first N, inclusive 1-based range START,END, "
            "or last N with -N (packs are sorted by name)"
        ),
    )
    ap.add_argument("--renderers", default=",".join(DEFAULT_RENDERERS))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open benchmark.html")
    args = ap.parse_args()

    if args.curve_tolerance is None:
        args.curve_tolerance = 0.125 if not args.no_accuracy else 0.3

    renderers = tuple(r.strip() for r in args.renderers.split(",") if r.strip())
    bad = [r for r in renderers if r not in RENDERERS]
    if bad:
        raise SystemExit(f"unknown renderers: {', '.join(bad)}")
    sizes = tuple(int(s) for s in args.sizes.split(",") if s)
    if args.reps <= 0:
        raise SystemExit("--reps must be positive")
    if args.frames < 0:
        raise SystemExit("--frames must be non-negative")
    if not math.isfinite(args.curve_tolerance) or args.curve_tolerance <= 0:
        raise SystemExit("--curve-tolerance must be a positive finite number")
    if args.accuracy_tolerance < 0:
        raise SystemExit("--accuracy-tolerance must be non-negative")
    if args.accuracy_diff_threshold < 0:
        raise SystemExit("--accuracy-diff-threshold must be non-negative")
    if args.save_diffs < 0:
        raise SystemExit("--save-diffs must be non-negative")
    if args.no_accuracy and args.save_diffs:
        raise SystemExit("--save-diffs requires accuracy; remove --no-accuracy")
    ensure_builds(args.skip_build)
    for r in renderers:
        if not LIBS[r].exists():
            raise SystemExit(f"missing {r} library: {LIBS[r]}")

    files = discover(args.input, args.limit)
    if args.packs is not None:
        all_packs = sorted({pack_of(args.input, file) for file in files})
        selected_packs = select_packs(all_packs, args.packs)
        keep = set(selected_packs)
        files = [f for f in files if pack_of(args.input, f) in keep]
        print(
            f"== packs limited to {len(keep)}/{len(all_packs)}: "
            + ", ".join(selected_packs),
            flush=True,
        )
    if not files:
        raise SystemExit(f"no .json files found under {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"== tlottie curve tolerance {args.curve_tolerance:g}px", flush=True)

    all_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    accuracy_by_pack: dict[str, dict[str, Any]] | None = None
    accuracy_size = args.accuracy_size or max(sizes)
    accuracy_frame_label = "all frames" if args.frames == 0 else f"first {args.frames} frame(s)"
    accuracy_renderers = ("tlottie", "rlottie", "thorvg")
    if not args.no_accuracy:
        for required in accuracy_renderers:
            if not LIBS[required].exists():
                raise SystemExit(f"missing {required} library for accuracy: {LIBS[required]}")
        print(
            f"== accuracy {accuracy_size}px {accuracy_frame_label}: {len(files)} files, "
            f"pixel tolerance {args.accuracy_tolerance}, "
            f"broken if diff > {args.accuracy_diff_threshold:g}% (separate pass)",
            flush=True,
        )

    energy_available = EnergySampler().available() or task_energy_nj() is not None
    if not energy_available:
        print("== energy counters unavailable; J columns will be n/a", flush=True)
    phase_count = len(sizes) * len(renderers) + (0 if args.no_accuracy else 1)
    overall_progress = ProgressDisplay("benchmark", len(files) * phase_count)
    for size in sizes:
        if not overall_progress.interactive:
            print(
                f"== {size}px: {len(files)} files, {args.jobs} workers, "
                f"{args.reps} reps, isolated renderers={','.join(renderers)}",
                flush=True,
            )
        for renderer in renderers:
            if not overall_progress.interactive:
                print(f"   renderer={renderer}", flush=True)
            size_rows, _ = run_size_batch(
                (renderer,),
                size,
                files,
                args.input,
                args.frames,
                args.jobs,
                args.reps,
                False,
                accuracy_size,
                args.accuracy_tolerance,
                args.accuracy_diff_threshold,
                args.curve_tolerance,
                overall_progress,
            )
            all_rows.extend(size_rows)

    if not args.no_accuracy:
        if not accuracy_rows:
            # Accuracy needs several renderers' frames at once, so keep it out
            # of the performance workers whose memory must remain isolated.
            if not overall_progress.interactive:
                print("== accuracy: rendering separate accuracy pass", flush=True)
            accuracy_rows = run_accuracy(
                files,
                args.input,
                accuracy_size,
                args.frames,
                args.accuracy_tolerance,
                args.accuracy_diff_threshold,
                args.jobs,
                "tlottie-vulkan" in renderers,
                args.curve_tolerance,
                overall_progress,
            )
        accuracy_by_pack = aggregate_accuracy(accuracy_rows)
    overall_progress.finish()
    if not args.no_accuracy:
        if args.save_diffs:
            diff_dir = args.diff_dir or (args.out / "diffs")
            print(
                f"== writing up to {args.save_diffs} diff grid(s) to {diff_dir}",
                flush=True,
            )
            diff_paths = save_diff_grids(
                accuracy_rows,
                args.input,
                diff_dir,
                args.save_diffs,
                accuracy_size,
                args.accuracy_tolerance,
                args.curve_tolerance,
            )
            print(f"wrote {len(diff_paths)} diff grid(s)", flush=True)

    file_rows = aggregate_file_rows(all_rows)
    pack_rows = aggregate_pack_rows(file_rows)
    pack_pivot = pivot_aggregate(pack_rows, ("pack", "size"))
    tgv = args.out / "benchmark.tgv"
    html_path = args.out / "benchmark.html"
    write_tgv(tgv, pack_pivot, renderers, ("pack", "size"))
    write_html(
        html_path,
        pack_rows,
        file_rows,
        renderers,
        energy_available,
        args.reps,
        accuracy_by_pack,
        accuracy_size,
        args.accuracy_tolerance,
        args.accuracy_diff_threshold,
        benchmark_invocation(args, accuracy_size),
        current_machine_details(),
    )
    raw = args.out / "benchmark.raw.json"
    accuracy_raw = args.out / "benchmark-accuracy.raw.json"
    if args.write_raw:
        raw.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        if accuracy_rows:
            accuracy_raw.write_text(json.dumps(accuracy_rows, indent=2), encoding="utf-8")
    print(f"wrote {tgv}")
    print(f"wrote {html_path}")
    if args.write_raw:
        print(f"wrote {raw}")
        if accuracy_rows:
            print(f"wrote {accuracy_raw}")
    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
