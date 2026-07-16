#!/usr/bin/env python3
"""Benchmark tlottie, rlottie variants, and ThorVG via native libraries.

The runner processes one renderer+canvas-size batch at a time so package energy
counters, when available, can be attributed to that batch. Within each batch it
uses many worker processes, and each worker loads the native library once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes as C
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import time
from typing import Any
import webbrowser
import zlib


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
DEFAULT_INPUT = Path.home() / "Documents" / "fixtures-full"
DEFAULT_OUT = ROOT / "target" / "benchmark"
DEFAULT_SIZES = (64, 320, 720)
RENDERERS = ("tlottie", "rlottie", "rlottie_2019", "rlottie_2019_patched", "thorvg")
RLOTTIE_RENDERERS = ("rlottie", "rlottie_2019", "rlottie_2019_patched")
PROJECT_DIRS = {
    "rlottie": PROJECTS / "rlottie",
    "rlottie_2019": PROJECTS / "rlottie_2019",
    "rlottie_2019_patched": PROJECTS / "rlottie_2019_patched",
    "thorvg": PROJECTS / "thorvg",
}

LIBS = {
    "tlottie": ROOT / "target" / "release" / "libtlottie_capi.so",
    "rlottie": PROJECT_DIRS["rlottie"] / "build-release" / "src" / "librlottie.so",
    "rlottie_2019": PROJECT_DIRS["rlottie_2019"] / "build-release" / "src" / "librlottie.so",
    "rlottie_2019_patched": PROJECT_DIRS["rlottie_2019_patched"]
    / "build-release"
    / "src"
    / "librlottie.so",
    "thorvg": PROJECT_DIRS["thorvg"] / "build-release" / "src" / "libthorvg-1.so",
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
    run(["cargo", "build", "-p", "tlottie-capi", "--release"], ROOT, env)

    meson = shutil_which("meson")
    if not meson:
        meson = "/tmp/tlottie-build-tools/bin/meson"
    if not Path(meson).exists() and "/" in meson:
        raise SystemExit("meson not found; install meson/ninja or use --skip-build")

    meson_env = os.environ.copy()
    meson_env["PATH"] = f"/tmp/tlottie-build-tools/bin:{meson_env.get('PATH', '')}"
    def setup_cmd(project: Path) -> list[str]:
        cmd = [meson, "setup"]
        if (project / "build-release").exists():
            cmd.append("--wipe")
        return cmd + [
            "build-release",
            ".",
            "-Dbuildtype=release",
            "-Db_lto=true",
            "-Dcpp_args=-march=native",
            "-Dc_args=-march=native",
        ]

    run(
        setup_cmd(PROJECTS / "thorvg")
        + [
            "-Dengines=cpu",
            "-Dloaders=lottie",
            "-Dbindings=capi",
            "-Dsimd=true",
            "-Dtools=",
            "-Dtests=false",
        ],
        PROJECT_DIRS["thorvg"],
        meson_env,
    )
    run([meson, "compile", "-C", "build-release"], PROJECT_DIRS["thorvg"], meson_env)

    run(
        setup_cmd(PROJECTS / "rlottie") + ["-Dexample=false", "-Dtest=false"],
        PROJECT_DIRS["rlottie"],
        meson_env,
    )
    run([meson, "compile", "-C", "build-release"], PROJECT_DIRS["rlottie"], meson_env)

    for project_name in ("rlottie_2019", "rlottie_2019_patched"):
        project = PROJECT_DIRS[project_name]
        run(
            setup_cmd(project)
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


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


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


class Tlottie:
    def __init__(self, path: Path) -> None:
        self.lib = C.CDLL(str(path))
        self.lib.tlottie_animation_new.argtypes = [C.c_void_p, C.c_size_t]
        self.lib.tlottie_animation_new.restype = C.c_void_p
        self.lib.tlottie_animation_drop.argtypes = [C.c_void_p]
        self.lib.tlottie_animation_frame_count.argtypes = [C.c_void_p]
        self.lib.tlottie_animation_frame_count.restype = C.c_uint32
        self.lib.tlottie_animation_render_argb.argtypes = [
            C.c_void_p,
            C.c_float,
            C.c_uint32,
            C.c_uint32,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_uint32,
        ]
        self.lib.tlottie_animation_render_argb.restype = C.c_int

    def measure(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str]:
        t0 = time.perf_counter_ns()
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_animation_new(buf, len(data))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_animation_frame_count(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            rc = self.lib.tlottie_animation_render_argb(
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
                    self.lib.tlottie_animation_drop(anim)
                    anim = self.lib.tlottie_animation_new(buf, len(data))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse"
                rc = self.lib.tlottie_animation_render_argb(
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
                self.lib.tlottie_animation_drop(anim)

    def render_argb(self, file: Path, size: int, frame: int) -> tuple[bool, list[int], str]:
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_animation_new(buf, len(data))
        if not anim:
            return False, [], "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_animation_frame_count(anim)))
            rc = self.lib.tlottie_animation_render_argb(
                anim, float(frame % count), size, size, pixels, size * size, 1
            )
            if rc != 0:
                return False, [], f"render:{rc}"
            return True, list(pixels), ""
        finally:
            if anim:
                self.lib.tlottie_animation_drop(anim)

    def render_frames_argb(self, file: Path, size: int) -> tuple[bool, list[list[int]], int, str]:
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_animation_new(buf, len(data))
        if not anim:
            return False, [], 0, "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_animation_frame_count(anim)))
            frames = []
            for frame in range(count):
                rc = self.lib.tlottie_animation_render_argb(
                    anim, float(frame), size, size, pixels, size * size, 1
                )
                if rc != 0:
                    return False, [], count, f"render:{rc}@{frame}"
                frames.append(list(pixels))
            return True, frames, count, ""
        finally:
            self.lib.tlottie_animation_drop(anim)

    def measure_frames_argb(
        self, file: Path, size: int, frames: int
    ) -> tuple[bool, float, float | None, int, float, float, str, list[list[int]], int]:
        t0 = time.perf_counter_ns()
        data = file.read_bytes()
        buf = C.create_string_buffer(data)
        anim = self.lib.tlottie_animation_new(buf, len(data))
        if not anim:
            return False, 0.0, None, 0, rss_mb(), rss_mb(), "parse", [], 0
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.tlottie_animation_frame_count(anim)))
            frames = count if frames <= 0 else frames
            rss_samples: list[float] = []
            out_frames: list[list[int]] = []
            rc = self.lib.tlottie_animation_render_argb(
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
                    self.lib.tlottie_animation_drop(anim)
                    anim = self.lib.tlottie_animation_new(buf, len(data))
                    if not anim:
                        render_ns += time.perf_counter_ns() - t1
                        return False, first_ms, None, i - 1, rss_mb(), rss_mb(), "parse", [], count
                rc = self.lib.tlottie_animation_render_argb(
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
            self.lib.tlottie_animation_drop(anim)


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

    def render_frames_argb(self, file: Path, size: int) -> tuple[bool, list[list[int]], int, str]:
        anim = self.lib.lottie_animation_from_file(os.fsencode(file))
        if not anim:
            return False, [], 0, "parse"
        pixels = (C.c_uint32 * (size * size))()
        try:
            count = max(1, int(self.lib.lottie_animation_get_totalframe(anim)))
            frames = []
            for frame in range(count):
                self.lib.lottie_animation_render(anim, frame, pixels, size, size, size * 4)
                frames.append(list(pixels))
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

    def render_frames_argb(self, file: Path, size: int) -> tuple[bool, list[list[int]], int, str]:
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
            for frame in range(count):
                self.lib.tvg_animation_set_frame(anim, float(frame))
                self.lib.tvg_canvas_update(canvas)
                self.lib.tvg_canvas_draw(canvas, True)
                self.lib.tvg_canvas_sync(canvas)
                frames.append(list(pixels))
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
) -> None:
    global _WORKER_RENDERERS, _WORKER_RENDERER_ORDER, _WORKER_SIZE, _WORKER_FRAMES, _WORKER_ROOT, _WORKER_REPS
    global _WORKER_ACCURACY_ENABLED, _WORKER_ACCURACY_SIZE, _WORKER_ACCURACY_TOLERANCE, _WORKER_ACCURACY_DIFF_THRESHOLD
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
    for renderer in renderers:
        lib = Path(libs[renderer])
        if renderer == "tlottie":
            _WORKER_RENDERERS[renderer] = Tlottie(lib)
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
            if capture_accuracy and rep == 0 and renderer in accuracy_renderers:
                (
                    ok,
                    first_frame_ms,
                    frame_ms,
                    other_frames,
                    mem_avg,
                    mem_max,
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
            else:
                (
                    ok,
                    first_frame_ms,
                    frame_ms,
                    other_frames,
                    mem_avg,
                    mem_max,
                    err,
                ) = _WORKER_RENDERERS[renderer].measure(file, _WORKER_SIZE, _WORKER_FRAMES)
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
                    "error": err,
                }
            )
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampler = EnergySampler()
    sampler.start()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    total = len(files)
    progress_every = progress_interval(total)
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
        ),
    ) as pool:
        for done, (file_rows, accuracy_row) in enumerate(
            pool.map(worker_measure, [str(p) for p in files], chunksize=1), 1
        ):
            rows.extend(file_rows)
            if accuracy_row:
                accuracy_rows.append(accuracy_row)
            if should_report_progress(done, total, progress_every):
                print(f"   measured {done}/{total} files", flush=True)
    elapsed = time.perf_counter() - t0
    energy_j = sampler.stop_j()
    total_ms = sum(r["measured_ms"] for r in rows if r["ok"])
    for row in rows:
        row["batch_elapsed_s"] = elapsed
        row["batch_energy_j"] = energy_j
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


def init_accuracy_worker(
    root: str, size: int, frames: int, tolerance: int, diff_threshold: float
) -> None:
    global _ACCURACY_RENDERERS, _ACCURACY_ROOT, _ACCURACY_SIZE, _ACCURACY_FRAMES, _ACCURACY_TOLERANCE, _ACCURACY_DIFF_THRESHOLD
    _ACCURACY_ROOT = Path(root)
    _ACCURACY_SIZE = size
    _ACCURACY_FRAMES = frames
    _ACCURACY_TOLERANCE = tolerance
    _ACCURACY_DIFF_THRESHOLD = diff_threshold
    _ACCURACY_RENDERERS = {
        "tlottie": Tlottie(LIBS["tlottie"]),
        "rlottie": Rlottie(LIBS["rlottie"]),
        "thorvg": Thorvg(LIBS["thorvg"]),
    }


def worker_accuracy(file_s: str) -> dict[str, Any]:
    file = Path(file_s)
    rendered: dict[str, list[list[int]]] = {}
    counts: dict[str, int] = {}
    errors = []
    for renderer in ("tlottie", "rlottie", "thorvg"):
        ok, frames, count, err = _ACCURACY_RENDERERS[renderer].render_frames_argb(
            file, _ACCURACY_SIZE
        )
        if not ok:
            errors.append(f"{renderer}:{err}")
        rendered[renderer] = frames
        counts[renderer] = count
    if _ACCURACY_FRAMES > 0:
        rendered = {renderer: frames[:_ACCURACY_FRAMES] for renderer, frames in rendered.items()}
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


def make_accuracy_row(
    file: Path,
    root: Path,
    size: int,
    rendered: dict[str, list[list[int]]],
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
    return row


def diff_from_consensus(
    candidate: list[int], a: list[int], b: list[int], tolerance: int
) -> tuple[int, int]:
    bad = 0
    consensus = 0
    for cp, ap, bp in zip(candidate, a, b):
        if not px_close(ap, bp, tolerance):
            continue
        consensus += 1
        if not px_close_to_avg(cp, ap, bp, tolerance):
            bad += 1
    return bad, consensus


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(files)
    progress_every = progress_interval(total)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=init_accuracy_worker,
        initargs=(str(root), size, frames, tolerance, diff_threshold),
    ) as pool:
        for done, row in enumerate(
            pool.map(worker_accuracy, [str(p) for p in files], chunksize=4), 1
        ):
            rows.append(row)
            if should_report_progress(done, total, progress_every):
                print(f"   accuracy {done}/{total} files", flush=True)
    return rows


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
        out[pack] = {
            "good": good,
            "total": len(items),
            "ratio": (good / len(items)) if items else None,
        }
    return out


def save_diff_grids(
    rows: list[dict[str, Any]],
    root: Path,
    out_dir: Path,
    limit: int,
    size: int,
    tolerance: int,
) -> list[Path]:
    selected = select_diff_rows(rows, limit)
    if not selected:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_diff_dir(out_dir)
    renderers = {
        "tlottie": Tlottie(LIBS["tlottie"]),
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
    for r in renderers:
        cols += [
            f"{r}_first_frame_ms",
            f"{r}_frame_ms",
            f"{r}_rss_avg_mb",
            f"{r}_rss_max_mb",
            f"{r}_energy_j",
            f"{r}_error",
        ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(format_cell(row.get(c)) for c in cols) + "\n")


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
) -> None:
    system_info = f"{platform.system()} {platform.release()} / {platform.machine()}"
    css = """
body{font:13px/1.38 system-ui,sans-serif;margin:24px;background:#11151b;color:#dbe3ee}
h1,h2{color:#f4f7fb}
a{color:#8ab4ff}
table{border-collapse:collapse;width:100%;margin:16px 0 32px;background:#161b22}
th,td{border-top:1px solid #2b3440;border-bottom:1px solid #2b3440;padding:5px 7px;text-align:right;white-space:nowrap}
th{background:#202936;color:#edf2f7;position:sticky;top:0;z-index:1}
th.renderer{background:#283447;text-align:center}
td.left,th.left{text-align:left;border-left:1px solid #2b3440;border-right:1px solid #2b3440}
th.metric,td.metric{border-left:1px solid #2b3440}
th.metric-last,td.metric-last{border-right:1px solid #2b3440}
tr:nth-child(even) td{background:#141a22}
.winner{background:#184d2b!important;color:#d8ffe5;font-weight:700}
.loser{background:#5a1f25!important;color:#ffd6dc}
.acc-badge{float:right;margin-left:16px;border-radius:4px;padding:1px 6px;font-weight:700}
.acc-ok{background:#163f27;color:#c9f7d5}
.acc-warn{background:#5a4717;color:#ffe7a3}
.acc-bad{background:#5a1f25;color:#ffd6dc}
.note{color:#aab6c5;max-width:980px}
.muted{color:#8391a3}
"""
    with path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Lottie Benchmark</title>")
        f.write(f"<style>{css}</style>")
        f.write(f"<p class='note'>Run on {esc(system_info)}.</p>")
        f.write(
            "<p class='note'>fms is load/init plus first-frame draw. "
            "ms is the average of subsequent frames only. "
            "Memory columns are process RSS samples inside benchmark workers, "
            "not isolated renderer-owned allocation.</p>"
        )
        if accuracy_by_pack:
            f.write(
                "<p class='note'>Pack names include good/all animations for tlottie "
                f"versus the rlottie+thorvg consensus at {accuracy_size}px over every frame. "
                f"An animation is broken if any frame differs by more than "
                f"{accuracy_diff_threshold:g}% of consensus pixels; "
                f"pixel tolerance {accuracy_tolerance} opacity-weighted ARGB distance.</p>"
            )
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


def write_grouped_table(
    f: Any,
    rows: list[dict[str, Any]],
    renderers: tuple[str, ...],
    labels: tuple[str, ...],
    include_size: bool,
    accuracy_by_pack: dict[str, dict[str, Any]] | None = None,
) -> None:
    f.write("<table><tr>")
    for label in labels:
        f.write(f"<th rowspan='2' class='left'>{esc(label)}</th>")
    if include_size:
        f.write("<th rowspan='2'>size</th>")
    for r in renderers:
        f.write(f"<th colspan='4' class='renderer'>{esc(r)}</th>")
    f.write("</tr><tr>")
    for _ in renderers:
        f.write(
            "<th class='metric'>fms</th><th>ms</th>"
            "<th>RSS MiB (avg/max)</th><th class='metric-last'>J</th>"
        )
    f.write("</tr>")
    for row in rows:
        f.write("<tr>")
        for label in labels:
            value = row[label]
            if label == "pack" and accuracy_by_pack:
                value = pack_label(value, accuracy_by_pack.get(value))
                f.write(f"<td class='left'>{value}</td>")
            else:
                f.write(f"<td class='left'>{esc(value)}</td>")
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
    f.write("</table>")


def pack_label(pack: str, accuracy: dict[str, Any] | None) -> str:
    if not accuracy:
        return esc(pack)
    good = int(accuracy.get("good", 0))
    total = int(accuracy.get("total", 0))
    ratio = accuracy.get("ratio")
    if not total or ratio is None:
        return esc(pack)
    cls = "acc-ok"
    if ratio < 0.5:
        cls = "acc-bad"
    elif good < total:
        cls = "acc-warn"
    return (
        f"{esc(pack)} "
        f"<span class='acc-badge {cls}'>{good}/{total}</span>"
    )


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
    ap.add_argument("--renderers", default=",".join(RENDERERS))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open benchmark.html")
    args = ap.parse_args()

    renderers = tuple(r.strip() for r in args.renderers.split(",") if r.strip())
    bad = [r for r in renderers if r not in RENDERERS]
    if bad:
        raise SystemExit(f"unknown renderers: {', '.join(bad)}")
    sizes = tuple(int(s) for s in args.sizes.split(",") if s)
    if args.reps <= 0:
        raise SystemExit("--reps must be positive")
    if args.frames < 0:
        raise SystemExit("--frames must be non-negative")
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
    if not files:
        raise SystemExit(f"no .json files found under {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    accuracy_by_pack: dict[str, dict[str, Any]] | None = None
    accuracy_size = args.accuracy_size or min(sizes)
    accuracy_frame_label = "all frames" if args.frames == 0 else f"first {args.frames} frame(s)"
    accuracy_renderers = ("tlottie", "rlottie", "thorvg")
    can_reuse_accuracy = (
        not args.no_accuracy
        and accuracy_size in sizes
        and all(r in renderers for r in accuracy_renderers)
    )
    if not args.no_accuracy:
        for required in accuracy_renderers:
            if not LIBS[required].exists():
                raise SystemExit(f"missing {required} library for accuracy: {LIBS[required]}")
        mode = "reused from performance pass" if can_reuse_accuracy else "separate pass"
        print(
            f"== accuracy {accuracy_size}px {accuracy_frame_label}: {len(files)} files, "
            f"pixel tolerance {args.accuracy_tolerance}, "
            f"broken if diff > {args.accuracy_diff_threshold:g}% ({mode})",
            flush=True,
        )

    energy_available = EnergySampler().available()
    for size in sizes:
        print(
            f"== {size}px: {len(files)} files, {args.jobs} workers, "
            f"{args.reps} interleaved reps, renderers={','.join(renderers)}",
            flush=True,
        )
        size_rows, size_accuracy_rows = run_size_batch(
            renderers,
            size,
            files,
            args.input,
            args.frames,
            args.jobs,
            args.reps,
            can_reuse_accuracy and size == accuracy_size,
            accuracy_size,
            args.accuracy_tolerance,
            args.accuracy_diff_threshold,
        )
        all_rows.extend(size_rows)
        accuracy_rows.extend(size_accuracy_rows)

    if not args.no_accuracy:
        if not accuracy_rows:
            print("== accuracy fallback: rendering separate accuracy pass", flush=True)
            accuracy_rows = run_accuracy(
                files,
                args.input,
                accuracy_size,
                args.frames,
                args.accuracy_tolerance,
                args.accuracy_diff_threshold,
                args.jobs,
            )
        accuracy_by_pack = aggregate_accuracy(accuracy_rows)
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
