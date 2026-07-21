// tlottie vs rlottie vs thorvg (cpu + webgpu) wasm demo. Every engine sits
// behind the same tiny C ABI (src/bindings/wasm.rs,
// rlottie_shim.cpp, tvg_shim.cpp); this file drives them symmetrically and
// shows per-engine draw times.
//
// tlottie loads as a raw wasm module (no glue); rlottie and thorvg go
// through emscripten-generated ES6 modules. JS-side rule for all: never
// cache views into wasm memory, re-derive them after every exported call
// (memory growth detaches buffers).

import initRlottie from './rlottie.mjs';
import initThorvg from './thorvg.mjs';

const drop = document.getElementById('drop');
const panels = document.getElementById('panels');
const info = document.getElementById('info');
const statsRow = document.getElementById('statsrow');
const globalStats = document.getElementById('stats');
const capFps = document.getElementById('capfps');
const antialias = document.getElementById('antialias');
const alphaOnly = document.getElementById('alphaonly');
const fitz = document.getElementById('fitz');
const curveTolerance = document.getElementById('curvetolerance');
const curveToleranceVal = document.getElementById('curvetoleranceval');
const sizeSlider = document.getElementById('sizeslider');
const sizeVal = document.getElementById('sizeval');
const err = document.getElementById('err');
const picker = document.getElementById('picker');

// thorvg's webgpu backend needs a GPUDevice created before module init
// (emdawnwebgpu's preinitializedWebGPUDevice hook). No WebGPU -> the twg
// panel reports unavailable and the other three engines run as usual.
let gpuDevice = null;
try {
  const adapter = await navigator.gpu?.requestAdapter();
  gpuDevice = adapter ? await adapter.requestDevice() : null;
} catch { /* no WebGPU */ }

const [{ instance }, rl, tv] = await Promise.all([
  WebAssembly.instantiateStreaming(fetch('tlottie.wasm'), {}),
  initRlottie(),
  initThorvg(gpuDevice ? { preinitializedWebGPUDevice: gpuDevice } : {}),
]);
const tl = instance.exports;

function engineDom(key) {
  return {
    key,
    panel: document.getElementById(`panel-${key}`),
    canvasEl: document.getElementById(`canvas-${key}`),
    statsEl: document.getElementById(`stats-${key}`),
  };
}

const engines = [
  {
    ...engineDom('tl'),
    alloc: (n) => tl.tlottie_alloc(n),
    free: (p, n) => tl.tlottie_free(p, n),
    heap: () => new Uint8Array(tl.memory.buffer),
    create: (p, n, fitzModifier = 0) =>
      tl.tlottie_new_with_options(p, n, fitzModifier, 0, 0),
    drop: (i) => tl.tlottie_drop(i),
    width: (i) => tl.tlottie_width(i),
    height: (i) => tl.tlottie_height(i),
    frameRate: (i) => tl.tlottie_frame_rate(i),
    frameCount: (i) => tl.tlottie_frame_count(i),
    render: (i, f, w, h, aa, tolerance) =>
      tl.tlottie_render_with_options(i, f, w, h, aa ? 1 : 0, tolerance),
    renderAlpha8: (i, f, w, h, aa, tolerance) =>
      tl.tlottie_render_alpha8_with_options(i, f, w, h, aa ? 1 : 0, tolerance),
    renderAlpha8Color: (i, f, w, h, aa, color, tolerance) =>
      tl.tlottie_render_alpha8_color_with_options(
        i, f, w, h, aa ? 1 : 0, color, tolerance),
  },
  {
    ...engineDom('rl'),
    alloc: (n) => rl._rl_alloc(n),
    free: (p, n) => rl._rl_free(p, n),
    heap: () => rl.HEAPU8,
    create: (p, n) => rl._rl_new(p, n),
    drop: (i) => rl._rl_drop(i),
    width: (i) => rl._rl_width(i),
    height: (i) => rl._rl_height(i),
    frameRate: (i) => rl._rl_frame_rate(i),
    frameCount: (i) => rl._rl_frame_count(i),
    render: (i, f, w, h) => rl._rl_render(i, f, w, h),
  },
  {
    ...engineDom('tvg'),
    alloc: (n) => tv._tv_alloc(n),
    free: (p, n) => tv._tv_free(p, n),
    heap: () => tv.HEAPU8,
    create: (p, n) => tv._tv_new(p, n, 0, 0),
    drop: (i) => tv._tv_drop(i),
    width: (i) => tv._tv_width(i),
    height: (i) => tv._tv_height(i),
    frameRate: (i) => tv._tv_frame_rate(i),
    frameCount: (i) => tv._tv_frame_count(i),
    render: (i, f, w, h) => tv._tv_render(i, f, w, h),
  },
  {
    ...engineDom('twg'),
    // Draws straight into #canvas-twg via WebGPU: no ImageData blit, and
    // draw time is measured submit -> onSubmittedWorkDone (GPU completion),
    // one frame in flight at a time.
    direct: true,
    unavailable: !gpuDevice,
    alloc: (n) => tv._tv_alloc(n),
    free: (p, n) => tv._tv_free(p, n),
    heap: () => tv.HEAPU8,
    create: (p, n) => {
      const sel = tv.stringToNewUTF8('#canvas-twg');
      const inst = tv._tv_new(p, n, 1, sel);
      tv._tv_free(sel, 0);
      return inst;
    },
    drop: (i) => tv._tv_drop(i),
    width: (i) => tv._tv_width(i),
    height: (i) => tv._tv_height(i),
    frameRate: (i) => tv._tv_frame_rate(i),
    frameCount: (i) => tv._tv_frame_count(i),
    render: (i, f, w, h) => tv._tv_render(i, f, w, h),
  },
];

// Cap the render target: huge comps would otherwise chew CPU for pixels
// the <canvas> just scales down anyway.
const MAX_DIM = 1024;
const WINDOW = 360;  // rolling-average window, in draws

let raf = 0;
let currentBytes = null;

function fitzModifierValue() {
  return Number(fitz.value);
}

function recreateTlottie() {
  if (!currentBytes) return;
  const modifier = fitzModifierValue();
  const engine = engines.find((candidate) => candidate.key === 'tl');
  const ptr = engine.alloc(currentBytes.length);
  if (!ptr) {
    err.textContent = 'tlottie allocation failed';
    return;
  }
  engine.heap().set(currentBytes, ptr);
  const next = engine.create(ptr, currentBytes.length, modifier);
  engine.free(ptr, currentBytes.length);
  if (!next) {
    err.textContent = 'tlottie rejected the selected fitz value';
    return;
  }
  if (engine.inst) engine.drop(engine.inst);
  engine.inst = next;
  engine.dead = false;
  engine.drawMs.length = 0;
  engine.statsEl.textContent = '';
  err.textContent = '';
  applySize();
}

fitz.addEventListener('change', recreateTlottie);

// Renderer tabs: 'both' shows every available engine; an engine key shows
// just that panel. Hidden engines are skipped in the render loop, so
// single-engine draw stats aren't polluted by the others sharing the tick.
let mode = 'both';
const tabs = document.getElementById('tabs');
tabs.addEventListener('click', (ev) => {
  const btn = ev.target.closest('button[data-mode]');
  if (!btn) return;
  mode = btn.dataset.mode;
  for (const b of tabs.querySelectorAll('button')) {
    b.classList.toggle('active', b === btn);
  }
  document.body.classList.toggle('single', mode !== 'both');
  for (const e of engines) {
    e.panel.style.display = active(e) ? '' : 'none';
    // Fresh timing windows: don't mix samples across visibility modes.
    if (e.drawMs) e.drawMs.length = 0;
  }
});
const active = (e) => !e.unavailable && (mode === 'both' || mode === e.key);

// Render-size slider: sets the long side of the render target (the canvas
// CSS size stays put — this changes how many pixels the engines rasterize,
// for perf testing at different resolutions). Draw-time windows reset:
// samples at different sizes must not mix.
function applySize() {
  const target = +sizeSlider.value;
  sizeVal.textContent = `${target} px`;
  for (const e of engines) {
    if (!e.inst) continue;
    const scale = target / Math.max(e.cw, e.ch);
    e.w = Math.max(1, Math.round(e.cw * scale));
    e.h = Math.max(1, Math.round(e.ch * scale));
    e.canvasEl.width = e.w;
    e.canvasEl.height = e.h;
    if (e.drawMs) e.drawMs.length = 0;
  }
}
sizeSlider.addEventListener('input', applySize);
antialias.addEventListener('change', () => {
  const tlottie = engines.find((e) => e.key === 'tl');
  if (tlottie?.drawMs) tlottie.drawMs.length = 0;
});
alphaOnly.addEventListener('change', () => {
  for (const e of engines) if (e.drawMs) e.drawMs.length = 0;
});
curveTolerance.addEventListener('input', () => {
  curveToleranceVal.textContent = `${Number(curveTolerance.value).toFixed(3)} px`;
  const tlottie = engines.find((e) => e.key === 'tl');
  if (tlottie?.drawMs) tlottie.drawMs.length = 0;
});

function loadLottie(bytes, name) {
  cancelAnimationFrame(raf);
  err.textContent = '';
  globalStats.textContent = '';
  currentBytes = bytes.slice();

  let loaded = null;  // engine-independent comp info from whichever parsed
  for (const e of engines) {
    if (e.inst) { e.drop(e.inst); e.inst = null; }
    e.dead = false;
    e.inFlight = false;
    e.drawMs = [];
    e.statsEl.textContent = '';
    if (e.unavailable) { e.statsEl.textContent = 'WebGPU unavailable'; continue; }

    const ptr = e.alloc(bytes.length);
    if (!ptr) { e.statsEl.textContent = 'allocation failed'; continue; }
    e.heap().set(bytes, ptr);
    const inst = e.create(ptr, bytes.length, e.key === 'tl' ? fitzModifierValue() ?? 0 : 0);
    e.free(ptr, bytes.length);
    if (!inst) { e.statsEl.textContent = `rejected ${name}`; continue; }

    e.inst = inst;
    e.ctx = e.direct ? null : e.canvasEl.getContext('2d');
    e.cw = e.width(inst);
    e.ch = e.height(inst);
    e.fps = e.frameRate(inst);
    e.frames = e.frameCount(inst);
    loaded ??= { cw: e.cw, ch: e.ch, fps: e.fps, frames: e.frames };
  }
  if (!loaded) { err.textContent = `all engines rejected ${name}`; return; }

  sizeSlider.value = Math.min(MAX_DIM, Math.max(loaded.cw, loaded.ch));
  applySize();

  panels.style.display = 'flex';
  tabs.style.display = 'flex';
  statsRow.style.display = 'flex';
  for (const e of engines) e.panel.style.display = active(e) ? '' : 'none';
  info.textContent =
    `${name} — ${loaded.cw}×${loaded.ch}, ${loaded.frames} frames @ ${loaded.fps} fps`;

  // One shared clock drives all engines. tlottie and thorvg take continuous
  // float frames (unique in-between poses every tick on fast displays);
  // rlottie's timeline is discrete, so its shim rounds to nearest.
  //
  // `window.timeScale` (console knob, default 1) scales playback speed:
  // e.g. timeScale = 0.2 plays at 1/5th speed. The playhead accumulates
  // dt * timeScale per tick, so changing it mid-play never jumps.
  let flip = false;   // rotate engine order per tick (fair interleave)
  let prevTick = 0;   // last rAF timestamp (playhead dt)
  let prevDraw = 0;   // last drawn tick timestamp (Hz readout)
  let nextDue = 0;    // fps-cap scheduler
  let lastCap = capFps.checked;
  const tickMs = [];
  let statsAt = 0;
  let pos = 0;  // playhead, in frames
  const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  const tick = (now) => {
    // Clamp tab-hidden pauses so the playhead doesn't leap on return.
    const dt = prevTick ? Math.min(now - prevTick, 100) : 0;
    prevTick = now;
    pos = (pos + dt / 1000 * loaded.fps * (window.timeScale ?? 1))
        % loaded.frames;

    // Toggling the cap invalidates both timing windows — reset, don't mix.
    if (capFps.checked !== lastCap) {
      lastCap = capFps.checked;
      nextDue = 0;
      tickMs.length = 0;
      for (const e of engines) if (e.drawMs) e.drawMs.length = 0;
    }

    // Cap: draw at most once per file-frame interval, so each engine
    // renders every frame exactly once — per-draw costs become
    // apples-to-apples (no cheap repeat-frame calls for rlottie, no extra
    // in-between poses for the float-frame engines). Quarter-interval
    // slack absorbs vsync jitter when display rate == file fps.
    const interval = 1000 / loaded.fps;
    let draw = true;
    if (capFps.checked) {
      if (now < nextDue - interval / 4) draw = false;
      else nextDue = Math.max(nextDue + interval, now);
    }

    if (draw) {
      // Rotate who renders first: the first engine after the vsync idle
      // wait runs on a cooler/downclocked core and measures slower (~2x
      // observed) — a fixed order would bias every comparison.
      flip = !flip;
      const order = flip ? [...engines].reverse() : engines;
      for (const e of order) {
        if (!e.inst || e.dead || !active(e)) continue;
        if (e.inFlight) continue;  // one GPU frame in flight at a time
        const frame = pos % e.frames;
        const d0 = performance.now();
        const alpha8 = e.key === 'tl' && alphaOnly.checked;
        const px = alpha8
          ? e.renderAlpha8Color(
              e.inst, frame, e.w, e.h, antialias.checked, 0xffffff,
              +curveTolerance.value)
          : e.render(e.inst, frame, e.w, e.h, antialias.checked, +curveTolerance.value);
        if (!px) { e.dead = true; e.statsEl.textContent = 'render failed'; continue; }
        if (e.direct) {
          // GPU submit done; sample completes when the queue drains.
          e.inFlight = true;
          gpuDevice.queue.onSubmittedWorkDone().then(() => {
            e.inFlight = false;
            const ms = performance.now() - d0;
            // >500ms means the tab was suspended mid-flight (the promise
            // resolves on the event loop), not real GPU time — discard.
            if (ms < 500) {
              e.drawMs.push(ms);
              if (e.drawMs.length > WINDOW) e.drawMs.shift();
            }
          });
          continue;
        }
        const rgba = new Uint8ClampedArray(e.heap().buffer, px, e.w * e.h * 4);
        e.ctx.putImageData(new ImageData(rgba, e.w, e.h), 0, 0);
        e.drawMs.push(performance.now() - d0);
        if (e.drawMs.length > WINDOW) e.drawMs.shift();
      }

      // Deltas over 500ms are tab-hidden pauses, not draw cadence; skip
      // them so the Hz readout recovers instantly after a tab switch.
      if (prevDraw && now - prevDraw < 500) {
        tickMs.push(now - prevDraw);
        if (tickMs.length > WINDOW) tickMs.shift();
      }
      prevDraw = now;
    }

    if (now - statsAt > 250) {
      statsAt = now;
      for (const e of engines) {
        if (e.inst && !e.dead && active(e) && e.drawMs.length) {
          e.statsEl.textContent =
            `avg draw ${avg(e.drawMs).toFixed(2)} ms (last ${e.drawMs.length})`;
        }
      }
      if (tickMs.length) {
        globalStats.textContent = `drawing at ${(1000 / avg(tickMs)).toFixed(1)} Hz`;
      }
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

async function loadFile(file) {
  let bytes = new Uint8Array(await file.arrayBuffer());
  // .tgs is gzipped Lottie JSON (magic 1f 8b).
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream()
      .pipeThrough(new DecompressionStream('gzip'));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  loadLottie(bytes, file.name);
}

drop.addEventListener('click', () => picker.click());
picker.addEventListener('change', () => {
  if (picker.files[0]) loadFile(picker.files[0]);
});
for (const ev of ['dragover', 'dragenter']) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
}
for (const ev of ['dragleave', 'drop']) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove('over');
  });
}
document.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) loadFile(file);
});

// Console hooks for driving the demo without a real drop (tests, quick loads):
//   loadLottieBytes(await (await fetch(url)).arrayBuffer(), 'name')
window.loadLottieBytes = (buf, name) =>
  loadLottie(new Uint8Array(buf), name ?? 'buffer');
window.loadLottieFile = loadFile;
