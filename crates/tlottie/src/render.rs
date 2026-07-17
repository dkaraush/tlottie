//! Frame rendering: evaluate the model at a frame, flatten geometry,
//! rasterize, and composite into a premultiplied ARGB32 buffer.

use crate::cells::CellRaster;
use crate::error::{Error, Result};
use crate::geometry::{
    clip_contour, clip_to_quad, dash_polyline, ellipse_contour, extract_by_length, flatten_path,
    polystar_path, quad_contains_box, rect_contour, round_polyline_corners, Contour,
};
use crate::limits::Limits;
use crate::math::{Color, Mat2x3, Vec2};
use crate::model::{
    Composition, DashElement, FillRule, FloatList, GradientKind, Layer, LayerKind, Shape,
    Transform, TrimMode,
};
use crate::raster::Rasterizer;
use crate::stroke::stroke_polyline;

/// Maximum group recursion while rendering (matches parse-side bound).
const MAX_RENDER_DEPTH: usize = 40;
/// Gradient LUT resolution; rlottie uses a 1024-entry table
/// (VGradient::colorTableSize) — 256 visibly quantizes steep ramps.
const GRADIENT_LUT_SIZE: usize = 1024;
/// Maximum precomp nesting during render.
const MAX_PRECOMP_DEPTH: usize = 16;

/// Reusable render-time buffers: rasterizer accumulators, offscreen pixel
/// planes, mask planes, gradient LUT memos. Owned by [`crate::Animation`]
/// (one per playing instance) so nothing is reallocated between frames —
/// at 720px a single frame otherwise allocates and zeroes multiple MB.
/// Dropping it frees everything; it carries no per-composition state.
#[derive(Default)]
pub(crate) struct RenderScratch {
    rasters: Vec<Rasterizer>,
    cells_pool: Vec<CellRaster>,
    bufs_u32: Vec<Vec<u32>>,
    bufs_u8: Vec<Vec<u8>>,
    /// Gradient LUT memoization: building a 1024-entry premultiplied table
    /// from the stop list is pure, and stop values repeat across frames
    /// (static gradients: every frame; animated: on hold segments and loop
    /// repeats). Keyed by the exact input bits — no collision risk.
    lut_cache: std::collections::HashMap<Vec<u32>, std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>>,
    lut_key: Vec<u32>,
    /// Recycled contour point buffers: stroke pieces + fill snapshots draw
    /// from here and return after their paint executes (measured: 2,683
    /// piece allocations per 64px frame on stroke-heavy files).
    pts_pool: Vec<Vec<Vec2>>,
    /// Memoized per-layer staticness (keyed by the Layer's stable address
    /// inside the Arc'd Composition).
    static_flags: std::collections::HashMap<usize, bool>,
    /// Static-layer job lists: replay the exact fill calls (by coverage
    /// key) without walking/evaluating/flattening the shape tree at all —
    /// the per-frame cost rlottie avoids via its own static detection.
    jobs_cache: std::collections::HashMap<u128, Vec<ReplayJob>>,
    /// Two-touch admission for jobs_cache: a replay key must be seen twice
    /// before recording (a static layer under an ANIMATED parent produces a
    /// fresh key every frame and must not flood the cache).
    jobs_seen: std::collections::HashSet<u128>,
    /// Content-addressed coverage cache (the GOALS design lever): keyed by
    /// the exact bits of a paint's clipped device-space geometry, valued by
    /// the rasterizer's coverage rows. Measured repeat rates on the heavy
    /// set: 74-96% within one loop, ~100% across loop replays. Coverage is
    /// deterministic per geometry, so replaying rows is bit-exact.
    cov_cache: CovCache,
}

/// Paint bbox extent (px, max dimension) ABOVE which the sparse cell/span
/// engine (mode S, cells.rs) rasterizes instead of the dense accumulator
/// (mode D, raster.rs). Initial value: 42, the measured crossover of the
/// rlottie small-mask patch (d9d6ad5: "≤42 → plain bitmap beats spans");
/// re-tuned by the RASTER_PLAN §5 benchmark matrix — change HERE only.
const MODE_S_MIN_EXTENT: usize = 42;

/// One uniform-coverage span, packed: y:20 | x0:20 | len:16 | cov:8.
#[inline]
fn pack_span(y: usize, x0: usize, len: usize, cov: u8) -> u64 {
    ((y as u64) << 44) | ((x0 as u64) << 24) | ((len as u64) << 8) | u64::from(cov)
}

#[inline]
fn unpack_span(s: u64) -> (usize, usize, usize, u8) {
    (
        (s >> 44) as usize,
        ((s >> 24) & 0xf_ffff) as usize,
        ((s >> 8) & 0xffff) as usize,
        (s & 0xff) as u8,
    )
}

/// Dev-only gradient instrumentation (zero cost when unread): pixel counts
/// per gradient kind and how many pixels the batched NEON kernels covered.
/// Indices: 0=linear 1=radial 2=focal 3=radial-batched 4=focal-batched.
pub(crate) static GRAD_STATS: [core::sync::atomic::AtomicU64; 5] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];

#[inline]
fn grad_stat(i: usize, n: usize) {
    if let Some(c) = GRAD_STATS.get(i) {
        c.fetch_add(n as u64, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Dev-only pixel-traffic counters (zero cost when unread), read via
/// tlottie::px_stats(). Slots:
/// 0=replay px (cov rows)  1=replay px (uniform spans)  2=fresh mode-S px
/// 3=fresh dense px        4=mask coverage px           5=mask plane-walk px
/// 6=offscreen clear px    7=composite px               8=fresh span count
/// 9=replay span count    10=modulate px               11=offscreen takes
pub(crate) static PX_STATS: [core::sync::atomic::AtomicU64; 12] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];

#[cfg(feature = "stats")]
#[inline]
pub(crate) fn px_stat(i: usize, n: usize) {
    if let Some(c) = PX_STATS.get(i) {
        c.fetch_add(n as u64, core::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(not(feature = "stats"))]
#[inline(always)]
pub(crate) fn px_stat(_i: usize, _n: usize) {}

/// Dev-only mode-selection counters: 0=mode S, 1=mode D (extent gate),
/// 2=mode D (density gate). Read via tlottie::mode_stats().
pub(crate) static MODE_STATS: [core::sync::atomic::AtomicU64; 3] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];

/// Mode-S guard against edge-dense content: the cell engine's cost is
/// ≈ perimeter px (deposit + sort), the dense engine's is ≈ area bytes.
/// Stroke piece unions (thousands of tiny quads) have perimeter of the
/// same order as area — sorting their cell piles loses to the plane
/// (measured: DogsEmoji@320 cold 1.4x, quicksort 21% of the profile).
/// S wins when `perimeter * DENSITY < bbox area`. Canvas-scaled per the
/// E156B matrix (2026-07-14): loosening to 12 won −6.4%/−4.9% at
/// 64/320px but cost +1.4% at 720px, so small canvases admit denser
/// paints to mode S than effects-class ones. Extent 32/42/64 measured
/// flat; 42 kept (the d9d6ad5 crossover).
const MODE_S_EDGE_DENSITY_SMALL: f32 = 12.0; // canvas ≤ 448x448
const MODE_S_EDGE_DENSITY_LARGE: f32 = 6.0;

/// Decides the rasterizer mode for one paint: sparse cells (mode S) for
/// large, edge-sparse paints; the dense accumulator (mode D) otherwise.
/// One pass over points — negligible next to rasterization; non-finite
/// points are ignored by f32 min/max.
fn mode_s_wins(contours: &[Contour], canvas_px: usize) -> bool {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut perim = 0.0f32;
    for c in contours {
        let mut prev: Option<Vec2> = c.points.last().copied();
        for p in &c.points {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
            if let Some(q) = prev {
                perim += (p.x - q.x).abs() + (p.y - q.y).abs();
            }
            prev = Some(*p);
        }
    }
    if !(x1 > x0 && y1 > y0) {
        return false;
    }
    if ((x1 - x0).max(y1 - y0)) <= MODE_S_MIN_EXTENT as f32 {
        grad_stat_arr(&MODE_STATS, 1, 1);
        return false;
    }
    let density = if canvas_px <= 448 * 448 {
        MODE_S_EDGE_DENSITY_SMALL
    } else {
        MODE_S_EDGE_DENSITY_LARGE
    };
    let s = perim * density < (x1 - x0) * (y1 - y0);
    grad_stat_arr(&MODE_STATS, if s { 0 } else { 2 }, 1);
    s
}

#[inline]
fn grad_stat_arr(arr: &[core::sync::atomic::AtomicU64; 3], i: usize, n: usize) {
    if let Some(c) = arr.get(i) {
        c.fetch_add(n as u64, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Rebuilds a row-plane cache entry from a span list (exact: spans arrive
/// y-ascending, x-ascending, gaps are zero coverage). Used when the span
/// list is FRAGMENTED (fine AA content — many len-1..3 spans): row replay
/// is one call per row, span replay pays per-span dispatch (measured 4.5x
/// on DogsEmoji@320 adaptive).
fn spans_to_cov_entry(spans: &[u64]) -> CovEntry {
    let mut entry = CovEntry::default();
    let PlaneData::Cov(data) = &mut entry.data else {
        return entry;
    };
    // Pre-size pass: count rows and covered bytes so the fill pass below
    // never reallocs (measured: the growth chains were half this
    // function's cost on stroke-heavy 720px frames). Tiny captures skip
    // it — doubling growth is cheaper than a second unpack walk there.
    if spans.len() >= 64 {
        pre_size(spans, &mut entry.rows, data);
    }
    let mut i = 0usize;
    while let Some(&first) = spans.get(i) {
        let (y, x0, len0, _) = unpack_span(first);
        let mut j = i;
        let mut x_end = x0 + len0;
        while let Some(&s) = spans.get(j) {
            let (yy, xx, ll, _) = unpack_span(s);
            if yy != y {
                break;
            }
            x_end = xx + ll;
            j += 1;
        }
        entry.rows.push((y as u32, x0 as u32, (x_end - x0) as u32));
        let base = data.len();
        data.resize(base + (x_end - x0), 0);
        for k in i..j {
            if let Some(&s) = spans.get(k) {
                let (_, xx, ll, cv) = unpack_span(s);
                let lo = base + (xx - x0);
                if let Some(seg) = data.get_mut(lo..lo + ll) {
                    seg.fill(cv);
                }
            }
        }
        i = j;
    }
    entry
}

/// Reservation pass for [`spans_to_cov_entry`]: walks the span list once
/// to count rows and covered bytes, then `reserve_exact`s both vectors.
fn pre_size(spans: &[u64], rows: &mut Vec<(u32, u32, u32)>, data: &mut Vec<u8>) {
    let mut nrows = 0usize;
    let mut nbytes = 0usize;
    {
        let mut i = 0usize;
        while let Some(&first) = spans.get(i) {
            let (y, x0, len0, _) = unpack_span(first);
            let mut x_end = x0 + len0;
            let mut j = i;
            while let Some(&s) = spans.get(j) {
                let (yy, xx, ll, _) = unpack_span(s);
                if yy != y {
                    break;
                }
                x_end = xx + ll;
                j += 1;
            }
            nrows += 1;
            nbytes += x_end - x0;
            i = j;
        }
    }
    rows.reserve_exact(nrows);
    data.reserve_exact(nbytes);
}

/// A span list larger than this can never pass [`CovCache::insert`]'s
/// COV_ENTRY_MAX admission (8 bytes/span) — capture stops early and the
/// entry is discarded instead of built and rejected (720px full-bleeds
/// pushed 10-30k spans per paint just to be dropped).
const SPAN_CAPTURE_MAX: usize = COV_ENTRY_MAX / 8;

/// Fragmentation test for a fresh span capture: average span shorter than
/// 4 px → store as rows (see [`spans_to_cov_entry`]).
fn spans_fragmented(spans: &[u64], px_total: usize) -> bool {
    spans.len() * 4 > px_total
}

/// Cached plane payload: rasterizer coverage bytes (mode D), a span list
/// (mode S — denser, replays as uniform-coverage blends), or a gradient
/// paint's premultiplied coverage-scaled SOURCE pixels (replayed as a pure
/// composite — bit-exact because the blend formula is identical).
/// The representation tag lives per entry: geometry keys are mode-blind,
/// but one geometry always has one extent, hence one mode.
enum PlaneData {
    Cov(Vec<u8>),
    Spans(Vec<u64>),
    Src(Vec<u32>),
}

impl Default for PlaneData {
    fn default() -> PlaneData {
        PlaneData::Cov(Vec::new())
    }
}

/// Rows for one cached plane: `(y, x0, len)` per row into `data`.
#[derive(Default)]
struct CovEntry {
    rows: Vec<(u32, u32, u32)>,
    data: PlaneData,
}

/// Byte-budgeted map from 128-bit geometry hash to coverage rows.
/// Collisions at 128 bits are negligible (the key deliberately does not
/// store the geometry itself — Joker-class files have ~18k distinct
/// geometries per loop and full keys would dwarf the coverage payload).
/// Eviction: whole-cache clear on budget overflow — periodic animations
/// refill within one loop, and the budget bounds per-instance memory.
#[derive(Default)]
struct CovCache {
    /// Canvas-scaled budget (see COV_CACHE_BUDGET docs); 0 = default.
    budget: usize,
    /// Canvas-scaled: trim entry Vec slack at insert (see the fleet gate
    /// note in [`CovCache::insert`]). Set alongside `budget`.
    shrink_entries: bool,
    /// Rotation guard: static-layer job replay pre-checks that every job's
    /// entry is present, then replays them one by one — but a replay-path
    /// insert may rotate generations and evict a LATER job's entry,
    /// silently dropping that paint (a fill given empty contours on a miss
    /// draws nothing). While pinned, rotation is deferred (young may
    /// briefly exceed budget by one layer's inserts).
    pinned: bool,
    /// Two generations: inserts go to `young`; the first time `young`
    /// exceeds half the budget it becomes `old` (nothing dropped), the
    /// second time the cache freezes (see below). Lookups hit either
    /// generation in place. A working set that fits half the budget never
    /// rotates at all; bigger sets freeze at ~budget resident.
    young: std::collections::HashMap<u128, CovEntry>,
    old: std::collections::HashMap<u128, CovEntry>,
    young_bytes: usize,
    /// Freeze policy: drop-rotation is the wrong eviction for periodic
    /// content whose loop working set exceeds the budget — every entry is
    /// evicted right before its once-per-loop reuse (sequential-scan
    /// pathology; measured: Peepo_Pepe@720 6.45→10.40ms and
    /// theopenemojis@320 hits≈inserts churn when the budget shrank below
    /// the loop set). Instead of dropping a generation on the second
    /// overflow, the cache FREEZES: the resident ~budget of entries stops
    /// changing (inserts skipped, no promotion), and every frozen entry
    /// hits once per loop — hit fraction becomes budget/working_set
    /// instead of ~0, with zero capture cost. tgs timelines are capped at
    /// 180 frames, so one [`FREEZE_ERA_FRAMES`] era always spans a full
    /// loop: at era end a frozen set that kept hitting (≥ half the
    /// resident entries) is renewed; a dead one (content moved on, or
    /// nothing ever repeats twice within the era) is cleared and the
    /// cache re-learns. This subsumes the earlier thrash detector
    /// (IceMan@720: its frozen ≤budget slice still replays across loops —
    /// capture cost stays zero, same win, plus the bonus hits) and fixes
    /// its cold-loop false positive (hits are judged over a full era, not
    /// over the first 2 rotations, which a small budget completes before
    /// loop 2 can produce a single cross-loop hit).
    hits: u32,
    inserts: u32,
    rotations: u32,
    frozen: bool,
    era_frames: u32,
    era_hits: u32,
}

/// Young-generation overflows before the resident set freezes (the first
/// overflow rotates an empty old generation away; the second would drop
/// real entries — freeze instead).
const FREEZE_ROTATIONS: u32 = 2;
/// Frames per frozen era; one era spans any full tgs loop (≤180 frames),
/// so every still-relevant frozen entry hits at least once per era.
const FREEZE_ERA_FRAMES: u32 = 180;

/// Default coverage-cache budget; the real value is canvas-scaled via
/// [`CovCache::set_budget_for_canvas`]. Measured curves:
/// - 64px (fleet case): entries are tiny, so RSS tracks the true working
///   set, not the budget — Joker@64's 5.4 MB loop replays entirely within
///   a total process RSS of ~9 MB. Budget 12 MB keeps those wins.
/// - 320px: 1 MB captures ~97% of the steady win (0.495 vs 0.483 ms/f at
///   12 MB); every budget MB past that costs ~2 MB peak RSS in dead
///   entries. Fully-animated giants (Joker@320: 28 MB/loop) are miss-bound
///   at ANY sane budget. Budget 3 MB.
/// - 720px: only sub-64KB entries cache; planes dominate RSS. Budget 2 MB.
const COV_CACHE_BUDGET: usize = 3 << 20;
/// Entries bigger than this are not cached — one entry would evict a whole
/// loop's worth of small ones. 256KB admits sticker-size (320px) gradient
/// source planes (measured: the 64KB cap left TableFontEmoji@320 recomputing
/// its gradients every frame while @64 flew).
const COV_ENTRY_MAX: usize = 256 << 10;

/// Two independent FNV/Murmur-style 64-bit streams -> 128-bit content key.
struct Hasher128 {
    h1: u64,
    h2: u64,
}

impl Hasher128 {
    fn new() -> Hasher128 {
        Hasher128 {
            h1: 0xcbf2_9ce4_8422_2325,
            h2: 0x9e37_79b9_7f4a_7c15,
        }
    }

    #[inline]
    fn mix(&mut self, w: u32) {
        self.h1 = (self.h1 ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3);
        self.h2 = (self.h2 ^ u64::from(w.rotate_left(16))).wrapping_mul(0xff51_afd7_ed55_8ccd);
    }

    fn finish(&self) -> u128 {
        (u128::from(self.h1) << 64) | u128::from(self.h2)
    }
}

impl CovCache {
    /// Looks a key up in either generation (in place — see the
    /// no-promotion note below).
    fn get(&mut self, key: u128) -> Option<&CovEntry> {
        if self.young.contains_key(&key) {
            self.hits = self.hits.saturating_add(1);
            if self.frozen {
                self.era_hits = self.era_hits.saturating_add(1);
            }
            return self.young.get(&key);
        }
        // Old-generation hits are returned IN PLACE — no promotion. Under
        // the freeze policy no generation is ever dropped (the second
        // overflow freezes instead of rotating; the only eviction is the
        // era clear), so promotion bought nothing and cost a map
        // remove+reinsert per hit plus re-inflated young_bytes: measured
        // on Joker@64 with a 4MB budget, rotation landed at the loop
        // seam and the entire next loop paid ~6.2k promotes and froze
        // mid-loop with two full-capacity maps (wall 0.43ms and RSS both
        // WORSE than at 2MB — the mid-budget anomaly in the freeze-era
        // curve). In place, the curve is monotonic.
        if let Some(e) = self.old.get(&key) {
            self.hits = self.hits.saturating_add(1);
            if self.frozen {
                self.era_hits = self.era_hits.saturating_add(1);
            }
            return Some(e);
        }
        None
    }

    fn contains(&self, key: u128) -> bool {
        self.young.contains_key(&key) || self.old.contains_key(&key)
    }

    fn size_of(e: &CovEntry) -> usize {
        // Count CAPACITIES (Vec growth over-allocates up to 2x) plus map
        // slot overhead — the RSS audit showed len-based accounting
        // understated the true footprint ~2x.
        let data = match &e.data {
            PlaneData::Cov(v) => v.capacity(),
            PlaneData::Spans(v) => v.capacity() * 8,
            PlaneData::Src(v) => v.capacity() * 4,
        };
        data + e.rows.capacity() * 12 + 64
    }

    fn set_budget_for_canvas(&mut self, w: usize, h: usize) {
        // Dev-only measurement hook (TLOTTIE_LEGACY_STROKER precedent):
        // TLOTTIE_COV_BUDGET_KB="a,b,c" overrides the three size-class
        // budgets (KB, fleet/mid/large) for budget-vs-RSS curve runs.
        // Benchmarked candidates are re-verified as const builds before
        // landing; this hook itself is never the shipped configuration.
        static OVERRIDE: std::sync::OnceLock<Option<[usize; 3]>> = std::sync::OnceLock::new();
        let ov = OVERRIDE.get_or_init(|| {
            let v = std::env::var("TLOTTIE_COV_BUDGET_KB").ok()?;
            let mut it = v.split(',').map(|s| s.trim().parse::<usize>());
            match (it.next(), it.next(), it.next()) {
                (Some(Ok(a)), Some(Ok(b)), Some(Ok(c))) => Some([a << 10, b << 10, c << 10]),
                _ => None,
            }
        });
        let px = w.saturating_mul(h);
        self.shrink_entries = px > 160 * 160;
        if let Some([a, b, c]) = ov {
            self.budget = if px <= 160 * 160 {
                *a
            } else if px <= 448 * 448 {
                *b
            } else {
                *c
            };
            return;
        }
        self.budget = if px <= 160 * 160 {
            // Fleet class. The 2026-07-16 full-pack 64px sweep showed the
            // old 4MB budget made tlottie keep a high RSS floor versus
            // rlottie (median +5.37MiB; 211/346 packs > +5MiB). 1MB keeps
            // tlottie well ahead on wall time in aggregate while bringing
            // memory back near parity (avg RSS 8.76 -> 5.53MiB, max
            // 15.02 -> 9.33MiB, no pack > +5MiB vs rlottie in the sweep).
            1 << 20
        } else if px <= 448 * 448 {
            // Device budget/RSS curve (2026-07-14, post-thrash-detector,
            // E156B): wall FLAT from 3MB down to 0.5MB on both the
            // low-hit set (ABC/News/Duck/TableFont) and the replay-heavy
            // guards (Woodpecker/RaccoonyDays/CuteNurse); 1MB takes
            // per-instance RSS from ~7.5-8.4 to ~5.1-6.0MB — production
            // rlottie parity. The thrash detector makes small budgets
            // degrade gracefully (capture off, not churn).
            1 << 20
        } else {
            // Same curve at 720: flat on IceMan/DeathNote/Godzi/Premium-
            // Gifts, RaccoonyDays -17% at 512KB; RSS -1..-3MB.
            512 << 10
        };
    }

    fn rotate_if_needed(&mut self) {
        // Frozen is a stable state (young_bytes stays over the rotation
        // threshold by construction) — only the era check may leave it.
        // Without this guard the replay-unpin path re-froze every batch,
        // resetting the era counters (measured: rotations=392 on
        // theopenemojis@320 and era_hits pinned at 0).
        if self.pinned || self.frozen {
            return;
        }
        let budget = if self.budget == 0 {
            COV_CACHE_BUDGET
        } else {
            self.budget
        };
        if self.young_bytes > budget / 2 {
            self.rotations += 1;
            if self.rotations >= FREEZE_ROTATIONS {
                // Young + old together hold ~budget of the most recent
                // working set — freeze it in place instead of dropping a
                // generation that is about to be reused.
                self.frozen = true;
                self.era_frames = 0;
                self.era_hits = 0;
            } else {
                self.old = core::mem::take(&mut self.young);
                self.young_bytes = 0;
            }
        }
    }

    /// True while the cache is still learning (capture pays for itself);
    /// false while frozen (the resident set replays, nothing new admits).
    #[inline]
    fn capture_enabled(&self) -> bool {
        !self.frozen
    }

    /// Per-frame tick: while frozen, count frames toward the era check —
    /// a frozen set that kept hitting (≥ half its entries over an era
    /// that always spans a full loop) is renewed; a dead one is cleared
    /// and the cache re-learns from scratch.
    fn frame_tick(&mut self) {
        if self.frozen {
            self.era_frames += 1;
            if self.era_frames >= FREEZE_ERA_FRAMES {
                let resident = self.young.len().saturating_add(self.old.len());
                if (self.era_hits as usize) >= (resident / 2).max(1) {
                    self.era_frames = 0;
                    self.era_hits = 0;
                } else {
                    self.young.clear();
                    self.old.clear();
                    self.young_bytes = 0;
                    self.frozen = false;
                    self.hits = 0;
                    self.inserts = 0;
                    self.rotations = 0;
                }
            }
        }
    }

    fn insert(&mut self, key: u128, mut entry: CovEntry) {
        if self.frozen {
            return;
        }
        // Entries are built by incremental push/extend, so their Vecs carry
        // growth slack (up to ~2x len). They are immutable after insert
        // (replay only reads) and under the freeze policy may live for the
        // whole animation — so the slack is pure dead RSS for the animation
        // lifetime. Trim it to the exact footprint once, here.
        //
        // Budget accounting: size_of() deliberately counts CAPACITY as a
        // proxy for real RSS. We measure it on the GROWN entry (before the
        // shrink) and keep counting capacity — i.e. the admission check and
        // young_bytes see exactly the same numbers HEAD saw. This is a
        // deliberate COMPENSATION, not an oversight: if we instead accounted
        // the shrunk (len-sized) entry, the same byte budget would admit
        // ~2x more entries, growing the frozen resident set and its HashMap
        // overhead — measured to push peak RSS UP 3-6% on the frozen
        // 320/720 cases (Joker@320 +5.6%), the opposite of the goal, and to
        // shift the freeze snapshot unpredictably (TableFontEmoji@320 lost
        // hits). Accounting the grown size keeps freeze timing, resident
        // entry count and admission bit-identical to HEAD, so the change is
        // pure RSS reduction (slack removed) with zero behavioural drift.
        //
        // Fleet gate: shrink is OFF for the ≤160² class. The realloc
        // scatters entries across allocator size bins and replay pays the
        // locality loss where entries are tiny and hit rates extreme —
        // referee 6-rep medians: UtyaDuck@64 +5.4% wall (distributions
        // non-overlapping), WallyOwl +6.1%, FestiveFont +3.8% against RSS
        // −4..−9%. At 320/720 wall is flat-to-better (TableFontEmoji@320
        // −7%) with RSS −1..−5%, so the trade only pays above fleet scale.
        let sz = Self::size_of(&entry);
        if sz > COV_ENTRY_MAX {
            return;
        }
        if self.shrink_entries {
            entry.rows.shrink_to_fit();
            match &mut entry.data {
                PlaneData::Cov(v) => v.shrink_to_fit(),
                PlaneData::Spans(v) => v.shrink_to_fit(),
                PlaneData::Src(v) => v.shrink_to_fit(),
            }
        }
        self.inserts = self.inserts.saturating_add(1);
        self.young_bytes += sz;
        self.rotate_if_needed();
        self.young.insert(key, entry);
    }
}

/// Bound on recycled point buffers (each typically 4-26 points).
const PTS_POOL_CAP: usize = 4096;

/// One recorded paint of a static layer: everything the fused execute loop
/// needs except geometry (which replays from the coverage cache by key).
enum ReplayJob {
    Solid {
        key: u128,
        rule: FillRule,
        color: Color,
        opacity: f32,
    },
    Gradient {
        key: u128,
        src_key: u128,
        rule: FillRule,
        lut: std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
        map: GradientMap,
    },
}

/// Bound on cached static-layer job lists.
const JOBS_CACHE_CAP: usize = 1024;

/// Bound on memoized gradient LUTs (4 KB each). A composition rarely has
/// more than a few dozen distinct gradients; continuously-animated stops
/// would otherwise grow the map without bound, so overflow clears it.
const LUT_CACHE_CAP: usize = 512;

/// Bound on pooled objects per kind — offscreen depth beyond this is rare
/// (matte pairs + a couple of nested composites); excess simply frees.
/// (Halved from 8 in the RSS audit: 8 pooled 320px rasterizers+planes are
/// ~6 MB of mostly-idle buffers.)
const SCRATCH_POOL_CAP: usize = 4;

impl RenderScratch {
    fn take_raster(&mut self, w: usize, h: usize) -> Rasterizer {
        match self.rasters.pop() {
            Some(mut r) => {
                r.reshape(w, h);
                r
            }
            None => Rasterizer::new(w, h),
        }
    }

    fn put_raster(&mut self, mut r: Rasterizer) {
        if self.rasters.len() < SCRATCH_POOL_CAP {
            r.reset();
            self.rasters.push(r);
        }
    }

    fn take_cells(&mut self, w: usize, h: usize) -> CellRaster {
        match self.cells_pool.pop() {
            Some(mut c) => {
                c.reshape(w, h);
                c
            }
            None => CellRaster::new(w, h),
        }
    }

    fn put_cells(&mut self, mut c: CellRaster) {
        if self.cells_pool.len() < SCRATCH_POOL_CAP {
            c.reset();
            self.cells_pool.push(c);
        }
    }

    fn take_u32(&mut self, n: usize) -> Vec<u32> {
        px_stat(6, n);
        px_stat(11, 1);
        let mut b = self.bufs_u32.pop().unwrap_or_default();
        b.clear();
        b.resize(n, 0);
        b
    }

    fn put_u32(&mut self, b: Vec<u32>) {
        if self.bufs_u32.len() < SCRATCH_POOL_CAP {
            self.bufs_u32.push(b);
        }
    }

    /// Returns a length-`n` u8 buffer WITHOUT a full-length fill: contents are
    /// unspecified (stale pool bytes) except that a freshly grown tail is
    /// zeroed. The only caller ([`RenderCtx::build_mask`]) reads just a bounded
    /// sub-rectangle (the offscreen dirty box) and seeds exactly that region
    /// itself via [`fill_rows_u8`], so skipping the O(n) fill is the whole
    /// point — in steady state the pooled buffer is already `n` long and no
    /// fill happens at all.
    fn take_u8_uninit(&mut self, n: usize) -> Vec<u8> {
        let mut b = self.bufs_u8.pop().unwrap_or_default();
        if b.len() != n {
            b.clear();
            b.resize(n, 0);
        }
        b
    }

    fn put_u8(&mut self, b: Vec<u8>) {
        if self.bufs_u8.len() < SCRATCH_POOL_CAP {
            self.bufs_u8.push(b);
        }
    }

    fn put_pts(&mut self, mut v: Vec<Vec2>) {
        if self.pts_pool.len() < PTS_POOL_CAP {
            v.clear();
            self.pts_pool.push(v);
        }
    }

    /// Returns the LUT plus a 64-bit id of its exact inputs (used in the
    /// gradient source-plane cache key).
    fn lut_for(
        &mut self,
        stops: &crate::model::FloatList,
        color_count: usize,
        opacity: f32,
    ) -> (std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>, u64) {
        self.lut_key.clear();
        self.lut_key.reserve(stops.0.len() + 2);
        self.lut_key.push(color_count as u32);
        self.lut_key.push(opacity.to_bits());
        for v in &stops.0 {
            self.lut_key.push(v.to_bits());
        }
        let mut h = Hasher128::new();
        for &w in &self.lut_key {
            h.mix(w);
        }
        let id = h.finish() as u64;
        if let Some(lut) = self.lut_cache.get(self.lut_key.as_slice()) {
            return (lut.clone(), id);
        }
        let lut: std::sync::Arc<[u32; GRADIENT_LUT_SIZE]> =
            std::sync::Arc::new(build_gradient_lut(stops, color_count, opacity));
        if self.lut_cache.len() >= LUT_CACHE_CAP {
            self.lut_cache.clear();
        }
        self.lut_cache.insert(self.lut_key.clone(), lut.clone());
        (lut, id)
    }
}

impl Composition {
    /// Renders frame `frame_index` (0-based, relative to the playable range)
    /// into `pixels`: row-major, premultiplied ARGB32 (0xAARRGGBB words),
    /// `width * height` entries. The buffer is fully overwritten (cleared to
    /// transparent first).
    ///
    /// One-off convenience: allocates its working buffers per call. Hosts
    /// rendering repeatedly should hold an [`crate::Animation`] instance,
    /// which reuses them across frames.
    pub fn render(
        &self,
        frame_index: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.render_with_options(
            frame_index,
            pixels,
            width,
            height,
            crate::RenderOptions::default(),
        )
    }

    /// One-off rendering with explicit [`crate::RenderOptions`]. Hosts that
    /// render repeatedly should prefer [`crate::Animation::render_with_options`]
    /// so working buffers persist across frames.
    pub fn render_with_options(
        &self,
        frame_index: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
        options: crate::RenderOptions,
    ) -> Result<()> {
        let mut scratch = RenderScratch::default();
        self.render_pooled(&mut scratch, frame_index, pixels, width, height, options)
    }

    /// The timeline is continuous: fractional `frame_index` renders an exact
    /// in-between pose (120 Hz playback of 60 fps files); integer positions
    /// match the u32 API bit-for-bit. (Precomp child time still quantizes to
    /// integers per rlottie parity.)
    pub(crate) fn render_pooled(
        &self,
        scratch: &mut RenderScratch,
        frame_index: f32,
        pixels: &mut [u32],
        width: u32,
        height: u32,
        options: crate::RenderOptions,
    ) -> Result<()> {
        let limits = Limits::default();
        if width == 0
            || height == 0
            || width > limits.max_dimension
            || height > limits.max_dimension
        {
            return Err(Error::InvalidLottie {
                offset: 0,
                what: "render size out of range",
            });
        }
        let need = (width as usize).saturating_mul(height as usize);
        let Some(buf) = pixels.get_mut(..need) else {
            return Err(Error::InvalidLottie {
                offset: 0,
                what: "pixel buffer too small",
            });
        };
        buf.fill(0);

        let max_frame = self.frame_count().saturating_sub(1) as f32;
        let frame_in_range = if frame_index.is_finite() {
            frame_index.clamp(0.0, max_frame)
        } else {
            0.0
        };
        let frame = self.in_point + frame_in_range;

        let base = Mat2x3::scale(
            width as f32 / self.width.max(1) as f32,
            height as f32 / self.height.max(1) as f32,
        );

        scratch
            .cov_cache
            .set_budget_for_canvas(width as usize, height as usize);
        scratch.cov_cache.frame_tick();
        let raster = scratch.take_raster(width as usize, height as usize);
        let cells = scratch.take_cells(width as usize, height as usize);
        let mut canvas = Canvas::with_raster(
            buf,
            width as usize,
            height as usize,
            raster,
            cells,
            options.antialias,
        );
        let ctx = RenderCtx {
            comp: self,
            continuous: frame_in_range.fract() != 0.0,
            antialias: options.antialias,
            curve_tolerance: options.curve_tolerance,
        };
        let res = ctx.render_layers(
            scratch,
            &mut canvas,
            &self.layers,
            base,
            frame,
            1.0,
            &Vec::new(),
            0,
        );
        scratch.put_raster(canvas.raster);
        scratch.put_cells(canvas.cells);
        res
    }
}

#[doc(hidden)]
pub mod vulkan {
    //! Unstable evaluated-shape access for the experimental Vulkan renderer.
    //!
    //! The data here is intentionally narrow: it exposes one evaluated frame's
    //! ordered shape paint ranges and device-space contours without committing
    //! tlottie to a general rendering backend API.

    use super::*;

    /// Device-space point.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Point {
        /// X coordinate in target pixels.
        pub x: f32,
        /// Y coordinate in target pixels.
        pub y: f32,
    }

    /// Fill rule for a vector paint.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Rule {
        /// Non-zero winding rule.
        NonZero,
        /// Even-odd winding rule.
        EvenOdd,
    }

    /// Premultiplied solid color paint metadata.
    #[derive(Clone, Copy, Debug)]
    pub struct SolidPaint {
        /// Fill rule used by this paint.
        pub rule: Rule,
        /// Premultiplied ARGB color after paint opacity has been folded in.
        pub argb: u32,
    }

    /// Device-to-gradient-local affine transform.
    #[derive(Clone, Copy, Debug)]
    pub struct GradientTransform {
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
        pub tx: f32,
        pub ty: f32,
    }

    /// Evaluated gradient coordinate parameters.
    #[derive(Clone, Copy, Debug)]
    pub enum GradientKind {
        Linear {
            sx: f32,
            sy: f32,
            dx: f32,
            dy: f32,
            inv_len_sq: f32,
        },
        Radial {
            sx: f32,
            sy: f32,
            inv_r: f32,
        },
        Focal {
            fx: f32,
            fy: f32,
            dx: f32,
            dy: f32,
            a: f32,
            r: f32,
        },
    }

    /// Premultiplied gradient LUT and evaluated coordinate map.
    #[derive(Clone, Debug)]
    pub struct GradientPaint {
        pub rule: Rule,
        pub lut: std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
        pub transform: GradientTransform,
        pub kind: GradientKind,
    }

    /// Paint metadata for the initial Vulkan fill baseline.
    #[derive(Clone, Debug)]
    pub enum Paint {
        /// Solid fill or expanded solid stroke.
        Solid(SolidPaint),
        /// Gradient fill or expanded gradient stroke. The geometry is exposed,
        /// but gradient shader data is not public yet.
        Gradient(GradientPaint),
        /// Starts a pixel-local isolated layer.
        BeginLayer,
        /// Composites the isolated layer once at the evaluated opacity.
        EndLayer { opacity: u8 },
        /// Starts a pixel-local matte source.
        BeginMatte,
        /// Saves the matte source and starts its target layer.
        BeginMatteTarget,
        /// Applies the matte and composites the target once.
        EndMatte { kind: u8, opacity: u8 },
    }

    /// One closed or open contour in device space.
    #[derive(Clone, Debug, Default)]
    pub struct WalkedContour {
        /// Points forming this contour.
        pub points: Vec<Point>,
        /// Whether the source contour is closed.
        pub closed: bool,
    }

    /// One paint operation. Jobs are returned in draw order, bottom to top.
    #[derive(Clone, Debug)]
    pub struct WalkedPaint {
        /// Paint metadata.
        pub paint: Paint,
        /// Index into [`WalkedFrame::contours`].
        pub start: usize,
        /// Exclusive end index into [`WalkedFrame::contours`].
        pub end: usize,
    }

    /// Evaluated shape data for a frame.
    #[derive(Clone, Debug, Default)]
    pub struct WalkedFrame {
        /// Contour arena. Paints reference ranges inside this array.
        pub contours: Vec<WalkedContour>,
        /// Ordered paint operations.
        pub paints: Vec<WalkedPaint>,
    }

    /// Evaluates visible shape content for one frame into Vulkan-consumable
    /// device-space geometry.
    ///
    /// Masks remain renderer work for a later phase. Solid fills, strokes,
    /// gradients, isolated layers, and matte commands preserve draw order.
    pub fn walk_frame(
        comp: &Composition,
        frame_index: f32,
        width: u32,
        height: u32,
        options: crate::RenderOptions,
    ) -> Result<WalkedFrame> {
        let limits = Limits::default();
        if width == 0
            || height == 0
            || width > limits.max_dimension
            || height > limits.max_dimension
        {
            return Err(Error::InvalidLottie {
                offset: 0,
                what: "render size out of range",
            });
        }

        let max_frame = comp.frame_count().saturating_sub(1) as f32;
        let frame_in_range = if frame_index.is_finite() {
            frame_index.clamp(0.0, max_frame)
        } else {
            0.0
        };
        let frame = comp.in_point + frame_in_range;
        let base = Mat2x3::scale(
            width as f32 / comp.width.max(1) as f32,
            height as f32 / comp.height.max(1) as f32,
        );

        let mut pixels = vec![0; (width as usize).saturating_mul(height as usize)];
        let mut scratch = RenderScratch::default();
        scratch
            .cov_cache
            .set_budget_for_canvas(width as usize, height as usize);
        let raster = scratch.take_raster(width as usize, height as usize);
        let cells = scratch.take_cells(width as usize, height as usize);
        let mut canvas = Canvas::with_raster(
            &mut pixels,
            width as usize,
            height as usize,
            raster,
            cells,
            options.antialias,
        );
        let ctx = RenderCtx {
            comp,
            continuous: frame_in_range.fract() != 0.0,
            antialias: options.antialias,
            curve_tolerance: options.curve_tolerance,
        };
        let mut out = WalkedFrame::default();
        let res = ctx.collect_layers(
            &mut scratch,
            &mut canvas,
            &comp.layers,
            base,
            frame,
            1.0,
            &Vec::new(),
            0,
            &mut out,
        );
        scratch.put_raster(canvas.raster);
        scratch.put_cells(canvas.cells);
        res?;
        Ok(out)
    }

    fn rule_of(rule: FillRule) -> Rule {
        match rule {
            FillRule::NonZero => Rule::NonZero,
            FillRule::EvenOdd => Rule::EvenOdd,
        }
    }

    fn premul_argb(color: Color, opacity: f32) -> u32 {
        let a = (color.a * opacity).clamp(0.0, 1.0);
        let scale = a;
        let ai = (a * 255.0 + 0.5) as u32;
        let ri = (color.r * scale * 255.0 + 0.5) as u32;
        let gi = (color.g * scale * 255.0 + 0.5) as u32;
        let bi = (color.b * scale * 255.0 + 0.5) as u32;
        (ai.min(255) << 24) | (ri.min(255) << 16) | (gi.min(255) << 8) | bi.min(255)
    }

    fn push_contours(out: &mut WalkedFrame, contours: &[Contour], closed: bool) -> (usize, usize) {
        let start = out.contours.len();
        for contour in contours {
            out.contours.push(WalkedContour {
                points: contour
                    .points
                    .iter()
                    .map(|p| Point { x: p.x, y: p.y })
                    .collect(),
                closed,
            });
        }
        (start, out.contours.len())
    }

    impl RenderCtx<'_> {
        #[allow(clippy::too_many_arguments)]
        fn collect_layers(
            &self,
            scratch: &mut RenderScratch,
            canvas: &mut Canvas<'_>,
            layers: &[Layer],
            base: Mat2x3,
            frame: f32,
            opacity: f32,
            clip: &ClipQuad,
            precomp_depth: usize,
            out: &mut WalkedFrame,
        ) -> Result<()> {
            if precomp_depth > MAX_PRECOMP_DEPTH {
                return Ok(());
            }
            let mut consumed_as_matte = vec![false; layers.len()];
            for (i, l) in layers.iter().enumerate() {
                if l.matte.is_some() {
                    if let Some(slot) = i.checked_sub(1).and_then(|j| consumed_as_matte.get_mut(j))
                    {
                        *slot = true;
                    }
                }
            }
            for (idx, layer) in layers.iter().enumerate().rev() {
                if consumed_as_matte.get(idx).copied().unwrap_or(false)
                    || layer.matte_src
                    || !self.layer_visible(layer, frame)
                {
                    continue;
                }
                if layer.matte.is_some() {
                    if let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) {
                        if !self.layer_visible(src, frame) {
                            continue;
                        }
                    }
                }
                let (layer_m, layer_opacity) = layer_transform_at(layer, frame);
                let m = base
                    .concat(parent_chain_matrix(layers, layer, frame))
                    .concat(layer_m);
                let combined_opacity = opacity * layer_opacity;
                let group_opacity = opacity_byte(combined_opacity);
                if group_opacity == 0 {
                    continue;
                }
                if let Some(kind) = layer.matte {
                    let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) else {
                        continue;
                    };
                    let at = out.contours.len();
                    out.paints.push(WalkedPaint {
                        paint: Paint::BeginMatte,
                        start: at,
                        end: at,
                    });
                    let (src_m, src_opacity) = layer_transform_at(src, frame);
                    let source_matrix = base
                        .concat(parent_chain_matrix(layers, src, frame))
                        .concat(src_m);
                    self.collect_layer_content(
                        scratch,
                        canvas,
                        src,
                        source_matrix,
                        frame,
                        src_opacity,
                        clip,
                        precomp_depth,
                        out,
                    )?;
                    let at = out.contours.len();
                    out.paints.push(WalkedPaint {
                        paint: Paint::BeginMatteTarget,
                        start: at,
                        end: at,
                    });
                    self.collect_layer_content(
                        scratch,
                        canvas,
                        layer,
                        m,
                        frame,
                        1.0,
                        clip,
                        precomp_depth,
                        out,
                    )?;
                    let at = out.contours.len();
                    out.paints.push(WalkedPaint {
                        paint: Paint::EndMatte {
                            kind,
                            opacity: group_opacity as u8,
                        },
                        start: at,
                        end: at,
                    });
                    continue;
                }
                let complex_precomp = if layer.kind == LayerKind::Precomp {
                    layer
                        .ref_id
                        .as_deref()
                        .and_then(|ref_id| self.comp.assets.iter().find(|asset| asset.id == ref_id))
                        .is_some_and(|asset| asset.layers.len() > 1)
                } else {
                    false
                };
                let isolate = group_opacity < 255 && complex_precomp;
                if isolate {
                    let at = out.contours.len();
                    out.paints.push(WalkedPaint {
                        paint: Paint::BeginLayer,
                        start: at,
                        end: at,
                    });
                }
                self.collect_layer_content(
                    scratch,
                    canvas,
                    layer,
                    m,
                    frame,
                    if isolate { 1.0 } else { combined_opacity },
                    clip,
                    precomp_depth,
                    out,
                )?;
                if isolate {
                    let at = out.contours.len();
                    out.paints.push(WalkedPaint {
                        paint: Paint::EndLayer {
                            opacity: group_opacity as u8,
                        },
                        start: at,
                        end: at,
                    });
                }
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn collect_layer_content(
            &self,
            scratch: &mut RenderScratch,
            canvas: &mut Canvas<'_>,
            layer: &Layer,
            m: Mat2x3,
            frame: f32,
            content_opacity: f32,
            clip: &ClipQuad,
            precomp_depth: usize,
            out: &mut WalkedFrame,
        ) -> Result<()> {
            if opacity_byte(content_opacity) == 0 {
                return Ok(());
            }
            match layer.kind {
                LayerKind::Shape => {
                    let mut walker = ShapeWalker {
                        canvas,
                        scratch,
                        frame,
                        clip,
                        curve_tolerance: self.curve_tolerance,
                    };
                    let (arena, pending) =
                        walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
                    walker.collect_shape_jobs(&arena, &pending, out);
                }
                LayerKind::Solid => {
                    if let Some((sw, sh, color)) = layer.solid {
                        let contour = rect_contour(
                            Vec2::new(sw * 0.5, sh * 0.5),
                            Vec2::new(sw, sh),
                            0.0,
                            false,
                            &m,
                            self.curve_tolerance,
                        );
                        let (start, end) = push_contours(out, &[contour], true);
                        out.paints.push(WalkedPaint {
                            paint: Paint::Solid(SolidPaint {
                                rule: Rule::NonZero,
                                argb: premul_argb(color, content_opacity),
                            }),
                            start,
                            end,
                        });
                    }
                }
                LayerKind::Precomp => {
                    let Some(ref_id) = layer.ref_id.as_deref() else {
                        return Ok(());
                    };
                    let Some(asset) = self.comp.assets.iter().find(|a| a.id == ref_id) else {
                        return Ok(());
                    };
                    let mut child_clip: ClipQuad = clip.clone();
                    if let Some((w, h)) = layer.precomp_size {
                        child_clip.push([
                            m.apply(Vec2::new(0.0, 0.0)),
                            m.apply(Vec2::new(w, 0.0)),
                            m.apply(Vec2::new(w, h)),
                            m.apply(Vec2::new(0.0, h)),
                        ]);
                    }
                    let sr = if layer.time_stretch.abs() > 1e-6 {
                        layer.time_stretch
                    } else {
                        1.0
                    };
                    let quant = |v: f32| if self.continuous { v } else { v.trunc() };
                    let child_frame = match &layer.time_remap {
                        Some(tm) => {
                            let dur = (self.comp.out_point - self.comp.in_point - 1.0).max(0.0);
                            let fr = self.comp.frame_rate.max(1e-6);
                            let pos = if dur > 0.0 {
                                (tm.eval(frame) * fr / dur).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            quant(pos * dur / sr)
                        }
                        None => quant((frame - layer.start_time) / sr),
                    };
                    self.collect_layers(
                        scratch,
                        canvas,
                        &asset.layers,
                        m,
                        child_frame,
                        content_opacity,
                        &child_clip,
                        precomp_depth + 1,
                        out,
                    )?;
                }
                LayerKind::Null | LayerKind::Other(_) => {}
            }
            Ok(())
        }
    }

    impl ShapeWalker<'_, '_> {
        fn collect_shape_jobs(
            &mut self,
            arena: &[(Contour, bool)],
            pending: &[PendingJob],
            out: &mut WalkedFrame,
        ) {
            for pj in pending.iter().rev() {
                match self.materialize(pj, arena) {
                    DrawJob::Solid {
                        contours,
                        rule,
                        color,
                        opacity,
                        ..
                    } => {
                        let (start, end) = push_contours(out, &contours, true);
                        out.paints.push(WalkedPaint {
                            paint: Paint::Solid(SolidPaint {
                                rule: rule_of(rule),
                                argb: premul_argb(color, opacity),
                            }),
                            start,
                            end,
                        });
                        for c in contours {
                            self.scratch.put_pts(c.points);
                        }
                    }
                    DrawJob::Gradient {
                        contours,
                        rule,
                        lut,
                        map,
                        ..
                    } => {
                        let (start, end) = push_contours(out, &contours, true);
                        out.paints.push(WalkedPaint {
                            paint: Paint::Gradient(GradientPaint {
                                rule: rule_of(rule),
                                lut,
                                transform: GradientTransform {
                                    a: map.inv.a,
                                    b: map.inv.b,
                                    c: map.inv.c,
                                    d: map.inv.d,
                                    tx: map.inv.tx,
                                    ty: map.inv.ty,
                                },
                                kind: match map.kind {
                                    super::GradientMapKind::Linear {
                                        sx,
                                        sy,
                                        dx,
                                        dy,
                                        inv_len_sq,
                                    } => GradientKind::Linear {
                                        sx,
                                        sy,
                                        dx,
                                        dy,
                                        inv_len_sq,
                                    },
                                    super::GradientMapKind::Radial { sx, sy, inv_r } => {
                                        GradientKind::Radial { sx, sy, inv_r }
                                    }
                                    super::GradientMapKind::Focal {
                                        fx,
                                        fy,
                                        dx,
                                        dy,
                                        a,
                                        r,
                                    } => GradientKind::Focal {
                                        fx,
                                        fy,
                                        dx,
                                        dy,
                                        a,
                                        r,
                                    },
                                },
                            }),
                            start,
                            end,
                        });
                        for c in contours {
                            self.scratch.put_pts(c.points);
                        }
                    }
                }
            }
        }
    }
}

/// Stack of convex clip quads (device-space precomp viewports). Nested
/// precomps INTERSECT their viewports (rlottie: `mask = clipper.rle() &
/// mask`, lottieitem.cpp renderHelper) — applying convex clips in sequence
/// is exactly their intersection. Empty = no precomp clip.
type ClipQuad = Vec<[Vec2; 4]>;

struct RenderCtx<'a> {
    comp: &'a Composition,
    /// True when the requested root frame is fractional. Integer frames keep
    /// rlottie's integer-truncated precomp child time (bit-exact parity);
    /// fractional frames evaluate the whole tree continuously — quantizing
    /// precomp interiors would defeat in-between rendering.
    continuous: bool,
    /// Whether edge coverage remains fractional or is thresholded to binary.
    antialias: bool,
    /// Maximum device-space error used while flattening cubic curves.
    curve_tolerance: f32,
}

impl RenderCtx<'_> {
    /// Renders a layer list (root composition or precomp asset) bottom-up.
    /// `frame` is in this list's local time; `base` maps this list's
    /// coordinate space to the device.
    #[allow(clippy::too_many_arguments)]
    fn render_layers(
        &self,
        scratch: &mut RenderScratch,
        canvas: &mut Canvas<'_>,
        layers: &[Layer],
        base: Mat2x3,
        frame: f32,
        opacity: f32,
        clip: &ClipQuad,
        precomp_depth: usize,
    ) -> Result<()> {
        if precomp_depth > MAX_PRECOMP_DEPTH {
            return Ok(()); // over-deep nesting is dropped, not fatal
        }
        // A layer with `tt` is matted by the layer directly above it in file
        // order; that source layer is not drawn on its own.
        let mut consumed_as_matte = vec![false; layers.len()];
        for (i, l) in layers.iter().enumerate() {
            if l.matte.is_some() {
                if let Some(slot) = i.checked_sub(1).and_then(|j| consumed_as_matte.get_mut(j)) {
                    *slot = true;
                }
            }
        }
        for (idx, layer) in layers.iter().enumerate().rev() {
            if consumed_as_matte.get(idx).copied().unwrap_or(false) {
                continue;
            }
            if layer.matte_src {
                continue; // matte-only layer without a consumer right below
            }
            if !self.layer_visible(layer, frame) {
                continue;
            }
            // Patched rlottie renders a matte consumer ONLY while its matte
            // source layer is itself visible (`if (matte->visible())
            // renderMatteLayer(...)` with no else) — a source whose lifetime
            // ends early takes the consumer with it (JollySanta lollipop,
            // source op one frame before comp op).
            if layer.matte.is_some() {
                if let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) {
                    if !self.layer_visible(src, frame) {
                        continue;
                    }
                }
            }
            let (layer_m, layer_opacity) = layer_transform_at(layer, frame);
            let m = base
                .concat(parent_chain_matrix(layers, layer, frame))
                .concat(layer_m);
            let combined_opacity = opacity * layer_opacity;
            let k = opacity_byte(combined_opacity);
            if k == 0 {
                continue;
            }

            // Opacity alone forces an offscreen composite ONLY for precomp
            // layers whose asset has ≥2 sublayers (rlottie complexContent,
            // lottieitem.cpp:606-627). Shape layers FOLD opacity into each
            // paint — overlapping paints double-composite, matching the
            // reference (verified: overlap alpha 190 vs offscreen 128).
            let complex_precomp = layer_opacity < 0.999
                && matches!(layer.kind, LayerKind::Precomp)
                && layer
                    .ref_id
                    .as_deref()
                    .and_then(|id| self.comp.assets.iter().find(|a| a.id == id))
                    .map(|a| a.layers.len() > 1)
                    .unwrap_or(false);
            let needs_offscreen =
                !layer.masks.is_empty() || layer.matte.is_some() || complex_precomp;
            if !needs_offscreen {
                self.draw_layer_content(
                    scratch,
                    canvas,
                    layer,
                    m,
                    frame,
                    combined_opacity,
                    clip,
                    precomp_depth,
                )?;
                continue;
            }

            // Offscreen path: render content at full opacity, modulate by
            // masks and matte, then composite with the layer opacity.
            // All buffers come from (and return to) the scratch pools.
            let (w, h) = (canvas.w, canvas.h);
            let mut buf_a = scratch.take_u32(w * h);
            let da; // content bounds of buf_a: outside it everything is 0
            {
                let raster = scratch.take_raster(w, h);
                let cells = scratch.take_cells(w, h);
                let mut off = Canvas::with_raster(&mut buf_a, w, h, raster, cells, self.antialias);
                let res = self.draw_layer_content(
                    scratch,
                    &mut off,
                    layer,
                    m,
                    frame,
                    1.0,
                    clip,
                    precomp_depth,
                );
                da = off.dirty;
                scratch.put_raster(off.raster);
                scratch.put_cells(off.cells);
                res?;
            }
            // Every following pass is a per-pixel function that maps 0 → 0
            // on the destination (mask multiply, matte multiply, source-
            // over of 0), so bounding all of them to buf_a's dirty box is
            // exact — and measured offscreens are often EMPTY (skip all).
            if !da.is_empty() {
                if !layer.masks.is_empty() {
                    // `da` is exactly the region the modulate below reads and
                    // the only region where buf_a is nonzero — bound the mask
                    // build to it (byte-exact; see build_mask).
                    let maskbuf = self.build_mask(scratch, layer, m, frame, w, h, da);
                    for_rows_boxed(&mut buf_a, w, da, |y, row| {
                        let lo = y * w + da.x0;
                        if let Some(mask_row) = maskbuf.get(lo..lo + row.len()) {
                            px_stat(10, row.len());
                            modulate(row, mask_row);
                        }
                    });
                    scratch.put_u8(maskbuf);
                }
                if layer.matte.is_some() {
                    if let Some(src) = idx.checked_sub(1).and_then(|j| layers.get(j)) {
                        let mut buf_b = scratch.take_u32(w * h);
                        if self.layer_visible(src, frame) {
                            let (src_m, src_op) = layer_transform_at(src, frame);
                            let sm = base
                                .concat(parent_chain_matrix(layers, src, frame))
                                .concat(src_m);
                            let raster = scratch.take_raster(w, h);
                            let cells = scratch.take_cells(w, h);
                            let mut off = Canvas::with_raster(
                                &mut buf_b,
                                w,
                                h,
                                raster,
                                cells,
                                self.antialias,
                            );
                            let res = self.draw_layer_content(
                                scratch,
                                &mut off,
                                src,
                                sm,
                                frame,
                                src_op,
                                clip,
                                precomp_depth,
                            );
                            // Source content bounds: buf_b is 0 outside `db`,
                            // where the mask modulate below is a no-op.
                            let db = off.dirty;
                            scratch.put_raster(off.raster);
                            scratch.put_cells(off.cells);
                            res?;
                            if !src.masks.is_empty() && !db.is_empty() {
                                let maskbuf = self.build_mask(scratch, src, sm, frame, w, h, db);
                                // Bound the modulate to `db` too — outside it
                                // buf_b is 0 (modulate maps 0 → 0), so this is
                                // byte-exact vs the former full-plane modulate.
                                for_rows_boxed(&mut buf_b, w, db, |y, row| {
                                    let lo = y * w + db.x0;
                                    if let Some(mask_row) = maskbuf.get(lo..lo + row.len()) {
                                        modulate(row, mask_row);
                                    }
                                });
                                scratch.put_u8(maskbuf);
                            }
                        }
                        let kind = layer.matte.unwrap_or(1);
                        for_rows_boxed(&mut buf_a, w, da, |y, row| {
                            let lo = y * w + da.x0;
                            if let Some(src_row) = buf_b.get(lo..lo + row.len()) {
                                apply_matte(row, src_row, kind);
                            }
                        });
                        scratch.put_u32(buf_b);
                    }
                }
                canvas.dirty.union(da);
                for_rows_boxed(canvas.pixels, w, da, |y, row| {
                    let lo = y * w + da.x0;
                    if let Some(src_row) = buf_a.get(lo..lo + row.len()) {
                        px_stat(7, row.len());
                        crate::simd::composite_over_span(row, src_row, k);
                    }
                });
            }
            scratch.put_u32(buf_a);
        }
        Ok(())
    }

    fn layer_visible(&self, layer: &Layer, frame: f32) -> bool {
        // Half-open lifetime [ip, op): the out-point frame is NOT drawn
        // (patched rlottie lottieitem.cpp LOTLayerItem::visible uses
        // `frameNo() < outFrame()`). Five independent BROKEN clusters were
        // single-frame spikes at exactly frame == op.
        !layer.hidden && frame >= layer.in_point && frame < layer.out_point
    }

    /// Draws one layer's content (shape tree / precomp / solid) into `canvas`.
    #[allow(clippy::too_many_arguments)]
    fn draw_layer_content(
        &self,
        scratch: &mut RenderScratch,
        canvas: &mut Canvas<'_>,
        layer: &Layer,
        m: Mat2x3,
        frame: f32,
        content_opacity: f32,
        clip: &ClipQuad,
        precomp_depth: usize,
    ) -> Result<()> {
        if opacity_byte(content_opacity) == 0 {
            return Ok(());
        }
        match layer.kind {
            LayerKind::Shape => {
                let mut walker = ShapeWalker {
                    canvas,
                    scratch,
                    frame,
                    clip,
                    curve_tolerance: self.curve_tolerance,
                };
                let lp = layer as *const Layer as usize;
                let is_static = *walker
                    .scratch
                    .static_flags
                    .entry(lp)
                    .or_insert_with(|| crate::model::shapes_static(&layer.shapes));
                if is_static {
                    // Static shape tree: the job list is a deterministic
                    // function of (device matrix, folded opacity, clip) —
                    // replay it without evaluating/flattening anything.
                    let mut h = walker.clip_sig();
                    h.mix(4); // replay-key tag
                    for v in [m.a, m.b, m.c, m.d, m.tx, m.ty] {
                        h.mix(v.to_bits());
                    }
                    h.mix(content_opacity.to_bits());
                    // Via u64: usize is 32-bit on wasm32, where `>> 32`
                    // on the pointer itself would overflow.
                    h.mix(lp as u64 as u32);
                    h.mix(((lp as u64) >> 32) as u32);
                    let rkey = h.finish();
                    if let Some(jobs) = walker.scratch.jobs_cache.remove(&rkey) {
                        let ok = walker.replay_jobs(&jobs);
                        walker.scratch.jobs_cache.insert(rkey, jobs);
                        if ok {
                            return Ok(());
                        }
                    }
                    let admit_now = !self.continuous && self.comp.frame_count() <= 1;
                    if admit_now || walker.scratch.jobs_seen.contains(&rkey) {
                        let mut rec = Vec::new();
                        let (arena, pending) =
                            walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
                        walker.render_shape_jobs_cpu(&arena, &pending, Some(&mut rec));
                        if walker.scratch.jobs_cache.len() >= JOBS_CACHE_CAP {
                            walker.scratch.jobs_cache.clear();
                        }
                        walker.scratch.jobs_cache.insert(rkey, rec);
                    } else {
                        if walker.scratch.jobs_seen.len() >= 4 * JOBS_CACHE_CAP {
                            walker.scratch.jobs_seen.clear();
                        }
                        walker.scratch.jobs_seen.insert(rkey);
                        let (arena, pending) =
                            walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
                        walker.render_shape_jobs_cpu(&arena, &pending, None);
                    }
                } else {
                    let (arena, pending) =
                        walker.walk_shapes(&layer.shapes, m, content_opacity, 0)?;
                    walker.render_shape_jobs_cpu(&arena, &pending, None);
                }
            }
            LayerKind::Solid => {
                if let Some((sw, sh, color)) = layer.solid {
                    let contour = rect_contour(
                        Vec2::new(sw * 0.5, sh * 0.5),
                        Vec2::new(sw, sh),
                        0.0,
                        false,
                        &m,
                        self.curve_tolerance,
                    );
                    let walker = ShapeWalker {
                        canvas,
                        scratch,
                        frame,
                        clip,
                        curve_tolerance: self.curve_tolerance,
                    };
                    let key = walker.fill_key(
                        core::slice::from_ref(&(contour.clone(), true)),
                        FillRule::NonZero,
                    );
                    let contours: Vec<Contour> = if walker.scratch.cov_cache.contains(key) {
                        Vec::new()
                    } else {
                        vec![walker.clip_all(&contour)]
                    };
                    walker.canvas.fill(
                        &mut walker.scratch.cov_cache,
                        key,
                        &contours,
                        FillRule::NonZero,
                        color,
                        content_opacity,
                    );
                }
            }
            LayerKind::Precomp => {
                let Some(ref_id) = layer.ref_id.as_deref() else {
                    return Ok(());
                };
                let Some(asset) = self.comp.assets.iter().find(|a| a.id == ref_id) else {
                    return Ok(());
                };
                let mut child_clip: ClipQuad = clip.clone();
                if let Some((w, h)) = layer.precomp_size {
                    child_clip.push([
                        m.apply(Vec2::new(0.0, 0.0)),
                        m.apply(Vec2::new(w, 0.0)),
                        m.apply(Vec2::new(w, h)),
                        m.apply(Vec2::new(0.0, h)),
                    ]);
                }
                // rlottie evaluates precomp children at INTEGER frames:
                // LOTLayerData::timeRemap returns int (lottiemodel.h), so
                // both branches truncate toward zero. The remap branch maps
                // tm's seconds through frameAtPos — pos clamped to [0,1]
                // over frameDuration = op − ip − 1 — and divides by the
                // time stretch (which tlottie previously omitted there).
                let sr = if layer.time_stretch.abs() > 1e-6 {
                    layer.time_stretch
                } else {
                    1.0
                };
                let quant = |v: f32| if self.continuous { v } else { v.trunc() };
                let child_frame = match &layer.time_remap {
                    Some(tm) => {
                        let dur = (self.comp.out_point - self.comp.in_point - 1.0).max(0.0);
                        let fr = self.comp.frame_rate.max(1e-6);
                        let pos = if dur > 0.0 {
                            (tm.eval(frame) * fr / dur).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        quant(pos * dur / sr)
                    }
                    None => quant((frame - layer.start_time) / sr),
                };
                self.render_layers(
                    scratch,
                    canvas,
                    &asset.layers,
                    m,
                    child_frame,
                    content_opacity,
                    &child_clip,
                    precomp_depth + 1,
                )?;
            }
            LayerKind::Null | LayerKind::Other(_) => {}
        }
        Ok(())
    }

    /// Rasterizes a layer's mask stack into a full-canvas u8 buffer, but only
    /// the `bound` sub-rectangle is computed exactly — every pixel OUTSIDE
    /// `bound` is left with unspecified (stale) contents. The sole callers read
    /// the mask only inside their offscreen's dirty box (`da` for a layer's own
    /// masks, `db` for a matte source's), which they pass as `bound`; the
    /// offscreen is transparent (0) everywhere outside that box, and both
    /// consumers (mask `modulate`, matte `modulate`) map dst 0 → 0, so a mask
    /// value there can never change a pixel. Bounding the three former
    /// full-plane passes (acc init-fill, per-mask `tmp` clear, per-mask
    /// accumulate) to `bound` is therefore byte-exact.
    ///
    /// Mode mapping follows the patched rlottie parser (lottieparser.cpp):
    /// 'a' Add, 's' Subtract, 'i' Intersect, 'f' Difference; 'd', 'l', 'n' and
    /// anything else are None — the mask contributes nothing, and a layer whose
    /// masks are all None draws nothing (empty maskRle).
    ///
    /// Per-mode "outside" value — the mask at a pixel inside `bound` but
    /// OUTSIDE every mask geometry (coverage t=0 for all masks). This is why we
    /// bound to `bound` (the dirty box) and NOT to the mask geometry bbox: the
    /// outside value is generally nonzero and mode-dependent, and the bounded
    /// accumulate reproduces it exactly because t=0 flows through the unchanged
    /// per-pixel body. With `c = round((invert? 255−t : t)·op/255)`, at t=0:
    /// non-inverted → c=0; inverted → c=op (= round(255·op/255)). Folding one
    /// mask over the running `cur`:
    ///   'a' Add   → c + round((255−c)·cur/255)   t0 non-inv: cur   inv: op+…·cur
    ///   's' Sub   → round(cur·(255−c)/255)        t0 non-inv: cur   inv: cur·(255−op)
    ///   'i' Isect → round(cur·c/255)              t0 non-inv: 0     inv: cur·op
    ///   'f' Diff  → |cur − c|                     t0 non-inv: cur   inv: |cur−op|
    /// `first_additive` seeds `cur`: 0 for a leading 'a'/'f' mask, 255 for a
    /// leading 's'/'i'. So a non-inverted intersect (or an all-add stack seeded
    /// at 0) collapses the outside toward 0, while any inverted mask leaves a
    /// nonzero outside — the outside region inside `bound` must be walked, not
    /// assumed. The bounded accumulate over `bound` does exactly that, pixel by
    /// pixel, identically to the former full-plane loop.
    #[allow(clippy::too_many_arguments)]
    fn build_mask(
        &self,
        scratch: &mut RenderScratch,
        layer: &Layer,
        m: Mat2x3,
        frame: f32,
        w: usize,
        h: usize,
        bound: DirtyBox,
    ) -> Vec<u8> {
        let effective = |mode: u8| matches!(mode, b'a' | b's' | b'i' | b'f');
        let first_additive = layer
            .masks
            .iter()
            .find(|mask| effective(mask.mode))
            .map(|mask| matches!(mask.mode, b'a' | b'f'))
            .unwrap_or(true); // all-None: acc stays 0 → layer hidden
                              // acc/tmp stay full-canvas length (indexed by y*w+x) but only `bound`
                              // is initialized/cleared/walked — see the note above. Outside `bound`
                              // they hold stale pool bytes that no consumer ever reads.
        let init = if first_additive { 0u8 } else { 255u8 };
        let mut acc: Vec<u8> = scratch.take_u8_uninit(w * h);
        fill_rows_u8(&mut acc, w, bound, init);
        let mut raster = scratch.take_raster(w, h);
        let mut cells = scratch.take_cells(w, h);
        let mut tmp: Vec<u8> = scratch.take_u8_uninit(w * h);
        for mask in &layer.masks {
            if !effective(mask.mode) {
                continue;
            }
            let opacity = (mask.opacity.eval(frame) / 100.0).clamp(0.0, 1.0);
            let data = mask.path.eval(frame);
            let contour = flatten_path(&data, &m, self.curve_tolerance);
            let clipped = clip_contour(&contour, w as f32, h as f32);
            // Clear only `bound`; the rasterizer sweep below overwrites the
            // mask's covered pixels, leaving `bound`-but-uncovered pixels at 0
            // (coverage t=0), exactly as the former `tmp.fill(0)` did there.
            fill_rows_u8(&mut tmp, w, bound, 0);
            if mode_s_wins(core::slice::from_ref(&clipped), w * h) {
                cells.reset();
                cells.fill_contours(core::slice::from_ref(&clipped));
                cells.sweep_spans(FillRule::NonZero, self.antialias, |y, x0, len, cov| {
                    let lo = y * w + x0;
                    if let Some(dst) = tmp.get_mut(lo..lo + len) {
                        px_stat(4, len);
                        dst.fill(cov);
                    }
                });
            } else {
                raster.reset();
                raster.fill_contours(core::slice::from_ref(&clipped));
                raster.sweep(FillRule::NonZero, self.antialias, |y, x0, cov_row| {
                    let lo = y * w + x0;
                    if let Some(dst) = tmp.get_mut(lo..lo + cov_row.len()) {
                        dst.copy_from_slice(cov_row);
                    }
                });
            }
            let op = (opacity * 255.0 + 0.5) as u32;
            // Full-plane accumulate, bounded to `bound`: the per-pixel body is
            // byte-identical to the former loop — only its iteration range
            // shrinks from w*h to the dirty box.
            for y in bound.y0..=bound.y1 {
                let lo = y * w + bound.x0;
                let hi = y * w + bound.x1 + 1;
                let (Some(acc_row), Some(tmp_row)) = (acc.get_mut(lo..hi), tmp.get(lo..hi)) else {
                    continue;
                };
                px_stat(5, acc_row.len());
                for (a, &t) in acc_row.iter_mut().zip(tmp_row.iter()) {
                    let mut c = u32::from(t);
                    if mask.invert {
                        c = 255 - c;
                    }
                    c = (c * op + 127) / 255;
                    let cur = u32::from(*a);
                    let next = match mask.mode {
                        b's' => (cur * (255 - c) + 127) / 255,
                        b'i' => (cur * c + 127) / 255,
                        b'f' => cur.abs_diff(c), // Difference (XOR-like)
                        // 'a' Add combines with SrcOver (rlottie blitSrcOver:
                        // b + (255−b)·a/255), NOT a saturating sum — overlapping
                        // partial-coverage masks differ (190 vs 255).
                        _ => c + ((255 - c) * cur + 127) / 255,
                    };
                    *a = next as u8;
                }
            }
        }
        scratch.put_u8(tmp);
        scratch.put_raster(raster);
        scratch.put_cells(cells);
        acc
    }
}

/// Fills the `bound` sub-rectangle of a `w`-stride u8 plane with `v`. Seeds
/// only the region build_mask actually walks (the offscreen dirty box),
/// replacing the former full-plane `take_u8` fill and per-mask `tmp.fill(0)`.
fn fill_rows_u8(buf: &mut [u8], w: usize, b: DirtyBox, v: u8) {
    if b.is_empty() {
        return;
    }
    for y in b.y0..=b.y1 {
        let lo = y * w + b.x0;
        let hi = (y * w + b.x1 + 1).min(buf.len());
        if lo >= hi {
            continue;
        }
        if let Some(row) = buf.get_mut(lo..hi) {
            row.fill(v);
        }
    }
}

/// Multiplies premultiplied ARGB pixels by a u8 coverage buffer.
/// Calls `f(y, row)` for each row of the dirty box, with `row` being the
/// box's column span `[b.x0, b.x1]` of that row.
fn for_rows_boxed(pixels: &mut [u32], w: usize, b: DirtyBox, mut f: impl FnMut(usize, &mut [u32])) {
    if b.is_empty() {
        return;
    }
    for y in b.y0..=b.y1 {
        let lo = y * w + b.x0;
        let hi = (y * w + b.x1 + 1).min(pixels.len());
        if lo >= hi {
            continue;
        }
        if let Some(row) = pixels.get_mut(lo..hi) {
            f(y, row);
        }
    }
}

fn modulate(pixels: &mut [u32], mask: &[u8]) {
    for (px, &mk) in pixels.iter_mut().zip(mask.iter()) {
        if mk == 255 {
            continue;
        }
        if mk == 0 {
            *px = 0;
            continue;
        }
        let m = u32::from(mk);
        let p = *px;
        let a = (((p >> 24) & 0xff) * m + 127) / 255;
        let r = (((p >> 16) & 0xff) * m + 127) / 255;
        let g = (((p >> 8) & 0xff) * m + 127) / 255;
        let b = ((p & 0xff) * m + 127) / 255;
        *px = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Applies a matte source (`src`) onto `dst` premultiplied pixels.
/// kind: 1 alpha, 2 inverted alpha, 3 luma, 4 inverted luma.
fn apply_matte(dst: &mut [u32], src: &[u32], kind: u8) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let factor = match kind {
            1 => (s >> 24) & 0xff,
            2 => 255 - ((s >> 24) & 0xff),
            3 => luma_premult(s),
            _ => 255 - luma_premult(s),
        };
        if factor == 255 {
            continue;
        }
        if factor == 0 {
            *d = 0;
            continue;
        }
        let p = *d;
        let a = (((p >> 24) & 0xff) * factor + 127) / 255;
        let r = (((p >> 16) & 0xff) * factor + 127) / 255;
        let g = (((p >> 8) & 0xff) * factor + 127) / 255;
        let b = ((p & 0xff) * factor + 127) / 255;
        *d = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Luma of a premultiplied pixel, rlottie semantics: unpremultiply, then
/// Rec.601 weights on the straight color (the matte's own alpha does not
/// scale the luma).
fn luma_premult(p: u32) -> u32 {
    let a = (p >> 24) & 0xff;
    if a == 0 {
        return 0;
    }
    let mut r = (p >> 16) & 0xff;
    let mut g = (p >> 8) & 0xff;
    let mut b = p & 0xff;
    if a != 255 {
        r = (r * 255) / a;
        g = (g * 255) / a;
        b = (b * 255) / a;
    }
    ((r * 299 + g * 587 + b * 114) / 1000).min(255)
}

/// Composites premultiplied `src` over `dst` with a global opacity factor.
/// Combined matrix of all ancestors of `layer` within `layers` (not
/// including the layer itself). Cycle-safe: walks at most `layers.len()`.
fn parent_chain_matrix(layers: &[Layer], layer: &Layer, frame: f32) -> Mat2x3 {
    let mut chain: Vec<Mat2x3> = Vec::new();
    let mut current = layer.parent;
    let mut steps = 0usize;
    while let Some(parent_ind) = current {
        steps += 1;
        if steps > layers.len() {
            break; // cycle; refuse to loop forever
        }
        let Some(parent) = layers.iter().find(|l| l.index == parent_ind) else {
            break;
        };
        let (m, _) = transform_at(&parent.transform, frame);
        chain.push(m);
        current = parent.parent;
    }
    let mut result = Mat2x3::IDENTITY;
    for m in chain.iter().rev() {
        result = result.concat(*m);
    }
    result
}

/// Layer-level transform: adds auto-orient rotation from the position path
/// derivative when the layer requests it.
fn layer_transform_at(layer: &Layer, frame: f32) -> (Mat2x3, f32) {
    let (m, opacity) = transform_at(&layer.transform, frame);
    if !layer.auto_orient {
        return (m, opacity);
    }
    let before = layer.transform.position.eval(frame - 0.5);
    let after = layer.transform.position.eval(frame + 0.5);
    let dx = after.x - before.x;
    let dy = after.y - before.y;
    if dx * dx + dy * dy < 1e-9 {
        return (m, opacity);
    }
    let angle = dy.atan2(dx).to_degrees();
    // Auto-orient rotates around the anchor, i.e. composes like rotation:
    // re-apply an extra rotation between position and the rest.
    layer.transform.anchor.eval(frame);
    let pos = layer.transform.position.eval(frame);
    let extra = Mat2x3::translate(pos.x, pos.y)
        .concat(Mat2x3::rotate(angle))
        .concat(Mat2x3::translate(-pos.x, -pos.y));
    (extra.concat(m), opacity)
}

/// Evaluates a Transform at a frame into (matrix, opacity 0..=1).
fn transform_at(tf: &Transform, frame: f32) -> (Mat2x3, f32) {
    let anchor = tf.anchor.eval(frame);
    let position = tf.position.eval(frame);
    let scale = tf.scale.eval(frame);
    let rotation = tf.rotation.eval(frame);
    let opacity = (tf.opacity.eval(frame) / 100.0).clamp(0.0, 1.0);
    // NOTE: skew (sk/sa) is parsed but NOT applied — rlottie (patched and
    // upstream) never reads those fields; its matrix is translate·rotate·
    // scale·translate(−anchor) only. Applying AE-correct shear diverged on
    // every skewed file (contract review: transforms).
    let m = Mat2x3::translate(position.x, position.y)
        .concat(Mat2x3::rotate(rotation))
        .concat(Mat2x3::scale(scale.x / 100.0, scale.y / 100.0))
        .concat(Mat2x3::translate(-anchor.x, -anchor.y));
    (m, opacity)
}

fn opacity_byte(opacity: f32) -> u32 {
    (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
}

/// Inclusive dirty rectangle of pixels a canvas has written. Everything
/// outside it is still the transparent clear color, so full-canvas
/// composite passes (mask modulate, matte, layer composite — measured at
/// ~20% of effects frames, mostly over EMPTY or ≤11%-covered offscreens)
/// can be bounded to it bit-exactly.
#[derive(Clone, Copy)]
struct DirtyBox {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl DirtyBox {
    fn empty() -> DirtyBox {
        DirtyBox {
            x0: usize::MAX,
            y0: usize::MAX,
            x1: 0,
            y1: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.x0 > self.x1 || self.y0 > self.y1
    }

    /// Marks the half-open column range `[x0, x1)` of row `y`.
    fn mark_row(&mut self, y: usize, x0: usize, x1: usize) {
        if x1 <= x0 {
            return;
        }
        self.x0 = self.x0.min(x0);
        self.x1 = self.x1.max(x1 - 1);
        self.y0 = self.y0.min(y);
        self.y1 = self.y1.max(y);
    }

    fn union(&mut self, other: DirtyBox) {
        if other.is_empty() {
            return;
        }
        self.x0 = self.x0.min(other.x0);
        self.y0 = self.y0.min(other.y0);
        self.x1 = self.x1.max(other.x1);
        self.y1 = self.y1.max(other.y1);
    }
}

struct Canvas<'a> {
    pixels: &'a mut [u32],
    w: usize,
    h: usize,
    antialias: bool,
    raster: Rasterizer,
    /// Mode-S sparse cell engine, selected per paint when the contours'
    /// bbox extent exceeds [`MODE_S_MIN_EXTENT`].
    cells: CellRaster,
    /// Union of all rows/columns written since creation (fills, gradient
    /// fills, and nested offscreen composites all mark it).
    dirty: DirtyBox,
    /// Uniform-coverage row scratch: mode-S gradient spans synthesize a
    /// constant cov row here so gradient_row's per-pixel math (and its
    /// bit-exactness contract with the cache replay) is reused verbatim.
    row_cov: Vec<u8>,
    /// All-255 coverage row, grown on demand: opaque uniform gradient
    /// spans (the common mode-S case) borrow it instead of re-filling
    /// row_cov per span.
    row_ones: Vec<u8>,
    /// Recycled mode-S span-capture buffer: taken at the top of each
    /// fresh fill, returned unless the buffer moved into a cache entry
    /// (chunky captures become `PlaneData::Spans` verbatim). Avoids a
    /// realloc-growth chain per fresh mode-S paint.
    span_buf: Vec<u64>,
}

impl<'a> Canvas<'a> {
    fn with_raster(
        pixels: &'a mut [u32],
        w: usize,
        h: usize,
        raster: Rasterizer,
        cells: CellRaster,
        antialias: bool,
    ) -> Self {
        Canvas {
            pixels,
            w,
            h,
            antialias,
            raster,
            cells,
            dirty: DirtyBox::empty(),
            row_cov: Vec::new(),
            row_ones: Vec::new(),
            span_buf: Vec::new(),
        }
    }
}

impl Canvas<'_> {
    /// Rasterizes `contours` and blends `color` (straight alpha, 0..=1
    /// components) with `opacity`, premultiplied source-over.
    fn fill(
        &mut self,
        cache: &mut CovCache,
        key: u128,
        contours: &[Contour],
        rule: crate::model::FillRule,
        color: Color,
        opacity: f32,
    ) {
        let alpha = (color.a * opacity).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        let (sr, sg, sb) = (
            (color.r.clamp(0.0, 1.0) * 255.0) as u32,
            (color.g.clamp(0.0, 1.0) * 255.0) as u32,
            (color.b.clamp(0.0, 1.0) * 255.0) as u32,
        );
        let sa = (alpha * 255.0 + 0.5) as u32;
        let w = self.w;
        let antialias = self.antialias;
        if let Some(e) = cache.get(key) {
            // Cache hit: replay coverage (identical bytes to a fresh
            // rasterization of the same geometry, whichever mode made it).
            match &e.data {
                PlaneData::Cov(data) => {
                    let mut off = 0usize;
                    for &(y, x0, len) in &e.rows {
                        let (y, x0, len) = (y as usize, x0 as usize, len as usize);
                        let lo = y.saturating_mul(w).saturating_add(x0);
                        let hi = lo.saturating_add(len);
                        let (Some(dst_row), Some(cov_row)) =
                            (self.pixels.get_mut(lo..hi), data.get(off..off + len))
                        else {
                            break;
                        };
                        self.dirty.mark_row(y, x0, x0 + len);
                        px_stat(0, len);
                        crate::simd::fill_span_solid(dst_row, cov_row, sr, sg, sb, sa);
                        off += len;
                    }
                }
                PlaneData::Spans(spans) => {
                    for &s in spans {
                        let (y, x0, len, cov) = unpack_span(s);
                        let lo = y.saturating_mul(w).saturating_add(x0);
                        let Some(dst_row) = self.pixels.get_mut(lo..lo.saturating_add(len)) else {
                            break;
                        };
                        self.dirty.mark_row(y, x0, x0 + len);
                        px_stat(1, len);
                        px_stat(9, 1);
                        crate::simd::fill_span_uniform(dst_row, cov, sr, sg, sb, sa);
                    }
                }
                PlaneData::Src(_) => {}
            }
            return;
        }
        if contours.is_empty() {
            return;
        }
        let pixels = &mut *self.pixels;
        let dirty = &mut self.dirty;
        if mode_s_wins(contours, w * self.h) {
            // Mode S: sparse cells — no w×h plane, cost ∝ edge crossings.
            self.cells.reset();
            self.cells.fill_contours(contours);
            let capture = cache.capture_enabled();
            let mut spans: Vec<u64> = core::mem::take(&mut self.span_buf);
            spans.clear();
            let mut px_total = 0usize;
            let mut overflow = false;
            self.cells.sweep_spans(rule, antialias, |y, x0, len, cov| {
                let lo = y.saturating_mul(w).saturating_add(x0);
                let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
                    return;
                };
                dirty.mark_row(y, x0, x0 + len);
                px_stat(2, len);
                px_stat(8, 1);
                if capture {
                    if spans.len() < SPAN_CAPTURE_MAX {
                        spans.push(pack_span(y, x0, len, cov));
                        px_total += len;
                    } else {
                        overflow = true;
                    }
                }
                crate::simd::fill_span_uniform(dst_row, cov, sr, sg, sb, sa);
            });
            // Chunky span lists cache as spans (denser, uniform replay);
            // fragmented ones as rows (fast row replay).
            if overflow || !capture {
                self.span_buf = spans; // recycle
                return; // uncacheable either way; skip entry construction
            }
            let entry = if spans_fragmented(&spans, px_total) {
                let e = spans_to_cov_entry(&spans);
                self.span_buf = spans; // recycle
                e
            } else {
                CovEntry {
                    rows: Vec::new(),
                    data: PlaneData::Spans(spans),
                }
            };
            cache.insert(key, entry);
            return;
        }
        self.raster.reset();
        self.raster.fill_contours(contours);
        let capture = cache.capture_enabled();
        let mut entry = CovEntry::default();
        self.raster.sweep(rule, antialias, |y, x0, cov_row| {
            let lo = y.saturating_mul(w).saturating_add(x0);
            let hi = lo.saturating_add(cov_row.len());
            let Some(dst_row) = pixels.get_mut(lo..hi) else {
                return;
            };
            dirty.mark_row(y, x0, x0 + cov_row.len());
            px_stat(3, cov_row.len());
            if capture {
                entry.rows.push((y as u32, x0 as u32, cov_row.len() as u32));
                if let PlaneData::Cov(d) = &mut entry.data {
                    d.extend_from_slice(cov_row);
                }
            }
            crate::simd::fill_span_solid(dst_row, cov_row, sr, sg, sb, sa);
        });
        if capture {
            cache.insert(key, entry);
        }
    }
}

/// Premultiplied ARGB LUT built from Lottie stop data.
///
/// Stop construction ports the PATCHED rlottie's LOTGradient::populate
/// (TMessagesProj lottiemodel.cpp:198, lottie-android lineage): opacity
/// stops are MERGED into the stop list with a persistent index walked
/// across the color points — an opacity stop before the next color stop
/// becomes its own stop carrying the pending color; color stops get
/// opacity interpolated from the bracketing opacity stops. Upstream
/// rlottie samples opacity per color stop instead; the two disagree
/// wildly on offset opacity ramps (HalloweenTeamEmoji glow washes).
fn build_gradient_lut(
    stops: &FloatList,
    color_count: usize,
    opacity: f32,
) -> [u32; GRADIENT_LUT_SIZE] {
    let data = &stops.0;
    let n = color_count.min(data.len() / 4);
    let mut lut = [0u32; GRADIENT_LUT_SIZE];
    // Opacity floats after the color stops; rlottie disables them when the
    // count is odd or implausibly small (populate's opacityArraySize gate).
    let opac = data.get(n * 4..).unwrap_or(&[]);
    let oas = if opac.len() % 2 != 0 || (n > opac.len() / 2 && opac.len() < 4) {
        0
    } else {
        opac.len()
    };
    let op_at = |k: usize| opac.get(k).copied().unwrap_or(0.0);

    // Merged (position, r, g, b, alpha) stop list per patched populate.
    // Stop alphas pack through rlottie's uchar cast, which WRAPS on
    // easing/extrapolation overshoot (opacity 1.638 → byte 162, verified
    // against the reference); interpolation then runs on wrapped values.
    let wrap = |a: f32| -> f32 { (((a * 255.0) as i32) as u8) as f32 / 255.0 };
    let mut merged: Vec<(f32, f32, f32, f32, f32)> = Vec::with_capacity(n + oas / 2);
    let mut j = 0usize;
    for i in 0..n {
        let base = i * 4;
        let cpos = data.get(base).copied().unwrap_or(0.0);
        let (cr, cg, cb) = (
            data.get(base + 1).copied().unwrap_or(0.0),
            data.get(base + 2).copied().unwrap_or(0.0),
            data.get(base + 3).copied().unwrap_or(0.0),
        );
        if oas == 0 {
            merged.push((cpos, cr, cg, cb, 1.0));
            continue;
        }
        if j == oas {
            // Past the last opacity stop: extend or interpolate the final
            // opacity segment (populate's j==opacityArraySize branch).
            let (s1, o1, s2, o2) = (
                op_at(j.saturating_sub(4)),
                op_at(j.saturating_sub(3)),
                op_at(j.saturating_sub(2)),
                op_at(j.saturating_sub(1)),
            );
            let a = if cpos > s2 {
                o2
            } else {
                // NO span guard: rlottie's populate divides by the raw stop
                // distance; COINCIDENT opacity stops (a hard discontinuity,
                // e.g. 0.5→0.0 at the same offset) give 0/0 = NaN, and
                // toColor(NaN) casts to alpha 0 on ARM — which is also the
                // authored intent. The old 1e-6 guard yielded 0.5 instead
                // and washed FroggoInLove's tear shimmer stack solid
                // (worst-region 49.8). Rust `NaN as i32` is a guaranteed 0,
                // so this is deterministic, unlike the C++ side.
                o1 + (cpos - s1) / (s2 - s1) * (o2 - o1)
            };
            merged.push((cpos, cr, cg, cb, wrap(a)));
            continue;
        }
        while j < oas {
            let ostop = op_at(j);
            if ostop < cpos {
                // Opacity stop before the color stop: emit it as its own
                // stop carrying the CURRENT color (populate's quirk).
                merged.push((ostop, cr, cg, cb, wrap(op_at(j + 1))));
                j += 2;
                continue;
            }
            let a = if j == 0 {
                op_at(j + 1)
            } else {
                // Unguarded like rlottie (see the coincident-stop comment
                // above): 0/0 → NaN → wrap() → alpha 0, matching both the
                // reference and the authored discontinuity.
                let span = op_at(j) - op_at(j - 2);
                op_at(j - 1) + (cpos - op_at(j - 2)) / span * (op_at(j + 1) - op_at(j - 1))
            };
            merged.push((cpos, cr, cg, cb, wrap(a)));
            j += 2;
            break;
        }
    }
    if merged.is_empty() {
        merged.push((0.0, 1.0, 1.0, 1.0, 1.0));
    }

    for (i, slot) in lut.iter_mut().enumerate() {
        let t = i as f32 / (GRADIENT_LUT_SIZE - 1) as f32;
        let first = merged.first().copied().unwrap_or((0.0, 0.0, 0.0, 0.0, 1.0));
        let last = merged.last().copied().unwrap_or(first);
        let (r, g, b, a) = if t <= first.0 {
            (first.1, first.2, first.3, first.4)
        } else if t >= last.0 {
            (last.1, last.2, last.3, last.4)
        } else {
            let mut out = (last.1, last.2, last.3, last.4);
            for w in merged.windows(2) {
                let (p0, p1) = match w {
                    [p0, p1] => (*p0, *p1),
                    _ => continue,
                };
                if t <= p1.0 {
                    let span = (p1.0 - p0.0).max(1e-6);
                    let f = ((t - p0.0) / span).clamp(0.0, 1.0);
                    out = (
                        p0.1 + (p1.1 - p0.1) * f,
                        p0.2 + (p1.2 - p0.2) * f,
                        p0.3 + (p1.3 - p0.3) * f,
                        p0.4 + (p1.4 - p0.4) * f,
                    );
                    break;
                }
            }
            out
        };
        let a = (a * opacity).clamp(0.0, 1.0);
        let pa = (a * 255.0 + 0.5) as u32;
        let pr = (r.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
        let pg = (g.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
        let pb = (b.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u32;
        *slot = (pa << 24) | (pr << 16) | (pg << 8) | pb;
    }
    lut
}

/// Gradient parametrization. The shape geometry (`sx/sy/…`) lives in LOCAL
/// (shape) space and each device pixel is pulled back through `inv`, the
/// inverse CTM — this is what makes a radial gradient an ellipse under
/// non-uniform scale/shear, matching rlottie (vdrawhelper.cpp setupMatrix:
/// spans are inverse-transformed before the radial distance is measured).
#[derive(Clone)]
struct GradientMap {
    inv: Mat2x3,
    kind: GradientMapKind,
}

#[derive(Clone)]
enum GradientMapKind {
    Linear {
        sx: f32,
        sy: f32,
        dx: f32,
        dy: f32,
        inv_len_sq: f32,
    },
    Radial {
        sx: f32,
        sy: f32,
        inv_r: f32,
    },
    /// Two-point (focal) radial, rlottie/Qt semantics: focal circle
    /// (fx,fy,r=0) → outer circle (C,r). `dx/dy = C−F`, `a = r² − |C−F|²`.
    Focal {
        fx: f32,
        fy: f32,
        dx: f32,
        dy: f32,
        a: f32,
        r: f32,
    },
}

impl GradientMap {
    /// Exact-bits content hash (for the source-plane cache key).
    fn content_hash(&self) -> u64 {
        let mut h = Hasher128::new();
        for v in [
            self.inv.a,
            self.inv.b,
            self.inv.c,
            self.inv.d,
            self.inv.tx,
            self.inv.ty,
        ] {
            h.mix(v.to_bits());
        }
        match &self.kind {
            GradientMapKind::Linear {
                sx,
                sy,
                dx,
                dy,
                inv_len_sq,
            } => {
                for v in [*sx, *sy, *dx, *dy, *inv_len_sq] {
                    h.mix(v.to_bits());
                }
                h.mix(1);
            }
            GradientMapKind::Radial { sx, sy, inv_r } => {
                for v in [*sx, *sy, *inv_r] {
                    h.mix(v.to_bits());
                }
                h.mix(2);
            }
            GradientMapKind::Focal {
                fx,
                fy,
                dx,
                dy,
                a,
                r,
            } => {
                for v in [*fx, *fy, *dx, *dy, *a, *r] {
                    h.mix(v.to_bits());
                }
                h.mix(3);
            }
        }
        h.finish() as u64
    }
}

/// Builds the radial/focal gradient map from LOCAL-space points; `inv` is
/// the inverse CTM used to pull device pixels back into that space.
fn radial_map(
    start: Vec2,
    end: Vec2,
    inv: Mat2x3,
    hl_len_pct: f32,
    hl_angle_deg: f32,
) -> GradientMap {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    // rlottie measures the radius with its approximate metric
    // (radial.cradius = VLine::length(start, end)).
    let r = crate::geometry::seg_len_rlottie(dx, dy);
    if hl_len_pct.abs() < 0.01 || r < 1e-6 {
        return GradientMap {
            inv,
            kind: GradientMapKind::Radial {
                sx: start.x,
                sy: start.y,
                inv_r: if r > 1e-6 { 1.0 / r } else { 0.0 },
            },
        };
    }
    // Focal point (rlottie lottiemodel.cpp): at highlight-length % of the
    // radius from the center, along the start→end direction rotated by the
    // highlight angle. Only exactly-100% is nudged to 99%.
    let mut progress = hl_len_pct / 100.0;
    if (progress - 1.0).abs() < 1e-4 {
        progress = 0.99;
    }
    let base_angle = dy.atan2(dx) + hl_angle_deg.to_radians();
    let fx = start.x + r * progress * base_angle.cos();
    let fy = start.y + r * progress * base_angle.sin();
    let cdx = start.x - fx;
    let cdy = start.y - fy;
    let a = r * r - (cdx * cdx + cdy * cdy);
    GradientMap {
        inv,
        kind: GradientMapKind::Focal {
            fx,
            fy,
            dx: cdx,
            dy: cdy,
            a,
            r,
        },
    }
}

impl Canvas<'_> {
    fn fill_gradient(
        &mut self,
        cache: &mut CovCache,
        key: u128,
        src_key: u128,
        contours: &[Contour],
        rule: FillRule,
        lut: &[u32; GRADIENT_LUT_SIZE],
        map: &GradientMap,
    ) {
        let w = self.w;
        let antialias = self.antialias;
        let dst_clear = self.dirty.is_empty();
        // Fastest path: the paint's premultiplied coverage-scaled SOURCE
        // pixels are cached (geometry + LUT + map all repeat) — replay as a
        // pure composite; bit-exact vs blend_gradient_px (same formula).
        if let Some(e) = cache.get(src_key) {
            let PlaneData::Src(data) = &e.data else {
                return;
            };
            let mut off = 0usize;
            for &(y, x0, len) in &e.rows {
                let (y, x0, len) = (y as usize, x0 as usize, len as usize);
                let lo = y.saturating_mul(w).saturating_add(x0);
                let hi = lo.saturating_add(len);
                let (Some(dst_row), Some(src_row)) =
                    (self.pixels.get_mut(lo..hi), data.get(off..off + len))
                else {
                    break;
                };
                self.dirty.mark_row(y, x0, x0 + len);
                if dst_clear {
                    dst_row.copy_from_slice(src_row);
                } else {
                    crate::simd::composite_over_span(dst_row, src_row, 255);
                }
                off += len;
            }
            return;
        }
        let mut src_entry = CovEntry {
            rows: Vec::new(),
            data: PlaneData::Src(Vec::new()),
        };
        let mut had_cov_hit = false;
        // Hoisted out of the `cache.get` borrow — `capture_enabled` must be
        // read before the mutable borrow extends through the cov entry.
        let capture_enabled = cache.capture_enabled();
        if let Some(e) = cache.get(key) {
            if let PlaneData::Spans(spans) = &e.data {
                // Mode-S coverage hit: replay spans through the identical
                // per-pixel gradient math via a synthesized uniform cov
                // row, and (size permitting) capture the source plane so
                // later frames replay as a pure composite — same
                // proven-to-repeat rule as the Cov arm (profiled:
                // gradient_srcs was the top cost at 720 without this).
                let px_total: usize = spans.iter().map(|&s| unpack_span(s).2).sum();
                // Only capture the source plane when it will actually be
                // admitted AND the cache is still learning — when frozen
                // `insert` is a no-op, so paying the two-pass
                // gradient_row_capture (src plane materialized to DRAM) for
                // every cov hit throws the bytes away. The fused gradient_row
                // is bit-identical and round-trip-free.
                let capture =
                    capture_enabled && px_total * 4 + spans.len() * 12 + 64 <= COV_ENTRY_MAX;
                for &s in spans {
                    let (y, x0, len, cov) = unpack_span(s);
                    let lo = y.saturating_mul(w).saturating_add(x0);
                    let Some(dst_row) = self.pixels.get_mut(lo..lo.saturating_add(len)) else {
                        break;
                    };
                    self.dirty.mark_row(y, x0, x0 + len);
                    if capture {
                        // Source-plane capture needs actual per-pixel bytes.
                        if cov == 255 && self.row_ones.len() < len {
                            self.row_ones.resize(len, 255);
                        }
                        if cov != 255 {
                            self.row_cov.clear();
                            self.row_cov.resize(len, cov);
                        }
                        let cr: &[u8] = if cov == 255 {
                            self.row_ones.get(..len).unwrap_or(&[])
                        } else {
                            &self.row_cov
                        };
                        src_entry.rows.push((y as u32, x0 as u32, len as u32));
                        if let PlaneData::Src(sd) = &mut src_entry.data {
                            if dst_clear {
                                gradient_row_capture_clear(dst_row, cr, y, x0, lut, map, sd);
                            } else {
                                gradient_row_capture(dst_row, cr, y, x0, lut, map, sd);
                            }
                        }
                    } else if dst_clear {
                        gradient_span_uniform_clear(dst_row, cov, y, x0, lut, map);
                    } else {
                        gradient_span_uniform(dst_row, cov, y, x0, lut, map);
                    }
                }
                if !capture {
                    return;
                }
                had_cov_hit = true;
            } else {
                let PlaneData::Cov(data) = &e.data else {
                    return;
                };
                had_cov_hit = true;
                // Only capture the source plane if it will actually be admitted:
                // 4 bytes/pixel, size known exactly from the coverage entry.
                // Oversized planes (720px gradients) previously re-captured every
                // frame just to be rejected by insert — measured at 49% of the
                // worst effects file.
                let capture =
                    capture_enabled && data.len() * 4 + e.rows.len() * 12 + 64 <= COV_ENTRY_MAX;
                let mut off = 0usize;
                for &(y, x0, len) in &e.rows {
                    let (y, x0, len) = (y as usize, x0 as usize, len as usize);
                    let lo = y.saturating_mul(w).saturating_add(x0);
                    let hi = lo.saturating_add(len);
                    let (Some(dst_row), Some(cov_row)) =
                        (self.pixels.get_mut(lo..hi), data.get(off..off + len))
                    else {
                        break;
                    };
                    self.dirty.mark_row(y, x0, x0 + len);
                    if capture {
                        src_entry.rows.push((y as u32, x0 as u32, len as u32));
                        if let PlaneData::Src(sd) = &mut src_entry.data {
                            if dst_clear {
                                gradient_row_capture_clear(dst_row, cov_row, y, x0, lut, map, sd);
                            } else {
                                gradient_row_capture(dst_row, cov_row, y, x0, lut, map, sd);
                            }
                        }
                    } else if dst_clear {
                        gradient_srcs(dst_row, cov_row, y, x0, lut, map);
                    } else {
                        gradient_row(dst_row, cov_row, y, x0, lut, map);
                    }
                    off += len;
                }
                if !capture {
                    return;
                }
            }
        }
        if had_cov_hit {
            cache.insert(src_key, src_entry);
            return;
        }
        if contours.is_empty() {
            return;
        }
        let pixels = &mut *self.pixels;
        let dirty = &mut self.dirty;
        if mode_s_wins(contours, w * self.h) {
            // Mode S: spans feed the same gradient_row math through a
            // synthesized uniform cov row.
            drop(src_entry);
            self.cells.reset();
            self.cells.fill_contours(contours);
            let capture = cache.capture_enabled();
            let mut spans: Vec<u64> = core::mem::take(&mut self.span_buf);
            spans.clear();
            let mut px_total = 0usize;
            let mut overflow = false;
            self.cells.sweep_spans(rule, antialias, |y, x0, len, cov| {
                let lo = y.saturating_mul(w).saturating_add(x0);
                let Some(dst_row) = pixels.get_mut(lo..lo.saturating_add(len)) else {
                    return;
                };
                dirty.mark_row(y, x0, x0 + len);
                if !capture {
                } else if spans.len() < SPAN_CAPTURE_MAX {
                    spans.push(pack_span(y, x0, len, cov));
                    px_total += len;
                } else {
                    overflow = true;
                }
                if dst_clear {
                    gradient_span_uniform_clear(dst_row, cov, y, x0, lut, map);
                } else {
                    gradient_span_uniform(dst_row, cov, y, x0, lut, map);
                }
            });
            if overflow || !capture {
                self.span_buf = spans; // recycle
                return; // uncacheable either way; skip entry construction
            }
            let entry = if spans_fragmented(&spans, px_total) {
                let e = spans_to_cov_entry(&spans);
                self.span_buf = spans; // recycle
                e
            } else {
                CovEntry {
                    rows: Vec::new(),
                    data: PlaneData::Spans(spans),
                }
            };
            cache.insert(key, entry);
            return;
        }
        self.raster.reset();
        self.raster.fill_contours(contours);
        let capture = cache.capture_enabled();
        let mut entry = CovEntry::default();
        // Fresh rasterization: do NOT capture the source plane here — most
        // fresh geometry is animated and never repeats, and the capture
        // (resize + per-pixel src writes) measured 13.6% of gradient-heavy
        // 320px frames. The src plane is captured on the first COV hit
        // instead, i.e. only for geometry proven to repeat.
        drop(src_entry);
        self.raster.sweep(rule, antialias, |y, x0, cov_row| {
            let lo = y.saturating_mul(w).saturating_add(x0);
            let hi = lo.saturating_add(cov_row.len());
            let Some(dst_row) = pixels.get_mut(lo..hi) else {
                return;
            };
            dirty.mark_row(y, x0, x0 + cov_row.len());
            px_stat(3, cov_row.len());
            if capture {
                entry.rows.push((y as u32, x0 as u32, cov_row.len() as u32));
                if let PlaneData::Cov(d) = &mut entry.data {
                    d.extend_from_slice(cov_row);
                }
            }
            if dst_clear {
                gradient_srcs(dst_row, cov_row, y, x0, lut, map);
            } else {
                gradient_row(dst_row, cov_row, y, x0, lut, map);
            }
        });
        if capture {
            cache.insert(key, entry);
        }
    }
}

/// gradient_row variant that also CAPTURES the premultiplied,
/// coverage-scaled source pixel per position (0 where the paint leaves the
/// destination untouched). Blending src over dst here uses the identical
/// integer formula as blend_gradient_px, so captured-replay via
/// composite_over_span(k=255) is bit-exact.
fn gradient_row_capture(
    dst_row: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
    out: &mut Vec<u32>,
) {
    let base = out.len();
    out.resize(base + dst_row.len(), 0);
    let Some(srcs) = out.get_mut(base..) else {
        return;
    };
    gradient_srcs(srcs, cov_row, y, x0, lut, map);
    crate::simd::composite_over_span(dst_row, srcs, 255);
}

/// Clear-destination form of [`gradient_row_capture`]: source-over onto
/// transparent pixels is exactly the coverage-scaled premultiplied source.
fn gradient_row_capture_clear(
    dst_row: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
    out: &mut Vec<u32>,
) {
    let base = out.len();
    out.resize(base + dst_row.len(), 0);
    let Some(srcs) = out.get_mut(base..) else {
        return;
    };
    gradient_srcs(srcs, cov_row, y, x0, lut, map);
    dst_row.copy_from_slice(srcs);
}

/// Computes the premultiplied coverage-scaled source pixels of one gradient
/// row into `srcs` (same per-pixel math as gradient_row's t/LUT sampling).
fn gradient_srcs(
    srcs: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    let inv = map.inv;
    let yf = y as f32 + 0.5;
    // Row origin: local coords at device column 0 (pixel-center x = 0.5).
    // Every position below is anchored to the ABSOLUTE device column X and
    // computed as one rounded `base + X·step`, so a pixel's bits do not
    // depend on the span/row/sub-run x0 it is reached through
    // (segmentation-invariant — the strong byte-exact-cache invariant).
    let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
    let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
    match &map.kind {
        GradientMapKind::Linear {
            sx,
            sy,
            dx,
            dy,
            inv_len_sq,
        } => {
            let n = srcs.len().min(cov_row.len());
            grad_stat(0, n);
            let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
            let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
            // Run-batched like the radial/focal arms; `row_base + X·dt` form
            // at absolute column X (corpus-gated association change).
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = srcs.get_mut(i..i + run) {
                        grad_stat(3, run);
                        crate::simd::linear_lut_fill(out, lut, row_base, dt, (x0 + i) as f32);
                    }
                    i += run;
                    continue;
                }
                let t = row_base + (x0 + i) as f32 * dt;
                if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
                    if cov == 255 {
                        *s = lut_sample(lut, t); // src_px(255, c) == c exactly
                    } else if cov != 0 {
                        *s = src_px(cov, lut_sample(lut, t));
                    } else {
                        // Explicit zero: lets gradient_row keep its scratch
                        // buffer dirty across rows (no per-row memset).
                        *s = 0;
                    }
                }
                i += 1;
            }
        }
        GradientMapKind::Radial { sx, sy, inv_r } => {
            // Full-coverage runs (interiors; every mode-S span) go through
            // the 4-lane kernel — the post-span profile had this loop as
            // the single largest 720px cost (gradient_srcs 74 samples).
            // `dd` positions are computed as `dd0 + X·step` at absolute
            // column X in BOTH paths (not sequential adds): sub-ULP
            // different from the historical loop, corpus-gated like round 2.
            let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
            let n = srcs.len().min(cov_row.len());
            grad_stat(1, n);
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = srcs.get_mut(i..i + run) {
                        grad_stat(3, run);
                        crate::simd::radial_lut_fill(
                            out,
                            lut,
                            dd0x,
                            dd0y,
                            inv.a,
                            inv.b,
                            *inv_r,
                            (x0 + i) as f32,
                        );
                    }
                    i += run;
                    continue;
                }
                let xf = (x0 + i) as f32;
                let ddx = dd0x + xf * inv.a;
                let ddy = dd0y + xf * inv.b;
                if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
                    if cov == 255 {
                        let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
                        *s = lut_sample(lut, t);
                    } else if cov != 0 {
                        let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
                        *s = src_px(cov, lut_sample(lut, t));
                    } else {
                        *s = 0; // see Linear arm: scratch stays dirty
                    }
                }
                i += 1;
            }
        }
        GradientMapKind::Focal {
            fx,
            fy,
            dx,
            dy,
            a,
            r,
        } => {
            if a.abs() < 1e-9 {
                srcs.fill(0); // callers assume every pixel was stored
                return;
            }
            let inv2a = 1.0 / (2.0 * a);
            // Same run-batched structure as the Radial arm; positions in
            // `g0 + X·step` form at absolute column X (corpus-gated).
            let (g0x, g0y) = (lx0 - fx, ly0 - fy);
            let n = srcs.len().min(cov_row.len());
            grad_stat(2, n);
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = srcs.get_mut(i..i + run) {
                        grad_stat(4, run);
                        crate::simd::focal_lut_fill(
                            out,
                            lut,
                            g0x,
                            g0y,
                            inv.a,
                            inv.b,
                            *dx,
                            *dy,
                            *a,
                            inv2a,
                            *r,
                            (x0 + i) as f32,
                        );
                    }
                    i += run;
                    continue;
                }
                let xf = (x0 + i) as f32;
                let gx = g0x + xf * inv.a;
                let gy = g0y + xf * inv.b;
                if let (Some(s), Some(&cov)) = (srcs.get_mut(i), cov_row.get(i)) {
                    // Every pixel gets a store (transparent cases write 0)
                    // so gradient_row's scratch stays dirty across rows.
                    let mut v = 0u32;
                    if cov != 0 {
                        let b = 2.0 * (gx * dx + gy * dy);
                        let gg = gx * gx + gy * gy;
                        let det = b * b + 4.0 * a * gg;
                        if det >= 0.0 {
                            let sq = det.sqrt();
                            let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
                            if r * sroot >= 0.0 {
                                let c = lut_sample(lut, sroot);
                                v = if cov == 255 { c } else { src_px(cov, c) };
                            }
                        }
                    }
                    *s = v;
                }
                i += 1;
            }
        }
    }
}

/// Coverage-scales a premultiplied LUT color: the s_* terms of
/// blend_gradient_px, packed.
#[inline(always)]
fn src_px(cov: u8, src: u32) -> u32 {
    let s_a0 = (src >> 24) & 0xff;
    if s_a0 == 0 {
        return 0;
    }
    let covu = u32::from(cov);
    let s_a = (s_a0 * covu + 127) / 255;
    let s_r = (((src >> 16) & 0xff) * covu + 127) / 255;
    let s_g = (((src >> 8) & 0xff) * covu + 127) / 255;
    let s_b = ((src & 0xff) * covu + 127) / 255;
    (s_a << 24) | (s_r << 16) | (s_g << 8) | s_b
}

/// Blends one coverage row of a gradient paint into `dst_row` (row `y`,
/// starting column `x0`) in a SINGLE fused pass: each source pixel's
/// coverage-scaled premultiplied color is computed (t/LUT stepping identical
/// to gradient_srcs) and immediately source-overed into `dst_row`, with NO
/// materialized src plane. Bit-for-bit identical to the historical
/// gradient_srcs + composite_over_span(k=255) two-pass — the same per-pixel
/// operations, only the intermediate DRAM buffer is elided (the measured
/// A76 win: gradient_srcs writes the plane, composite reads it back, and on
/// the stacked-gradient packs that round-trip is DRAM-bandwidth-bound).
fn gradient_row(
    dst_row: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    let n = dst_row.len().min(cov_row.len());
    // Short rows: the tight per-pixel fused loop (blend_gradient_px ==
    // src_px + blend_premult_px) — no run scan, no runs long enough to reach
    // the batched kernels anyway.
    if n < 32 {
        let (Some(d), Some(c)) = (dst_row.get_mut(..n), cov_row.get(..n)) else {
            return;
        };
        gradient_row_scalar(d, c, y, x0, lut, map);
        return;
    }
    let (Some(d), Some(c)) = (dst_row.get_mut(..n), cov_row.get(..n)) else {
        return;
    };
    gradient_over(d, c, y, x0, lut, map);
}

/// Blends a mode-S uniform-coverage gradient span directly. This is the
/// same math as `gradient_row` over a synthetic constant coverage row, but
/// full-coverage spans can jump straight to the fused LUT-over kernels and
/// partial spans avoid allocating/filling a temporary coverage slice.
fn gradient_span_uniform(
    dst_row: &mut [u32],
    cov: u8,
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    if cov == 0 || dst_row.is_empty() {
        return;
    }
    let inv = map.inv;
    let yf = y as f32 + 0.5;
    let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
    let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
    match &map.kind {
        GradientMapKind::Linear {
            sx,
            sy,
            dx,
            dy,
            inv_len_sq,
        } => {
            let n = dst_row.len();
            grad_stat(0, n);
            let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
            let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
            if cov == 255 {
                grad_stat(3, n);
                crate::simd::linear_lut_over(dst_row, lut, row_base, dt, x0 as f32);
                return;
            }
            for (i, dst) in dst_row.iter_mut().enumerate() {
                let t = row_base + (x0 + i) as f32 * dt;
                blend_gradient_px(dst, cov, lut_sample(lut, t));
            }
        }
        GradientMapKind::Radial { sx, sy, inv_r } => {
            let n = dst_row.len();
            grad_stat(1, n);
            let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
            if cov == 255 {
                grad_stat(3, n);
                crate::simd::radial_lut_over(
                    dst_row, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, x0 as f32,
                );
                return;
            }
            for (i, dst) in dst_row.iter_mut().enumerate() {
                let xf = (x0 + i) as f32;
                let ddx = dd0x + xf * inv.a;
                let ddy = dd0y + xf * inv.b;
                let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
                blend_gradient_px(dst, cov, lut_sample(lut, t));
            }
        }
        GradientMapKind::Focal {
            fx,
            fy,
            dx,
            dy,
            a,
            r,
        } => {
            if a.abs() < 1e-9 {
                return;
            }
            let n = dst_row.len();
            grad_stat(2, n);
            let inv2a = 1.0 / (2.0 * a);
            let (g0x, g0y) = (lx0 - fx, ly0 - fy);
            if cov == 255 {
                grad_stat(4, n);
                crate::simd::focal_lut_over(
                    dst_row, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, x0 as f32,
                );
                return;
            }
            for (i, dst) in dst_row.iter_mut().enumerate() {
                let xf = (x0 + i) as f32;
                let gx = g0x + xf * inv.a;
                let gy = g0y + xf * inv.b;
                let b = 2.0 * (gx * dx + gy * dy);
                let gg = gx * gx + gy * gy;
                let det = b * b + 4.0 * a * gg;
                if det >= 0.0 {
                    let sq = det.sqrt();
                    let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
                    if r * sroot >= 0.0 {
                        blend_gradient_px(dst, cov, lut_sample(lut, sroot));
                    }
                }
            }
        }
    }
}

/// Clear-destination form of [`gradient_span_uniform`].
fn gradient_span_uniform_clear(
    dst_row: &mut [u32],
    cov: u8,
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    if cov == 0 || dst_row.is_empty() {
        return;
    }
    if cov == 255 {
        match &map.kind {
            GradientMapKind::Linear {
                sx,
                sy,
                dx,
                dy,
                inv_len_sq,
            } => {
                let inv = map.inv;
                let yf = y as f32 + 0.5;
                let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
                let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
                let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
                let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
                grad_stat(0, dst_row.len());
                grad_stat(3, dst_row.len());
                crate::simd::linear_lut_fill(dst_row, lut, row_base, dt, x0 as f32);
            }
            GradientMapKind::Radial { sx, sy, inv_r } => {
                let inv = map.inv;
                let yf = y as f32 + 0.5;
                let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
                let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
                let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
                grad_stat(1, dst_row.len());
                grad_stat(3, dst_row.len());
                crate::simd::radial_lut_fill(
                    dst_row, lut, dd0x, dd0y, inv.a, inv.b, *inv_r, x0 as f32,
                );
            }
            GradientMapKind::Focal {
                fx,
                fy,
                dx,
                dy,
                a,
                r,
            } => {
                if a.abs() < 1e-9 {
                    return;
                }
                let inv = map.inv;
                let yf = y as f32 + 0.5;
                let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
                let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
                let inv2a = 1.0 / (2.0 * a);
                let (g0x, g0y) = (lx0 - fx, ly0 - fy);
                grad_stat(2, dst_row.len());
                grad_stat(4, dst_row.len());
                crate::simd::focal_lut_fill(
                    dst_row, lut, g0x, g0y, inv.a, inv.b, *dx, *dy, *a, inv2a, *r, x0 as f32,
                );
            }
        }
        return;
    }

    if dst_row.len() > 1024 {
        gradient_span_uniform(dst_row, cov, y, x0, lut, map);
        return;
    }
    let mut cov_row = [0u8; 1024];
    if let Some(row) = cov_row.get_mut(..dst_row.len()) {
        row.fill(cov);
        gradient_srcs(dst_row, row, y, x0, lut, map);
    }
}

/// Fused generate+blend of one gradient coverage row — the single-pass form
/// of `gradient_srcs` + `composite_over_span(k=255)`. Full-coverage runs
/// (>=16px, the bulk of gradient pixels) go through the fused `*_lut_over`
/// NEON kernels; partial-coverage and short-run pixels blend scalar via
/// `blend_gradient_px`. The per-pixel LUT math is byte-identical to
/// gradient_srcs (same segmentation-invariant `base + X·step` form), and the
/// blend is the identical integer source-over composite_over_span uses, so
/// the output matches the two-pass path bit-for-bit.
fn gradient_over(
    dst_row: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    let inv = map.inv;
    let yf = y as f32 + 0.5;
    // Row origin: local coords at device column 0 (pixel-center x = 0.5) —
    // identical anchoring to gradient_srcs/gradient_row_scalar.
    let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
    let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
    match &map.kind {
        GradientMapKind::Linear {
            sx,
            sy,
            dx,
            dy,
            inv_len_sq,
        } => {
            let n = dst_row.len().min(cov_row.len());
            grad_stat(0, n);
            let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
            let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = dst_row.get_mut(i..i + run) {
                        grad_stat(3, run);
                        crate::simd::linear_lut_over(out, lut, row_base, dt, (x0 + i) as f32);
                    }
                    i += run;
                    continue;
                }
                if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
                    if cov != 0 {
                        let t = row_base + (x0 + i) as f32 * dt;
                        blend_gradient_px(d, cov, lut_sample(lut, t));
                    }
                }
                i += 1;
            }
        }
        GradientMapKind::Radial { sx, sy, inv_r } => {
            let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
            let n = dst_row.len().min(cov_row.len());
            grad_stat(1, n);
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = dst_row.get_mut(i..i + run) {
                        grad_stat(3, run);
                        crate::simd::radial_lut_over(
                            out,
                            lut,
                            dd0x,
                            dd0y,
                            inv.a,
                            inv.b,
                            *inv_r,
                            (x0 + i) as f32,
                        );
                    }
                    i += run;
                    continue;
                }
                if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
                    if cov != 0 {
                        let xf = (x0 + i) as f32;
                        let ddx = dd0x + xf * inv.a;
                        let ddy = dd0y + xf * inv.b;
                        let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
                        blend_gradient_px(d, cov, lut_sample(lut, t));
                    }
                }
                i += 1;
            }
        }
        GradientMapKind::Focal {
            fx,
            fy,
            dx,
            dy,
            a,
            r,
        } => {
            if a.abs() < 1e-9 {
                return; // gradient_srcs fills 0 (all transparent) → dst untouched
            }
            let inv2a = 1.0 / (2.0 * a);
            let (g0x, g0y) = (lx0 - fx, ly0 - fy);
            let n = dst_row.len().min(cov_row.len());
            grad_stat(2, n);
            let mut i = 0usize;
            while i < n {
                let run = cov_row
                    .get(i..)
                    .map(|c| c.iter().take_while(|&&v| v == 255).count())
                    .unwrap_or(0);
                if run >= 16 {
                    if let Some(out) = dst_row.get_mut(i..i + run) {
                        grad_stat(4, run);
                        crate::simd::focal_lut_over(
                            out,
                            lut,
                            g0x,
                            g0y,
                            inv.a,
                            inv.b,
                            *dx,
                            *dy,
                            *a,
                            inv2a,
                            *r,
                            (x0 + i) as f32,
                        );
                    }
                    i += run;
                    continue;
                }
                if let (Some(d), Some(&cov)) = (dst_row.get_mut(i), cov_row.get(i)) {
                    if cov != 0 {
                        let xf = (x0 + i) as f32;
                        let gx = g0x + xf * inv.a;
                        let gy = g0y + xf * inv.b;
                        let b = 2.0 * (gx * dx + gy * dy);
                        let gg = gx * gx + gy * gy;
                        let det = b * b + 4.0 * a * gg;
                        if det >= 0.0 {
                            let sq = det.sqrt();
                            let sroot = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
                            if r * sroot >= 0.0 {
                                blend_gradient_px(d, cov, lut_sample(lut, sroot));
                            }
                        }
                    }
                }
                i += 1;
            }
        }
    }
}

/// Fused scalar row blend (short rows; also the reference form for
/// gradient_srcs' batched kernels).
fn gradient_row_scalar(
    dst_row: &mut [u32],
    cov_row: &[u8],
    y: usize,
    x0: usize,
    lut: &[u32; GRADIENT_LUT_SIZE],
    map: &GradientMap,
) {
    let inv = map.inv;
    let yf = y as f32 + 0.5;
    // Row origin at device column 0 (pixel-center x = 0.5). Positions are
    // anchored to the ABSOLUTE device column X = x0 + i and computed as one
    // rounded `base + X·step` per pixel — identical to gradient_srcs, so a
    // pixel's bits are the same via this short-row path, the batched
    // kernels, or any span/row segmentation (segmentation-invariant).
    let lx0 = inv.a * 0.5 + inv.c * yf + inv.tx;
    let ly0 = inv.b * 0.5 + inv.d * yf + inv.ty;
    match &map.kind {
        GradientMapKind::Linear {
            sx,
            sy,
            dx,
            dy,
            inv_len_sq,
        } => {
            let row_base = ((lx0 - sx) * dx + (ly0 - sy) * dy) * inv_len_sq;
            let dt = (inv.a * dx + inv.b * dy) * inv_len_sq;
            for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
                if cov != 0 {
                    let t = row_base + (x0 + i) as f32 * dt;
                    blend_gradient_px(dst, cov, lut_sample(lut, t));
                }
            }
        }
        GradientMapKind::Radial { sx, sy, inv_r } => {
            let (dd0x, dd0y) = (lx0 - sx, ly0 - sy);
            for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
                if cov != 0 {
                    let xf = (x0 + i) as f32;
                    let ddx = dd0x + xf * inv.a;
                    let ddy = dd0y + xf * inv.b;
                    let t = (ddx * ddx + ddy * ddy).sqrt() * inv_r;
                    blend_gradient_px(dst, cov, lut_sample(lut, t));
                }
            }
        }
        GradientMapKind::Focal {
            fx,
            fy,
            dx,
            dy,
            a,
            r,
        } => {
            if a.abs() < 1e-9 {
                return; // rlottie: vIsZero(a) → transparent
            }
            let inv2a = 1.0 / (2.0 * a);
            let (g0x, g0y) = (lx0 - fx, ly0 - fy);
            for (i, (dst, &cov)) in dst_row.iter_mut().zip(cov_row.iter()).enumerate() {
                if cov != 0 {
                    // rlottie fetch_radial_gradient: solve
                    // a·s² + b·s − |g|² = 0, take the LARGER root; no
                    // real solution / behind the focal cone → skip
                    // (transparent).
                    let xf = (x0 + i) as f32;
                    let gx = g0x + xf * inv.a;
                    let gy = g0y + xf * inv.b;
                    let b = 2.0 * (gx * dx + gy * dy);
                    let gg = gx * gx + gy * gy;
                    let det = b * b + 4.0 * a * gg;
                    if det >= 0.0 {
                        let sq = det.sqrt();
                        let s = ((-b - sq) * inv2a).max((-b + sq) * inv2a);
                        if r * s >= 0.0 {
                            blend_gradient_px(dst, cov, lut_sample(lut, s));
                        }
                    }
                }
            }
        }
    }
}

/// Clamped LUT sample at gradient position `t`. Non-finite `t` (degenerate
/// transform, focal edge cases) yields transparent, like the old NaN skip.
#[inline(always)]
fn lut_sample(lut: &[u32; GRADIENT_LUT_SIZE], t: f32) -> u32 {
    if !t.is_finite() {
        return 0;
    }
    let t = t.clamp(0.0, 1.0);
    lut.get((t * (GRADIENT_LUT_SIZE - 1) as f32 + 0.5) as usize)
        .copied()
        .unwrap_or(0)
}

/// Coverage-modulated premultiplied source-over of one gradient pixel.
#[inline(always)]
fn blend_gradient_px(dst: &mut u32, cov: u8, src: u32) {
    let s_a0 = (src >> 24) & 0xff;
    if s_a0 == 0 {
        return;
    }
    let covu = u32::from(cov);
    let s_a = (s_a0 * covu + 127) / 255;
    let s_r = (((src >> 16) & 0xff) * covu + 127) / 255;
    let s_g = (((src >> 8) & 0xff) * covu + 127) / 255;
    let s_b = ((src & 0xff) * covu + 127) / 255;
    let d = *dst;
    let inv = 255 - s_a;
    let o_a = s_a + (((d >> 24) & 0xff) * inv + 127) / 255;
    let o_r = s_r + (((d >> 16) & 0xff) * inv + 127) / 255;
    let o_g = s_g + (((d >> 8) & 0xff) * inv + 127) / 255;
    let o_b = s_b + ((d & 0xff) * inv + 127) / 255;
    *dst = (o_a.min(255) << 24) | (o_r.min(255) << 16) | (o_g.min(255) << 8) | o_b.min(255);
}

struct ShapeWalker<'a, 'b> {
    canvas: &'a mut Canvas<'b>,
    scratch: &'a mut RenderScratch,
    frame: f32,
    clip: &'a ClipQuad,
    curve_tolerance: f32,
}

/// A paint recorded during the walk. `range` indexes the geometry arena;
/// contours are snapshotted only AFTER the whole walk, so path modifiers
/// (trim) that appear later in the tree still affect earlier paints —
/// rlottie mutates paths first and paints afterwards.
enum PendingPaint {
    Solid {
        rule: FillRule,
        color: Color,
        opacity: f32,
    },
    Gradient {
        rule: FillRule,
        lut: std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
        lut_id: u64,
        map: GradientMap,
    },
    Stroke {
        color: Option<Color>,
        lut: Option<(std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>, u64, GradientMap)>,
        opacity: f32,
        hw: f32,
        cap: crate::stroke::Cap,
        join: crate::stroke::Join,
        miter_limit: f32,
        pattern: Vec<f32>,
        dash_offset: f32,
    },
}

struct PendingJob {
    paint: PendingPaint,
    start: usize,
    end: usize,
}

/// Materialized draw operation, executed in reverse walk order
/// (first-listed item paints on top).
enum DrawJob {
    Solid {
        key: u128,
        contours: Vec<Contour>,
        rule: FillRule,
        color: Color,
        opacity: f32,
    },
    Gradient {
        key: u128,
        src_key: u128,
        contours: Vec<Contour>,
        rule: FillRule,
        lut: std::sync::Arc<[u32; GRADIENT_LUT_SIZE]>,
        map: GradientMap,
    },
}

impl ShapeWalker<'_, '_> {
    fn walk_shapes(
        &mut self,
        shapes: &[Shape],
        m: Mat2x3,
        opacity: f32,
        depth: usize,
    ) -> Result<(Vec<(Contour, bool)>, Vec<PendingJob>)> {
        let mut arena: Vec<(Contour, bool)> = Vec::new();
        let mut pending: Vec<PendingJob> = Vec::new();
        self.walk(shapes, m, opacity, depth, &mut arena, &mut pending)?;
        Ok((arena, pending))
    }

    fn render_shape_jobs_cpu(
        &mut self,
        arena: &[(Contour, bool)],
        pending: &[PendingJob],
        mut record: Option<&mut Vec<ReplayJob>>,
    ) {
        // Materialize AFTER all modifiers ran, execute in reverse. Fused:
        // materialize is pure per-job (the arena is immutable once the walk
        // finished), so each job's geometry is built, drawn, and freed before
        // the next — nothing forces all jobs' contours to coexist.
        for pj in pending.iter().rev() {
            let contours = match self.materialize(pj, arena) {
                DrawJob::Solid {
                    key,
                    contours,
                    rule,
                    color,
                    opacity,
                } => {
                    self.canvas.fill(
                        &mut self.scratch.cov_cache,
                        key,
                        &contours,
                        rule,
                        color,
                        opacity,
                    );
                    if let Some(rec) = record.as_deref_mut() {
                        rec.push(ReplayJob::Solid {
                            key,
                            rule,
                            color,
                            opacity,
                        });
                    }
                    contours
                }
                DrawJob::Gradient {
                    key,
                    src_key,
                    contours,
                    rule,
                    lut,
                    map,
                } => {
                    self.canvas.fill_gradient(
                        &mut self.scratch.cov_cache,
                        key,
                        src_key,
                        &contours,
                        rule,
                        &lut,
                        &map,
                    );
                    if let Some(rec) = record.as_deref_mut() {
                        rec.push(ReplayJob::Gradient {
                            key,
                            src_key,
                            rule,
                            lut: lut.clone(),
                            map: map.clone(),
                        });
                    }
                    contours
                }
            };
            for c in contours {
                self.scratch.put_pts(c.points);
            }
        }
    }

    /// Replays a static layer's recorded paints straight from the coverage
    /// cache. Returns false (touching nothing) if any needed cache entry
    /// was evicted — the caller then takes the normal path.
    fn replay_jobs(&mut self, jobs: &[ReplayJob]) -> bool {
        let all_present = jobs.iter().all(|j| match j {
            ReplayJob::Solid { key, .. } => self.scratch.cov_cache.contains(*key),
            ReplayJob::Gradient { key, src_key, .. } => {
                self.scratch.cov_cache.contains(*src_key) || self.scratch.cov_cache.contains(*key)
            }
        });
        if !all_present {
            return false;
        }
        self.scratch.cov_cache.pinned = true;
        for j in jobs {
            match j {
                ReplayJob::Solid {
                    key,
                    rule,
                    color,
                    opacity,
                } => {
                    self.canvas.fill(
                        &mut self.scratch.cov_cache,
                        *key,
                        &[],
                        *rule,
                        *color,
                        *opacity,
                    );
                }
                ReplayJob::Gradient {
                    key,
                    src_key,
                    rule,
                    lut,
                    map,
                } => {
                    self.canvas.fill_gradient(
                        &mut self.scratch.cov_cache,
                        *key,
                        *src_key,
                        &[],
                        *rule,
                        lut,
                        map,
                    );
                }
            }
        }
        self.scratch.cov_cache.pinned = false;
        self.scratch.cov_cache.rotate_if_needed();
        true
    }

    fn materialize(&mut self, pj: &PendingJob, arena: &[(Contour, bool)]) -> DrawJob {
        let slice = arena.get(pj.start..pj.end.min(arena.len())).unwrap_or(&[]);
        match &pj.paint {
            PendingPaint::Solid {
                rule,
                color,
                opacity,
            } => {
                let key = self.fill_key(slice, *rule);
                DrawJob::Solid {
                    key,
                    contours: if self.scratch.cov_cache.contains(key) {
                        Vec::new() // hit: coverage replays, geometry unneeded
                    } else {
                        let mut v = Vec::with_capacity(slice.len());
                        for (c, _) in slice {
                            let copy = self.pooled_copy(c);
                            v.push(self.clip_all_owned(copy));
                        }
                        v
                    },
                    rule: *rule,
                    color: *color,
                    opacity: *opacity,
                }
            }
            PendingPaint::Gradient {
                rule,
                lut,
                lut_id,
                map,
            } => {
                let key = self.fill_key(slice, *rule);
                let src_key = Self::src_key_of(key, *lut_id, map);
                DrawJob::Gradient {
                    key,
                    src_key,
                    contours: if self.scratch.cov_cache.contains(src_key)
                        || self.scratch.cov_cache.contains(key)
                    {
                        Vec::new()
                    } else {
                        let mut v = Vec::with_capacity(slice.len());
                        for (c, _) in slice {
                            let copy = self.pooled_copy(c);
                            v.push(self.clip_all_owned(copy));
                        }
                        v
                    },
                    rule: *rule,
                    lut: lut.clone(),
                    map: map.clone(),
                }
            }
            PendingPaint::Stroke {
                color,
                lut,
                opacity,
                hw,
                cap,
                join,
                miter_limit,
                pattern,
                dash_offset,
            } => {
                let stroke_key =
                    self.stroke_key(slice, *hw, *cap, *join, *miter_limit, pattern, *dash_offset);
                let gradient_keys = lut
                    .as_ref()
                    .map(|(_, lut_id, map)| Self::src_key_of(stroke_key, *lut_id, map));
                let hit = match gradient_keys {
                    Some(src_key) => {
                        self.scratch.cov_cache.contains(src_key)
                            || self.scratch.cov_cache.contains(stroke_key)
                    }
                    None => self.scratch.cov_cache.contains(stroke_key),
                };
                if hit {
                    // Hit: coverage replays from the cache — skip dashing,
                    // stroking and clipping wholesale (measured 12-13% of
                    // 64px frames spent stroking geometry whose coverage
                    // was already cached).
                    return match lut {
                        Some((lut, _lut_id, map)) => DrawJob::Gradient {
                            key: stroke_key,
                            src_key: gradient_keys.unwrap_or(stroke_key),
                            contours: Vec::new(),
                            rule: FillRule::NonZero,
                            lut: lut.clone(),
                            map: map.clone(),
                        },
                        None => DrawJob::Solid {
                            key: stroke_key,
                            contours: Vec::new(),
                            rule: FillRule::NonZero,
                            color: color.unwrap_or(Color::BLACK),
                            opacity: *opacity,
                        },
                    };
                }
                let solo = slice.len() == 1 && pattern.is_empty();
                let mut contours: Vec<Contour> = Vec::new();
                let mut pieces: Vec<Contour> = Vec::new();
                for (contour, closed) in slice {
                    // Every stroke emission (segment rect, join wedge or
                    // miter tip, cap) stays within hw·max(miter_limit, √2)
                    // of the source polyline, so ONE inflated-bbox test on
                    // the source proves every piece clip-free — replacing
                    // thousands of per-piece bbox scans (measured 12% of
                    // stroke-heavy 64px frames).
                    let margin = hw * miter_limit.max(1.5);
                    let skip_clip = self.contour_clip_is_noop_inflated(contour, margin);
                    pieces.clear();
                    if pattern.is_empty() {
                        pieces.extend(stroke_polyline(
                            &contour.points,
                            &contour.anchors,
                            *closed,
                            *hw,
                            *cap,
                            *join,
                            *miter_limit,
                            &mut self.scratch.pts_pool,
                            solo,
                        ));
                    } else {
                        for (piece, piece_anchors) in dash_polyline(
                            &contour.points,
                            &contour.anchors,
                            *closed,
                            pattern,
                            *dash_offset,
                        ) {
                            pieces.extend(stroke_polyline(
                                &piece,
                                &piece_anchors,
                                false,
                                *hw,
                                *cap,
                                *join,
                                *miter_limit,
                                &mut self.scratch.pts_pool,
                                false,
                            ));
                        }
                    }
                    if skip_clip {
                        contours.append(&mut pieces);
                    } else {
                        for p in pieces.drain(..) {
                            let clipped = self.clip_all_owned(p);
                            contours.push(clipped);
                        }
                    }
                }
                match lut {
                    Some((lut, _lut_id, map)) => DrawJob::Gradient {
                        key: stroke_key,
                        src_key: gradient_keys.unwrap_or(stroke_key),
                        contours,
                        rule: FillRule::NonZero,
                        lut: lut.clone(),
                        map: map.clone(),
                    },
                    None => DrawJob::Solid {
                        key: stroke_key,
                        contours,
                        rule: FillRule::NonZero,
                        color: color.unwrap_or(Color::BLACK),
                        opacity: *opacity,
                    },
                }
            }
        }
    }

    /// Forward pass. Geometry goes into `arena` (device space, unclipped);
    /// paints record their scope's arena range; trims mutate the arena
    /// range of their scope eagerly.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        shapes: &[Shape],
        m: Mat2x3,
        opacity: f32,
        depth: usize,
        arena: &mut Vec<(Contour, bool)>,
        pending: &mut Vec<PendingJob>,
    ) -> Result<()> {
        if depth > MAX_RENDER_DEPTH {
            return Ok(());
        }
        let scope_start = arena.len();
        let jobs_start = pending.len();
        for shape in shapes {
            match shape {
                Shape::Group(g) => {
                    let (gm, gop) = transform_at(&g.transform, self.frame);
                    let child_m = m.concat(gm);
                    self.walk(&g.shapes, child_m, opacity * gop, depth + 1, arena, pending)?;
                }
                Shape::Path(p) => {
                    let data = p.path.eval(self.frame);
                    let closed = data.closed;
                    arena.push((flatten_path(&data, &m, self.curve_tolerance), closed));
                }
                Shape::Rect(r) => {
                    let pos = r.position.eval(self.frame);
                    let size = r.size.eval(self.frame);
                    let radius = r.radius.eval(self.frame);
                    arena.push((
                        rect_contour(pos, size, radius, r.reversed, &m, self.curve_tolerance),
                        true,
                    ));
                }
                Shape::Ellipse(e) => {
                    let pos = e.position.eval(self.frame);
                    let size = e.size.eval(self.frame);
                    arena.push((
                        ellipse_contour(pos, size, e.reversed, &m, self.curve_tolerance),
                        true,
                    ));
                }
                Shape::Polystar(ps) => {
                    let data = polystar_path(
                        ps.star,
                        ps.reversed,
                        ps.points.eval(self.frame),
                        ps.position.eval(self.frame),
                        ps.rotation.eval(self.frame),
                        ps.inner_radius.eval(self.frame),
                        ps.outer_radius.eval(self.frame),
                        ps.inner_roundness.eval(self.frame),
                        ps.outer_roundness.eval(self.frame),
                    );
                    arena.push((flatten_path(&data, &m, self.curve_tolerance), true));
                }
                Shape::RoundCorners(rc) => {
                    let radius = rc.radius.eval(self.frame);
                    let det = (m.a * m.d - m.b * m.c).abs();
                    let r = radius * det.sqrt();
                    if r > 0.0 {
                        if let Some(range) = arena.get_mut(scope_start..) {
                            for (contour, closed) in range {
                                *contour = round_polyline_corners(contour, *closed, r);
                            }
                        }
                    }
                }
                Shape::Trim(tr) => {
                    self.apply_trim(tr, arena, pending, scope_start);
                }
                Shape::Repeater(rp) => {
                    self.apply_repeater(rp, m, arena, pending, scope_start, jobs_start);
                }
                Shape::Fill(f) => {
                    let color = f.color.eval(self.frame);
                    let fill_opacity = (f.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
                    let paint_opacity = opacity * fill_opacity;
                    if opacity_byte(color.a * paint_opacity) == 0 {
                        continue;
                    }
                    pending.push(PendingJob {
                        paint: PendingPaint::Solid {
                            rule: f.rule,
                            color,
                            opacity: paint_opacity,
                        },
                        start: scope_start,
                        // rlottie: a paint covers only geometry that
                        // precedes it in the walk (verified: a fill/stroke
                        // listed before a path or nested group does NOT
                        // paint that later geometry).
                        end: arena.len(), // resolved at scope end
                    });
                }
                Shape::GradientFill(gf) => {
                    // Gradient geometry stays in local space; pixels are
                    // pulled back through the inverse CTM at fill time.
                    let fill_opacity = (gf.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
                    let paint_opacity = opacity * fill_opacity;
                    if opacity_byte(paint_opacity) == 0 {
                        continue;
                    }
                    let start_p = gf.start.eval(self.frame);
                    let end_p = gf.end.eval(self.frame);
                    let inv = m.inverse();
                    let stops = gf.stops.eval(self.frame);
                    let (lut, lut_id) = self.scratch.lut_for(&stops, gf.color_count, paint_opacity);
                    let map = match gf.kind {
                        GradientKind::Linear => linear_map(start_p, end_p, inv),
                        GradientKind::Radial => radial_map(
                            start_p,
                            end_p,
                            inv,
                            gf.highlight_len.eval(self.frame),
                            gf.highlight_angle.eval(self.frame),
                        ),
                    };
                    pending.push(PendingJob {
                        paint: PendingPaint::Gradient {
                            rule: gf.rule,
                            lut,
                            lut_id,
                            map,
                        },
                        start: scope_start,
                        // rlottie: a paint covers only geometry that
                        // precedes it in the walk (verified: a fill/stroke
                        // listed before a path or nested group does NOT
                        // paint that later geometry).
                        end: arena.len(),
                    });
                }
                Shape::Stroke(st) => {
                    let stroke_opacity = (st.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
                    let paint_opacity = opacity * stroke_opacity;
                    let color = st.color.eval(self.frame);
                    if opacity_byte(color.a * paint_opacity) == 0 {
                        continue;
                    }
                    let width = st.width.eval(self.frame);
                    let scale = stroke_scale(&m);
                    let hw = 0.5 * width * scale;
                    if hw <= 0.0 || !hw.is_finite() {
                        continue;
                    }
                    let (pattern, dash_offset) = self.dash_pattern(&st.dashes, scale);
                    pending.push(PendingJob {
                        paint: PendingPaint::Stroke {
                            color: Some(color),
                            lut: None,
                            opacity: paint_opacity,
                            hw,
                            cap: st.cap,
                            join: st.join,
                            miter_limit: st.miter_limit,
                            pattern,
                            dash_offset,
                        },
                        start: scope_start,
                        // rlottie: a paint covers only geometry that
                        // precedes it in the walk (verified: a fill/stroke
                        // listed before a path or nested group does NOT
                        // paint that later geometry).
                        end: arena.len(),
                    });
                }
                Shape::GradientStroke(gs) => {
                    let stroke_opacity = (gs.opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
                    let paint_opacity = opacity * stroke_opacity;
                    if opacity_byte(paint_opacity) == 0 {
                        continue;
                    }
                    let width = gs.width.eval(self.frame);
                    let scale = stroke_scale(&m);
                    let hw = 0.5 * width * scale;
                    if hw <= 0.0 || !hw.is_finite() {
                        continue;
                    }
                    let start_p = gs.start.eval(self.frame);
                    let end_p = gs.end.eval(self.frame);
                    let inv = m.inverse();
                    let stops = gs.stops.eval(self.frame);
                    let (lut, lut_id) = self.scratch.lut_for(&stops, gs.color_count, paint_opacity);
                    let map = match gs.kind {
                        GradientKind::Linear => linear_map(start_p, end_p, inv),
                        GradientKind::Radial => radial_map(
                            start_p,
                            end_p,
                            inv,
                            gs.highlight_len.eval(self.frame),
                            gs.highlight_angle.eval(self.frame),
                        ),
                    };
                    let (pattern, dash_offset) = self.dash_pattern(&gs.dashes, scale);
                    pending.push(PendingJob {
                        paint: PendingPaint::Stroke {
                            color: None,
                            lut: Some((lut, lut_id, map)),
                            opacity: opacity * stroke_opacity,
                            hw,
                            cap: gs.cap,
                            join: gs.join,
                            miter_limit: gs.miter_limit,
                            pattern,
                            dash_offset,
                        },
                        start: scope_start,
                        // rlottie: a paint covers only geometry that
                        // precedes it in the walk (verified: a fill/stroke
                        // listed before a path or nested group does NOT
                        // paint that later geometry).
                        end: arena.len(),
                    });
                }
            }
        }
        let _ = jobs_start;
        Ok(())
    }

    /// Dash array, rlottie semantics (model::Dash::getDashInfo): values are
    /// consumed POSITIONALLY in file order — the `n` role tags are ignored —
    /// and the LAST value is the offset. An even-length list gets rlottie's
    /// fixup first: the last value moves to the end as the offset and the
    /// second-to-last is duplicated into its place as a synthesized gap
    /// ([d,g] → dash=d, gap=d, offset=g — NOT period d+g).
    fn dash_pattern(&self, dashes: &[DashElement], scale: f32) -> (Vec<f32>, f32) {
        let mut raw: Vec<f32> = dashes
            .iter()
            .map(|d| d.value.eval(self.frame) * scale)
            .collect();
        if raw.len() <= 1 {
            return (Vec::new(), 0.0);
        }
        if raw.len() % 2 == 0 {
            let last = raw.last().copied().unwrap_or(0.0);
            let prev = raw.get(raw.len() - 2).copied().unwrap_or(0.0);
            if let Some(slot) = raw.last_mut() {
                *slot = prev; // duplicate previous dash as the missing gap
            }
            raw.push(last); // original last value becomes the offset
        }
        let offset = raw.pop().unwrap_or(0.0);
        for v in raw.iter_mut() {
            *v = v.max(0.0);
        }
        // AE "draw the dash even if dash value is 0" quirk: rlottie
        // (lottieitem.cpp LOTStrokeItem::updateRenderNode) forces the first
        // dash length to 0.1 when it is zero, AFTER scaling. This flips
        // VDasher's mNoLength false so a zero-dash/zero-gap array (`[d:0,o:0]`,
        // e.g. ShibaInu's heart outlines) reaches the `mNoGap → return solid
        // path` branch and renders as a SOLID stroke rather than nothing.
        if let Some(first) = raw.first_mut() {
            if first.abs() < 1e-6 {
                *first = 0.1;
            }
        }
        (raw, offset)
    }

    /// Repeater: replaces this scope's geometry with transformed copies and
    /// duplicates this scope's earlier paints per copy.
    fn apply_repeater(
        &self,
        rp: &crate::model::Repeater,
        m: Mat2x3,
        arena: &mut Vec<(Contour, bool)>,
        pending: &mut Vec<PendingJob>,
        scope_start: usize,
        jobs_start: usize,
    ) {
        let copies = rp.copies.eval(self.frame).clamp(0.0, 64.0) as usize;
        if copies <= 1 {
            return;
        }
        let offset = rp.offset.eval(self.frame);
        let so = (rp.start_opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
        let eo = (rp.end_opacity.eval(self.frame) / 100.0).clamp(0.0, 1.0);
        // rlottie builds each copy's matrix PARAMETRICALLY with multiplier
        // mult = offset + i (LOTRepeaterTransform::matrix): position and
        // rotation scale linearly, scale is raised to the mult power, the
        // anchor applies once. A matrix power step^k is NOT the same once
        // rotation is nonzero (it spirals; the reference draws a row).
        let rp_anchor = rp.transform.anchor.eval(self.frame);
        let rp_pos = rp.transform.position.eval(self.frame);
        let rp_scale = rp.transform.scale.eval(self.frame);
        let rp_rot = rp.transform.rotation.eval(self.frame);
        let m_inv = m.inverse();
        let base_end = arena.len();
        let base_len = base_end - scope_start;
        let prior_jobs: Vec<(usize, usize, usize)> = pending
            .iter()
            .enumerate()
            .skip(jobs_start)
            .map(|(i, pj)| (i, pj.start, pj.end))
            .collect();
        for i in 0..copies {
            let mult = offset + i as f32;
            let t_local = Mat2x3::translate(rp_pos.x * mult, rp_pos.y * mult)
                .concat(Mat2x3::translate(rp_anchor.x, rp_anchor.y))
                .concat(Mat2x3::scale(
                    (rp_scale.x / 100.0).powf(mult),
                    (rp_scale.y / 100.0).powf(mult),
                ))
                .concat(Mat2x3::rotate(rp_rot * mult))
                .concat(Mat2x3::translate(-rp_anchor.x, -rp_anchor.y));
            let t_dev = m.concat(t_local).concat(m_inv);
            let alpha = if copies > 1 {
                so + (eo - so) * (i as f32 / (copies - 1) as f32)
            } else {
                so
            };
            let block_start = arena.len();
            for idx in scope_start..base_end {
                let (contour, closed) = match arena.get(idx) {
                    Some((c, cl)) => (
                        Contour {
                            points: c.points.iter().map(|p| t_dev.apply(*p)).collect(),
                            anchors: c.anchors.clone(),
                            // Repeater copies carry an extra device-space
                            // conjugated transform; the flatten matrix no
                            // longer describes them (device measure).
                            inv_lin: None,
                        },
                        *cl,
                    ),
                    None => continue,
                };
                arena.push((contour, closed));
            }
            for &(ji, js, je) in &prior_jobs {
                if let Some(orig) = pending.get(ji) {
                    let shift = block_start - scope_start;
                    let np = clone_paint(&orig.paint, alpha);
                    pending.push(PendingJob {
                        paint: np,
                        start: js + shift,
                        end: if je == usize::MAX {
                            usize::MAX
                        } else {
                            je + shift
                        },
                    });
                }
            }
        }
        // Originals are replaced by the copies: blank the base geometry and
        // disarm the original paint jobs.
        if let Some(range) = arena.get_mut(scope_start..base_end) {
            for (c, _) in range {
                c.points.clear();
            }
        }
        let _ = base_len;
        for pj in pending.iter_mut().skip(jobs_start).take(prior_jobs.len()) {
            pj.end = pj.start; // empty range
        }
    }

    fn apply_trim(
        &self,
        tr: &crate::model::Trim,
        arena: &mut Vec<(Contour, bool)>,
        pending: &mut Vec<PendingJob>,
        scope_start: usize,
    ) {
        let start_pct = tr.start.eval(self.frame) / 100.0;
        let end_pct = tr.end.eval(self.frame) / 100.0;
        // rlottie LOTTrimData::segment() (lottiemodel.h): offset is
        // fmod(deg,360)/360 and a window pushed past the path end WRAPS,
        // yielding TWO ranges — a "loop" segment with ss > ee.
        let offset = (tr.offset.eval(self.frame) % 360.0) / 360.0;
        let diff = (start_pct - end_pct).abs();
        if diff <= 1e-6 {
            if let Some(geoms) = arena.get_mut(scope_start..) {
                for (c, _) in geoms {
                    c.points.clear();
                }
            }
            return;
        }
        if diff >= 1.0 - 1e-6 {
            return; // full path
        }
        let s = start_pct + offset;
        let e = end_pct + offset;
        let noloop = |a: f32, b: f32| (a.min(b), a.max(b));
        let loopf = |a: f32, b: f32| (a.max(b), a.min(b)); // ss > ee marks wrap
        let (ss, ee) = if offset >= 0.0 {
            if s <= 1.0 && e <= 1.0 {
                noloop(s, e)
            } else if s > 1.0 && e > 1.0 {
                noloop(s - 1.0, e - 1.0)
            } else if s > 1.0 {
                loopf(s - 1.0, e)
            } else {
                loopf(s, e - 1.0)
            }
        } else if s >= 0.0 && e >= 0.0 {
            noloop(s, e)
        } else if s < 0.0 && e < 0.0 {
            noloop(1.0 + s, 1.0 + e)
        } else if s < 0.0 {
            loopf(1.0 + s, e)
        } else {
            loopf(s, 1.0 + e)
        };
        // Fractional ranges along the path; two when the window wraps.
        let ranges: [(f32, f32); 2] = if ss <= ee {
            [(ss, ee), (0.0, 0.0)]
        } else {
            [(0.0, ee), (ss, 1.0)]
        };
        let range_count = if ss <= ee { 1 } else { 2 };

        match tr.mode {
            TrimMode::Simultaneous => {
                let mut i = scope_start;
                while i < arena.len() {
                    let Some((contour, closed)) = arena.get(i) else {
                        break;
                    };
                    let closed = *closed;
                    let total = polyline_length(&contour.points, closed, contour.inv_lin);
                    if closed && range_count == 2 {
                        // Closed path: the wrapped window is one continuous
                        // piece across the seam; extract_by_length wraps.
                        let Some((contour, cl)) = arena.get_mut(i) else {
                            break;
                        };
                        let (pts, anc) = extract_by_length(
                            &contour.points,
                            &contour.anchors,
                            true,
                            ss * total,
                            (1.0 + ee) * total,
                            contour.inv_lin,
                        );
                        contour.points = pts;
                        contour.anchors = anc;
                        *cl = false;
                        i += 1;
                        continue;
                    }
                    let mut pieces: Vec<(Vec<Vec2>, Vec<bool>)> = Vec::with_capacity(range_count);
                    for &(lo, hi) in ranges.iter().take(range_count) {
                        if hi > lo + 1e-6 {
                            let Some((contour, _)) = arena.get(i) else {
                                break;
                            };
                            let piece = extract_by_length(
                                &contour.points,
                                &contour.anchors,
                                closed,
                                lo * total,
                                hi * total,
                                contour.inv_lin,
                            );
                            if piece.0.len() >= 2 {
                                pieces.push(piece);
                            }
                        }
                    }
                    i = splice_trimmed(arena, pending, i, pieces);
                }
            }
            TrimMode::Individual => {
                let totals: Vec<f32> = arena
                    .get(scope_start..)
                    .unwrap_or(&[])
                    .iter()
                    .map(|(c, cl)| polyline_length(&c.points, *cl, c.inv_lin))
                    .collect();
                let grand: f32 = totals.iter().sum();
                if grand <= 1e-6 {
                    return;
                }
                let mut i = scope_start;
                let mut acc = 0.0f32;
                let mut ti = 0usize;
                while i < arena.len() {
                    let total = totals.get(ti).copied().unwrap_or(0.0);
                    let Some((_, closed)) = arena.get(i) else {
                        break;
                    };
                    let closed = *closed;
                    let mut pieces: Vec<(Vec<Vec2>, Vec<bool>)> = Vec::new();
                    for &(lo, hi) in ranges.iter().take(range_count) {
                        let c0 = (lo * grand - acc).clamp(0.0, total);
                        let c1 = (hi * grand - acc).clamp(0.0, total);
                        if c1 > c0 + 1e-6 {
                            let Some((contour, _)) = arena.get(i) else {
                                break;
                            };
                            let piece = extract_by_length(
                                &contour.points,
                                &contour.anchors,
                                closed,
                                c0,
                                c1,
                                contour.inv_lin,
                            );
                            if piece.0.len() >= 2 {
                                pieces.push(piece);
                            }
                        }
                    }
                    acc += total;
                    ti += 1;
                    i = splice_trimmed(arena, pending, i, pieces);
                }
            }
        }
    }

    /// True when clipping is a bit-exact no-op for this contour: its bbox
    /// is inside the viewport AND inside every precomp clip quad (all under
    /// the clippers' own non-strict inside tests, so the Sutherland–Hodgman
    /// passes would return the polygon unchanged). Measured: on precomp-
    /// heavy files 100% of clip calls were fully inside — this one bbox
    /// pass replaces `1 + |clip|` full S-H passes and their allocations.
    fn clip_is_noop(&self, c: &Contour) -> bool {
        let wf = self.canvas.w as f32;
        let hf = self.canvas.h as f32;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &c.points {
            // NaN/inf must take the slow path: min/max ignore NaN, but the
            // clippers' `>= 0.0` comparisons treat NaN as OUTSIDE and cut.
            if !(p.x.is_finite() && p.y.is_finite()) {
                return false;
            }
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        c.points.is_empty()
            || (x0 >= 0.0
                && y0 >= 0.0
                && x1 <= wf
                && y1 <= hf
                && self
                    .clip
                    .iter()
                    .all(|q| quad_contains_box(q, x0, y0, x1, y1)))
    }

    /// Hash of everything besides the paint's own geometry that determines
    /// coverage: canvas dims + the precomp clip-quad stack.
    fn clip_sig(&self) -> Hasher128 {
        let mut h = Hasher128::new();
        h.mix(self.canvas.w as u32);
        h.mix(self.canvas.h as u32);
        h.mix(u32::from(self.canvas.antialias));
        for q in self.clip.iter() {
            for p in q {
                h.mix(p.x.to_bits());
                h.mix(p.y.to_bits());
            }
        }
        h
    }

    /// Source-level coverage key for a fill: exact bits of the UNclipped
    /// arena geometry + rule + clip signature. Coverage is a deterministic
    /// function of these, so a hit can skip snapshot+clip+raster wholesale.
    fn fill_key(&self, slice: &[(Contour, bool)], rule: FillRule) -> u128 {
        let mut h = self.clip_sig();
        h.mix(1); // paint kind tag: fill
        h.mix(match rule {
            FillRule::NonZero => 1,
            FillRule::EvenOdd => 2,
        });
        for (c, _) in slice {
            for p in &c.points {
                h.mix(p.x.to_bits());
                h.mix(p.y.to_bits());
            }
            h.mix(c.points.len() as u32);
        }
        h.finish()
    }

    /// Source-plane key: coverage key + LUT id + gradient map bits.
    fn src_key_of(key: u128, lut_id: u64, map: &GradientMap) -> u128 {
        let mut h = Hasher128::new();
        h.mix(3); // plane kind tag: gradient source
        h.mix(key as u32);
        h.mix((key >> 32) as u32);
        h.mix((key >> 64) as u32);
        h.mix((key >> 96) as u32);
        h.mix(lut_id as u32);
        h.mix((lut_id >> 32) as u32);
        let mh = map.content_hash();
        h.mix(mh as u32);
        h.mix((mh >> 32) as u32);
        h.finish()
    }

    /// Source-level coverage key for a stroke: geometry (points + anchor
    /// flags + closed — all three feed the stroker) + every stroke
    /// parameter + clip signature.
    #[allow(clippy::too_many_arguments)]
    fn stroke_key(
        &self,
        slice: &[(Contour, bool)],
        hw: f32,
        cap: crate::stroke::Cap,
        join: crate::stroke::Join,
        miter_limit: f32,
        pattern: &[f32],
        dash_offset: f32,
    ) -> u128 {
        let mut h = self.clip_sig();
        h.mix(2); // paint kind tag: stroke
        h.mix(hw.to_bits());
        h.mix(miter_limit.to_bits());
        h.mix(match cap {
            crate::stroke::Cap::Butt => 1,
            crate::stroke::Cap::Round => 2,
            crate::stroke::Cap::Square => 3,
        });
        h.mix(match join {
            crate::stroke::Join::Miter => 1,
            crate::stroke::Join::Round => 2,
            crate::stroke::Join::Bevel => 3,
        });
        for v in pattern {
            h.mix(v.to_bits());
        }
        h.mix(pattern.len() as u32);
        h.mix(dash_offset.to_bits());
        for (c, closed) in slice {
            for p in &c.points {
                h.mix(p.x.to_bits());
                h.mix(p.y.to_bits());
            }
            let mut bits: u32 = 0;
            for (i, &a) in c.anchors.iter().enumerate() {
                bits = (bits << 1) | u32::from(a);
                if i % 32 == 31 {
                    h.mix(bits);
                    bits = 0;
                }
            }
            h.mix(bits);
            h.mix(c.anchors.len() as u32);
            h.mix(c.points.len() as u32);
            h.mix(u32::from(*closed));
        }
        h.finish()
    }

    /// Copies a borrowed arena contour into a pooled buffer (anchors are
    /// not carried: nothing after materialization reads them).
    fn pooled_copy(&mut self, c: &Contour) -> Contour {
        let mut v = self.scratch.pts_pool.pop().unwrap_or_default();
        v.clear();
        v.extend_from_slice(&c.points);
        Contour {
            points: v,
            anchors: Vec::new(),
            inv_lin: None,
        }
    }

    /// clip_is_noop for a source polyline whose derived geometry may
    /// extend up to `margin` beyond it (stroke pieces): tests the inflated
    /// bbox against the same non-strict viewport + quad conditions.
    fn contour_clip_is_noop_inflated(&self, c: &Contour, margin: f32) -> bool {
        if !margin.is_finite() {
            return false;
        }
        let wf = self.canvas.w as f32;
        let hf = self.canvas.h as f32;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &c.points {
            if !(p.x.is_finite() && p.y.is_finite()) {
                return false;
            }
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        if c.points.is_empty() {
            return true;
        }
        let (x0, y0, x1, y1) = (x0 - margin, y0 - margin, x1 + margin, y1 + margin);
        x0 >= 0.0
            && y0 >= 0.0
            && x1 <= wf
            && y1 <= hf
            && self
                .clip
                .iter()
                .all(|q| quad_contains_box(q, x0, y0, x1, y1))
    }

    fn clip_all(&self, c: &Contour) -> Contour {
        if self.clip_is_noop(c) {
            return c.clone();
        }
        let wf = self.canvas.w as f32;
        let hf = self.canvas.h as f32;
        let mut c = c.clone();
        for quad in self.clip.iter() {
            c = clip_to_quad(&c, quad);
        }
        clip_contour(&c, wf, hf)
    }

    /// clip_all for OWNED temporaries (stroke pieces): moves the contour
    /// through unchanged when nothing clips — the borrowing variant clones
    /// every piece, which the profiler showed as pure allocator churn.
    fn clip_all_owned(&self, c: Contour) -> Contour {
        if self.clip_is_noop(&c) {
            return c;
        }
        let wf = self.canvas.w as f32;
        let hf = self.canvas.h as f32;
        let mut c = c;
        for quad in self.clip.iter() {
            c = clip_to_quad(&c, quad);
        }
        clip_contour(&c, wf, hf)
    }
}

/// Replaces `arena[idx]` with the trimmed `pieces` (0, 1, or 2 open
/// polylines). Extra pieces are INSERTED right after `idx` so they stay
/// inside every paint range that covered the original contour; recorded
/// job indices past the insertion point are shifted to compensate.
/// Returns the index of the next original entry.
fn splice_trimmed(
    arena: &mut Vec<(Contour, bool)>,
    pending: &mut Vec<PendingJob>,
    idx: usize,
    mut pieces: Vec<(Vec<Vec2>, Vec<bool>)>,
) -> usize {
    let (first, first_anchors) = if pieces.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        pieces.remove(0)
    };
    if let Some((contour, closed)) = arena.get_mut(idx) {
        contour.points = first;
        contour.anchors = first_anchors;
        *closed = false;
    }
    let extra = pieces.len();
    for (k, (pts, anc)) in pieces.into_iter().enumerate() {
        let at = (idx + 1 + k).min(arena.len());
        arena.insert(
            at,
            (
                Contour {
                    points: pts,
                    anchors: anc,
                    inv_lin: None,
                },
                false,
            ),
        );
    }
    if extra > 0 {
        for pj in pending.iter_mut() {
            if pj.start > idx {
                pj.start += extra;
            }
            if pj.end != usize::MAX && pj.end > idx {
                pj.end += extra;
            }
        }
    }
    idx + 1 + extra
}

/// Clones a pending paint with opacity scaled (repeater copies).
fn clone_paint(p: &PendingPaint, alpha: f32) -> PendingPaint {
    match p {
        PendingPaint::Solid {
            rule,
            color,
            opacity,
        } => PendingPaint::Solid {
            rule: *rule,
            color: *color,
            opacity: opacity * alpha,
        },
        PendingPaint::Gradient {
            rule,
            lut,
            lut_id,
            map,
        } => PendingPaint::Gradient {
            rule: *rule,
            lut: lut.clone(),
            lut_id: *lut_id,
            map: map.clone(),
        },
        PendingPaint::Stroke {
            color,
            lut,
            opacity,
            hw,
            cap,
            join,
            miter_limit,
            pattern,
            dash_offset,
        } => PendingPaint::Stroke {
            color: *color,
            lut: lut.clone(),
            opacity: opacity * alpha,
            hw: *hw,
            cap: *cap,
            join: *join,
            miter_limit: *miter_limit,
            pattern: pattern.clone(),
            dash_offset: *dash_offset,
        },
    }
}

/// Scalar stroke-width scale for a (possibly non-uniform) CTM, matching
/// rlottie's getScale() (lottieitem.cpp): |M·(√2,√2)|/2, i.e. the RMS of the
/// mapped axis lengths — NOT the geometric mean det.sqrt(). Equal for
/// uniform scale; diverges exactly where non-uniform-scale strokes did.
fn stroke_scale(m: &Mat2x3) -> f32 {
    (((m.a + m.c) * (m.a + m.c) + (m.b + m.d) * (m.b + m.d)) * 0.5).sqrt()
}

fn linear_map(start: Vec2, end: Vec2, inv: Mat2x3) -> GradientMap {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let len_sq = dx * dx + dy * dy;
    GradientMap {
        inv,
        kind: GradientMapKind::Linear {
            sx: start.x,
            sy: start.y,
            dx,
            dy,
            inv_len_sq: if len_sq > 1e-9 { 1.0 / len_sq } else { 0.0 },
        },
    }
}

/// Polyline length in rlottie's metric (see geometry::seg_len_rlottie) so
/// trim fractions land where rlottie puts them along curves.
fn polyline_length(points: &[crate::math::Vec2], closed: bool, inv: Option<[f32; 4]>) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let seg_count = if closed {
        points.len()
    } else {
        points.len() - 1
    };
    let mut total = 0.0f32;
    for i in 0..seg_count {
        let Some(a) = points.get(i) else { break };
        let Some(b) = points.get(if i + 1 == points.len() { 0 } else { i + 1 }) else {
            break;
        };
        total += crate::geometry::seg_len_measured(b.x - a.x, b.y - a.y, inv);
    }
    total
}

#[cfg(test)]
mod cov_freeze_tests {
    use super::*;

    fn entry(bytes: usize) -> CovEntry {
        CovEntry {
            rows: Vec::new(),
            data: PlaneData::Cov(vec![0u8; bytes]),
        }
    }

    fn cache(budget: usize) -> CovCache {
        CovCache {
            budget,
            ..CovCache::default()
        }
    }

    /// Working set that fits the budget never freezes: capture stays on
    /// (the fleet case — Joker@64's 5.4MB loop in a 12MB budget).
    #[test]
    fn fitting_set_never_freezes() {
        let mut c = cache(1 << 20);
        for loop_pass in 0..3 {
            for k in 0..40u128 {
                if c.get(k).is_none() {
                    assert_eq!(loop_pass, 0, "warm loops must hit");
                    c.insert(k, entry(4 << 10));
                }
            }
            for _ in 0..60 {
                c.frame_tick();
            }
        }
        assert!(c.capture_enabled());
        assert!(c.hits >= 80, "two warm loops of 40 keys");
    }

    /// Loop working set over budget: freezes on the second overflow, the
    /// resident slice keeps hitting every loop, and the era check renews
    /// the frozen set (never clears a productive one).
    #[test]
    fn oversized_loop_freezes_and_keeps_resident_hits() {
        let mut c = cache(64 << 10);
        // 3 "loops" of 40 distinct keys x 4KB = 160KB/loop > 64KB budget.
        let mut warm_hits = 0u32;
        for loop_pass in 0..4 {
            for k in 0..40u128 {
                if c.get(k).is_some() {
                    if loop_pass > 0 {
                        warm_hits += 1;
                    }
                } else {
                    c.insert(k, entry(4 << 10));
                }
            }
            for _ in 0..45 {
                c.frame_tick(); // 4 loops x 45 = 180 = one era boundary
            }
        }
        assert!(c.frozen, "over-budget periodic loop must freeze");
        assert!(warm_hits > 0, "frozen slice must replay across loops");
        // The era boundary fired once during the warm loops; a productive
        // set must have been RENEWED, not cleared.
        assert!(
            !c.young.is_empty() || !c.old.is_empty(),
            "productive frozen set survived the era check"
        );
    }

    /// A frozen set that stops hitting (content moved on) is cleared at
    /// the era boundary and the cache re-learns.
    #[test]
    fn dead_frozen_set_clears_and_relearns() {
        let mut c = cache(64 << 10);
        for k in 0..40u128 {
            c.insert(k, entry(4 << 10));
        }
        assert!(c.frozen, "overflow x2 freezes");
        // A full era with zero lookups on the frozen keys.
        for _ in 0..FREEZE_ERA_FRAMES {
            c.frame_tick();
        }
        assert!(!c.frozen, "dead set unfreezes");
        assert!(c.young.is_empty() && c.old.is_empty(), "dead set cleared");
        // Re-learning admits inserts again.
        c.insert(1000, entry(4 << 10));
        assert!(c.get(1000).is_some());
    }

    /// Old-generation hits are in-place in the LEARNING state too: after
    /// the first rotation, replaying the rotated entries must not churn
    /// them back into young (the promote pathology: with rotation landing
    /// at a loop seam, the whole next loop paid a map remove+reinsert per
    /// hit and re-inflated young_bytes into a premature freeze).
    #[test]
    fn learning_old_hits_are_in_place() {
        let mut c = cache(1 << 20);
        // One overflow only: rotation 1 moves everything to old.
        for k in 0..70u128 {
            c.insert(k, entry(8 << 10));
        }
        assert_eq!(c.rotations, 1);
        assert!(!c.frozen);
        let yb = c.young_bytes;
        let olen = c.old.len();
        for _ in 0..3 {
            for k in 0..70u128 {
                c.get(k);
            }
        }
        assert_eq!(c.young_bytes, yb, "no promote re-accounting");
        assert_eq!(c.old.len(), olen, "old generation untouched");
        assert!(!c.frozen, "hits alone must not push toward freeze");
    }

    /// Frozen lookups must not mutate the resident structure (no
    /// promotion): repeated old-generation hits keep young_bytes stable.
    #[test]
    fn frozen_hits_do_not_promote() {
        let mut c = cache(64 << 10);
        for k in 0..40u128 {
            c.insert(k, entry(4 << 10));
        }
        assert!(c.frozen);
        let yb = c.young_bytes;
        let ylen = c.young.len();
        let olen = c.old.len();
        for _ in 0..3 {
            for k in 0..40u128 {
                c.get(k);
            }
        }
        assert_eq!(c.young_bytes, yb);
        assert_eq!(c.young.len(), ylen);
        assert_eq!(c.old.len(), olen);
    }
}

#[cfg(test)]
mod mask_bound_tests {
    //! Byte-exactness of the offscreen-dirty-box bounding in `build_mask`.
    //!
    //! The property under test: for any bound box `B`, `build_mask(.., B)`
    //! produces bytes inside `B` identical to `build_mask(.., FULL)` (the whole
    //! canvas). Since the full-canvas bound runs the accumulate over every
    //! pixel with the unchanged per-pixel body, `build_mask(.., FULL)` is the
    //! former (pre-bounding) result, so equality inside `B` is exactly the
    //! byte-identical gate — proven here for every mask mode, inverted masks,
    //! multi-mask stacks, and the geometry edge cases (mask larger than canvas,
    //! empty mask, mask fully outside canvas).
    use super::*;
    use crate::math::Vec2;
    use crate::model::{Composition, Mask, PathData, Position, Transform};
    use crate::property::Property;

    fn full_box(w: usize, h: usize) -> DirtyBox {
        DirtyBox {
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
        }
    }

    fn rect_path(x0: f32, y0: f32, x1: f32, y1: f32) -> PathData {
        PathData {
            vertices: vec![
                Vec2::new(x0, y0),
                Vec2::new(x1, y0),
                Vec2::new(x1, y1),
                Vec2::new(x0, y1),
            ],
            in_tangents: vec![Vec2::ZERO; 4],
            out_tangents: vec![Vec2::ZERO; 4],
            closed: true,
        }
    }

    fn mask(mode: u8, invert: bool, path: PathData, opacity: f32) -> Mask {
        Mask {
            mode,
            invert,
            path: Property::Static(path),
            opacity: Property::Static(opacity),
        }
    }

    fn layer_with_masks(masks: Vec<Mask>) -> Layer {
        Layer {
            kind: LayerKind::Shape,
            index: 0,
            parent: None,
            in_point: 0.0,
            out_point: 100.0,
            start_time: 0.0,
            time_stretch: 1.0,
            hidden: false,
            transform: Transform::identity(),
            shapes: Vec::new(),
            ref_id: None,
            precomp_size: None,
            masks,
            matte: None,
            matte_src: false,
            solid: None,
            time_remap: None,
            auto_orient: false,
        }
    }

    fn empty_comp() -> Composition {
        Composition {
            width: 0,
            height: 0,
            frame_rate: 60.0,
            in_point: 0.0,
            out_point: 100.0,
            layers: Vec::new(),
            assets: Vec::new(),
        }
    }

    /// Builds the mask over `bound` with a fresh scratch (so pooled stale bytes
    /// never leak between calls, making the comparison unambiguous).
    fn build(layer: &Layer, w: usize, h: usize, bound: DirtyBox) -> Vec<u8> {
        let comp = empty_comp();
        let ctx = RenderCtx {
            comp: &comp,
            continuous: false,
            antialias: true,
            curve_tolerance: 0.05,
        };
        let mut scratch = RenderScratch::default();
        ctx.build_mask(&mut scratch, layer, Mat2x3::IDENTITY, 0.0, w, h, bound)
    }

    /// Asserts every pixel inside `b` matches the full-canvas reference.
    fn assert_bound_matches(layer: &Layer, w: usize, h: usize, b: DirtyBox) {
        let full = build(layer, w, h, full_box(w, h));
        let bounded = build(layer, w, h, b);
        for y in b.y0..=b.y1 {
            for x in b.x0..=b.x1 {
                let i = y * w + x;
                assert_eq!(
                    full.get(i),
                    bounded.get(i),
                    "mismatch at ({x},{y}) i={i} for bound {:?}",
                    (b.x0, b.y0, b.x1, b.y1)
                );
            }
        }
    }

    // A representative selection of bounds: interior slice of the mask, a box
    // straddling the mask edge, a box entirely outside the mask geometry, a
    // one-pixel box, and the full canvas. Every box is clamped into the canvas
    // (a real DirtyBox is always marked from in-canvas pixel coords, so
    // x1 < w and y1 < h always hold) and degenerate boxes are dropped.
    fn sample_bounds(w: usize, h: usize) -> Vec<DirtyBox> {
        [
            full_box(w, h),
            DirtyBox {
                x0: 20,
                y0: 20,
                x1: 30,
                y1: 30,
            }, // inside the mask rect
            DirtyBox {
                x0: 8,
                y0: 8,
                x1: 18,
                y1: 18,
            }, // straddles the edge
            DirtyBox {
                x0: 40,
                y0: 40,
                x1: 55,
                y1: 55,
            }, // outside the mask rect
            DirtyBox {
                x0: 0,
                y0: 0,
                x1: 4,
                y1: 4,
            }, // corner, outside
            DirtyBox {
                x0: 33,
                y0: 33,
                x1: 33,
                y1: 33,
            }, // single pixel, outside
        ]
        .into_iter()
        .filter_map(|b| {
            let (x1, y1) = (b.x1.min(w - 1), b.y1.min(h - 1));
            (b.x0 <= x1 && b.y0 <= y1).then_some(DirtyBox { x1, y1, ..b })
        })
        .collect()
    }

    #[test]
    fn all_modes_and_inversion_are_bound_exact() {
        let (w, h) = (64, 64);
        // Mask geometry: a rect [10,10]-[35,35] (well inside the canvas).
        for &mode in &[b'a', b's', b'i', b'f'] {
            for &invert in &[false, true] {
                let layer = layer_with_masks(vec![mask(
                    mode,
                    invert,
                    rect_path(10.0, 10.0, 35.0, 35.0),
                    100.0,
                )]);
                for b in sample_bounds(w, h) {
                    assert_bound_matches(&layer, w, h, b);
                }
            }
        }
    }

    #[test]
    fn partial_opacity_is_bound_exact() {
        let (w, h) = (64, 64);
        for &mode in &[b'a', b's', b'i', b'f'] {
            let layer = layer_with_masks(vec![mask(
                mode,
                false,
                rect_path(10.0, 10.0, 35.0, 35.0),
                50.0,
            )]);
            for b in sample_bounds(w, h) {
                assert_bound_matches(&layer, w, h, b);
            }
        }
    }

    #[test]
    fn multi_mask_stack_is_bound_exact() {
        let (w, h) = (64, 64);
        // Add, then subtract an overlapping rect, then intersect a third, with
        // one inverted member — exercises the running-`cur` fold under bounds.
        let layer = layer_with_masks(vec![
            mask(b'a', false, rect_path(8.0, 8.0, 40.0, 40.0), 100.0),
            mask(b's', false, rect_path(20.0, 20.0, 50.0, 50.0), 75.0),
            mask(b'i', true, rect_path(5.0, 5.0, 45.0, 45.0), 100.0),
            mask(b'f', false, rect_path(15.0, 25.0, 30.0, 38.0), 50.0),
        ]);
        for b in sample_bounds(w, h) {
            assert_bound_matches(&layer, w, h, b);
        }
    }

    #[test]
    fn mask_larger_than_canvas_is_bound_exact() {
        let (w, h) = (48, 48);
        // Rect covers the whole canvas and spills far past every edge: after
        // clipping, coverage is 255 everywhere.
        for &mode in &[b'a', b's', b'i', b'f'] {
            for &invert in &[false, true] {
                let layer = layer_with_masks(vec![mask(
                    mode,
                    invert,
                    rect_path(-100.0, -100.0, 200.0, 200.0),
                    100.0,
                )]);
                for b in sample_bounds(w, h) {
                    assert_bound_matches(&layer, w, h, b);
                }
                // Interior pixel: full coverage (t=255). Add non-inverted ⇒ 255.
                if mode == b'a' && !invert {
                    let m = build(
                        &layer,
                        w,
                        h,
                        DirtyBox {
                            x0: 24,
                            y0: 24,
                            x1: 24,
                            y1: 24,
                        },
                    );
                    assert_eq!(m.get(24 * w + 24), Some(&255));
                }
            }
        }
    }

    #[test]
    fn empty_mask_stack_is_bound_exact() {
        let (w, h) = (32, 32);
        // No effective masks (mode 'n'): acc keeps its seed (first_additive
        // defaults true ⇒ 0). Also the truly-empty vec.
        for masks in [
            vec![mask(b'n', false, rect_path(4.0, 4.0, 20.0, 20.0), 100.0)],
            Vec::new(),
        ] {
            let layer = layer_with_masks(masks);
            for b in sample_bounds(w, h)
                .into_iter()
                .filter(|b| b.x1 < w && b.y1 < h)
            {
                assert_bound_matches(&layer, w, h, b);
            }
            // Every sampled pixel is 0 (layer hidden).
            let m = build(
                &layer,
                w,
                h,
                DirtyBox {
                    x0: 5,
                    y0: 5,
                    x1: 15,
                    y1: 15,
                },
            );
            for y in 5..=15 {
                for x in 5..=15 {
                    assert_eq!(m.get(y * w + x), Some(&0));
                }
            }
        }
    }

    #[test]
    fn mask_fully_outside_canvas_is_bound_exact() {
        let (w, h) = (32, 32);
        // Rect lives at (1000,1000)-(1100,1100): clips to nothing, coverage 0.
        for &mode in &[b'a', b's', b'i', b'f'] {
            for &invert in &[false, true] {
                let layer = layer_with_masks(vec![mask(
                    mode,
                    invert,
                    rect_path(1000.0, 1000.0, 1100.0, 1100.0),
                    100.0,
                )]);
                for b in sample_bounds(w, h)
                    .into_iter()
                    .filter(|b| b.x1 < w && b.y1 < h)
                {
                    assert_bound_matches(&layer, w, h, b);
                }
            }
        }
    }

    /// Directly validates the per-mode OUTSIDE-value analysis documented on
    /// `build_mask`: a pixel inside `bound` but outside all mask geometry
    /// (coverage t=0) must equal the analytically derived constant O, and the
    /// bounded build must reproduce it (never leaving stale/zero bytes there).
    #[test]
    fn outside_value_matches_per_mode_analysis() {
        let (w, h) = (64, 64);
        // op = 1.0 ⇒ at t=0: non-inverted c=0, inverted c=255. Expected O:
        //   a: noninv 0,  inv 255      s: noninv 255, inv 0
        //   i: noninv 0,  inv 255      f: noninv 0,   inv 255
        let cases: &[(u8, bool, u8)] = &[
            (b'a', false, 0),
            (b'a', true, 255),
            (b's', false, 255),
            (b's', true, 0),
            (b'i', false, 0),
            (b'i', true, 255),
            (b'f', false, 0),
            (b'f', true, 255),
        ];
        // Bound entirely outside the mask rect [10,10]-[35,35].
        let outside = DirtyBox {
            x0: 44,
            y0: 44,
            x1: 50,
            y1: 50,
        };
        for &(mode, invert, expected) in cases {
            let layer = layer_with_masks(vec![mask(
                mode,
                invert,
                rect_path(10.0, 10.0, 35.0, 35.0),
                100.0,
            )]);
            let m = build(&layer, w, h, outside);
            for y in outside.y0..=outside.y1 {
                for x in outside.x0..=outside.x1 {
                    assert_eq!(
                        m.get(y * w + x),
                        Some(&expected),
                        "mode {} invert {invert} outside value",
                        mode as char
                    );
                }
            }
            // And the same value must appear in the full build (consistency).
            let full = build(&layer, w, h, full_box(w, h));
            assert_eq!(full.get(47 * w + 47), Some(&expected));
        }
    }
}
