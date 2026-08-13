use super::*;
use alloc::vec::Vec;

/// Paint bbox extent (px, max dimension) ABOVE which the sparse cell/span
/// engine rasterizes instead of the dense accumulator. The threshold is
/// centralized here so benchmark-driven tuning cannot diverge by call site.
pub(super) const MODE_S_MIN_EXTENT: usize = 42;

/// One uniform-coverage span, packed: y:20 | x0:20 | len:16 | cov:8.
#[inline]
pub(crate) fn pack_span(y: usize, x0: usize, len: usize, cov: u8) -> u64 {
  ((y as u64) << 44) | ((x0 as u64) << 24) | ((len as u64) << 8) | u64::from(cov)
}

#[inline]
pub(crate) fn unpack_span(s: u64) -> (usize, usize, usize, u8) {
  ((s >> 44) as usize, ((s >> 24) & 0xf_ffff) as usize, ((s >> 8) & 0xffff) as usize, (s & 0xff) as u8)
}

/// Mode-S guard against edge-dense content: the cell engine's cost is
/// ≈ perimeter px (deposit + sort), the dense engine's is ≈ area bytes.
/// Stroke piece unions (thousands of tiny quads) have perimeter of the
/// same order as area — sorting their cell piles loses to the plane
/// (measured: DogsEmoji@320 cold 1.4x, quicksort 21% of the profile).
/// S wins when `perimeter * DENSITY < bbox area`. Canvas-scaled per E156B
/// measurements: 12 remains best at 64, 18 is the selected medium-size
/// crossover after the opaque-row changes, and 6 remains best at 720.
/// Extent 32/42/64 measured flat; 42 kept (the d9d6ad5 crossover).
pub(super) const MODE_S_EDGE_DENSITY_SMALL: f32 = 12.0; // canvas ≤ 160x160
pub(super) const MODE_S_EDGE_DENSITY_MEDIUM: f32 = 18.0; // canvas ≤ 448x448
pub(super) const MODE_S_EDGE_DENSITY_LARGE: f32 = 6.0;

/// Decides the rasterizer mode for one paint: sparse cells (mode S) for
/// large, edge-sparse paints; the dense accumulator (mode D) otherwise.
/// One pass over points — negligible next to rasterization; non-finite
/// points are ignored by f32 min/max.
pub(crate) fn mode_s_wins(contours: &[Contour], canvas_px: usize) -> bool {
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
    return false;
  }
  let density = if canvas_px <= 160 * 160 {
    MODE_S_EDGE_DENSITY_SMALL
  } else if canvas_px <= 448 * 448 {
    MODE_S_EDGE_DENSITY_MEDIUM
  } else {
    MODE_S_EDGE_DENSITY_LARGE
  };
  let s = perim * density < (x1 - x0) * (y1 - y0);
  s
}

/// Rebuilds a row-plane cache entry from a span list (exact: spans arrive
/// y-ascending, x-ascending, gaps are zero coverage). Used when the span
/// list is FRAGMENTED (fine AA content — many len-1..3 spans): row replay
/// is one call per row, span replay pays per-span dispatch (measured 4.5x
/// on DogsEmoji@320 adaptive).
pub(super) fn spans_to_cov_entry(spans: &[u64]) -> CovEntry {
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
pub(super) fn pre_size(spans: &[u64], rows: &mut Vec<(u32, u32, u32)>, data: &mut Vec<u8>) {
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
pub(crate) const SPAN_CAPTURE_MAX: usize = COV_ENTRY_MAX / 8;

/// Fragmentation test for a fresh span capture: average span shorter than
/// 4 px → store as rows (see [`spans_to_cov_entry`]).
pub(super) fn spans_fragmented(spans: &[u64], px_total: usize) -> bool {
  spans.len() * 4 > px_total
}

/// Cached plane payload: rasterizer coverage bytes (mode D), a span list
/// (mode S — denser, replays as uniform-coverage blends), or a gradient
/// paint's premultiplied coverage-scaled SOURCE pixels (replayed as a pure
/// composite — bit-exact because the blend formula is identical).
/// The representation tag lives per entry: geometry keys are mode-blind,
/// but one geometry always has one extent, hence one mode.
pub(crate) enum PlaneData {
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
pub(crate) struct CovEntry {
  pub(crate) rows: Vec<(u32, u32, u32)>,
  pub(crate) data: PlaneData,
}

/// Byte-budgeted map from 128-bit geometry hash to coverage rows.
/// Collisions at 128 bits are negligible (the key deliberately does not
/// store the geometry itself — Joker-class files have ~18k distinct
/// geometries per loop and full keys would dwarf the coverage payload).
/// Eviction: whole-cache clear on budget overflow — periodic animations
/// refill within one loop, and the budget bounds per-instance memory.
pub(super) type CoverageMap = crate::compat::HashMap<u128, CovEntry, core::hash::BuildHasherDefault<CoverageKeyHasher>>;

/// Coverage keys are already two independently mixed 64-bit content hashes.
/// Running SipHash over them at every lookup is redundant and shows up on
/// small Android frames, so fold and avalanche the existing key directly.
#[derive(Default)]
pub(super) struct CoverageKeyHasher(u64);

impl core::hash::Hasher for CoverageKeyHasher {
  #[inline]
  fn finish(&self) -> u64 {
    self.0
  }

  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
      hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    self.0 = avalanche(hash);
  }

  #[inline]
  fn write_u128(&mut self, key: u128) {
    self.0 = avalanche(key as u64 ^ (key >> 64) as u64);
  }
}

#[inline]
fn avalanche(mut value: u64) -> u64 {
  value ^= value >> 33;
  value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
  value ^= value >> 33;
  value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
  value ^ (value >> 33)
}

#[derive(Default)]
pub(crate) struct CovCache {
  /// Canvas-scaled budget (see COV_CACHE_BUDGET docs); 0 = default.
  pub(super) budget: usize,
  /// Canvas-scaled: trim entry Vec slack at insert (see the fleet gate
  /// note in [`CovCache::insert`]). Set alongside `budget`.
  pub(super) shrink_entries: bool,
  /// Rotation guard: static-layer job replay pre-checks that every job's
  /// entry is present, then replays them one by one — but a replay-path
  /// insert may rotate generations and evict a LATER job's entry,
  /// silently dropping that paint (a fill given empty contours on a miss
  /// draws nothing). While pinned, rotation is deferred (young may
  /// briefly exceed budget by one layer's inserts).
  pub(super) pinned: bool,
  /// Two generations: inserts go to `young`; the first time `young`
  /// exceeds half the budget it becomes `old` (nothing dropped), the
  /// second time the cache freezes (see below). Lookups hit either
  /// generation in place. A working set that fits half the budget never
  /// rotates at all; bigger sets freeze at ~budget resident.
  pub(super) young: CoverageMap,
  pub(super) old: CoverageMap,
  pub(super) young_bytes: usize,
  /// Dedicated byte pool for gradient SOURCE planes (≥720px planes are >1MB
  /// and would otherwise blow the shared cov budget / COV_ENTRY_MAX and evict
  /// useful coverage entries — measured: a blanket 2MB COV_ENTRY_MAX cut
  /// LedScreenEmoji@720 −16.5% and GlowingFont −21% but regressed
  /// prtyparrot@320 +7.5% via small-entry eviction). Src entries live in
  /// the same two maps but are counted here and capped by [`SRC_BUDGET`], so
  /// caching a big gradient plane never evicts coverage entries.
  pub(super) src_bytes: usize,
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
  pub(super) hits: u32,
  pub(super) inserts: u32,
  pub(super) rotations: u32,
  pub(super) frozen: bool,
  pub(super) era_frames: u32,
  pub(super) era_hits: u32,
}

/// Young-generation overflows before the resident set freezes (the first
/// overflow rotates an empty old generation away; the second would drop
/// real entries — freeze instead).
pub(super) const FREEZE_ROTATIONS: u32 = 2;
/// Frames per frozen era; one era spans any full tgs loop (≤180 frames),
/// so every still-relevant frozen entry hits at least once per era.
pub(super) const FREEZE_ERA_FRAMES: u32 = 180;

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
pub(super) const COV_CACHE_BUDGET: usize = 3 << 20;
/// Entries bigger than this are not cached — one entry would evict a whole
/// loop's worth of small ones. 256KB admits sticker-size (320px) gradient
/// source planes (measured: the 64KB cap left TableFontEmoji@320 recomputing
/// its gradients every frame while @64 flew).
pub(super) const COV_ENTRY_MAX: usize = 256 << 10;
/// Big gradient source planes are allowed up to this size (a 720×720 src
/// plane is 2MB) but live in the dedicated [`CovCache::src_bytes`] pool, so
/// they never count toward or evict the coverage budget.
pub(super) const SRC_ENTRY_MAX: usize = 4 << 20;
/// Dedicated byte budget for cached gradient source planes.
pub(super) const SRC_BUDGET: usize = 8 << 20;

/// Two independent FNV/Murmur-style 64-bit streams -> 128-bit content key.
pub(super) struct Hasher128 {
  pub(super) h1: u64,
  pub(super) h2: u64,
}

impl Hasher128 {
  pub(super) fn new() -> Hasher128 {
    Hasher128 {
      h1: 0xcbf2_9ce4_8422_2325,
      h2: 0x9e37_79b9_7f4a_7c15,
    }
  }

  #[inline]
  pub(super) fn mix(&mut self, w: u32) {
    self.h1 = (self.h1 ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3);
    self.h2 = (self.h2 ^ u64::from(w.rotate_left(16))).wrapping_mul(0xff51_afd7_ed55_8ccd);
  }

  pub(super) fn finish(&self) -> u128 {
    (u128::from(self.h1) << 64) | u128::from(self.h2)
  }
}

impl CovCache {
  /// Looks a key up in either generation (in place — see the
  /// no-promotion note below).
  pub(crate) fn get(&mut self, key: u128) -> Option<&CovEntry> {
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

  pub(crate) fn contains(&self, key: u128) -> bool {
    self.young.contains_key(&key) || self.old.contains_key(&key)
  }

  pub(crate) fn contains_coverage(&self, key: u128) -> bool {
    self
      .young
      .get(&key)
      .or_else(|| self.old.get(&key))
      .is_some_and(|entry| matches!(entry.data, PlaneData::Cov(_) | PlaneData::Spans(_)))
  }

  pub(super) fn size_of(e: &CovEntry) -> usize {
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

  pub(crate) fn set_budget_for_canvas(&mut self, w: usize, h: usize) {
    // Development hook for overriding the three size-class cache budgets.
    // Reading the environment is the one thing this crate does that core
    // cannot, so no_std builds simply take the compiled-in budgets.
    #[cfg(feature = "std")]
    let ov = {
      static OVERRIDE: std::sync::OnceLock<Option<[usize; 3]>> = std::sync::OnceLock::new();
      OVERRIDE.get_or_init(|| {
        let v = std::env::var("TLOTTIE_COV_BUDGET_KB").ok()?;
        let mut it = v.split(',').map(|s| s.trim().parse::<usize>());
        match (it.next(), it.next(), it.next()) {
          (Some(Ok(a)), Some(Ok(b)), Some(Ok(c))) => Some([a << 10, b << 10, c << 10]),
          _ => None,
        }
      })
    };
    #[cfg(not(feature = "std"))]
    let ov = &None::<[usize; 3]>;
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

  pub(super) fn rotate_if_needed(&mut self) {
    // Frozen is a stable state (young_bytes stays over the rotation
    // threshold by construction) — only the era check may leave it.
    // Without this guard the replay-unpin path re-froze every batch,
    // resetting the era counters (measured: rotations=392 on
    // theopenemojis@320 and era_hits pinned at 0).
    if self.pinned || self.frozen {
      return;
    }
    let budget = if self.budget == 0 { COV_CACHE_BUDGET } else { self.budget };
    // Src planes have a dedicated budget (see SRC_BUDGET). Both young
    // bytes and src bytes share the same rotation, so a gradient-heavy pack
    // ages its plane generations instead of either (a) pooling bytes forever
    // or (b) trashing on an overflow flush.
    if self.young_bytes > budget / 2 || self.src_bytes > SRC_BUDGET / 2 {
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
        // Tiny-entry canvases normally grow the next generation to roughly
        // the same entry count. Seed it with the capacity the first one
        // already proved useful, avoiding a second allocate-and-rehash ladder
        // during first playback. Large canvases keep gradual growth: their
        // fewer, bigger entries did not repay the eager table allocation.
        if !self.shrink_entries {
          self.young = CoverageMap::with_capacity_and_hasher(self.old.capacity(), core::hash::BuildHasherDefault::default());
        }
        self.young_bytes = 0;
        self.src_bytes = 0;
      }
    }
  }

  /// True while the cache is still learning (capture pays for itself);
  /// false while frozen (the resident set replays, nothing new admits).
  #[inline]
  pub(crate) fn capture_enabled(&self) -> bool {
    !self.frozen
  }

  /// Per-frame tick: while frozen, count frames toward the era check —
  /// a frozen set that kept hitting (≥ half its entries over an era
  /// that always spans a full loop) is renewed; a dead one is cleared
  /// and the cache re-learns from scratch.
  pub(crate) fn frame_tick(&mut self) {
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
          self.src_bytes = 0;
          self.frozen = false;
          self.hits = 0;
          self.inserts = 0;
          self.rotations = 0;
        }
      }
    }
  }

  pub(crate) fn insert(&mut self, key: u128, mut entry: CovEntry) {
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
    if matches!(entry.data, PlaneData::Src(_)) {
      // Gradient source planes live in a dedicated pool (see `src_bytes`);
      // they never count toward the coverage budget, and they age out through
      // the same young/old rotation (triggered by their own pressure) instead
      // of pooling until an overflow flush — a hard flush made big-720 packs
      // re-capture every frame (deadline: GlowingFont@720 regressed to
      // -0.08% from -21% under the blanket raise).
      if sz > SRC_ENTRY_MAX {
        return;
      }
      if self.shrink_entries {
        entry.rows.shrink_to_fit();
        if let PlaneData::Src(v) = &mut entry.data {
          v.shrink_to_fit();
        }
      }
      self.inserts = self.inserts.saturating_add(1);
      self.src_bytes += sz;
      self.rotate_if_needed();
      self.young.insert(key, entry);
      return;
    }
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
