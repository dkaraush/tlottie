use super::*;
use alloc::vec;
use alloc::vec::Vec;

fn entry(bytes: usize) -> CovEntry {
  CovEntry {
    rows: Vec::new(),
    data: PlaneData::Cov(vec![0u8; bytes]),
  }
}

fn cache(budget: usize) -> CovCache {
  CovCache { budget, ..CovCache::default() }
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
  assert!(!c.young.is_empty() || !c.old.is_empty(), "productive frozen set survived the era check");
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
  assert!(c.young.capacity() >= c.old.capacity(), "tiny-entry successor inherits proven table capacity");
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
