//! Denial-of-service regression cases.
//!
//! Each test is a self-contained piece of untrusted JSON that once made
//! tlottie do unbounded (or super-linear) work. Every case runs on a worker
//! thread through [`assert_lottie!`], which fails the test if the work does not
//! finish inside a **time** budget (a CPU hang) OR crosses a **memory** budget
//! (an allocation blow-up), so a regression that reintroduces either failure
//! mode trips a budget instead of wedging the run or OOM-ing the machine. The
//! fixed code finishes each case in a few ms holding a few MB; the original
//! behavior ran for seconds-to-forever or ballooned to many GB.

// A budget breach is reported by panicking from the wait loop (not a #[test]
// fn, so the usual test exemption does not apply); that panic is the assertion.
#![allow(clippy::panic)]
// The accounting allocator below is the crate's only test-side `unsafe`; it
// just forwards to the system allocator while counting live bytes.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::{CPURenderer, Composition, Error, Limits, RenderOptions};

const FRAME_TIME_BUDGET: Duration = Duration::from_millis(1_500);
const MEM_BUDGET: usize = 32 << 20;

thread_local! {
  /// Per-thread pointer to the live-byte counter of the worker running on
  /// this thread (null on every other thread: the main test thread, other
  /// tests' workers, rustc's own threads). Const-initialized so that reading
  /// it inside the allocator never itself allocates: no reentrancy, and each
  /// worker accounts only to its OWN counter, so concurrently-running cases
  /// never contaminate each other's measurement.
  static WORKER_BYTES: Cell<*const AtomicUsize> = const { Cell::new(std::ptr::null()) };
}

/// Adds `delta` to the current thread's worker counter, if one is installed,
/// and freezes the thread if that pushes it over the memory budget.
fn account(delta: isize) {
  let counter = WORKER_BYTES.with(Cell::get);
  if counter.is_null() {
    return;
  }
  // SAFETY: a non-null pointer here was installed by an [`assert_lottie!`]
  // worker from an `Arc<AtomicUsize>` it (and the waiting main thread) keep
  // alive for at least as long as this thread runs.
  let counter = unsafe { &*counter };
  if delta < 0 {
    counter.fetch_sub(delta.unsigned_abs(), Ordering::Relaxed);
    return;
  }
  let total = counter.fetch_add(delta as usize, Ordering::Relaxed) + delta as usize;
  if total > MEM_BUDGET {
    // This worker just blew the budget. The wait loop will read the counter
    // and fail the case; park the worker HERE before it can allocate any
    // further, so a detached memory-DoS worker holds ~MEM_BUDGET forever
    // instead of racing to an OOM while slower cases keep the process alive.
    // sleep/park never allocate, so this cannot re-enter the allocator.
    loop {
      thread::sleep(Duration::from_secs(3600));
    }
  }
}

/// System allocator that attributes each allocation to the worker running on
/// the allocating thread, so a case that balloons memory fails its assertion
/// instead of OOM-ing the test binary.
struct Accounting;

unsafe impl GlobalAlloc for Accounting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    let ptr = unsafe { System.alloc(layout) };
    if !ptr.is_null() {
      account(layout.size() as isize);
    }
    ptr
  }
  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) };
    account(-(layout.size() as isize));
  }
  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
    if !new_ptr.is_null() {
      account(new_size as isize - layout.size() as isize);
    }
    new_ptr
  }
}

#[global_allocator]
static ACCOUNTING: Accounting = Accounting;

/// Parses `$json` and renders frames `[frame_start, frame_end)` at 64x64,
/// asserting the whole pipeline succeeds without hanging. `frame_start` and
/// `frame_end` default to `0` and `1` (a single frame); the timeout budget
/// scales to `frame_end - frame_start` frames' worth of [`FRAME_TIME_BUDGET`].
macro_rules! assert_lottie {
  ($json: expr) => {
    assert_lottie!($json, 64, 0, 1, RenderOptions::default());
  };
  ($json: expr, $size: expr) => {
    assert_lottie!($json, $size, 0, 1, RenderOptions::default(), false);
  };
  ($json: expr, $size: expr, $frame_start: expr, $frame_end: expr) => {
    assert_lottie!($json, $size, $frame_start, $frame_end, RenderOptions::default(), false);
  };
  ($json: expr, $size: expr, $frame_start: expr, $frame_end: expr, $options: expr) => {
    assert_lottie!($json, $size, $frame_start, $frame_end, $options, false);
  };
  ($json: expr, $size: expr, $frame_start: expr, $frame_end: expr, $options: expr, $expect_error: expr) => {
    let json = $json;
    let size: u32 = $size;
    let frame_start: u32 = $frame_start;
    let frame_end: u32 = $frame_end;
    let options: RenderOptions = $options;
    let expect_error: bool = $expect_error;
    let budget = (frame_end - frame_start) * FRAME_TIME_BUDGET;
    let counter = Arc::new(AtomicUsize::new(0));
    let worker_counter = Arc::clone(&counter);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
      WORKER_BYTES.with(|slot| slot.set(Arc::as_ptr(&worker_counter)));
      let result = (|| -> Result<(), Error> {
        let composition = Composition::parse(json.as_bytes(), &Limits::default())?;
        let mut cpu = CPURenderer::new(composition);
        if options.alpha_only {
          let mut alpha = vec![0u8; (size * size).try_into().unwrap()];
          for frame in frame_start..frame_end {
            cpu.render_alpha8(frame as f32, &mut alpha, size, size, options)?;
          }
        } else {
          let mut pixels = vec![0u32; (size * size).try_into().unwrap()];
          for frame in frame_start..frame_end {
            cpu.render(frame as f32, &mut pixels, size, size, options)?;
          }
        }
        Ok(())
      })();
      WORKER_BYTES.with(|slot| slot.set(std::ptr::null()));
      let _ = tx.send(result);
    });
    let start = Instant::now();
    let rendered = loop {
      match rx.try_recv() {
        Ok(result) => break result,
        Err(TryRecvError::Disconnected) => panic!("render worker vanished before reporting"),
        Err(TryRecvError::Empty) => {}
      }
      let used = counter.load(Ordering::Relaxed);
      assert!(used <= MEM_BUDGET, "render allocated {used} bytes, over the {MEM_BUDGET}-byte budget");
      assert!(start.elapsed() <= budget, "render did not finish within the frame budget");
      thread::sleep(Duration::from_millis(1));
    };
    if expect_error {
      assert!(rendered.is_err(), "malformed input unexpectedly rendered successfully");
    } else {
      assert!(rendered.is_ok() || matches!(rendered, Err(Error::LimitExceeded(_))), "{rendered:?}");
    }
  };
}

fn cumulative_shape_paints_json(pairs: usize) -> String {
  let mut items = String::new();
  for i in 0..pairs {
    items.push_str(&format!(
      r#"{{"ty":"rc","p":{{"a":0,"k":[{},{}]}},"s":{{"a":0,"k":[{},{}]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},"#,
      i % 64,
      (i / 64) % 64,
      1.0 + (i % 251) as f32 * 0.1,
      1.0 + (i % 251) as f32 * 0.1
    ));
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":240,"layers":[{{"ty":4,"ind":0,"ip":0,"op":240,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{items}{{"ty":"tr"}}]}}]}}]}}"#)
}

fn focal_radial_gradients_json(fills_count: usize) -> String {
  let mut fills = String::new();
  for i in 0..fills_count {
    fills.push_str(&format!(
      r#"{{"ty":"gf","t":2,"s":{{"a":0,"k":[256,256]}},"e":{{"a":0,"k":[{},{}]}},"h":{{"a":0,"k":99}},"a":{{"a":0,"k":{}}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":0,"k":100}}}},"#,
      256 + (i % 17),
      512 - (i % 19),
      i % 360
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":512,"h":512,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[256,256]}},"s":{{"a":0,"k":[512,512]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn focal_radial_gradient_strokes_json(strokes_count: usize) -> String {
  let mut strokes = String::new();
  for i in 0..strokes_count {
    strokes.push_str(&format!(
      r#"{{"ty":"gs","t":2,"s":{{"a":0,"k":[256,256]}},"e":{{"a":0,"k":[{},{}]}},"h":{{"a":0,"k":99}},"a":{{"a":0,"k":{}}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":512}},"lc":2,"lj":2}},"#,
      256 + (i % 17),
      512 - (i % 19),
      i % 360
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":512,"h":512,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,256],[512,256]],"i":[],"o":[]}}}}}},{strokes}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn reused_precomp_focal_radial_gradients_json(refs_count: usize, fills_count: usize) -> String {
  let mut refs = String::new();
  for i in 0..refs_count {
    if i > 0 {
      refs.push(',');
    }
    refs.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"radial","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let mut fills = String::new();
  for i in 0..fills_count {
    fills.push_str(&format!(
      r#"{{"ty":"gf","t":2,"s":{{"a":0,"k":[256,256]}},"e":{{"a":0,"k":[{},{}]}},"h":{{"a":0,"k":99}},"a":{{"a":0,"k":{}}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":0,"k":100}}}},"#,
      256 + (i % 17),
      512 - (i % 19),
      i % 360
    ));
  }

  format!(
    r#"{{"v":"5.5.0","w":512,"h":512,"fr":60,"ip":0,"op":60,"layers":[{refs}],"assets":[{{"id":"radial","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[256,256]}},"s":{{"a":0,"k":[512,512]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn reused_precomp_animated_linear_gradients_json(refs_count: usize, fills_count: usize) -> String {
  let mut refs = String::new();
  for i in 0..refs_count {
    if i > 0 {
      refs.push(',');
    }
    refs.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"linear","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let mut fills = String::new();
  for _ in 0..fills_count {
    fills.push_str(r#"{"ty":"gf","s":{"a":0,"k":[0,0]},"e":{"a":0,"k":[256,256]},"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"o":{"a":1,"k":[{"t":0,"s":[100]},{"t":1,"s":[100]}]}},"#);
  }

  format!(
    r#"{{"v":"5.5.0","w":256,"h":256,"fr":60,"ip":0,"op":60,"layers":[{refs}],"assets":[{{"id":"linear","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[128,128]}},"s":{{"a":0,"k":[256,256]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn reused_precomp_animated_linear_gradient_strokes_json(refs_count: usize, strokes_count: usize) -> String {
  let mut refs = String::new();
  for i in 0..refs_count {
    if i > 0 {
      refs.push(',');
    }
    refs.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"linear-strokes","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let mut strokes = String::new();
  for _ in 0..strokes_count {
    strokes.push_str(r#"{"ty":"gs","s":{"a":0,"k":[0,0]},"e":{"a":0,"k":[256,256]},"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"o":{"a":1,"k":[{"t":0,"s":[100]},{"t":1,"s":[100]}]},"w":{"a":0,"k":256},"lc":2,"lj":2},"#);
  }

  format!(
    r#"{{"v":"5.5.0","w":256,"h":256,"fr":60,"ip":0,"op":60,"layers":[{refs}],"assets":[{{"id":"linear-strokes","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,128],[256,128]],"i":[],"o":[]}}}}}},{strokes}{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn clipped_dashed_strokes_json(strokes_count: usize, line_cap: u32) -> String {
  let mut items = String::new();
  for i in 0..strokes_count {
    let y = i % 64;
    items.push_str(&format!(
      r#"{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[-1000,{y}],[1000,{y}]],"i":[],"o":[]}}}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":16}},"lc":{line_cap},"lj":1,"d":[{{"n":"d","v":{{"a":0,"k":1}}}},{{"n":"g","v":{{"a":0,"k":1}}}}]}},"#
    ));
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{items}{{"ty":"tr"}}]}}]}}]}}"#)
}

fn clipped_dashed_zigzag_json(vertices_count: usize) -> String {
  let mut verts = String::new();
  for i in 0..vertices_count {
    if i > 0 {
      verts.push(',');
    }
    let y = if i % 2 == 0 { -1000 } else { 1000 };
    verts.push_str(&format!("[{},{}]", i % 64, y));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":16}},"lc":2,"lj":2,"d":[{{"n":"d","v":{{"a":0,"k":1}}}},{{"n":"g","v":{{"a":0,"k":1}}}}]}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn animated_tiny_dash_period_json() -> &'static str {
  r#"{
    "v":"5.5.0","w":720,"h":720,"fr":60,"ip":0,"op":2,
    "layers":[{"ty":4,"ind":1,"ip":0,"op":2,"st":0,"ks":{},"shapes":[{"ty":"gr","it":[
      {"ty":"sh","ks":{"a":0,"k":{"c":false,"v":[[-140000,360],[140000,360]],"i":[[0,0],[0,0]],"o":[[0,0],[0,0]]}}},
      {"ty":"st","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"w":{"a":0,"k":1},"lc":1,"lj":1,
       "d":[
         {"n":"d","v":{"a":1,"k":[{"t":0,"s":[1000],"e":[0.0011]},{"t":1,"s":[0.0011]}]}},
         {"n":"g","v":{"a":1,"k":[{"t":0,"s":[1000],"e":[0.0011]},{"t":1,"s":[0.0011]}]}},
         {"n":"d","v":{"a":1,"k":[{"t":0,"s":[1000],"e":[0.0011]},{"t":1,"s":[0.0011]}]}},
         {"n":"g","v":{"a":1,"k":[{"t":0,"s":[1000],"e":[0.0011]},{"t":1,"s":[0.0011]}]}},
         {"n":"o","v":{"a":0,"k":0}}
       ]},
      {"ty":"tr"}
    ]}]}]
  }"#
}

fn animated_opacity_tail_gradient_json(opacity_pairs: usize) -> String {
  let mut stops0 = String::from("0,1,0,0");
  let mut stops1 = String::from("0,1,0,0");
  let denom = (opacity_pairs - 1) as f64;
  for i in 0..opacity_pairs {
    let offset = format!("{:.8}", i as f64 / denom);
    stops0.push(',');
    stops0.push_str(&offset);
    stops0.push_str(",1");
    stops1.push(',');
    stops1.push_str(&offset);
    stops1.push(',');
    stops1.push(if i % 2 == 0 { '0' } else { '1' });
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":120,"layers":[{{"ty":4,"ind":0,"ip":0,"op":120,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":1,"k":{{"a":1,"k":[{{"t":0,"s":[{stops0}]}},{{"t":1,"s":[{stops1}]}}]}}}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn animated_unique_lut_gradient_fills_json(fills_count: usize) -> String {
  let mut fills = String::new();
  for i in 0..fills_count {
    let r0 = i as f32 / (fills_count + 1) as f32;
    let r1 = (i + 1) as f32 / (fills_count + 1) as f32;
    fills.push_str(&format!(
      r#"{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":1,"k":[{{"t":0,"s":[0,{r0},0,0,1,0,{},1]}},{{"t":1,"s":[0,{r1},0,0,1,0,{},1]}}]}}}},"o":{{"a":0,"k":100}}}},"#,
      1.0 - r0,
      1.0 - r1
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn animated_gradient_map_fills_json(fills_count: usize) -> String {
  let mut fills = String::new();
  for i in 0..fills_count {
    let x0 = i % 64;
    let x1 = (i + 1) % 64;
    fills.push_str(&format!(
      r#"{{"ty":"gf","s":{{"a":1,"k":[{{"t":0,"s":[{x0},0]}},{{"t":1,"s":[{x1},0]}}]}},"e":{{"a":1,"k":[{{"t":0,"s":[64,{x0}]}},{{"t":1,"s":[64,{x1}]}}]}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":0,"k":100}}}},"#
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn inverted_difference_masks_json(masks_count: usize) -> String {
  let mut masks = String::new();
  for i in 0..masks_count {
    if i > 0 {
      masks.push(',');
    }
    masks.push_str(&format!(
      r#"{{"mode":"f","inv":true,"o":{{"a":0,"k":100}},"pt":{{"a":0,"k":{{"c":true,"v":[[{},{}],[{},{}],[{},{}],[{},{}]],"i":[],"o":[]}}}}}}"#,
      i % 512,
      (i * 3) % 512,
      (i % 512) + 1,
      (i * 3) % 512,
      (i % 512) + 1,
      ((i * 3) % 512) + 1,
      i % 512,
      ((i * 3) % 512) + 1
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":512,"h":512,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"hasMask":true,"masksProperties":[{masks}],"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[256,256]}},"s":{{"a":0,"k":[512,512]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn dashed_duplicate_run_path_json(vertices_count: usize) -> String {
  let mut verts = String::new();
  for i in 0..vertices_count {
    if i > 0 {
      verts.push(',');
    }
    let run = i / 2;
    let y = if run % 2 == 0 { 30 } else { 34 };
    verts.push_str(&format!("[{},{}]", run % 64, y));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":8}},"lc":2,"lj":2,"d":[{{"n":"d","v":{{"a":0,"k":1}}}},{{"n":"g","v":{{"a":0,"k":1}}}}]}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn dashed_polystar_json(points: usize, copies: usize) -> String {
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sr","sy":1,"pt":{{"a":0,"k":{points}}},"p":{{"a":0,"k":[32,32]}},"r":{{"a":0,"k":0}},"ir":{{"a":0,"k":256}},"or":{{"a":0,"k":512}},"is":{{"a":0,"k":0}},"os":{{"a":0,"k":0}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":12}},"lc":2,"lj":2,"d":[{{"n":"d","v":{{"a":0,"k":1}}}},{{"n":"g","v":{{"a":0,"k":1}}}}]}},{{"ty":"rp","c":{{"a":0,"k":{copies}}},"o":{{"a":0,"k":0}},"tr":{{"r":{{"a":0,"k":0.5}}}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn animated_repeater_product_json() -> &'static str {
  r#"{
    "v":"5.5.0","w":720,"h":720,"fr":60,"ip":0,"op":2,
    "layers":[{"ty":4,"ind":1,"ip":0,"op":2,"st":0,"ks":{},"shapes":[{"ty":"gr","it":[
      {"ty":"rc","p":{"a":0,"k":[360,360]},"s":{"a":0,"k":[8,8]},"r":{"a":0,"k":0}},
      {"ty":"rp","c":{"a":1,"k":[{"t":0,"s":[0],"e":[64]},{"t":1,"s":[64]}]},"o":{"a":0,"k":0},"tr":{"p":{"a":0,"k":[1,0]}}},
      {"ty":"rp","c":{"a":1,"k":[{"t":0,"s":[0],"e":[64]},{"t":1,"s":[64]}]},"o":{"a":0,"k":0},"tr":{"p":{"a":0,"k":[1,0]}}},
      {"ty":"rp","c":{"a":1,"k":[{"t":0,"s":[0],"e":[64]},{"t":1,"s":[64]}]},"o":{"a":0,"k":0},"tr":{"p":{"a":0,"k":[1,0]}}},
      {"ty":"rp","c":{"a":1,"k":[{"t":0,"s":[0],"e":[64]},{"t":1,"s":[64]}]},"o":{"a":0,"k":0},"tr":{"p":{"a":0,"k":[1,0]}}},
      {"ty":"rp","c":{"a":1,"k":[{"t":0,"s":[0],"e":[64]},{"t":1,"s":[64]}]},"o":{"a":0,"k":0},"tr":{"p":{"a":0,"k":[1,0]}}},
      {"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},
      {"ty":"tr"}
    ]}]}]
  }"#
}

fn rounded_polystar_many_paints_json(paints_count: usize, strokes: bool) -> String {
  let mut paints = String::new();
  for _ in 0..paints_count {
    if strokes {
      paints.push_str(r#"{"ty":"st","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"w":{"a":0,"k":4},"lc":2,"lj":2},"#);
    } else {
      paints.push_str(r#"{"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},"#);
    }
  }
  format!(
    r#"{{"v":"5.5.0","w":128,"h":128,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sr","sy":1,"pt":{{"a":0,"k":256}},"p":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}},"ir":{{"a":0,"k":32}},"or":{{"a":0,"k":64}},"is":{{"a":0,"k":100}},"os":{{"a":0,"k":100}}}},{paints}{{"ty":"rp","c":{{"a":0,"k":64}},"o":{{"a":0,"k":0}},"tr":{{}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn trim_repeater_many_paints_json(paints_count: usize, strokes: bool) -> String {
  let mut verts = String::new();
  for i in 0..512 {
    if i > 0 {
      verts.push(',');
    }
    let y = if i % 2 == 0 { 8 } else { 120 };
    verts.push_str(&format!("[{},{}]", i % 128, y));
  }
  let mut trims = String::new();
  for i in 0..500 {
    trims.push_str(&format!(
      r#"{{"ty":"tm","s":{{"a":0,"k":{}}},"e":{{"a":0,"k":{}}},"o":{{"a":0,"k":0}},"m":2}},"#,
      i % 100,
      (i + 50) % 100
    ));
  }
  let mut paints = String::new();
  for _ in 0..paints_count {
    if strokes {
      paints.push_str(r#"{"ty":"st","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"w":{"a":0,"k":4},"lc":2,"lj":2},"#);
    } else {
      paints.push_str(r#"{"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},"#);
    }
  }
  format!(
    r#"{{"v":"5.5.0","w":128,"h":128,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{verts}],"i":[],"o":[]}}}}}},{trims}{paints}{{"ty":"rp","c":{{"a":0,"k":64}},"o":{{"a":0,"k":0}},"tr":{{}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn static_translated_paint_pairs_json(pairs: usize) -> String {
  let mut items = String::new();
  for i in 0..pairs {
    items.push_str(&format!(
      r#"{{"ty":"rc","p":{{"a":0,"k":[{},{}]}},"s":{{"a":0,"k":[2,2]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},"#,
      i % 96,
      (i / 96) % 96
    ));
  }
  format!(
    r#"{{"v":"5.5.0","w":96,"h":96,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"p":{{"a":1,"k":[{{"t":0,"s":[0,0]}},{{"t":1,"s":[30,0]}}]}}}},"shapes":[{{"ty":"gr","it":[{items}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn animated_same_gradient_fills_json(fills_count: usize) -> String {
  let mut items = String::new();
  for i in 0..fills_count {
    let x = (i * 13) % 246 + 5;
    let y = (i * 17) % 246 + 5;
    items.push_str(&format!(
      r#"{{"ty":"rc","p":{{"a":0,"k":[{x},{y}]}},"s":{{"a":0,"k":[10,10]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[256,256]}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":1,"k":[{{"t":0,"s":[100]}},{{"t":1,"s":[100]}}]}}}},"#
    ));
  }
  format!(r#"{{"v":"5.5.0","w":256,"h":256,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{items}{{"ty":"tr"}}]}}]}}]}}"#)
}

fn auto_orient_split_position_json(refs_count: usize) -> String {
  let mut kfs = String::from(r#"{"t":-1,"s":[0]},{"t":0,"s":[0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0]}"#);
  }
  kfs.push_str(r#",{"t":1000000,"s":[0]}"#);
  let mut layers = String::new();
  for i in 0..refs_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"orientsplit","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"orientsplit","layers":[{{"ty":4,"ind":0,"ao":1,"ip":0,"op":60,"st":0,"ks":{{"p":{{"s":true,"x":{{"a":1,"k":[{kfs}]}},"y":{{"a":1,"k":[{kfs}]}}}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn solid_layers_json(layers_count: usize, layer_size: usize) -> String {
  let mut layers = String::new();
  for i in 0..layers_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(
      r##"{{"ty":1,"ind":{i},"sw":{layer_size},"sh":{layer_size},"sc":"#ff0000","ip":0,"op":60,"st":0,"ks":{{"p":{{"a":0,"k":[{},{}]}},"a":{{"a":0,"k":[{},{}]}},"s":{{"a":0,"k":[100,100]}},"o":{{"a":0,"k":100}}}}}}"##,
      layer_size / 2,
      layer_size / 2,
      layer_size / 2,
      layer_size / 2
    ));
  }
  format!(r#"{{"v":"5.5.0","w":{layer_size},"h":{layer_size},"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#)
}

fn shape_layers_json(layers_count: usize, layer_size: usize) -> String {
  let mut layers = String::new();
  for i in 0..layers_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(
      r#"{{"ty":4,"ind":{i},"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[{},{}]}},"s":{{"a":0,"k":[{layer_size},{layer_size}]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}"#,
      layer_size / 2,
      layer_size / 2
    ));
  }
  format!(r#"{{"v":"5.5.0","w":{layer_size},"h":{layer_size},"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#)
}

fn curve_flattening_json(n: usize) -> String {
  let mut verts = String::new();
  let mut in_t = String::new();
  let mut out_t = String::new();
  for i in 0..n {
    if i > 0 {
      verts.push(',');
      in_t.push(',');
      out_t.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 50, (i / 50) % 50));
    out_t.push_str("[2000,2000]");
    in_t.push_str("[-2000,-2000]");
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[{in_t}],"o":[{out_t}]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn many_fills_over_large_path_json() -> String {
  let mut verts = String::new();
  for i in 0..100_000 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i / 64) % 64));
  }
  let mut fills = String::new();
  for _ in 0..3900 {
    fills.push_str(r#"{"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},"#);
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[],"o":[]}}}}}},{fills}{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn repeater_clones_prior_paint_jobs_json() -> String {
  let mut fills = String::new();
  for _ in 0..3900 {
    fills.push_str(r#"{"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},"#);
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[10,10]}},"r":{{"a":0,"k":0}}}},{fills}{{"ty":"rp","c":{{"a":0,"k":64}},"o":{{"a":0,"k":0}},"tr":{{}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn matte_source_expensive_precomp_json() -> String {
  let mut pairs = String::new();
  for i in 0..1500 {
    if i > 0 {
      pairs.push(',');
    }
    let src = 2 * i;
    let dst = src + 1;
    pairs.push_str(&format!(
      r#"{{"ty":0,"ind":{src},"td":1,"refId":"a","ip":0,"op":60,"st":0,"ks":{{}}}},{{"ty":4,"ind":{dst},"tt":1,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}"#
    ));
  }

  let mut kfs = String::from(r#"{"t":-1,"s":[0]},{"t":0,"s":[0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0]}"#);
  }
  kfs.push_str(r#",{"t":10000000,"s":[0]}"#);

  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{pairs}],"assets":[{{"id":"a","layers":[{{"ty":0,"ind":0,"refId":"leaf","ip":0,"op":60,"st":0,"ks":{{}},"tm":{{"a":1,"k":[{kfs}]}}}}]}},{{"id":"leaf","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn gradient_opacity_stop_lut_json(stop_count: usize) -> String {
  let mut stops = String::from("0,1,0,0");
  for i in 0..stop_count {
    stops.push(',');
    stops.push_str(&(i as f32 / (stop_count - 1) as f32).to_string());
    stops.push_str(",1");
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":1,"k":{{"a":0,"k":[{stops}]}}}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn dense_in_bounds_raster_edge_storm_json() -> String {
  let mut verts = String::new();
  for i in 0..700_000 {
    if i > 0 {
      verts.push(',');
    }
    let x = if i % 2 == 0 { 1 } else { 63 };
    let y = 1 + i % 62;
    verts.push_str(&format!("[{x},{y}]"));
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn luma_matte_pairs_json(pairs_count: usize, size: usize) -> String {
  let mut layers = String::new();
  for i in 0..pairs_count {
    if i > 0 {
      layers.push(',');
    }
    let src = i * 2;
    let dst = src + 1;
    layers.push_str(&format!(
      r##"{{"ty":1,"ind":{src},"td":1,"sw":64,"sh":64,"sc":"#ffffff","ip":0,"op":60,"st":0,"ks":{{"p":{{"a":0,"k":[32,32]}},"a":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[100,100]}},"o":{{"a":0,"k":100}}}}}},{{"ty":1,"ind":{dst},"tt":3,"sw":64,"sh":64,"sc":"#00ff00","ip":0,"op":60,"st":0,"ks":{{"p":{{"a":0,"k":[32,32]}},"a":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[100,100]}},"o":{{"a":0,"k":100}}}}}}"##
    ));
  }
  format!(r#"{{"v":"5.5.0","w":{size},"h":{size},"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#)
}

fn unsorted_scalar_keyframes_json(count: usize, value: &str) -> String {
  let mut kfs = format!(r#"{{"t":-1,"s":[{value}]}},{{"t":0,"s":[{value}]}}"#);
  for _ in 0..count {
    kfs.push_str(&format!(r#",{{"t":-1,"s":[{value}]}}"#));
  }
  kfs.push_str(&format!(r#",{{"t":1000000,"s":[{value}]}}"#));
  kfs
}

fn unsorted_color_keyframes_json(count: usize) -> String {
  let mut kfs = String::from(r#"{"t":-1,"s":[1,0,0]},{"t":0,"s":[1,0,0]}"#);
  for _ in 0..count {
    kfs.push_str(r#",{"t":-1,"s":[1,0,0]}"#);
  }
  kfs.push_str(r#",{"t":1000000,"s":[1,0,0]}"#);
  kfs
}

fn reused_precomp_fill_opacity_timeline_json(refs_count: usize, keyframes_count: usize) -> String {
  let mut layers = String::new();
  for i in 0..refs_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"fillprop","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let kfs = unsorted_scalar_keyframes_json(keyframes_count, "100");
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"fillprop","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":1,"k":[{kfs}]}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn reused_precomp_fill_color_timeline_json(refs_count: usize, keyframes_count: usize) -> String {
  let mut layers = String::new();
  for i in 0..refs_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"fillcolor","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let kfs = unsorted_color_keyframes_json(keyframes_count);
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"fillcolor","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":1,"k":[{kfs}]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn reused_precomp_layer_opacity_timeline_json(refs_count: usize, keyframes_count: usize) -> String {
  let mut layers = String::new();
  for i in 0..refs_count {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"layeropacity","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let kfs = unsorted_scalar_keyframes_json(keyframes_count, "100");
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"layeropacity","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"o":{{"a":1,"k":[{kfs}]}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  )
}

fn easing_handle_scalar_tail_json(tail_count: usize) -> String {
  let mut tail = String::from("0.5");
  for _ in 0..tail_count {
    tail.push_str(",0.5");
  }
  format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":1,"k":[{{"t":0,"s":[32,32],"e":[33,33],"o":{{"x":[{tail}],"y":0.5}},"i":{{"x":0.5,"y":0.5}}}},{{"t":1,"s":[33,33]}}]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  )
}

fn transform_position_vec2_tail_json(tail_count: usize) -> String {
  let mut k = String::from("[1,2");
  for _ in 0..tail_count {
    k.push_str(",1");
  }
  k.push(']');
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"p":{{"s":false,"k":{k}}}}},"shapes":[]}}]}}"#)
}

fn unknown_field_junk(fields: usize) -> String {
  let mut junk = String::new();
  for j in 0..fields {
    junk.push_str(&format!(r#","u{j}":1"#));
  }
  junk
}

fn asset_layers_delayed_dispatch_json(layers_count: usize, junk_fields: usize) -> String {
  let mut inner_layers = String::new();
  for i in 0..layers_count {
    if i > 0 {
      inner_layers.push(',');
    }
    let junk = unknown_field_junk(junk_fields);
    inner_layers.push_str(&format!(r#"{{"ty":4,"ind":{i},"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[] {junk}}}"#));
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":0,"refId":"a","ip":0,"op":60,"st":0,"ks":{{}}}}],"assets":[{{"id":"a","layers":[{inner_layers}]}}]}}"#)
}

fn group_it_before_ty_delayed_dispatch_json(items_count: usize, junk_fields: usize) -> String {
  let mut items = String::new();
  for i in 0..items_count {
    if i > 0 {
      items.push(',');
    }
    let junk = unknown_field_junk(junk_fields);
    items.push_str(&format!(r#"{{"ty":"tr"{junk}}}"#));
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"it":[{items}],"ty":"gr"}}]}}]}}"#)
}

fn layer_shapes_before_ty_delayed_dispatch_json(items_count: usize, junk_fields: usize) -> String {
  let mut shapes = String::new();
  for i in 0..items_count {
    if i > 0 {
      shapes.push(',');
    }
    let junk = unknown_field_junk(junk_fields);
    shapes.push_str(&format!(r#"{{"ty":"tr"{junk}}}"#));
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"shapes":[{shapes}],"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}}}}]}}"#)
}

fn many_minimal_masks_json(masks_count: usize) -> String {
  let mut masks = String::new();
  for i in 0..masks_count {
    if i > 0 {
      masks.push(',');
    }
    masks.push_str(r#"{"mode":"a","pt":{"a":0,"k":{"c":false,"v":[[0,0]],"i":[],"o":[]}}}"#);
  }
  format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"masksProperties":[{masks}],"hasMask":true,"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[]}}]}}"#)
}

#[test]
fn precomp_reference_fan_out_is_rejected() {
  // Asset "a" holds four precomp layers that each reference "a" again, so a
  // naive walk instantiates ~4^16 layers. Parse must reject it, fast.
  assert_lottie!(
    r#"{
      "v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":1000,
      "layers":[{"ty":0,"refId":"a","ip":0,"op":1000,"st":0,"ks":{}}],
      "assets":[{"id":"a","layers":[
        {"ty":0,"refId":"a","ip":0,"op":1000,"st":0,"ks":{}},
        {"ty":0,"refId":"a","ip":0,"op":1000,"st":0,"ks":{}},
        {"ty":0,"refId":"a","ip":0,"op":1000,"st":0,"ks":{}},
        {"ty":0,"refId":"a","ip":0,"op":1000,"st":0,"ks":{}}
      ]}]
    }"#
  );
}

#[test]
fn compounding_repeaters_are_rejected() {
  // Three Repeaters (`rp`) stacked in one group, each with 64 copies. A
  // repeater duplicates every shape earlier in the group, and later repeaters
  // see the copies the earlier ones produced, so the counts multiply instead
  // of adding: 64 * 64 * 64 = ~262k copies of the rect from a handful of `rp`
  // items. The per-repeater copy count is clamped (64) but nothing caps the
  // compounded product across stacked repeaters, so this must be rejected fast.
  assert_lottie!(
    r#"{
      "v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,
      "layers":[{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{},"shapes":[
        {"ty":"gr","it":[
          {"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[10,10]},"r":{"a":0,"k":0}},
          {"ty":"rp","c":{"a":0,"k":64},"o":{"a":0,"k":0},"tr":{}},
          {"ty":"rp","c":{"a":0,"k":64},"o":{"a":0,"k":0},"tr":{}},
          {"ty":"rp","c":{"a":0,"k":64},"o":{"a":0,"k":0},"tr":{}},
          {"ty":"tr"}
        ]}
      ]}]
    }"#
  );
}

#[test]
fn animated_repeater_product_stays_bounded() {
  // Repeater product is checked with `copies.eval(0.0)`. These five repeaters
  // have zero copies at frame 0, then each reaches the runtime clamp ceiling of
  // 64 at frame 1, bypassing the parse-time compounded-product budget.
  assert_lottie!(animated_repeater_product_json(), 720, 0, 2);
}

#[test]
fn dash_amplification_stays_bounded() {
  // A 100M-long stroked path with a unit dash period would emit ~5e7 dash
  // pieces; the dasher budget bounds it so the frame still renders.
  assert_lottie!(
    r#"{
      "v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,
      "layers":[{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{},"shapes":[
        {"ty":"gr","it":[
          {"ty":"sh","ks":{"a":0,"k":{"c":false,"v":[[0,0],[100000000,0]],"i":[[0,0],[0,0]],"o":[[0,0],[0,0]]}}},
          {"ty":"st","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"w":{"a":0,"k":2},
           "d":[{"n":"d","v":{"a":0,"k":1}},{"n":"g","v":{"a":0,"k":1}}]},
          {"ty":"tr"}
        ]}
      ]}]
    }"#
  );
}

#[test]
fn animated_dash_period_shrink_stays_bounded() {
  // The parser estimates dash work at frame 0. This path uses a safe 1000px dash
  // period at frame 0, then shrinks each dash/gap to 0.0011px at frame 1; without
  // a render-time dash budget, frame 1 materializes millions of tiny pieces and
  // can spend seconds in dash phase transitions even at a normal 720px output.
  assert_lottie!(animated_tiny_dash_period_json(), 720, 0, 2);
}

#[test]
fn unique_easing_handle_cache_stays_bounded() {
  // Every animated keyframe interns its temporal easing handles into a
  // per-composition HashMap keyed by formatted control points. A hostile
  // property can make nearly every keyframe carry a distinct easing key, so
  // parser work and cache memory grow with max_keyframes even though the
  // rendered shape is trivial.
  let mut rotation_kfs = String::from(r#"{"t":0,"s":[0],"o":{"x":0,"y":0.5},"i":{"x":0.5,"y":0.5}}"#);
  let mut opacity_kfs = String::from(r#"{"t":0,"s":[100],"o":{"x":0,"y":0.5},"i":{"x":0.51,"y":0.5}}"#);
  for t in 1..65_000 {
    let x = (t % 10_000) as f32 / 100.0;
    let y = (t / 10_000) as f32 / 100.0;
    rotation_kfs.push_str(&format!(r#",{{"t":{t},"s":[{}],"o":{{"x":{x:.2},"y":{y:.2}}},"i":{{"x":0.5,"y":0.5}}}}"#, t % 360));
    opacity_kfs.push_str(&format!(r#",{{"t":{t},"s":[{}],"o":{{"x":{x:.2},"y":{y:.2}}},"i":{{"x":0.51,"y":0.5}}}}"#, t % 100));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"r":{{"a":1,"k":[{rotation_kfs}]}},"o":{{"a":1,"k":[{opacity_kfs}]}}}},"shapes":[]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn dashed_duplicate_run_path_stays_bounded() {
  // Consecutive duplicate vertices look sanitizer-friendly, but dashing runs
  // before stroke dedupe can collapse them and materializes too many pieces.
  let json = dashed_duplicate_run_path_json(44_000);
  assert_lottie!(json);
}

#[test]
fn many_clipped_dashed_strokes_stay_bounded() {
  // Many short dashed strokes that straddle the viewport force dash splitting,
  // stroke outlining, and clipping per shape. This is a small-input variant of
  // dash pressure that does not rely on one absurdly long path.
  let json = clipped_dashed_strokes_json(60, 3);
  assert_lottie!(json);
}

#[test]
fn deep_parent_chain_stays_bounded() {
  // 4000 layers, each parented to the previous. This is rejected by parse-time
  // layer/parent budgets instead of reaching the render-time parent walk.
  let mut layers = String::from(r#"{"ty":4,"ind":0,"ip":0,"op":1000,"st":0,"ks":{},"shapes":[]}"#);
  for i in 1..4000 {
    layers.push_str(&format!(r#",{{"ty":4,"ind":{i},"parent":{},"ip":0,"op":1000,"st":0,"ks":{{}},"shapes":[]}}"#, i - 1));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":1000,"layers":[{layers}]}}"#);
  assert_lottie!(json);
}

#[test]
fn max_legal_deep_parent_chain_stays_bounded() {
  // Same parent-chain shape as the over-limit regression, but kept exactly at
  // the default layer cap. It stays bounded through the parent-chain depth cap,
  // not by adding render-time parent caches.
  let mut layers = String::from(r#"{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{},"shapes":[]}"#);
  for i in 1..Limits::default().max_layers {
    layers.push_str(&format!(r#",{{"ty":4,"ind":{i},"parent":{},"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[]}}"#, i - 1));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#);
  assert_lottie!(json);
}

#[test]
fn many_max_depth_parent_chains_stay_bounded() {
  // The parser caps a single parent chain and the total parent-chain depth in
  // one layer list, so repeated legal-depth chains are rejected before render.
  let chain_len = Limits::default().max_parent_chain_depth + 1;
  let chains = Limits::default().max_layers / chain_len;
  let mut layers = String::new();
  for c in 0..chains {
    for i in 0..chain_len {
      if !layers.is_empty() {
        layers.push(',');
      }
      let ind = c * chain_len + i;
      let parent = if i == 0 { String::new() } else { format!(r#","parent":{}"#, ind - 1) };
      layers.push_str(&format!(r#"{{"ty":4,"ind":{ind}{parent},"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[]}}"#));
    }
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#);
  assert_lottie!(json);
}

#[test]
fn mask_heavy_layer_stays_bounded() {
  // Canary, not a hard cap: mask cost is linear in count (~tens of us each),
  // so this only checks that a moderate count stays bounded. Real corpus files
  // use at most one mask per layer.
  let mut masks = String::new();
  for i in 0..5000 {
    if i > 0 {
      masks.push(',');
    }
    masks.push_str(r#"{"mode":"a","o":{"a":0,"k":100},"pt":{"a":0,"k":{"c":true,"v":[[0,0],[20,0],[10,20]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}}"#);
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":128,"h":128,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"hasMask":true,"masksProperties":[{masks}],"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[64,64]}},"s":{{"a":0,"k":[128,128]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn many_layers_with_max_masks_stay_bounded() {
  // Mask count is capped per layer. A hostile file can still place that max
  // mask count on many legal layers, forcing repeated mask flatten/clip/combine
  // work over otherwise tiny content.
  let mask = r#"{"mode":"a","o":{"a":0,"k":100},"pt":{"a":0,"k":{"c":true,"v":[[0,0],[64,0],[64,64],[0,64]],"i":[],"o":[]}}}"#;
  let masks = (0..Limits::default().max_masks_per_layer).map(|_| mask).collect::<Vec<_>>().join(",");
  let mut layers = String::new();
  for i in 0..900 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(
      r#"{{"ty":4,"ind":{i},"ip":0,"op":60,"st":0,"ks":{{}},"hasMask":true,"masksProperties":[{masks}],"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}"#
    ));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}]}}"#);
  assert_lottie!(json);
}

#[test]
fn inverted_difference_masks_large_dirty_box_stay_bounded() {
  // Difference/inverted masks must still accumulate over the full content
  // dirty box: outside the tiny mask geometry, the inverted mode contributes
  // nonzero coverage. A stack of tiny masks over a full-canvas fill therefore
  // forces many full-box mask-combine passes. The same JSON is checked through
  // the public RGBA, alpha-only, and no-antialias CPU render paths.
  let json = inverted_difference_masks_json(1600);
  assert_lottie!(json, 512);
  let json = inverted_difference_masks_json(1600);
  assert_lottie!(
    json,
    512,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
  let json = inverted_difference_masks_json(1600);
  assert_lottie!(
    json,
    512,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn precomp_dag_fan_out_stays_bounded() {
  // Precomp fan-out with NO self-reference: 16 DISTINCT assets a0..a15, each
  // holding four precomp layers that reference the *next* asset, plus a leaf
  // a16. Every path a0->a1->...->a16 visits distinct ids, so a cycle/self-reference
  // guard never fires, yet the render re-instantiates the leaf ~4^16 (~4.3e9)
  // times before MAX_PRECOMP_DEPTH stops it. Distinct from the self-referential
  // fan-out case: the bound has to be a total-instantiation budget (or memoized
  // subtree cost), not just cycle detection. Buffers are reused across the
  // fan-out, so this trips the TIME budget (a pure CPU hang), not memory.
  let mut assets = String::new();
  for i in 0..16 {
    let child = format!("a{}", i + 1);
    let layer = format!(r#"{{"ty":0,"refId":"{child}","ip":0,"op":60,"st":0,"ks":{{}}}}"#);
    assets.push_str(&format!(r#"{{"id":"a{i}","layers":[{layer},{layer},{layer},{layer}]}},"#));
  }
  assets.push_str(
    r#"{"id":"a16","layers":[{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{},"shapes":[{"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[32,32]},"s":{"a":0,"k":[10,10]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[1,0,0]},"o":{"a":0,"k":100}},{"ty":"tr"}]}]}]}"#,
  );
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":0,"refId":"a0","ip":0,"op":60,"st":0,"ks":{{}}}}],"assets":[{assets}]}}"#);
  assert_lottie!(json);
}

#[test]
fn stacked_round_corners_stay_bounded() {
  // Stacked Round-Corners (`rd`) modifiers on a sharp zig-zag path. Each `rd`
  // rewrites every corner of every contour in scope into ~5 points, and the
  // fresh corners of a tight zig-zag stay sharp enough (turn > the smooth-skip
  // threshold) that the NEXT `rd` re-rounds them again, so k stacked `rd`
  // shapes expand the point count ~5^k. Each `rd` is one shape item (bounded
  // only by max_shapes_per_layer=4096), and nothing caps a contour's point
  // count downstream, so a few dozen bytes of `rd` shapes explode geometry.
  // Trips the MEMORY budget (the contour balloons past 64 MiB in ms).
  let mut zig = String::new();
  for i in 0..40 {
    if i > 0 {
      zig.push(',');
    }
    let y = if i % 2 == 0 { 8 } else { 56 };
    zig.push_str(&format!("[{},{}]", 2 + i, y));
  }
  let mut rds = String::new();
  for _ in 0..40 {
    rds.push_str(r#"{"ty":"rd","r":{"a":0,"k":3}},"#);
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{zig}],"i":[],"o":[]}}}}}},{rds}{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":2}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn keyframe_value_inheritance_stays_bounded() {
  // Parse-time amplification: a keyframed path whose FIRST keyframe carries a
  // large value, followed by tens of thousands of empty `{"t":n}` keyframes. A
  // keyframe with no `s` inherits the previous keyframe's value by DEEP-CLONING
  // it, so every one of the ~65k keyframes ends up holding its own full copy of
  // the big path: O(keyframes * value_size) time and memory. The keyframe
  // COUNT is capped (max_keyframes=65536) but the per-keyframe value size is
  // not, so a ~2 MB input balloons to many GB during parse. Trips the MEMORY
  // budget while parsing, before a single pixel is drawn.
  let mut verts = String::new();
  for i in 0..4000 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 50, (i / 50) % 50));
  }
  let mut kfs = format!(r#"{{"t":0,"s":[{{"c":true,"v":[{verts}]}}]}}"#);
  for t in 1..65000 {
    kfs.push_str(&format!(r#",{{"t":{t}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":1,"k":[{kfs}]}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn repeater_amplifies_large_geometry_stays_bounded() {
  // A single, ordinary Repeater (`rp`, 64 copies, the clamp ceiling) applied
  // to a LARGE authored path. `apply_repeater` clones the base scope's geometry
  // once per copy, so output points = copies * base_points with no cap on the
  // base: a ~1.6 MB path (200k vertices) balloons to ~64x that in device-space
  // contour copies. The copy count is clamped (64) but the per-copy payload is
  // not, so a normal-looking repeater over a big path exhausts memory. Trips
  // the memory budget while the copies accumulate in the geometry arena.
  let mut verts = String::new();
  for i in 0..200000 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 60, (i / 60) % 60));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"rp","c":{{"a":0,"k":64}},"o":{{"a":0,"k":0}},"tr":{{"p":{{"a":0,"k":[1,1]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn curve_flattening_amplification_stays_bounded() {
  // A single authored path whose every segment is a cubic with enormous
  // tangents. flatten_cubic caps subdivision at MAX_SEGS=128 PER cubic, but
  // nothing caps the number of cubics in one `sh` path (the vertex list is
  // uncapped), so N curved segments flatten to up to 128*N device-space points
  // - all materialized in the geometry arena BEFORE any viewport clip. A ~4 MB
  // path (100k curved vertices) expands to ~12.8M points (~200 MB). Trips the
  // memory budget while flattening, well before rasterization clips anything.
  // The alpha-only render option must not bypass the same bound.
  let json = curve_flattening_json(100_000);
  assert_lottie!(json);
  let json = curve_flattening_json(100_000);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn gradient_keyframe_value_inheritance_stays_bounded() {
  // Same parser amplification as path keyframe inheritance, but through a
  // gradient's `g.k` FloatList: one large stop table, then many keyframes
  // without `s`. Resolving missing values deep-clones the previous FloatList,
  // so a compact keyframe tail multiplies the stop payload at parse time.
  let mut stops = String::new();
  for i in 0..3000 {
    if i > 0 {
      stops.push(',');
    }
    stops.push_str(match i % 4 {
      0 => "0",
      1 => "1",
      2 => "0.5",
      _ => "0.25",
    });
  }
  let mut kfs = format!(r#"{{"t":0,"s":[{stops}]}}"#);
  for t in 1..65000 {
    kfs.push_str(&format!(r#",{{"t":{t}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":1,"k":[{kfs}]}}}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn oversized_static_gradient_stop_table_stays_bounded() {
  // `g.p` is clamped to 64 color stops, but the backing `g.k` float list is
  // parsed and later copied into the gradient LUT cache key without a length
  // cap. A few MB of JSON therefore creates tens of MB of FloatList + LUT-key
  // vectors before the renderer can ignore the excess stops.
  let mut stops = String::new();
  for i in 0..4_500_000 {
    if i > 0 {
      stops.push(',');
    }
    stops.push(if i % 2 == 0 { '0' } else { '1' });
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":0,"k":[{stops}]}}}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_shape_arrays_do_not_bypass_shape_budget() {
  // Once `ty:4` is known, every `shapes` field in the layer object is parsed
  // immediately. A hostile object can repeat the key many times; each array
  // gets its own fresh max_shapes_per_layer counter, and only the final array
  // is retained. The JSON below therefore forces hundreds of thousands of
  // shape items through the parser while leaving one tiny valid shape tree.
  let mut spam_array = String::from("[");
  for i in 0..4096 {
    if i > 0 {
      spam_array.push(',');
    }
    spam_array.push_str(r#"{"ty":"tr"}"#);
  }
  spam_array.push(']');

  let mut duplicate_fields = String::new();
  for _ in 0..320 {
    duplicate_fields.push_str(r#","shapes":"#);
    duplicate_fields.push_str(&spam_array);
  }

  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}}{duplicate_fields},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[10,10]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn delayed_dispatch_shape_lists_with_unknown_fields_stay_bounded() {
  // When `shapes`/group `it` appear before their dispatching `ty`, the parser
  // first skips the whole value and later forks back to parse it. Unknown-field
  // junk in each legal shape item therefore gets walked twice from compact JSON.
  let json = layer_shapes_before_ty_delayed_dispatch_json(4095, 300);
  assert_lottie!(json);
  let json = group_it_before_ty_delayed_dispatch_json(4095, 300);
  assert_lottie!(json);
}

#[test]
fn asset_layers_delayed_dispatch_stays_bounded() {
  // Asset `layers` are recorded while parsing the asset object, skipped once,
  // then parsed from a fork after the id/layers scan completes. Keep the
  // max-legal layer-list variant live for untrusted precomp JSON.
  let json = asset_layers_delayed_dispatch_json(4095, 300);
  assert_lottie!(json);
}

#[test]
fn oversized_dash_metadata_stays_bounded() {
  // Dash entries are not shape items, keyframes, or points, so they do not
  // trip an existing structural cap. The renderer later walks the whole list
  // to build the dash pattern even for a two-point path.
  let mut dashes = String::new();
  for i in 0..250_000 {
    if i > 0 {
      dashes.push(',');
    }
    let n = if i % 2 == 0 { "d" } else { "g" };
    dashes.push_str(&format!(r#"{{"n":"{n}","v":{{"a":0,"k":1}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,32],[64,32]],"i":[],"o":[]}}}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":2}},"d":[{dashes}]}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn shared_unsorted_time_remap_stays_bounded() {
  // Unsorted keyframes evaluate by linear file-order scan. Reuse one precomp
  // asset thousands of times, and make its child precomp carry a 65k-entry
  // non-monotonic `tm` timeline whose matching segment is at the very end.
  // Every root instance re-enters the same asset and pays the full scan during
  // render, even though the rendered leaf is just one tiny rectangle.
  let mut root_layers = String::new();
  for i in 0..4090 {
    if i > 0 {
      root_layers.push(',');
    }
    root_layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"a","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let mut kfs = String::from(r#"{"t":-1,"s":[0]},{"t":0,"s":[0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0]}"#);
  }
  kfs.push_str(r#",{"t":10000000,"s":[0]}"#);

  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{root_layers}],"assets":[{{"id":"a","layers":[{{"ty":0,"ind":0,"refId":"leaf","ip":0,"op":60,"st":0,"ks":{{}},"tm":{{"a":1,"k":[{kfs}]}}}}]}},{{"id":"leaf","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn reused_precomp_cyclic_parent_graph_stays_bounded() {
  // Parent cycles do not infinite-loop: parent_chain_matrix stops after
  // layers.len() steps. But each step still linearly searches the layer list,
  // so one cyclic 1000-layer precomp costs O(1000^2) just to discover the
  // bounded cycle. Reusing that asset 3000 times multiplies a "safe" cycle
  // guard into a render-time CPU DoS.
  let mut root_layers = String::new();
  for i in 0..3000 {
    if i > 0 {
      root_layers.push(',');
    }
    root_layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"cycle","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let mut cycle_layers = String::new();
  for i in 0..1000 {
    if i > 0 {
      cycle_layers.push(',');
    }
    let parent = if i == 999 { 0 } else { i + 1 };
    cycle_layers.push_str(&format!(r#"{{"ty":3,"ind":{i},"parent":{parent},"ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }

  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{root_layers}],"assets":[{{"id":"cycle","layers":[{cycle_layers}]}}]}}"#);
  assert_lottie!(json);
}

#[test]
fn static_path_tangent_padding_stays_bounded() {
  // Path vertices are uncapped, and parse_path_object pads missing `i` and
  // `o` tangent arrays to match `v.len()`. A path with only vertices therefore
  // silently triples the model allocation before rendering starts.
  let mut verts = String::new();
  for i in 0..1_400_000 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str("[0,0]");
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn transform_position_vec2_tail_stays_bounded() {
  // Transform position is a delayed-dispatch property: the huge vec2 tail is
  // skipped while recording `ks.p`, then reparsed through parse_position. The
  // host supplied only normal JSON and default Limits.
  let json = transform_position_vec2_tail_json(8_000_000);
  assert_lottie!(json);
}

#[test]
fn top_level_fitz_metadata_flood_stays_bounded() {
  // The top-level `fitz` table is parsed unconditionally, even when default
  // parse options do not request a Fitzpatrick replacement. Its entries are
  // not capped by any Lottie resource limit, so metadata alone can allocate
  // tens of MB before being dropped.
  let mut fitz = String::new();
  for i in 0..1_800_000 {
    if i > 0 {
      fitz.push(',');
    }
    fitz.push_str(r#"{"o":0}"#);
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"fitz":[{fitz}],"layers":[]}}"#);
  assert_lottie!(json);
}

#[test]
fn repeater_clones_prior_paint_jobs_stays_bounded() {
  // Repeaters clone both geometry and every prior paint job in their scope.
  // One rect plus thousands of fill items is legal under max_shapes_per_layer;
  // applying a 64-copy repeater turns those few thousand pending paints into
  // roughly 250k render jobs. The alpha-only path shares the cloning work.
  let json = repeater_clones_prior_paint_jobs_json();
  assert_lottie!(json);
  let json = repeater_clones_prior_paint_jobs_json();
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn many_fills_over_large_path_stays_bounded() {
  // Coverage caching can skip repeated rasterization, but every fill still
  // hashes/scans the entire source contour to build its fill key. A single
  // large static path under thousands of fill paints therefore burns CPU in
  // key construction before the cache can help. Alpha-only rendering still
  // evaluates the same paint/cache keys.
  let json = many_fills_over_large_path_json();
  assert_lottie!(json);
  let json = many_fills_over_large_path_json();
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn max_legal_cumulative_shape_paints_stay_bounded() {
  // This stays just under max_shapes_per_layer: 2047 rect/fill pairs plus the
  // group transform. Lottie paint semantics make each fill cover all geometry
  // accumulated before it, so the legal shape list creates triangular work.
  // Alpha-only rendering uses the same legal JSON payload.
  let json = cumulative_shape_paints_json(2047);
  assert_lottie!(json);
  let json = cumulative_shape_paints_json(2047);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn repeated_individual_trims_over_large_paths_stay_bounded() {
  // Individual trim first sums every contour, then extracts per-contour
  // pieces. Stacking thousands of individual trims over dozens of large
  // paths recomputes the same arena-wide lengths and slices again and again.
  let mut path_items = String::new();
  for p in 0..80 {
    if p > 0 {
      path_items.push(',');
    }
    let mut verts = String::new();
    for i in 0..2000 {
      if i > 0 {
        verts.push(',');
      }
      verts.push_str(&format!("[{},{}]", i % 64, (p + i / 64) % 64));
    }
    path_items.push_str(&format!(r#"{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{verts}],"i":[],"o":[]}}}}}}"#));
  }
  let mut trims = String::new();
  for _ in 0..3900 {
    trims.push_str(r#"{"ty":"tm","s":{"a":0,"k":1},"e":{"a":0,"k":99},"o":{"a":0,"k":0},"m":2},"#);
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{path_items},{trims}{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn curved_mask_flattening_stays_bounded() {
  // Mask geometry is flattened before clipping just like visible paths, but
  // mask paths are outside the shape-item budget. A single mask with many
  // high-curvature cubic vertices expands into millions of temporary points
  // while the layer content itself is just a small rectangle.
  let n = 40_000;
  let mut verts = String::new();
  let mut in_t = String::new();
  let mut out_t = String::new();
  for i in 0..n {
    if i > 0 {
      verts.push(',');
      in_t.push(',');
      out_t.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i / 64) % 64));
    in_t.push_str("[-2000,-2000]");
    out_t.push_str("[2000,2000]");
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"hasMask":true,"masksProperties":[{{"mode":"a","o":{{"a":0,"k":100}},"pt":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[{in_t}],"o":[{out_t}]}}}}}}],"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[12,12]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn many_minimal_masks_stay_bounded() {
  // Mask count has no explicit default limit. Even one-point masks can balloon
  // parser/model memory when a hostile layer carries a very long masks array.
  let json = many_minimal_masks_json(120_000);
  assert_lottie!(json);
}

#[test]
fn gradient_opacity_stop_lut_scan_stays_bounded() {
  // `g.p` caps color-stop interpretation, but opacity stops after the color
  // stops are retained and scanned while building the 1024-entry gradient LUT.
  // A monotonic opacity-stop tail makes every LUT sample walk a long prefix.
  // Alpha-only rendering still builds gradient LUT/source data.
  let json = gradient_opacity_stop_lut_json(200_000);
  assert_lottie!(json);
  let json = gradient_opacity_stop_lut_json(220_000);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn animated_gradient_opacity_tail_lut_scan_stays_bounded() {
  // Animated gradient stops keep large opacity tails on both keyframes. Even a
  // single rendered frame has to build a LUT from that retained tail. The same
  // JSON is checked through RGBA, alpha-only, and no-antialias render options.
  let json = animated_opacity_tail_gradient_json(525_000);
  assert_lottie!(json);
  let json = animated_opacity_tail_gradient_json(525_000);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
  let json = animated_opacity_tail_gradient_json(525_000);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn multi_frame_animated_gradient_lut_work_stays_bounded() {
  // A smaller animated opacity tail can still exceed the aggregate frame
  // budget when two frames rebuild/cache distinct LUT state.
  let json = animated_opacity_tail_gradient_json(250_000);
  assert_lottie!(json, 64, 0, 2);
}

#[test]
fn matte_source_expensive_precomp_stays_bounded() {
  // Track mattes render their source layer for each consumer. Make every matte
  // source a precomp whose child has an expensive unsorted `tm` timeline; the
  // source is never drawn directly (`td:1`) but it is repeatedly evaluated as
  // matte input. Alpha-only rendering still walks matte/precomp/time-remap
  // structure for the same hostile JSON.
  let json = matte_source_expensive_precomp_json();
  assert_lottie!(json);
  let json = matte_source_expensive_precomp_json();
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn duplicate_path_ks_fields_stay_bounded() {
  // Once a shape item's `ty:"sh"` is known, each later duplicate `ks` field is
  // parsed immediately. Only the last one is retained, so duplicate keys can
  // multiply path parse/model work outside the layer-level `shapes` budget.
  let mut verts = String::new();
  for i in 0..9000 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i / 64) % 64));
  }
  let ks = format!(r#""ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[],"o":[]}}}}"#);
  let mut duplicate_ks = String::new();
  for _ in 0..120 {
    duplicate_ks.push(',');
    duplicate_ks.push_str(&ks);
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh"{duplicate_ks}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_path_vertex_fields_stay_bounded() {
  // A path object's `v` field is parsed immediately every time it appears.
  // Each duplicate list below is max-legal, so no single path trips the point
  // cap, but only the final list is retained. The rest are pure parse/allocation
  // work from duplicate JSON keys inside one otherwise ordinary shape.
  let mut verts = String::new();
  for i in 0..4096 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i / 64) % 64));
  }
  let v = format!(r#""v":[{verts}]"#);
  let duplicate_v = (0..260).map(|_| v.clone()).collect::<Vec<_>>().join(",");
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,{duplicate_v},"i":[],"o":[]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_keyframe_path_value_fields_stay_bounded() {
  // Duplicate `s` fields inside one keyframe are each parsed as a complete
  // path value before the last one wins. This bypasses the keyframe-count cap:
  // the JSON contains one animated keyframe object, but it rebuilds hundreds
  // of max-legal path payloads during parse.
  let mut verts = String::new();
  for i in 0..4096 {
    if i > 0 {
      verts.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i / 64) % 64));
  }
  let s = format!(r#""s":[{{"c":true,"v":[{verts}],"i":[],"o":[]}}]"#);
  let duplicate_s = (0..220).map(|_| s.clone()).collect::<Vec<_>>().join(",");
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":1,"k":[{{"t":0,{duplicate_s}}},{{"t":1,"s":[{{"c":true,"v":[[0,0]],"i":[],"o":[]}}]}}]}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn gradient_stroke_dash_metadata_stays_bounded() {
  // Mirror the dash-metadata vector bomb through gradient strokes (`gs`), which
  // has its own shape arm but shares the unbounded dash parsing/render hashing
  // surface with solid strokes.
  let mut dashes = String::new();
  for i in 0..250_000 {
    if i > 0 {
      dashes.push(',');
    }
    let n = if i % 2 == 0 { "d" } else { "g" };
    dashes.push_str(&format!(r#"{{"n":"{n}","v":{{"a":0,"k":1}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,32],[64,32]],"i":[],"o":[]}}}}}},{{"ty":"gs","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":0,"k":[0,1,0,0,1,0,0,1]}}}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":2}},"d":[{dashes}]}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_layer_ks_fields_stay_bounded() {
  // Layer `ks` is parsed immediately every time it appears. Repeating a large
  // animated transform under duplicate `ks` keys multiplies parse/model work
  // even though the final transform is the only one retained.
  let mut kfs = String::from(r#"{"t":0,"s":[0,0]}"#);
  for t in 1..1500 {
    kfs.push_str(&format!(r#",{{"t":{t},"s":[{},{}]}}"#, t % 64, (t / 64) % 64));
  }
  let ks = format!(r#""ks":{{"p":{{"a":1,"k":[{kfs}]}}}}"#);
  let mut duplicate_ks = String::new();
  for _ in 0..180 {
    duplicate_ks.push(',');
    duplicate_ks.push_str(&ks);
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0{duplicate_ks},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn oversized_color_array_tail_stays_bounded() {
  // A fill color keeps only rgb, but parse_color still parses every component
  // in the array. One fixed-width-looking field can therefore spend the whole
  // frame budget parsing ignored numeric tail values.
  let mut comps = String::new();
  for i in 0..5_000_000 {
    if i > 0 {
      comps.push(',');
    }
    comps.push(if i % 2 == 0 { '0' } else { '1' });
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[{comps}]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn oversized_vec2_array_tail_stays_bounded() {
  // `parse_vec2` keeps only x/y but still parses every extra array component.
  // A rectangle position with a huge numeric tail burns parse time before any
  // renderer work begins.
  let mut comps = String::new();
  for i in 0..5_000_000 {
    if i > 0 {
      comps.push(',');
    }
    comps.push('1');
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[{comps}]}},"s":{{"a":0,"k":[10,10]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_transform_subfields_stay_bounded() {
  // Duplicate fields inside a single transform object are all parsed. Eight
  // duplicate animated rotations each carry a full keyframe list; only the
  // final value is semantically retained.
  let mut kfs = String::from(r#"{"t":0,"s":[0]}"#);
  for t in 1..65_000 {
    kfs.push_str(&format!(r#",{{"t":{t},"s":[{}]}}"#, t % 360));
  }
  let r = format!(r#""r":{{"a":1,"k":[{kfs}]}}"#);
  let fields = (0..8).map(|_| r.clone()).collect::<Vec<_>>().join(",");
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{{fields}}},"shapes":[]}}]}}"#);
  assert_lottie!(json);
}

#[test]
fn duplicate_stroke_miter_property_objects_stay_bounded() {
  // Stroke `ml` is usually a number, but object form is accepted and parsed as
  // a property. Duplicate `ml` fields therefore reparse full animated property
  // objects repeatedly while keeping only the final miter limit.
  let mut kfs = String::from(r#"{"t":0,"s":0}"#);
  for t in 1..65_000 {
    kfs.push_str(&format!(r#",{{"t":{t},"s":{}}}"#, t % 16));
  }
  let ml = format!(r#""ml":{{"a":1,"k":[{kfs}]}}"#);
  let mut fields = String::from(r#""ty":"st","c":{"a":0,"k":[1,0,0]},"w":{"a":0,"k":1}"#);
  for _ in 0..8 {
    fields.push(',');
    fields.push_str(&ml);
  }
  let json =
    format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{{fields}}},{{"ty":"tr"}}]}}]}}]}}"#);
  assert_lottie!(json);
}

#[test]
fn gradient_stroke_static_stop_table_stays_bounded() {
  // Gradient strokes share the unbounded `g.k` stop table surface with
  // gradient fills, but through a separate `gs` parser/render arm.
  let mut stops = String::new();
  for i in 0..4_500_000 {
    if i > 0 {
      stops.push(',');
    }
    stops.push(if i % 2 == 0 { '0' } else { '1' });
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,32],[64,32]],"i":[],"o":[]}}}}}},{{"ty":"gs","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":0,"k":[{stops}]}}}},"w":{{"a":0,"k":1}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn gradient_stroke_opacity_stop_lut_scan_stays_bounded() {
  // Same opacity-stop LUT scan as gradient fills, but through `gs`.
  let mut stops = String::from("0,1,0,0");
  for i in 0..200_000 {
    stops.push(',');
    stops.push_str(&(i as f32 / 199_999.0).to_string());
    stops.push_str(",1");
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[[0,32],[64,32]],"i":[],"o":[]}}}}}},{{"ty":"gs","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":1,"k":{{"a":0,"k":[{stops}]}}}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":8}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn many_focal_radial_gradient_strokes_over_large_surface_stay_bounded() {
  // Focal radial gradient strokes use the same per-pixel radial sampling as
  // fills, but they are a separate shape type. Wide strokes keep the coverage
  // full-surface while staying under the per-layer paint/source-item caps.
  let json = focal_radial_gradient_strokes_json(900);
  assert_lottie!(json, 512);
}

#[test]
fn many_focal_radial_gradients_over_large_surface_stay_bounded() {
  // Focal radial gradients pay per-pixel quadratic sampling. Many legal
  // gradient fills over one full-surface rect should still fit one frame.
  // No-antialias uses the same payload through the alternate raster option.
  let json = focal_radial_gradients_json(900);
  assert_lottie!(json, 512);
  let json = focal_radial_gradients_json(900);
  assert_lottie!(
    json,
    512,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn static_translated_paint_pairs_stay_bounded() {
  // A static shape subtree under animated layer translation produces many
  // distinct replay/cache keys across frames while the JSON itself is compact.
  // Keep default, alpha-only, and no-antialias variants together; the latter
  // two use the larger count needed to reproduce their current budget breach.
  let json = static_translated_paint_pairs_json(800);
  assert_lottie!(json, 192, 0, 2);
  let json = static_translated_paint_pairs_json(1200);
  assert_lottie!(
    json,
    192,
    0,
    2,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
  let json = static_translated_paint_pairs_json(1200);
  assert_lottie!(
    json,
    192,
    0,
    2,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn reused_precomp_focal_radial_gradients_stay_bounded() {
  // A single precomp asset can stay within the focal-radial-gradient cap while
  // root precomp refs replay it up to the expansion limit. This keeps the JSON
  // legal but multiplies per-pixel radial sampling through precomp reuse.
  let limits = Limits::default();
  let json = reused_precomp_focal_radial_gradients_json(limits.max_precomp_expansion / 2, limits.max_focal_radial_gradients_per_layer);
  assert_lottie!(json, 512);
}

#[test]
fn reused_precomp_animated_linear_gradients_stay_bounded() {
  // Animated linear-gradient opacity disables static shape replay even though
  // both keyframes evaluate to the same value. Reusing that precomp multiplies
  // ordinary gradient-map/source work without tripping the focal-radial cap.
  let json = reused_precomp_animated_linear_gradients_json(30, 30);
  assert_lottie!(json, 256);
}

#[test]
fn reused_precomp_animated_linear_gradient_strokes_stay_bounded() {
  // Same animated-opacity static-replay bypass as gradient fills, but through
  // the gradient-stroke renderer path with a full-surface stroke.
  let json = reused_precomp_animated_linear_gradient_strokes_json(30, 30);
  assert_lottie!(json, 256);
}

#[test]
fn animated_same_gradient_source_replay_stays_bounded() {
  // Gradient opacity is technically animated but evaluates to the same value
  // at both frames, so the source/replay path sees repeated output from a
  // non-static layer and still has to stay within a frame budget.
  let json = animated_same_gradient_fills_json(1700);
  assert_lottie!(json, 224, 0, 2);
}

#[test]
fn animated_unique_lut_gradient_fills_stay_bounded() {
  // Max-legal paint count with animated, distinct gradient stops churns many
  // LUT keys across frames over a larger CPU surface. No-antialias keeps the
  // same JSON on the alternate raster option.
  let json = animated_unique_lut_gradient_fills_json(3900);
  assert_lottie!(json, 320, 0, 2);
  let json = animated_unique_lut_gradient_fills_json(3900);
  assert_lottie!(
    json,
    320,
    0,
    2,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn animated_gradient_map_fills_stay_bounded() {
  // Animated start/end points keep the static stop table small but churn the
  // per-gradient map/source keys over thousands of fills. No-antialias is the
  // same payload through a different render option.
  let json = animated_gradient_map_fills_json(3900);
  assert_lottie!(json, 320, 0, 2);
  let json = animated_gradient_map_fills_json(3900);
  assert_lottie!(
    json,
    320,
    0,
    2,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn nested_precomp_clip_stack_over_large_contour_stays_bounded() {
  // Nested precomp viewports push clip quads. A large contour that straddles
  // the viewport makes clipping a real Sutherland-Hodgman pass at each depth.
  let mut verts = String::new();
  for i in 0..200_000 {
    if i > 0 {
      verts.push(',');
    }
    let x = if i % 2 == 0 { -32 } else { 96 };
    verts.push_str(&format!("[{},{}]", x, i % 64));
  }
  let leaf = format!(
    r#"{{"id":"a15","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  let mut assets = String::new();
  for i in 0..15 {
    if i > 0 {
      assets.push(',');
    }
    assets.push_str(&format!(
      r#"{{"id":"a{i}","layers":[{{"ty":0,"ind":0,"refId":"a{}","w":64,"h":64,"ip":0,"op":60,"st":0,"ks":{{}}}}]}}"#,
      i + 1
    ));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":0,"ind":0,"refId":"a0","w":64,"h":64,"ip":0,"op":60,"st":0,"ks":{{}}}}],"assets":[{assets},{leaf}]}}"#);
  assert_lottie!(json);
}

#[test]
fn round_stroke_outline_many_turns_stays_bounded() {
  // No dashes, no repeaters: just a huge zig-zag open path with round joins
  // and caps. The plain stroke outline materialization creates heavy CPU work.
  let mut verts = String::new();
  for i in 0..200_000 {
    if i > 0 {
      verts.push(',');
    }
    let y = if i % 2 == 0 { 8 } else { 56 };
    verts.push_str(&format!("[{},{}]", i % 64, y));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"sh","ks":{{"a":0,"k":{{"c":false,"v":[{verts}],"i":[],"o":[]}}}}}},{{"ty":"st","c":{{"a":0,"k":[1,0,0,1]}},"o":{{"a":0,"k":100}},"w":{{"a":0,"k":8}},"lc":2,"lj":2}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn clipped_dashed_round_join_zigzag_stays_bounded() {
  // Dashes plus round joins/caps materialize a large stroked outline before
  // clipping can discard the off-screen halves of each alternating segment.
  let json = clipped_dashed_zigzag_json(80);
  assert_lottie!(json);
}

#[test]
fn dashed_polystar_single_copy_stays_bounded() {
  // Polystar points are capped, but a max-legal generated star has enough
  // perimeter that tiny dashes can exhaust memory without a large JSON path.
  let json = dashed_polystar_json(256, 1);
  assert_lottie!(json);
}

#[test]
fn dashed_polystar_repeater_stays_bounded() {
  // Lower point count plus a legal repeater keeps memory lower but pushes the
  // dash/stroke work over a frame budget.
  let json = dashed_polystar_json(160, 16);
  assert_lottie!(json);
}

#[test]
fn gradient_stroke_dash_amplification_stays_bounded() {
  // The tiny-dash long-path explosion also exists for gradient strokes.
  assert_lottie!(
    r#"{
      "v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,
      "layers":[{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{},"shapes":[{"ty":"gr","it":[
        {"ty":"sh","ks":{"a":0,"k":{"c":false,"v":[[0,32],[100000000,32]],"i":[],"o":[]}}},
        {"ty":"gs","s":{"a":0,"k":[0,0]},"e":{"a":0,"k":[64,64]},"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"o":{"a":0,"k":100},"w":{"a":0,"k":2},
         "d":[{"n":"d","v":{"a":0,"k":1}},{"n":"g","v":{"a":0,"k":1}}]},
        {"ty":"tr"}
      ]}]}]
    }"#
  );
}

#[test]
fn long_prefix_missing_asset_lookup_stays_bounded() {
  // Missing precomp refs scan the full asset list and compare every id. Long
  // shared-prefix ids make each failed comparison expensive while staying
  // under the input-size cap.
  let id_len = 3200;
  let prefix = "a".repeat(id_len - 1);
  let missing = format!("{prefix}Z");
  let mut layers = String::new();
  for i in 0..3600 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"{missing}","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let mut assets = String::new();
  for i in 0..512 {
    if i > 0 {
      assets.push(',');
    }
    let suffix = (b'A' + (i % 25) as u8) as char;
    assets.push_str(&format!(r#"{{"id":"{prefix}{suffix}","layers":[]}}"#));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{assets}]}}"#);
  assert_lottie!(json);
}

#[test]
fn duplicate_asset_ids_before_late_target_stay_bounded() {
  // Duplicate asset ids are semantically decoys here: every root ref scans
  // hundreds of same-prefix assets before reaching the late matching target.
  let id_len = 3200;
  let prefix = "a".repeat(id_len - 1);
  let target = format!("{prefix}Z");
  let decoy = format!("{prefix}A");
  let mut layers = String::new();
  for i in 0..3600 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"{target}","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let mut assets = String::new();
  for i in 0..511 {
    if i > 0 {
      assets.push(',');
    }
    assets.push_str(&format!(r#"{{"id":"{decoy}","layers":[]}}"#));
  }
  let mut child_layers = String::new();
  for i in 0..400 {
    if i > 0 {
      child_layers.push(',');
    }
    child_layers.push_str(&format!(r#"{{"ty":3,"ind":{i},"ip":0,"op":60,"st":0,"hd":true,"ks":{{}}}}"#));
  }
  assets.push_str(&format!(r#",{{"id":"{target}","layers":[{child_layers}]}}"#));
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{assets}]}}"#);
  assert_lottie!(json);
}

#[test]
fn reused_precomp_split_position_timelines_stay_bounded() {
  // Split position (`p.s:true`) stores x/y as separate scalar timelines.
  // Reusing one precomp many times makes render repeatedly evaluate both
  // unsorted lists while the hostile JSON only stores the timelines once.
  let mut kfs = String::from(r#"{"t":-1,"s":[0]},{"t":0,"s":[0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0]}"#);
  }
  kfs.push_str(r#",{"t":1000000,"s":[0]}"#);
  let mut layers = String::new();
  for i in 0..2200 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"split","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"split","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"p":{{"s":true,"x":{{"a":1,"k":[{kfs}]}},"y":{{"a":1,"k":[{kfs}]}}}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn reused_precomp_spatial_position_timelines_stay_bounded() {
  // Combined position timelines can carry spatial tangents (`to`/`ti`).
  // The unsorted list forces a linear scan, then the selected segment samples
  // the cubic 32 times to map eased progress by arc length.
  let mut kfs = String::from(r#"{"t":-1,"s":[0,0]},{"t":0,"s":[0,0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0,0]}"#);
  }
  kfs.push_str(r#",{"t":-1,"s":[0,0],"e":[64,64],"to":[1000,1000],"ti":[-1000,-1000]}"#);
  kfs.push_str(r#",{"t":1000000,"s":[64,64]}"#);
  let mut layers = String::new();
  for i in 0..2200 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"spatial","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"spatial","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{"p":{{"a":1,"k":[{kfs}]}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn reused_precomp_auto_orient_spatial_timeline_stays_bounded() {
  // Auto-orient samples position before, at, and after the current frame.
  // Reusing one precomp with one unsorted spatial timeline multiplies that
  // extra evaluation work from a compact JSON payload.
  let mut kfs = String::from(r#"{"t":-1,"s":[0,0]},{"t":0,"s":[0,0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0,0]}"#);
  }
  kfs.push_str(r#",{"t":-1,"s":[0,0],"e":[64,64],"to":[1000,1000],"ti":[-1000,-1000]}"#);
  kfs.push_str(r#",{"t":1000000,"s":[64,64]}"#);
  let mut layers = String::new();
  for i in 0..1800 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"orient","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"orient","layers":[{{"ty":4,"ind":0,"ao":1,"ip":0,"op":60,"st":0,"ks":{{"p":{{"a":1,"k":[{kfs}]}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn reused_precomp_auto_orient_split_position_timelines_stay_bounded() {
  // Split-position tracks evaluate x and y separately; auto-orient samples
  // the position three times, multiplying both unsorted scalar scans.
  // Alpha-only rendering must stay bounded for the same hostile JSON.
  let json = auto_orient_split_position_json(1700);
  assert_lottie!(json);
  let json = auto_orient_split_position_json(1700);
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn shifted_stretched_precomp_auto_orient_timeline_stays_bounded() {
  // `st` and `sr` shift root frame 30 back to child frame 0. The auto-orient
  // child then pays the unsorted spatial-position scan for every precomp ref.
  let mut kfs = String::from(r#"{"t":-1,"s":[0,0]},{"t":0,"s":[0,0]}"#);
  for _ in 0..65_000 {
    kfs.push_str(r#",{"t":-1,"s":[0,0]}"#);
  }
  kfs.push_str(r#",{"t":10000000,"s":[64,64],"to":[1000,1000],"ti":[-1000,-1000]}"#);
  let mut layers = String::new();
  for i in 0..4095 {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"shift","ip":0,"op":60,"st":30,"sr":2,"ks":{{}}}}"#));
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{layers}],"assets":[{{"id":"shift","layers":[{{"ty":4,"ind":0,"ao":1,"ip":0,"op":60,"st":0,"ks":{{"p":{{"a":1,"k":[{kfs}]}}}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json, 64, 30, 31);
}

#[test]
fn reused_precomp_missing_parent_graph_stays_bounded() {
  // Every reused child layer has a missing parent id, so parent-chain
  // resolution scans the asset layer list and misses for each child, each time
  // the precomp is referenced from the root.
  let mut root_layers = String::new();
  for i in 0..3000 {
    if i > 0 {
      root_layers.push(',');
    }
    root_layers.push_str(&format!(r#"{{"ty":0,"ind":{i},"refId":"missparents","ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let mut child_layers = String::new();
  for i in 0..1000 {
    if i > 0 {
      child_layers.push(',');
    }
    child_layers.push_str(&format!(r#"{{"ty":3,"ind":{i},"parent":999999,"ip":0,"op":60,"st":0,"ks":{{}}}}"#));
  }
  let json = format!(r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{root_layers}],"assets":[{{"id":"missparents","layers":[{child_layers}]}}]}}"#);
  assert_lottie!(json);
}

#[test]
fn matte_reuses_masked_precomp_stays_bounded() {
  // Small input, big repeated work: each matte source re-renders the same
  // precomp, and that precomp has a curved mask which flattens before clipping.
  let mut pairs = String::new();
  for i in 0..40 {
    if i > 0 {
      pairs.push(',');
    }
    let src = i * 2;
    let dst = src + 1;
    pairs.push_str(&format!(
      r#"{{"ty":0,"ind":{src},"td":1,"refId":"masked","ip":0,"op":60,"st":0,"ks":{{}}}},{{"ty":4,"ind":{dst},"tt":1,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[4,4]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}"#
    ));
  }
  let mut verts = String::new();
  let mut in_t = String::new();
  let mut out_t = String::new();
  for i in 0..2500 {
    if i > 0 {
      verts.push(',');
      in_t.push(',');
      out_t.push(',');
    }
    verts.push_str(&format!("[{},{}]", i % 64, (i * 7) % 64));
    in_t.push_str("[-2000,-2000]");
    out_t.push_str("[2000,2000]");
  }
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{pairs}],"assets":[{{"id":"masked","layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"hasMask":true,"masksProperties":[{{"mode":"a","o":{{"a":0,"k":100}},"pt":{{"a":0,"k":{{"c":true,"v":[{verts}],"i":[{in_t}],"o":[{out_t}]}}}}}}],"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn rounded_polystar_many_fills_repeater_stays_bounded() {
  // Rounded polystars are generated geometry, not JSON-authored path points.
  // Combining one with many paints and max-legal repeater copies covers the
  // polystar + paint/repeater gap without relying on huge input.
  let json = rounded_polystar_many_paints_json(190, false);
  assert_lottie!(json, 128);
}

#[test]
fn rounded_polystar_many_strokes_repeater_stays_bounded() {
  let json = rounded_polystar_many_paints_json(190, true);
  assert_lottie!(json, 128);
}

#[test]
fn trim_repeater_many_fills_stay_bounded() {
  // Individual trims, many paints, and repeater cloning each have standalone
  // coverage; this keeps the combined legal shape stack as a live repro too.
  let json = trim_repeater_many_paints_json(3000, false);
  assert_lottie!(json, 128);
}

#[test]
fn trim_repeater_many_strokes_stay_bounded() {
  let json = trim_repeater_many_paints_json(3000, true);
  assert_lottie!(json, 128);
}

#[test]
fn many_luma_matte_pairs_large_surface_stays_bounded() {
  // Luma matte pairs are color-dependent and expensive over a larger CPU
  // surface. This covers matte compositing loops outside precomp/time-remap.
  // Alpha-only falls back to color rendering when luma mattes are present, then
  // copies alpha out, so keep that option beside the same payload family.
  let json = luma_matte_pairs_json(2048, 512);
  assert_lottie!(json, 512);
  let json = luma_matte_pairs_json(1800, 256);
  assert_lottie!(
    json,
    256,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn dense_in_bounds_raster_edge_storm_stays_bounded() {
  // This path stays in-bounds and uses straight edges, so it reaches dense
  // raster draw/sweep work instead of failing earlier in curve flattening.
  // Check the same JSON through RGBA, alpha-only, and no-antialias paths.
  let json = dense_in_bounds_raster_edge_storm_json();
  assert_lottie!(json);
  let json = dense_in_bounds_raster_edge_storm_json();
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      alpha_only: true,
      ..RenderOptions::default()
    }
  );
  let json = dense_in_bounds_raster_edge_storm_json();
  assert_lottie!(
    json,
    64,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn many_full_canvas_solid_layers_stay_bounded() {
  // Max-legal solid layers over a large output exercise the straight backend
  // compositing path without masks, gradients, or shape geometry. No-antialias
  // keeps the same full-canvas layer JSON on the alternate raster option.
  let json = solid_layers_json(4096, 512);
  assert_lottie!(json, 512);
  let json = solid_layers_json(4096, 512);
  assert_lottie!(
    json,
    512,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn max_legal_full_canvas_solid_layers_large_surface_stay_bounded() {
  // Keep the layer count legal under the current parser cap, but make every
  // solid cover a 1024x1024 output surface so plain compositing work dominates.
  let json = solid_layers_json(Limits::default().max_layers, 1024);
  assert_lottie!(json, 1024);
}

#[test]
fn many_full_canvas_shape_layers_stay_bounded() {
  // Same max-layer pressure as solid layers, but through the shape walker,
  // coverage cache, and fill backend. No-antialias keeps the same JSON in the
  // alternate raster option.
  let json = shape_layers_json(4096, 512);
  assert_lottie!(json, 512);
  let json = shape_layers_json(4096, 512);
  assert_lottie!(
    json,
    512,
    0,
    1,
    RenderOptions {
      antialias: false,
      ..RenderOptions::default()
    }
  );
}

#[test]
fn max_legal_full_canvas_shape_layers_stay_bounded() {
  // Keep painted shape layers legal under the current default cap, but make
  // every layer a full-canvas fill over a larger surface.
  let json = shape_layers_json(Limits::default().max_painted_shape_layers, 512);
  assert_lottie!(json, 512);
}

#[test]
fn reused_precomp_unsorted_fill_properties_stay_bounded() {
  // Reusing one precomp multiplies file-order scans of unsorted animated
  // properties. Time-remap and position tracks already have coverage; this
  // keeps ordinary fill opacity and color eval as live CPU-renderer PoCs.
  let json = reused_precomp_fill_opacity_timeline_json(2200, 65_000);
  assert_lottie!(json);
  let json = reused_precomp_fill_color_timeline_json(2200, 65_000);
  assert_lottie!(json);
}

#[test]
fn reused_precomp_layer_opacity_timeline_stay_bounded() {
  // Layer transform opacity is evaluated while walking reused precomp layers,
  // so one unsorted opacity timeline can be rescanned thousands of times from
  // compact JSON that stores the timeline only once.
  let json = reused_precomp_layer_opacity_timeline_json(2200, 65_000);
  assert_lottie!(json);
}

#[test]
fn easing_handle_scalar_tail_stays_bounded() {
  // Easing handles accept scalar-array form (`x: [0.5, ...]`) and keep only
  // the first element. The ignored numeric tail is still fully parsed/skipped
  // while constructing one otherwise tiny animated property.
  let json = easing_handle_scalar_tail_json(3_000_000);
  assert_lottie!(json);
}

#[test]
fn duplicate_gradient_stop_property_k_fields_stay_bounded() {
  // `parse_property` parses every duplicate `k` field immediately, keeping
  // only the last property semantically. Repeating a large gradient stop table
  // under one property object multiplies parse/model work without adding
  // distinct shapes or layers.
  let mut stops = String::new();
  for i in 0..220_000 {
    if i > 0 {
      stops.push(',');
    }
    stops.push(if i % 2 == 0 { '0' } else { '1' });
  }
  let k = format!(r#""k":[{stops}]"#);
  let duplicate_k = (0..18).map(|_| k.clone()).collect::<Vec<_>>().join(",");
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"gf","s":{{"a":0,"k":[0,0]}},"e":{{"a":0,"k":[64,64]}},"g":{{"p":2,"k":{{"a":0,{duplicate_k}}}}},"o":{{"a":0,"k":100}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}

#[test]
fn duplicate_scalar_property_k_keyframes_stay_bounded() {
  // Same duplicate-`k` parser surface, but through scalar animated properties:
  // each duplicate carries a legal keyframe list and is fully parsed before the
  // final retained value replaces the previous one.
  let mut kfs = String::from(r#"{"t":0,"s":[100]}"#);
  for t in 1..15_000 {
    kfs.push_str(&format!(r#",{{"t":{t},"s":[{}]}}"#, t % 100));
  }
  let k = format!(r#""k":[{kfs}]"#);
  let duplicate_k = (0..16).map(|_| k.clone()).collect::<Vec<_>>().join(",");
  let json = format!(
    r#"{{"v":"5.5.0","w":64,"h":64,"fr":60,"ip":0,"op":60,"layers":[{{"ty":4,"ind":0,"ip":0,"op":60,"st":0,"ks":{{}},"shapes":[{{"ty":"gr","it":[{{"ty":"rc","p":{{"a":0,"k":[32,32]}},"s":{{"a":0,"k":[64,64]}},"r":{{"a":0,"k":0}}}},{{"ty":"fl","c":{{"a":0,"k":[1,0,0]}},"o":{{"a":1,{duplicate_k}}}}},{{"ty":"tr"}}]}}]}}]}}"#
  );
  assert_lottie!(json);
}
