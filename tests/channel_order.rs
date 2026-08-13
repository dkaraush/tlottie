//! `ChannelOrder::Bgra` must be exactly `ChannelOrder::Rgba` with red and
//! blue exchanged — nothing else may move.
//!
//! This is the whole safety net for the parse-time swap: if any color-bearing
//! node in the model is missed (a gradient stop quad, a keyframe's explicit
//! `end` value, a solid layer, a nested group), the two renders stop being
//! byte-swaps of each other and these tests fail.

use tlottie::{ChannelOrder, Composition, Limits, ParseOptions, RenderOptions};

/// `0xAABBGGRR` -> `0xAARRGGBB`; green and alpha stay put.
fn swap_rb(pixel: u32) -> u32 {
  (pixel & 0xff00_ff00) | ((pixel & 0x00ff_0000) >> 16) | ((pixel & 0x0000_00ff) << 16)
}

fn render(json: &[u8], order: ChannelOrder, frame: f32, size: u32) -> Option<Vec<u32>> {
  let options = ParseOptions {
    channel_order: order,
    ..ParseOptions::default()
  };
  let comp = Composition::parse_with_options(json, &Limits::default(), &options).ok()?;
  let mut renderer = tlottie::CPURenderer::new(comp);
  let mut pixels = vec![0u32; (size as usize) * (size as usize)];
  renderer.render(frame, &mut pixels, size, size, RenderOptions::default()).ok()?;
  Some(pixels)
}

/// Returns the number of pixels that break the invariant.
fn mismatches(json: &[u8], frame: f32, size: u32) -> Option<usize> {
  let rgba = render(json, ChannelOrder::Rgba, frame, size)?;
  let bgra = render(json, ChannelOrder::Bgra, frame, size)?;
  Some(rgba.iter().zip(bgra.iter()).filter(|(r, b)| swap_rb(**r) != **b).count())
}

/// Solid fill, stroke, a linear gradient with an alpha ramp, a solid layer,
/// and a luma track matte — one of each color-bearing node, animated so the
/// keyframe paths are exercised too.
const MIXED: &str = r##"{
  "v":"5.5.7","fr":30,"ip":0,"op":30,"w":64,"h":64,
  "layers":[
    {"ddd":0,"ind":0,"ty":4,"nm":"matte-source","td":1,"sr":1,
     "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[32,32,0]},"a":{"a":0,"k":[0,0,0]},"s":{"a":0,"k":[100,100,100]}},
     "shapes":[
       {"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[52,52]},"r":{"a":0,"k":0}},
         {"ty":"gf","t":1,"s":{"a":0,"k":[-26,0]},"e":{"a":0,"k":[26,0]},
          "g":{"p":2,"k":{"a":0,"k":[0,0.95,0.05,0.05, 1,0.05,0.05,0.95]}},
          "o":{"a":0,"k":100}},
         {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"o":{"a":0,"k":100}}
       ]}
     ],"ip":0,"op":30,"st":0},
    {"ddd":0,"ind":1,"ty":4,"nm":"matted","tt":3,"sr":1,
     "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[32,32,0]},"a":{"a":0,"k":[0,0,0]},"s":{"a":0,"k":[100,100,100]}},
     "shapes":[
       {"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[40,40]},"r":{"a":0,"k":4}},
         {"ty":"gf","t":1,"s":{"a":0,"k":[0,0]},"e":{"a":0,"k":[40,0]},
          "g":{"p":3,"k":{"a":0,"k":[0,0.9,0.1,0.2, 0.5,0.2,0.8,0.3, 1,0.1,0.2,0.95, 0,1, 0.5,0.4, 1,0.8]}},
          "o":{"a":0,"k":100}},
         {"ty":"st","c":{"a":1,"k":[
             {"t":0,"s":[0.95,0.25,0.1,1],"e":[0.1,0.3,0.9,1]},
             {"t":30,"s":[0.1,0.3,0.9,1]}]},
          "o":{"a":0,"k":100},"w":{"a":0,"k":3},"lc":1,"lj":1},
         {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"o":{"a":0,"k":100}}
       ]}
     ],"ip":0,"op":30,"st":0},
    {"ddd":0,"ind":2,"ty":1,"nm":"solid","sr":1,"sc":"#3c78d8","sw":64,"sh":64,
     "ks":{"o":{"a":0,"k":80},"p":{"a":0,"k":[32,32,0]},"a":{"a":0,"k":[32,32,0]},"s":{"a":0,"k":[100,100,100]}},
     "ip":0,"op":30,"st":0},
    {"ddd":0,"ind":3,"ty":4,"nm":"fill","sr":1,
     "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[32,32,0]},"a":{"a":0,"k":[0,0,0]},"s":{"a":0,"k":[100,100,100]}},
     "shapes":[
       {"ty":"gr","it":[
         {"ty":"el","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[50,50]}},
         {"ty":"fl","c":{"a":0,"k":[0.85,0.15,0.35,1]},"o":{"a":0,"k":100}},
         {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"o":{"a":0,"k":100}}
       ]}
     ],"ip":0,"op":30,"st":0}
  ]
}"##;

#[test]
fn bgra_is_rgba_with_red_and_blue_swapped() {
  for frame in [0.0, 7.0, 15.5, 29.0] {
    let bad = mismatches(MIXED.as_bytes(), frame, 64).expect("fixture renders in both orders");
    assert_eq!(bad, 0, "frame {frame} broke the swap invariant on {bad} pixels");
  }
}

#[test]
fn swap_is_not_a_no_op() {
  // Guards the test itself: if the animation were greyscale the invariant
  // above would hold trivially and prove nothing.
  let rgba = render(MIXED.as_bytes(), ChannelOrder::Rgba, 7.0, 64).expect("renders");
  let bgra = render(MIXED.as_bytes(), ChannelOrder::Bgra, 7.0, 64).expect("renders");
  assert!(
    rgba.iter().zip(bgra.iter()).any(|(r, b)| r != b),
    "fixture has no red/blue asymmetry, so it cannot detect a missed swap"
  );
}

/// Sweeps a fixture corpus when `TLOTTIE_FIXTURES` points at a directory of
/// Lottie JSON. Skipped otherwise, so CI stays self-contained.
#[test]
fn corpus_upholds_the_swap_invariant() {
  let Ok(root) = std::env::var("TLOTTIE_FIXTURES") else {
    eprintln!("TLOTTIE_FIXTURES unset - skipping corpus sweep");
    return;
  };
  let mut files = Vec::new();
  collect_json(std::path::Path::new(&root), &mut files);
  assert!(!files.is_empty(), "no .json fixtures under {root}");

  let (mut checked, mut skipped) = (0usize, 0usize);
  let mut failures = Vec::new();
  for path in &files {
    let Ok(bytes) = std::fs::read(path) else {
      skipped += 1;
      continue;
    };
    match mismatches(&bytes, 0.0, 96) {
      Some(0) => checked += 1,
      Some(bad) => {
        checked += 1;
        if failures.len() < 20 {
          failures.push(format!("{} ({bad} px)", path.display()));
        }
      }
      // Does not parse in either order - not this feature's business.
      None => skipped += 1,
    }
  }
  eprintln!("corpus: {checked} checked, {skipped} unparsable, {} broken", failures.len());
  assert!(failures.is_empty(), "swap invariant broken by:\n{}", failures.join("\n"));
}

fn collect_json(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return;
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_dir() {
      collect_json(&path, out);
    } else if path.extension().is_some_and(|e| e == "json") {
      out.push(path);
    }
  }
}
