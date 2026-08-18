use super::*;
use alloc::vec::Vec;

#[derive(Default)]
struct Counter {
  commands: usize,
  draws: usize,
  contours: usize,
  points: usize,
}

impl FrameRenderer for Counter {
  fn save_layer(&mut self) {
    self.commands += 1;
  }

  fn draw(&mut self, geometry: Geometry<'_>, _paint: Paint<'_>) {
    self.commands += 1;
    self.draws += 1;
    self.contours += geometry.len();
    self.points += geometry.contours().map(|contour| contour.points().count()).sum::<usize>();
  }

  fn apply_mask(&mut self, _geometry: Geometry<'_>, _mode: u8, _inverted: bool, _opacity: u8, _first: bool, _last: bool) {
    self.commands += 1;
  }

  fn end_layer(&mut self, _composite: Composite) {
    self.commands += 1;
  }
}

#[test]
fn solid_premultiplication_matches_cpu_byte_math() {
  let color = Color { r: 0.5, g: 0.25, b: 1.0, a: 0.5 };
  assert_eq!(premul_rgba(color, 0.5), 0x3f3f_101f);
}

#[test]
fn borrowed_sink_receives_evaluated_geometry() {
  let json = br#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
            {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
             "ks":{"o":{"a":0,"k":50},"p":{"a":0,"k":[50,50]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
             "shapes":[{"ty":"gr","it":[
                {"ty":"sh","ks":{"a":0,"k":{"c":true,"v":[[0,0],[10,0],[10,10]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}},
                {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1},
                {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}}
             ]}]}
        ]}"#;
  let comp = Composition::parse(json, &Limits::default()).unwrap();
  let mut counter = Counter::default();
  walk_frame_into(&comp, 0.0, 100, 100, crate::RenderOptions::default(), &mut counter).unwrap();

  assert_eq!(counter.commands, 1);
  assert_eq!(counter.draws, 1);
  assert_eq!(counter.contours, 1);
  assert!(counter.points >= 3);
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Trace {
  draws: Vec<Vec<u32>>,
}

impl FrameRenderer for Trace {
  fn save_layer(&mut self) {}

  fn draw(&mut self, geometry: Geometry<'_>, paint: Paint<'_>) {
    let mut words = Vec::new();
    match paint {
      Paint::Solid(paint) => words.extend([
        0,
        paint.rgba,
        paint.opacity.to_bits(),
        paint.color.r.to_bits(),
        paint.color.g.to_bits(),
        paint.color.b.to_bits(),
        paint.color.a.to_bits(),
      ]),
      Paint::Gradient(paint) => {
        words.extend([
          1,
          paint.transform.a.to_bits(),
          paint.transform.b.to_bits(),
          paint.transform.c.to_bits(),
          paint.transform.d.to_bits(),
          paint.transform.tx.to_bits(),
          paint.transform.ty.to_bits(),
        ]);
        words.extend(paint.lut.iter().copied());
      }
    }
    for contour in geometry.contours() {
      for point in contour.points() {
        words.extend([point.x.to_bits(), point.y.to_bits()]);
      }
      words.push(u32::MAX);
    }
    self.draws.push(words);
  }

  fn apply_mask(&mut self, _geometry: Geometry<'_>, _mode: u8, _inverted: bool, _opacity: u8, _first: bool, _last: bool) {}

  fn end_layer(&mut self, _composite: Composite) {}

  // Always false models a backend whose coverage was evicted between every
  // command. Cached flattened jobs must therefore carry complete geometry.
  fn retains_geometry(&self, _cache_key: u128) -> bool {
    false
  }
}

fn static_triangle() -> Composition {
  let json = br#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
    {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
     "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[50,50]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
     "shapes":[{"ty":"gr","it":[
       {"ty":"sh","ks":{"a":0,"k":{"c":true,"v":[[0,0],[10,0],[10,10]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}},
       {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1},
       {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}}
     ]}]}
  ]}"#;
  Composition::parse(json, &Limits::default()).unwrap()
}

#[test]
fn static_job_replay_is_bit_exact_without_retained_coverage() {
  let comp = static_triangle();
  let mut walker = FrameWalker::default();
  let mut traces = Vec::new();
  for _ in 0..3 {
    let mut trace = Trace::default();
    walker.render(&comp, 0.0, 100, 100, crate::RenderOptions::default(), &mut trace).unwrap();
    traces.push(trace);
  }
  assert_eq!(traces[0], traces[1]);
  assert_eq!(traces[0], traces[2]);
  assert_eq!(walker.static_jobs.hits, 1, "third render should use the two-touch cache");
}

#[test]
fn static_job_cache_invalidates_on_render_context_change() {
  let comp = static_triangle();
  let mut walker = FrameWalker::default();
  for size in [100, 100, 100, 200, 100] {
    walker.render(&comp, 0.0, size, size, crate::RenderOptions::default(), &mut Trace::default()).unwrap();
  }
  assert_eq!(walker.static_jobs.hits, 2, "the changed size must miss while the original exact context remains reusable");
  assert!(walker.static_jobs.bytes <= STATIC_JOB_CACHE_BYTES);
  assert!(walker.static_jobs.entries.len() <= STATIC_JOB_CACHE_ENTRIES);
}

#[test]
fn translating_static_layer_matches_a_fresh_walker() {
  let json = br#"{"fr":30,"ip":0,"op":20,"w":100,"h":100,"layers":[
    {"ty":4,"ind":1,"ip":0,"op":20,"st":0,
     "ks":{"o":{"a":0,"k":100},"p":{"a":1,"k":[{"t":0,"s":[20,30],"e":[60,55]},{"t":10,"s":[60,55]}]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
     "shapes":[{"ty":"gr","it":[
       {"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[20,12]},"r":{"a":0,"k":0}},
       {"ty":"gf","o":{"a":0,"k":100},"r":1,"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"s":{"a":0,"k":[-10,0]},"e":{"a":0,"k":[10,0]},"t":1},
       {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}}
     ]}]}
  ]}"#;
  // A reused walker must render every frame exactly as a fresh one would.
  // The static-job cache is an optimisation, never a change in output.
  let comp = Composition::parse(json, &Limits::default()).unwrap();
  let mut cached = FrameWalker::default();
  for frame in [0.0, 1.0, 2.5, 3.0, 4.0, 7.5, 10.0, 12.0] {
    let mut actual = Trace::default();
    cached.render(&comp, frame, 100, 100, crate::RenderOptions::default(), &mut actual).unwrap();

    let mut expected = Trace::default();
    FrameWalker::default().render(&comp, frame, 100, 100, crate::RenderOptions::default(), &mut expected).unwrap();
    assert_eq!(actual, expected, "frame {frame}");
  }
  // Deliberately no hit-count assertion: a layer that only translates used
  // to be captured at a canonical origin so the translation became a replay
  // parameter, which is what this asserted. That capture had to run
  // unbounded, and the unclipped geometry it produced dropped scanlines on
  // the recording frame. The equivalence above is the property that matters.
}
