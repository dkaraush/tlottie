use super::*;

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
  assert_eq!(premul_argb(color, 0.5), 0x3f1f_103f);
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
