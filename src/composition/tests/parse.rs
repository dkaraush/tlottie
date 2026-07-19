use super::*;

const MINIMAL: &str = r#"{"v":"5.5.2","fr":60,"ip":0,"op":180,"w":512,"h":512,"nm":"t","layers":[]}"#;

fn parse(s: &str) -> Result<Composition> {
  parse_composition(s.as_bytes(), &Limits::default())
}

#[test]
fn minimal_composition() {
  let comp = parse(MINIMAL).unwrap();
  assert_eq!(comp.width, 512);
  assert_eq!(comp.height, 512);
  assert_eq!(comp.frame_rate, 60.0);
  assert_eq!(comp.frame_count(), 180);
}

#[test]
fn skips_unknown_fields() {
  let comp = parse(r#"{"junk":{"a":[1,2,{"b":"\" }"}]},"fr":30,"ip":5,"op":65,"w":100,"h":50}"#).unwrap();
  assert_eq!((comp.width, comp.height), (100, 50));
  assert_eq!(comp.frame_count(), 60);
}

#[test]
fn rejects_truncated_input() {
  let bytes = MINIMAL.as_bytes();
  for cut in 1..bytes.len() {
    let sliced = &bytes[..cut];
    assert!(parse_composition(sliced, &Limits::default()).is_err(), "accepted truncation at {cut}");
  }
}

#[test]
fn rejects_missing_header_fields() {
  assert!(matches!(parse(r#"{"fr":30,"ip":0,"op":60,"w":100}"#), Err(Error::InvalidLottie { .. })));
}

#[test]
fn rejects_deep_nesting() {
  let mut s = String::from(r#"{"a":"#);
  for _ in 0..1000 {
    s.push('[');
  }
  assert!(matches!(parse(&s), Err(Error::LimitExceeded(Limit::NestingDepth))));
}

#[test]
fn rejects_oversized_dimensions() {
  assert!(matches!(parse(r#"{"fr":30,"ip":0,"op":60,"w":1e9,"h":100}"#), Err(Error::LimitExceeded(Limit::CompositionSize))));
}

#[test]
fn rejects_trailing_data() {
  let with_trailer = format!("{MINIMAL}x");
  assert!(matches!(
    parse(&with_trailer),
    Err(Error::Json {
      kind: JsonErrorKind::TrailingData,
      ..
    })
  ));
}

#[test]
fn rejects_non_finite_numbers() {
  assert!(parse(r#"{"fr":1e999,"ip":0,"op":60,"w":10,"h":10}"#).is_err());
}

#[test]
fn parses_shape_layer() {
  let comp = parse(
    r#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
              {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
               "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[50,50]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
               "shapes":[{"ty":"gr","it":[
                  {"ty":"sh","ks":{"a":0,"k":{"c":true,"v":[[0,0],[10,0],[10,10]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}},
                  {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1},
                  {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}}
               ]}]}
          ]}"#,
  )
  .unwrap();
  assert_eq!(comp.layers.len(), 1);
  let Layer { kind, shapes, .. } = comp.layers.first().unwrap();
  assert_eq!(*kind, LayerKind::Shape);
  assert_eq!(shapes.len(), 1);
  let Some(Shape::Group(g)) = shapes.first() else {
    panic!("expected group");
  };
  assert_eq!(g.shapes.len(), 2); // path + fill; tr became the group transform
  assert!(matches!(g.shapes.first(), Some(Shape::Path(_))));
  assert!(matches!(g.shapes.get(1), Some(Shape::Fill(_))));
}

#[test]
fn parses_round_corners_modifier() {
  let comp = parse(
    r#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":30,"st":0,"ks":{},"shapes":[
        {"ty":"gr","it":[
          {"ty":"sh","ks":{"a":0,"k":{"c":true,"v":[[0,0],[20,0],[20,20]],"i":[[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0]]}}},
          {"ty":"rd","r":{"a":0,"k":4}},
          {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1}
        ]}
      ]}
    ]}"#,
  )
  .unwrap();
  let Some(Shape::Group(group)) = comp.layers[0].shapes.first() else {
    panic!("expected shape group");
  };
  let Some(Shape::RoundCorners(round)) = group.shapes.get(1) else {
    panic!("expected round-corners modifier");
  };
  assert_eq!(round.radius.eval(0.0), 4.0);
}

#[test]
fn parses_animated_position() {
  let comp = parse(
    r#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
              {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
               "ks":{"p":{"a":1,"k":[
                  {"t":0,"s":[0,0],"i":{"x":[0.5],"y":[0.5]},"o":{"x":[0.5],"y":[0.5]}},
                  {"t":30,"s":[100,100]}
               ]}},
               "shapes":[]}
          ]}"#,
  )
  .unwrap();
  let layer = comp.layers.first().unwrap();
  let p0 = layer.transform.position.eval(0.0);
  let p30 = layer.transform.position.eval(30.0);
  assert_eq!((p0.x, p0.y), (0.0, 0.0));
  assert_eq!((p30.x, p30.y), (100.0, 100.0));
}
