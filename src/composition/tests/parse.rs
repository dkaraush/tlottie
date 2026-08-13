use super::*;
use crate::FitzModifier;
use alloc::format;
use alloc::string::String;
use alloc::vec;

const MINIMAL: &str = r#"{"v":"5.5.2","fr":60,"ip":0,"op":180,"w":512,"h":512,"nm":"t","layers":[]}"#;

fn parse(s: &str) -> Result<Composition> {
  parse_composition(s.as_bytes(), &Limits::default(), &ParseOptions::default())
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
    assert!(parse_composition(sliced, &Limits::default(), &ParseOptions::default()).is_err(), "accepted truncation at {cut}");
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
fn rejects_too_many_solid_layers() {
  let mut layers = String::new();
  for i in 0..=Limits::default().max_solid_layers {
    if i > 0 {
      layers.push(',');
    }
    layers.push_str(&format!(r##"{{"ty":1,"ind":{i},"sw":64,"sh":64,"sc":"#ff0000","ip":0,"op":60,"st":0,"ks":{{}}}}"##));
  }
  let json = format!(r#"{{"fr":30,"ip":0,"op":60,"w":64,"h":64,"layers":[{layers}]}}"#);
  assert!(matches!(parse(&json), Err(Error::LimitExceeded(Limit::SolidLayers))));
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
  assert!(comp.is_static());
  assert_eq!(comp.frame_count(), 1);
}

#[test]
fn primitive_direction_is_order_independent() {
  let comp = parse(
    r#"{"fr":30,"ip":0,"op":30,"w":100,"h":100,"layers":[
              {"ty":4,"ind":1,"ip":0,"op":30,"st":0,"ks":{},
               "shapes":[
                  {"d":3,"ty":"el","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[10,10]}},
                  {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1}
               ]}
          ]}"#,
  )
  .unwrap();
  let Some(Shape::Ellipse(ellipse)) = comp.layers[0].shapes.first() else {
    panic!("expected ellipse");
  };
  assert!(ellipse.reversed);
}

#[test]
fn applies_fitz_table_once_during_parse() {
  let json = r#"{"fr":30,"ip":0,"op":30,"w":16,"h":16,
    "fitz":[{"o":16711680,"f3":255}],
    "layers":[{"ty":4,"nm":"Skin","ind":1,"ip":0,"op":30,"ks":{},"shapes":[
      {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
    ]}]}"#;
  let options = ParseOptions {
    fitz_modifier: FitzModifier::Type3,
    ..ParseOptions::default()
  };
  let comp = parse_composition(json.as_bytes(), &Limits::default(), &options).unwrap();
  let Some(Shape::Fill(fill)) = comp.layers[0].shapes.first() else {
    panic!("expected fill");
  };
  assert_eq!(fill.color.eval(0.0), Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 });
}

#[test]
fn replaces_exact_source_colors_once_during_parse() {
  let json = r#"{"fr":30,"ip":0,"op":30,"w":16,"h":16,
    "layers":[
      {"ty":4,"nm":"Paints","ind":1,"ip":0,"op":30,"ks":{},"shapes":[
        {"ty":"fl","c":{"a":0,"k":[0.2,0.4,0.6,0.5]},"o":{"a":0,"k":100}},
        {"ty":"st","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"w":{"a":0,"k":1}},
        {"ty":"gf","s":{"a":0,"k":[0,0]},"e":{"a":0,"k":[16,0]},
         "g":{"p":2,"k":{"a":0,"k":[0,0.2,0.4,0.6,1,1,0,0]}},"o":{"a":0,"k":100},"r":1}
      ]}
    ]}"#;
  let options = ParseOptions {
    source_color_replacements: vec![SourceColorReplacement {
      source_color: 0x7f33_6699,
      target_color: 0x80aa_bbcc,
    }],
    ..ParseOptions::default()
  };
  let comp = parse_composition(json.as_bytes(), &Limits::default(), &options).unwrap();
  let Shape::Fill(fill) = &comp.layers[0].shapes[0] else {
    panic!("expected fill");
  };
  assert_eq!(
    fill.color.eval(0.0),
    Color {
      r: 0xaa as f32 / 255.0,
      g: 0xbb as f32 / 255.0,
      b: 0xcc as f32 / 255.0,
      a: 1.0
    }
  );
  let Shape::Stroke(stroke) = &comp.layers[0].shapes[1] else {
    panic!("expected stroke");
  };
  assert_eq!(stroke.color.eval(0.0), Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 });
  let Shape::GradientFill(gradient) = &comp.layers[0].shapes[2] else {
    panic!("expected gradient");
  };
  let stops = gradient.stops.eval(0.0);
  assert_eq!(&stops.0[1..4], &[0xaa as f32 / 255.0, 0xbb as f32 / 255.0, 0xcc as f32 / 255.0]);
  assert_eq!(&stops.0[5..8], &[1.0, 0.0, 0.0]);
}

#[test]
fn layer_prefix_color_is_full_argb_and_wins_over_fitz() {
  let json = r#"{"fr":30,"ip":0,"op":30,"w":16,"h":16,
    "fitz":[{"o":16711680,"f3":255}],
    "layers":[
      {"ty":4,"nm":"Accent primary","ind":1,"ip":0,"op":30,"ks":{},"shapes":[
        {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
      ]},
      {"ty":4,"nm":"Other","ind":2,"ip":0,"op":30,"ks":{},"shapes":[
        {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
      ]}
    ]}"#;
  let options = ParseOptions {
    fitz_modifier: FitzModifier::Type3,
    layer_color_replacements: vec![LayerColorReplacement {
      layer_name_prefix: "Accent".into(),
      color: 0x8040_80c0,
    }],
    ..ParseOptions::default()
  };
  let comp = parse_composition(json.as_bytes(), &Limits::default(), &options).unwrap();
  assert_eq!(
    comp.layers[0].color_override,
    Some(Color {
      r: 64.0 / 255.0,
      g: 128.0 / 255.0,
      b: 192.0 / 255.0,
      a: 128.0 / 255.0
    })
  );
  let Some(Shape::Fill(other)) = comp.layers[1].shapes.first() else {
    panic!("expected other fill");
  };
  assert_eq!(other.color.eval(0.0), Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 });
}

#[cfg(feature = "cpu")]
#[test]
fn layer_prefix_color_propagates_through_a_precomp_instance() {
  let json = r##"{"fr":30,"ip":0,"op":30,"w":1,"h":1,
    "assets":[{"id":"asset","layers":[
      {"ty":1,"nm":"shared solid","ind":1,"ip":0,"op":30,"sw":1,"sh":1,"sc":"#ff0000","ks":{}}
    ]}],
    "layers":[{"ty":0,"nm":"Comp 1","ind":1,"ip":0,"op":30,"w":1,"h":1,"refId":"asset","ks":{}}]
  }"##;
  let options = ParseOptions {
    layer_color_replacements: vec![LayerColorReplacement {
      layer_name_prefix: "Comp 1".into(),
      color: 0x8000_ff00,
    }],
    ..ParseOptions::default()
  };
  let comp = parse_composition(json.as_bytes(), &Limits::default(), &options).unwrap();
  let mut renderer = crate::CPURenderer::new(comp);
  let mut pixel = [0u32];
  renderer
    .render(
      0.0,
      &mut pixel,
      1,
      1,
      crate::RenderOptions {
        antialias: false,
        ..crate::RenderOptions::default()
      },
    )
    .unwrap();
  assert_eq!(pixel[0], 0x8000_8000);
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
  assert!(!comp.is_static());
  assert_eq!(comp.frame_count(), 30);
}

#[test]
fn static_content_with_a_visibility_transition_keeps_declared_frames() {
  let comp = parse(
    r##"{"fr":30,"ip":0,"op":30,"w":16,"h":16,"layers":[
      {"ty":1,"ind":1,"sw":16,"sh":16,"sc":"#ffffff","ip":10,"op":30,"st":0,"ks":{}}
    ]}"##,
  )
  .unwrap();
  assert!(!comp.is_static());
  assert_eq!(comp.frame_count(), 30);
}
