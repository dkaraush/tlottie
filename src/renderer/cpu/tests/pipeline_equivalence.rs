use crate::renderer::cpu::executor::{render_pooled, RenderScratch};
use crate::renderer::frame::FrameRenderer;
use crate::{Composition, Limits, RenderOptions};
use alloc::format;
use alloc::vec;

#[test]
fn single_mask_fast_path_matches_accumulator_formula() {
  for &mode in b"asif" {
    for inverted in [false, true] {
      for opacity in [0u8, 1, 63, 128, 254, 255] {
        for sample in [0u8, 1, 63, 127, 128, 254, 255] {
          let mut actual = [sample];
          super::prepare_single_mask(&mut actual, mode, inverted, opacity);

          let mut contribution = u32::from(sample);
          if inverted {
            contribution = 255 - contribution;
          }
          contribution = (contribution * u32::from(opacity) + 127) / 255;
          let initial = if matches!(mode, b'a' | b'f') { 0u32 } else { 255 };
          let expected = match mode {
            b's' => (initial * (255 - contribution) + 127) / 255,
            b'i' => (initial * contribution + 127) / 255,
            b'f' => initial.abs_diff(contribution),
            _ => contribution + ((255 - contribution) * initial + 127) / 255,
          } as u8;
          assert_eq!(actual[0], expected, "mode={} inverted={inverted} opacity={opacity} sample={sample}", mode as char);
        }
      }
    }
  }
}

fn empty_renderer() -> crate::CPURenderer {
  let composition = Composition::parse(br#"{"fr":30,"ip":0,"op":1,"w":2,"h":2,"layers":[]}"#, &Limits::default()).unwrap();
  crate::CPURenderer::new(composition)
}

#[test]
fn render_options_clear_destination_by_default() {
  assert!(RenderOptions::default().clear);
  let mut renderer = empty_renderer();
  let mut pixels = [0x8040_2010; 4];
  renderer.render(0.0, &mut pixels, 2, 2, RenderOptions::default()).unwrap();
  assert_eq!(pixels, [0; 4]);
}

#[test]
fn disabled_clear_preserves_an_empty_animation_destination() {
  let mut renderer = empty_renderer();
  let original = [0x8040_2010, 0xff30_2010, 0x4020_1008, 0];
  let mut pixels = original;
  renderer
    .render(
      0.0,
      &mut pixels,
      2,
      2,
      RenderOptions {
        clear: false,
        ..RenderOptions::default()
      },
    )
    .unwrap();
  assert_eq!(pixels, original);
}

fn assert_renders_over_existing_pixels(json: &[u8], width: u32, height: u32) {
  let composition = Composition::parse(json, &Limits::default()).unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let len = width as usize * height as usize;
  let mut source = vec![0; len];
  renderer.render(0.0, &mut source, width, height, RenderOptions::default()).unwrap();

  for background in [0xff60_4020, 0xff10_3050] {
    let mut expected = vec![background; len];
    crate::simd::composite_over_span(&mut expected, &source, 255);
    let mut actual = vec![background; len];
    renderer
      .render(
        0.0,
        &mut actual,
        width,
        height,
        RenderOptions {
          clear: false,
          ..RenderOptions::default()
        },
      )
      .unwrap();
    assert_eq!(actual, expected);
  }
}

#[test]
fn disabled_clear_uses_source_over_for_static_solid_content() {
  assert_renders_over_existing_pixels(
    br##"{"fr":30,"ip":0,"op":1,"w":4,"h":4,"layers":[
      {"ty":1,"ind":1,"sw":4,"sh":4,"sc":"#ff0000","ip":0,"op":1,"st":0,
       "ks":{"o":{"a":0,"k":50},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}}
    ]}"##,
    4,
    4,
  );
}

#[test]
fn disabled_clear_uses_source_over_for_cached_gradients() {
  assert_renders_over_existing_pixels(
    br#"{"fr":30,"ip":0,"op":1,"w":8,"h":8,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":1,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[4,4]},"s":{"a":0,"k":[6,6]},"r":{"a":0,"k":1}},
         {"ty":"gf","o":{"a":0,"k":70},"r":1,"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"s":{"a":0,"k":[1,1]},"e":{"a":0,"k":[7,7]},"t":1}
       ]}]}
    ]}"#,
    8,
    8,
  );
}

#[test]
fn bitmap_binding_rejects_nested_targets() {
  let mut renderer = empty_renderer();
  let mut outer = [0u32; 4];
  let mut inner = [0u32; 4];
  let result = renderer.with_bitmap(&mut outer, 2, 2, RenderOptions::default(), |renderer| {
    renderer.with_bitmap(&mut inner, 2, 2, RenderOptions::default(), |_| Ok(()))
  });
  assert!(result.is_err());
  assert!(renderer.bitmap.is_none());
}

#[test]
fn bitmap_binding_recycles_unfinished_frame_state_on_error() {
  let mut renderer = empty_renderer();
  let mut pixels = [0u32; 4];
  let result = renderer.with_bitmap(&mut pixels, 2, 2, RenderOptions::default(), |renderer| {
    renderer.save_layer();
    Err::<(), _>(crate::Error::InvalidLottie { offset: 0, what: "test error" })
  });
  assert!(result.is_err());
  assert!(renderer.bitmap.is_none());
  assert!(renderer.surfaces.is_empty());
  assert!(renderer.surface_dirty.is_empty());
  assert!(renderer.surface_rows.is_empty());
  assert!(renderer.mask_accumulator.is_none());
}

#[test]
fn faint_shape_opacity_is_truncated_to_a_byte() {
  let composition = Composition::parse(
    br#"{"fr":30,"ip":0,"op":1,"w":2,"h":2,"layers":[{"ty":4,"ind":1,"ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]}},"shapes":[{"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[1,1]},"s":{"a":0,"k":[2,2]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100},"r":1},{"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":0.68}}]}]}]}"#,
    &Limits::default(),
  )
  .unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut pixels = [0u32; 4];
  renderer.render(0.0, &mut pixels, 2, 2, RenderOptions::default()).unwrap();
  assert_eq!(pixels[0] >> 24, 1);
}

#[test]
fn shape_layer_opacity_is_applied_after_paints_are_flattened() {
  let composition = Composition::parse(
    br#"{"fr":30,"ip":0,"op":1,"w":4,"h":4,"layers":[{"ty":4,"ind":1,"ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":50},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]}},"shapes":[{"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[2,2]},"s":{"a":0,"k":[4,4]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100},"r":1}]},{"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[2,2]},"s":{"a":0,"k":[4,4]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100},"r":1}]}]}]}"#,
    &Limits::default(),
  )
  .unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut pixels = [0u32; 16];
  renderer.render(0.0, &mut pixels, 4, 4, RenderOptions::default()).unwrap();
  assert_eq!(pixels[5] >> 24, 127);
}

#[test]
fn matte_source_precomp_opacity_is_applied_after_flattening() {
  let composition = Composition::parse(
    br##"{"fr":30,"ip":0,"op":1,"w":4,"h":4,
      "assets":[{"id":"matte","w":4,"h":4,"layers":[
        {"ty":1,"ind":1,"sw":4,"sh":4,"sc":"#ffffff","ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}},
        {"ty":1,"ind":2,"sw":4,"sh":4,"sc":"#ffffff","ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}}
      ]}],
      "layers":[
        {"ty":0,"ind":1,"td":1,"refId":"matte","w":4,"h":4,"ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":50},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]}}},
        {"ty":1,"ind":2,"tt":1,"sw":4,"sh":4,"sc":"#00ff00","ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}}
      ]}"##,
    &Limits::default(),
  )
  .unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut pixels = [0u32; 16];
  renderer.render(0.0, &mut pixels, 4, 4, RenderOptions::default()).unwrap();
  assert_eq!(pixels[5] >> 24, 127);
}

#[test]
fn inactive_matte_source_makes_consumer_transparent() {
  let composition = Composition::parse(
    br##"{"fr":30,"ip":0,"op":2,"w":4,"h":4,"layers":[
      {"ty":1,"ind":1,"td":1,"sw":4,"sh":4,"sc":"#ffffff","ip":0,"op":1,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}},
      {"ty":1,"ind":2,"tt":1,"sw":4,"sh":4,"sc":"#00ff00","ip":0,"op":2,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}}
    ]}"##,
    &Limits::default(),
  )
  .unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut pixels = [0u32; 16];
  renderer.render(1.0, &mut pixels, 4, 4, RenderOptions::default()).unwrap();
  assert_eq!(pixels[5], 0);
}

#[test]
fn fractional_matte_out_point_includes_rounded_boundary_frame() {
  let composition = Composition::parse(
    br##"{"fr":29.97,"ip":0,"op":2,"w":4,"h":4,"layers":[
      {"ty":1,"ind":1,"td":1,"sw":4,"sh":4,"sc":"#ffffff","ip":0,"op":1.001,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}},
      {"ty":1,"ind":2,"tt":1,"sw":4,"sh":4,"sc":"#00ff00","ip":0,"op":2,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[2,2]},"a":{"a":0,"k":[2,2]},"s":{"a":0,"k":[100,100]}}}
    ]}"##,
    &Limits::default(),
  )
  .unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut pixels = [0u32; 16];
  renderer.render(1.0, &mut pixels, 4, 4, RenderOptions::default()).unwrap();
  assert_eq!(pixels[5], 0xff00_ff00);
}

#[test]
fn full_color_output_is_premultiplied_rgba_in_memory() {
  fn center_pixel(color: &str) -> u32 {
    let json = format!(
      r##"{{"fr":30,"ip":0,"op":1,"w":4,"h":4,"layers":[{{"ty":1,"ind":1,"sw":4,"sh":4,"sc":"#{color}","ip":0,"op":1,"st":0,"ks":{{"o":{{"a":0,"k":100}},"p":{{"a":0,"k":[2,2]}},"a":{{"a":0,"k":[2,2]}},"s":{{"a":0,"k":[100,100]}}}}}}]}}"##,
    );
    let composition = Composition::parse(json.as_bytes(), &Limits::default()).unwrap();
    let mut renderer = crate::CPURenderer::new(composition);
    let mut pixels = [0u32; 16];
    renderer.render(0.0, &mut pixels, 4, 4, RenderOptions::default()).unwrap();
    pixels[5]
  }

  assert_eq!(center_pixel("ff0000"), 0xff00_00ff);
  assert_eq!(center_pixel("0000ff"), 0xffff_0000);
}

fn assert_matches_direct(json: &[u8]) {
  let composition = Composition::parse(json, &Limits::default()).unwrap();
  let mut direct = vec![0u32; 64 * 64];
  let mut streamed = vec![0u32; 64 * 64];
  let mut direct_scratch = RenderScratch::default();
  render_pooled(&composition, &mut direct_scratch, 0.0, &mut direct, 64, 64, RenderOptions::default()).unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  renderer.render(0.0, &mut streamed, 64, 64, RenderOptions::default()).unwrap();
  assert_eq!(streamed, direct);
}

fn assert_alpha8_matches_rgba(json: &[u8]) {
  let composition = Composition::parse(json, &Limits::default()).unwrap();
  let mut renderer = crate::CPURenderer::new(composition);
  let mut rgba = vec![0u32; 32 * 32];
  let mut alpha8 = vec![0u8; 32 * 32];
  renderer.render(0.0, &mut rgba, 32, 32, RenderOptions::default()).unwrap();
  renderer.render_alpha8(0.0, &mut alpha8, 32, 32, RenderOptions::default()).unwrap();
  for (index, (&pixel, &alpha)) in rgba.iter().zip(&alpha8).enumerate() {
    assert_eq!(alpha, (pixel >> 24) as u8, "pixel {index}");
  }
}

#[test]
fn alpha8_matches_rgba_alpha_for_opaque_and_varying_opacity_gradients() {
  assert_alpha8_matches_rgba(
    br#"{"fr":30,"ip":0,"op":1,"w":32,"h":32,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":1,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[16,16]},"s":{"a":0,"k":[24,24]},"r":{"a":0,"k":3}},
         {"ty":"gf","o":{"a":0,"k":100},"r":1,"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"s":{"a":0,"k":[4,4]},"e":{"a":0,"k":[28,28]},"t":1}
       ]}]},
      {"ty":4,"ind":2,"ip":0,"op":1,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[16,16]},"s":{"a":0,"k":[16,16]},"r":{"a":0,"k":0}},
         {"ty":"gf","o":{"a":0,"k":100},"r":1,"g":{"p":2,"k":{"a":0,"k":[0,0,0,1,1,1,1,1,0,0,1,1]}},"s":{"a":0,"k":[8,16]},"e":{"a":0,"k":[24,16]},"t":1}
       ]}]}
    ]}"#,
  );
}

#[test]
fn streamed_cpu_matches_direct_for_masked_shape() {
  assert_matches_direct(
    br##"{"fr":30,"ip":0,"op":30,"w":64,"h":64,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":75},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "masksProperties":[{"mode":"a","inv":false,"o":{"a":0,"k":100},"pt":{"a":0,"k":{"c":true,"v":[[8,8],[40,8],[40,40],[8,40]],"i":[[0,0],[0,0],[0,0],[0,0]],"o":[[0,0],[0,0],[0,0],[0,0]]}}}],
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[32,32]},"s":{"a":0,"k":[48,48]},"r":{"a":0,"k":0}},
         {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1}
       ]}]}
    ]}"##,
  );
}

#[test]
fn streamed_cpu_matches_direct_for_gradient() {
  assert_matches_direct(
    br#"{"fr":30,"ip":0,"op":30,"w":64,"h":64,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[32,32]},"s":{"a":0,"k":[48,48]},"r":{"a":0,"k":4}},
         {"ty":"gf","o":{"a":0,"k":80},"r":2,"g":{"p":2,"k":{"a":0,"k":[0,1,0,0,1,0,0,1]}},"s":{"a":0,"k":[8,8]},"e":{"a":0,"k":[56,56]},"t":1}
       ]}]}
    ]}"#,
  );
}

#[test]
fn streamed_cpu_composites_gradient_over_existing_content() {
  assert_matches_direct(
    br##"{"fr":30,"ip":0,"op":30,"w":64,"h":64,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[{"ty":"gr","it":[
         {"ty":"rc","p":{"a":0,"k":[32,32]},"s":{"a":0,"k":[40,40]},"r":{"a":0,"k":0}},
         {"ty":"gf","o":{"a":0,"k":50},"r":1,"g":{"p":2,"k":{"a":0,"k":[0,0,0,1,1,0,1,0]}},"s":{"a":0,"k":[12,12]},"e":{"a":0,"k":[52,52]},"t":1}
       ]}]},
      {"ty":1,"ind":2,"sw":64,"sh":64,"sc":"#ff0000","ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[32,32]},"a":{"a":0,"k":[32,32]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}}
    ]}"##,
  );
}

#[test]
fn streamed_cpu_matches_direct_for_matte() {
  assert_matches_direct(
    br##"{"fr":30,"ip":0,"op":30,"w":64,"h":64,"layers":[
      {"ty":1,"ind":1,"td":1,"sw":32,"sh":64,"sc":"#ffffff","ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[32,32]},"a":{"a":0,"k":[16,32]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}},
      {"ty":1,"ind":2,"tt":1,"sw":64,"sh":64,"sc":"#00ff00","ip":0,"op":30,"st":0,
       "ks":{"o":{"a":0,"k":65},"p":{"a":0,"k":[32,32]},"a":{"a":0,"k":[32,32]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}}
    ]}"##,
  );
}

#[test]
fn streamed_cpu_matches_direct_for_precomp_group_opacity() {
  assert_matches_direct(
    br##"{"fr":30,"ip":0,"op":30,"w":64,"h":64,
      "assets":[{"id":"nested","w":64,"h":64,"layers":[
        {"ty":1,"ind":1,"sw":40,"sh":40,"sc":"#ff0000","ip":0,"op":30,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[24,32]},"a":{"a":0,"k":[20,20]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}},
        {"ty":1,"ind":2,"sw":40,"sh":40,"sc":"#0000ff","ip":0,"op":30,"st":0,"ks":{"o":{"a":0,"k":100},"p":{"a":0,"k":[40,32]},"a":{"a":0,"k":[20,20]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}}
      ]}],
      "layers":[{"ty":0,"ind":1,"refId":"nested","w":64,"h":64,"ip":0,"op":30,"st":0,
        "ks":{"o":{"a":0,"k":50},"p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}}}]
    }"##,
  );
}

#[test]
fn recycled_offscreen_rows_match_fresh_renderer_after_motion() {
  let composition = Composition::parse(
    br##"{"fr":30,"ip":0,"op":2,"w":16,"h":4,"layers":[
      {"ty":4,"ind":1,"ip":0,"op":2,"st":0,
       "ks":{"o":{"a":0,"k":50},"p":{"a":1,"k":[{"t":0,"s":[2,2],"e":[14,2]},{"t":1,"s":[14,2]}]},"a":{"a":0,"k":[0,0]},"s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0}},
       "shapes":[
         {"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[4,4]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100},"r":1}]},
         {"ty":"gr","it":[{"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[2,2]},"r":{"a":0,"k":0}},{"ty":"fl","c":{"a":0,"k":[0,0,1,1]},"o":{"a":0,"k":100},"r":1}]}
       ]}
    ]}"##,
    &Limits::default(),
  )
  .unwrap();
  let mut reused = crate::CPURenderer::new(composition.clone());
  let mut first = [0u32; 16 * 4];
  let mut actual = [0u32; 16 * 4];
  reused.render(0.0, &mut first, 16, 4, RenderOptions::default()).unwrap();
  reused.render(1.0, &mut actual, 16, 4, RenderOptions::default()).unwrap();

  let mut fresh = crate::CPURenderer::new(composition);
  let mut expected = [0u32; 16 * 4];
  fresh.render(1.0, &mut expected, 16, 4, RenderOptions::default()).unwrap();
  assert_eq!(actual, expected);
}
