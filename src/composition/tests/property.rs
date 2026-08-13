use crate::composition::property::{Easing, Keyframe, Property, Timeline};
use alloc::vec;
use alloc::vec::Vec;

fn animated(kfs: Vec<Keyframe<f32>>) -> Property<f32> {
  let mut it = kfs.into_iter();
  let first = it.next().unwrap();
  Property::Animated(Timeline {
    first,
    rest: it.collect(),
    sorted: true,
  })
}

#[test]
fn static_eval() {
  let p = Property::Static(5.0f32);
  assert_eq!(p.eval(10.0), 5.0);
}

#[test]
fn linear_two_keyframes() {
  let p = animated(vec![
    Keyframe {
      t: 0.0,
      value: 0.0f32,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
    Keyframe {
      t: 10.0,
      value: 100.0,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
  ]);
  assert_eq!(p.eval(-1.0), 0.0);
  assert_eq!(p.eval(0.0), 0.0);
  assert!((p.eval(5.0) - 50.0).abs() < 0.01);
  assert_eq!(p.eval(10.0), 100.0);
  assert_eq!(p.eval(11.0), 100.0);
}

#[test]
fn three_segment_lookup() {
  let p = animated(vec![
    Keyframe {
      t: 0.0,
      value: 0.0f32,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
    Keyframe {
      t: 10.0,
      value: 10.0,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
    Keyframe {
      t: 20.0,
      value: 30.0,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
  ]);
  assert!((p.eval(5.0) - 5.0).abs() < 0.01);
  assert!((p.eval(15.0) - 20.0).abs() < 0.01);
}

#[test]
fn hold_keyframe() {
  let p = animated(vec![
    Keyframe {
      t: 0.0,
      value: 1.0f32,
      end: None,
      easing: Easing::HOLD,
      spatial: None,
    },
    Keyframe {
      t: 10.0,
      value: 2.0,
      end: None,
      easing: Easing::LINEAR,
      spatial: None,
    },
  ]);
  assert_eq!(p.eval(9.9), 1.0);
  assert_eq!(p.eval(10.0), 2.0);
}

#[test]
fn eased_segment_monotonic() {
  let ease = Easing {
    ox: 0.42,
    oy: 0.0,
    ix: 0.58,
    iy: 1.0,
    hold: false,
  };
  let mut prev = -1.0f32;
  for i in 0..=20 {
    let v = ease.ease(i as f32 / 20.0);
    assert!(v >= prev - 1e-4, "not monotonic at {i}: {v} < {prev}");
    prev = v;
  }
  assert!(ease.ease(0.0).abs() < 1e-3);
  assert!((ease.ease(1.0) - 1.0).abs() < 1e-3);
}
