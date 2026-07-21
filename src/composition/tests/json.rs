use super::*;

#[test]
fn integer_fast_path_matches_general_f64_edges() {
  for token in ["0", "-0", "42", "-42", "9007199254740992", "9007199254740993", "1.25", "1e3"] {
    let mut cursor = Cursor::new(token.as_bytes(), 8);
    let parsed = cursor.parse_f64().expect("valid number");
    let expected = token.parse::<f64>().expect("valid number");
    assert_eq!(parsed.to_bits(), expected.to_bits(), "{token}");
  }
}

#[test]
fn easing_cache_reuses_first_curve_across_forked_cursors() {
  let cursor = Cursor::new(b"", 8);
  let fork = cursor.fork_at(0);
  let first = [0.809, 0.0, 0.667, 1.0];
  assert_eq!(cursor.intern_easing(first), first);
  // Both outgoing X values format to 0.81, matching the compatibility
  // cache key used by rlottie and ThorVG.
  assert_eq!(fork.intern_easing([0.814, 0.0, 0.667, 1.0]), first);
  assert_eq!(fork.intern_easing([0.821, 0.0, 0.667, 1.0]), [0.821, 0.0, 0.667, 1.0]);
}
