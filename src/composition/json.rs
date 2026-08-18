//! Minimal hand-rolled JSON cursor: the foundation the Lottie parser walks on.
//!
//! Design rules:
//! - zero dependencies, zero panics: every byte access is bounds-checked,
//!   every failure is a typed [`Error`] with a byte offset;
//! - no recursion: skipping nested values uses an explicit depth counter
//!   bounded by [`Limits::max_nesting_depth`](crate::Limits);
//! - values we don't care about are *skipped*, not validated in depth —
//!   full validation happens where a subtree is actually parsed.

use crate::compat::HashMap;
use crate::error::{Error, JsonErrorKind, Limit, Result};
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use core::cell::{Cell, RefCell};

pub(crate) struct Cursor<'a> {
  bytes: &'a [u8],
  pos: usize,
  max_depth: usize,
  easing_cache: Rc<RefCell<HashMap<String, [f32; 4]>>>,
  saw_animated_property: Rc<Cell<bool>>,
}

impl<'a> Cursor<'a> {
  pub fn new(bytes: &'a [u8], max_depth: usize) -> Self {
    Self {
      bytes,
      pos: 0,
      max_depth,
      easing_cache: Rc::new(RefCell::new(HashMap::new())),
      saw_animated_property: Rc::new(Cell::new(false)),
    }
  }

  pub fn pos(&self) -> usize {
    self.pos
  }

  /// A new cursor over the same input, positioned at `pos`. Used to
  /// re-parse a value whose position was recorded while scanning an object
  /// (JSON object fields have no guaranteed order).
  pub fn fork_at(&self, pos: usize) -> Cursor<'a> {
    Cursor {
      bytes: self.bytes,
      pos,
      max_depth: self.max_depth,
      easing_cache: Rc::clone(&self.easing_cache),
      saw_animated_property: Rc::clone(&self.saw_animated_property),
    }
  }

  pub fn mark_animated_property(&self) {
    self.saw_animated_property.set(true);
  }

  pub fn properties_are_static(&self) -> bool {
    !self.saw_animated_property.get()
  }

  /// Matches the per-composition interpolator cache used by rlottie and
  /// ThorVG. Nearby curves share the first exact controls registered under
  /// their two-decimal cache key.
  pub fn intern_easing(&self, controls: [f32; 4]) -> [f32; 4] {
    let key = format!("{:.2}_{:.2}_{:.2}_{:.2}", controls[0], controls[1], controls[2], controls[3]);
    let mut cache = self.easing_cache.borrow_mut();
    if let Some(easing) = cache.get(&key) {
      return *easing;
    }
    cache.insert(key, controls);
    controls
  }

  fn err(&self, kind: JsonErrorKind) -> Error {
    Error::Json { offset: self.pos, kind }
  }

  pub fn peek(&self) -> Option<u8> {
    self.bytes.get(self.pos).copied()
  }

  pub fn bump(&mut self) -> Option<u8> {
    let b = self.peek();
    if b.is_some() {
      self.pos += 1;
    }
    b
  }

  pub fn skip_ws(&mut self) {
    while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
      self.pos += 1;
    }
  }

  pub fn expect(&mut self, expected: u8) -> Result<()> {
    match self.peek() {
      Some(b) if b == expected => {
        self.pos += 1;
        Ok(())
      }
      Some(b) => Err(self.err(JsonErrorKind::UnexpectedByte(b))),
      None => Err(self.err(JsonErrorKind::UnexpectedEof)),
    }
  }

  /// Reads a string and returns its raw (unescaped) byte contents.
  ///
  /// Escapes are *not* decoded; `\X` pairs are skipped so an escaped quote
  /// cannot terminate the string early. Good enough for key matching —
  /// a key containing escapes simply won't match any known name and its
  /// value gets skipped.
  pub fn read_string_bytes(&mut self) -> Result<&'a [u8]> {
    self.expect(b'"')?;
    let start = self.pos;
    loop {
      match self.bump() {
        Some(b'"') => {
          return self.bytes.get(start..self.pos - 1).ok_or_else(|| self.err(JsonErrorKind::BadString));
        }
        Some(b'\\') => {
          if self.bump().is_none() {
            return Err(self.err(JsonErrorKind::UnexpectedEof));
          }
        }
        Some(_) => {}
        None => return Err(self.err(JsonErrorKind::UnexpectedEof)),
      }
    }
  }

  /// Parses a JSON number. Delegates to `str::parse::<f64>` after scanning
  /// the token extent — correct and panic-free; a bespoke fast path can
  /// replace it later behind the same signature.
  pub fn parse_f64(&mut self) -> Result<f64> {
    let start = self.pos;
    let negative = self.peek() == Some(b'-');
    if negative {
      self.pos += 1;
    }
    let mut saw_digit = false;
    while matches!(self.peek(), Some(b'0'..=b'9')) {
      self.pos += 1;
      saw_digit = true;
    }
    let mut integer_token = true;
    if self.peek() == Some(b'.') {
      integer_token = false;
      self.pos += 1;
      while matches!(self.peek(), Some(b'0'..=b'9')) {
        self.pos += 1;
        saw_digit = true;
      }
    }
    if !saw_digit {
      return Err(self.err(JsonErrorKind::BadNumber));
    }
    if matches!(self.peek(), Some(b'e' | b'E')) {
      integer_token = false;
      self.pos += 1;
      if matches!(self.peek(), Some(b'+' | b'-')) {
        self.pos += 1;
      }
      let mut exp_digit = false;
      while matches!(self.peek(), Some(b'0'..=b'9')) {
        self.pos += 1;
        exp_digit = true;
      }
      if !exp_digit {
        return Err(self.err(JsonErrorKind::BadNumber));
      }
    }
    let token = self.bytes.get(start..self.pos).ok_or_else(|| self.err(JsonErrorKind::BadNumber))?;
    if integer_token {
      let digits = if negative { token.get(1..).unwrap_or(&[]) } else { token };
      let mut value = 0u64;
      let mut exact = true;
      for &digit in digits {
        let Some(next) = value.checked_mul(10).and_then(|v| v.checked_add(u64::from(digit - b'0'))) else {
          exact = false;
          break;
        };
        value = next;
      }
      if exact && value <= (1u64 << 53) {
        let value = value as f64;
        return Ok(if negative { -value } else { value });
      }
    }
    core::str::from_utf8(token)
      .ok()
      .and_then(|s| s.parse::<f64>().ok())
      .filter(|v| v.is_finite())
      .ok_or_else(|| self.err(JsonErrorKind::BadNumber))
  }

  fn expect_keyword(&mut self, rest: &[u8]) -> Result<()> {
    for &expected in rest {
      match self.bump() {
        Some(b) if b == expected => {}
        Some(b) => return Err(self.err(JsonErrorKind::UnexpectedByte(b))),
        None => return Err(self.err(JsonErrorKind::UnexpectedEof)),
      }
    }
    Ok(())
  }

  /// Skips one JSON value of any kind. Iterative; nesting depth is bounded.
  pub fn skip_value(&mut self) -> Result<()> {
    self.skip_ws();
    match self.peek() {
      Some(b'"') => self.read_string_bytes().map(|_| ()),
      Some(b'{' | b'[') => {
        let mut depth: usize = 0;
        loop {
          match self.bump() {
            Some(b'{' | b'[') => {
              depth += 1;
              if depth > self.max_depth {
                return Err(Error::LimitExceeded(Limit::NestingDepth));
              }
            }
            Some(b'}' | b']') => {
              depth -= 1;
              if depth == 0 {
                return Ok(());
              }
            }
            Some(b'"') => {
              self.pos -= 1;
              self.read_string_bytes()?;
            }
            Some(_) => {}
            None => return Err(self.err(JsonErrorKind::UnexpectedEof)),
          }
        }
      }
      Some(b't') => {
        self.pos += 1;
        self.expect_keyword(b"rue")
      }
      Some(b'f') => {
        self.pos += 1;
        self.expect_keyword(b"alse")
      }
      Some(b'n') => {
        self.pos += 1;
        self.expect_keyword(b"ull")
      }
      Some(b'-' | b'0'..=b'9') => self.parse_f64().map(|_| ()),
      Some(b) => Err(self.err(JsonErrorKind::UnexpectedByte(b))),
      None => Err(self.err(JsonErrorKind::UnexpectedEof)),
    }
  }
}

#[cfg(test)]
#[path = "tests/json.rs"]
mod tests;
