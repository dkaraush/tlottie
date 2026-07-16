//! Minimal hand-rolled JSON cursor: the foundation the Lottie parser walks on.
//!
//! Design rules (see GOALS.md):
//! - zero dependencies, zero panics: every byte access is bounds-checked,
//!   every failure is a typed [`Error`] with a byte offset;
//! - no recursion: skipping nested values uses an explicit depth counter
//!   bounded by [`Limits::max_nesting_depth`](crate::Limits);
//! - values we don't care about are *skipped*, not validated in depth —
//!   full validation happens where a subtree is actually parsed.

use crate::error::{Error, JsonErrorKind, Limit, Result};

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    max_depth: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8], max_depth: usize) -> Self {
        Self {
            bytes,
            pos: 0,
            max_depth,
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
        }
    }

    fn err(&self, kind: JsonErrorKind) -> Error {
        Error::Json {
            offset: self.pos,
            kind,
        }
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
                    return self
                        .bytes
                        .get(start..self.pos - 1)
                        .ok_or_else(|| self.err(JsonErrorKind::BadString));
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
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut saw_digit = false;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
            saw_digit = true;
        }
        if self.peek() == Some(b'.') {
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
        let token = self
            .bytes
            .get(start..self.pos)
            .ok_or_else(|| self.err(JsonErrorKind::BadNumber))?;
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
