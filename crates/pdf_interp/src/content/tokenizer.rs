//! PDF content stream tokenizer.
//!
//! Converts raw content stream bytes into a sequence of [`Token`]s.
//! The PDF content stream grammar is simple: zero or more operand tokens
//! followed by a single operator keyword, repeated until end-of-stream.
//!
//! Inline image data (`BI … ID … EI`) is handled as a single [`Token::InlineImage`]
//! so the caller never sees a bare `BI` operator keyword.

/// A single token from a PDF content stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    /// Integer or real number operand.
    Number(f64),
    /// Boolean literal (`true` / `false`).
    Bool(bool),
    /// Name operand (`/Foo`) — slice into the source buffer, no leading `/`.
    Name(&'a [u8]),
    /// Literal string operand (`(…)`) or hex string (`<…>`), decoded.
    String(Vec<u8>),
    /// Array operand (`[…]`), elements already decoded.
    Array(Vec<Token<'a>>),
    /// Operator keyword — slice into the source buffer.
    Op(&'a [u8]),
    /// Inline image: parameter dictionary bytes + raw pixel data bytes.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between `BI` and `ID`).
        params: &'a [u8],
        /// Raw image data bytes (between `ID` and `EI`).
        data: &'a [u8],
    },
}

/// Iterates over tokens in a PDF content stream byte slice.
///
/// The tokenizer borrows its source slice for its lifetime; zero-copy for
/// names and operator keywords.
pub struct Tokenizer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    /// Create a tokenizer over `src`.
    #[must_use]
    pub fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn remaining(&self) -> &'a [u8] {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.src.len() {
            self.pos += 1;
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                // PDF §7.2.2: NUL (0x00) is also whitespace.
                Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C' | b'\0') => self.advance(),
                Some(b'%') => {
                    // Comment runs to end of line (CR, LF, or CRLF).
                    while !matches!(self.peek(), None | Some(b'\r' | b'\n')) {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn read_name(&mut self) -> &'a [u8] {
        // Caller has verified the current byte is '/'; skip it.
        self.advance();
        let start = self.pos;
        while let Some(b) = self.peek() {
            if is_delimiter(b) || is_whitespace(b) {
                break;
            }
            self.advance();
        }
        &self.src[start..self.pos]
    }

    fn read_literal_string(&mut self) -> Vec<u8> {
        // Caller has verified the current byte is '('; skip it.
        self.advance();
        let mut out = Vec::new();
        let mut depth = 1usize;
        while let Some(b) = self.peek() {
            self.advance();
            match b {
                b'(' => {
                    depth += 1;
                    out.push(b'(');
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(b')');
                }
                b'\\' => match self.peek() {
                    Some(b'n')  => { self.advance(); out.push(b'\n'); }
                    Some(b'r')  => { self.advance(); out.push(b'\r'); }
                    Some(b't')  => { self.advance(); out.push(b'\t'); }
                    Some(b'b')  => { self.advance(); out.push(b'\x08'); }
                    Some(b'f')  => { self.advance(); out.push(b'\x0C'); }
                    Some(b'(')  => { self.advance(); out.push(b'('); }
                    Some(b')')  => { self.advance(); out.push(b')'); }
                    Some(b'\\') => { self.advance(); out.push(b'\\'); }
                    Some(b'\r') => {
                        self.advance();
                        // CRLF line continuation — skip both bytes.
                        if self.peek() == Some(b'\n') { self.advance(); }
                    }
                    Some(b'\n') => { self.advance(); }
                    Some(d) if d.is_ascii_digit() && d < b'8' => {
                        // Up to 3 octal digits (PDF §7.3.4.2).
                        // Max octal value is \377 (255); mask to u8 is safe.
                        let mut val: u16 = 0;
                        for _ in 0..3 {
                            match self.peek() {
                                Some(c) if c.is_ascii_digit() && c < b'8' => {
                                    val = val * 8 + u16::from(c - b'0');
                                    self.advance();
                                }
                                _ => break,
                            }
                        }
                        // Truncate to low byte; values >255 are malformed PDF.
                        out.push(val as u8);
                    }
                    _ => {} // Unrecognised escape — PDF spec says ignore the backslash.
                },
                other => out.push(other),
            }
        }
        out
    }

    fn read_hex_string(&mut self) -> Vec<u8> {
        // Caller has verified the current byte is '<'; skip it.
        self.advance();
        let mut out = Vec::new();
        let mut hi: Option<u8> = None;
        while let Some(b) = self.peek() {
            self.advance();
            if b == b'>' {
                break;
            }
            if is_whitespace(b) {
                continue;
            }
            let nibble = hex_nibble(b);
            match hi {
                None => hi = Some(nibble),
                Some(h) => {
                    out.push((h << 4) | nibble);
                    hi = None;
                }
            }
        }
        // A trailing unpaired nibble is treated as the high nibble with low=0
        // (PDF §7.3.4.3).
        if let Some(h) = hi {
            out.push(h << 4);
        }
        out
    }

    fn read_array(&mut self) -> Vec<Token<'a>> {
        // Caller has verified the current byte is '['; skip it.
        self.advance();
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => break,
                Some(b']') => {
                    self.advance();
                    break;
                }
                _ => {
                    if let Some(tok) = self.next_token() {
                        items.push(tok);
                    }
                }
            }
        }
        items
    }

    /// Read an inline image block starting immediately after the `BI` keyword.
    ///
    /// The PDF spec (§8.9.7) requires:
    /// - Parameter key/value pairs between `BI` and `ID`
    /// - A single whitespace byte immediately after `ID`
    /// - Image data bytes up to `EI`, which must be preceded by whitespace
    ///
    /// On a malformed stream (no `ID` or no `EI`), returns the bytes consumed
    /// so far rather than panicking.
    fn read_inline_image(&mut self) -> Token<'a> {
        // Scan for the `ID` keyword. We do NOT call skip_whitespace_and_comments
        // here because we need the raw param bytes to include any whitespace
        // that is part of the dict; `param_end` is set at the start of `ID`.
        let param_start = self.pos;
        let param_end = loop {
            if self.pos >= self.src.len() {
                break self.pos;
            }
            if self.remaining().starts_with(b"ID")
                && self.remaining().get(2).map_or(true, |&b| is_whitespace(b) || is_delimiter(b))
            {
                let end = self.pos;
                self.pos += 2; // consume `ID`
                // Consume exactly one whitespace byte after `ID` (PDF §8.9.7).
                if matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C')) {
                    self.advance();
                }
                break end;
            }
            self.pos += 1;
        };

        // Scan for `EI` preceded by whitespace. The byte *before* `EI` is
        // whitespace and is not part of the image data.
        let data_start = self.pos;
        let data_end = loop {
            if self.pos >= self.src.len() {
                break self.pos;
            }
            // Check that the byte before `EI` is whitespace; at pos==data_start
            // there is no preceding byte, so we treat that as whitespace (pos 0
            // relative to data).
            let prev_is_ws = self.pos == data_start
                || is_whitespace(self.src[self.pos - 1]);
            if prev_is_ws
                && self.remaining().starts_with(b"EI")
                && self.remaining().get(2).map_or(true, |&b| is_whitespace(b) || is_delimiter(b))
            {
                // data_end is the position of the preceding whitespace byte,
                // which is not part of the image data.
                let end = self.pos.saturating_sub(1);
                self.pos += 2; // consume `EI`
                break end;
            }
            self.pos += 1;
        };

        Token::InlineImage {
            params: &self.src[param_start..param_end],
            data: &self.src[data_start..data_end],
        }
    }

    /// Return the next token, or `None` at end of stream.
    pub fn next_token(&mut self) -> Option<Token<'a>> {
        self.skip_whitespace_and_comments();
        let b = self.peek()?;

        match b {
            b'/' => Some(Token::Name(self.read_name())),
            b'(' => Some(Token::String(self.read_literal_string())),
            b'<' if self.src.get(self.pos + 1) == Some(&b'<') => {
                // Inline dictionary (`<< … >>`). Rare in content streams (only
                // appears in some inline-image parameter dicts). Emit as a raw
                // Op slice so the dispatcher can surface a clear error instead
                // of silently producing wrong output.
                let start = self.pos;
                self.pos += 2;
                let mut depth = 1usize;
                while self.pos < self.src.len() {
                    if self.src[self.pos..].starts_with(b"<<") {
                        depth += 1;
                        self.pos += 2;
                    } else if self.src[self.pos..].starts_with(b">>") {
                        depth -= 1;
                        self.pos += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        self.pos += 1;
                    }
                }
                Some(Token::Op(&self.src[start..self.pos]))
            }
            b'<' => Some(Token::String(self.read_hex_string())),
            b'[' => Some(Token::Array(self.read_array())),
            b']' => {
                // Stray closing bracket — skip iteratively (not recursively) to
                // avoid unbounded call depth on malformed input.
                self.advance();
                self.next_token()
            }
            _ => {
                // Number, boolean, or operator keyword — all start with an
                // ASCII character that is not a delimiter.
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if is_whitespace(c) || is_delimiter(c) {
                        break;
                    }
                    self.advance();
                }
                let word = &self.src[start..self.pos];

                if word == b"true" {
                    return Some(Token::Bool(true));
                }
                if word == b"false" {
                    return Some(Token::Bool(false));
                }

                // Try parsing as an integer or real number.
                if let Ok(s) = std::str::from_utf8(word) {
                    if let Ok(n) = s.parse::<f64>() {
                        return Some(Token::Number(n));
                    }
                }

                // Inline image: consume params + data into a single token so
                // the caller never has to handle a bare `BI` operator.
                if word == b"BI" {
                    return Some(self.read_inline_image());
                }

                Some(Token::Op(word))
            }
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

/// Returns `true` for the six PDF whitespace characters (PDF §7.2.2).
fn is_whitespace(b: u8) -> bool {
    matches!(b, b'\0' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

/// Returns `true` for the ten PDF delimiter characters (PDF §7.2.2).
fn is_delimiter(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

/// Convert a single ASCII hex character to its nibble value.
/// Non-hex characters return 0 (matches Acrobat's lenient behaviour).
fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(src: &[u8]) -> Vec<Token<'_>> {
        Tokenizer::new(src).collect()
    }

    #[test]
    fn numbers() {
        let t = tokens(b"1 2.5 -3 +4.0");
        assert_eq!(t[0], Token::Number(1.0));
        assert_eq!(t[1], Token::Number(2.5));
        assert_eq!(t[2], Token::Number(-3.0));
        assert_eq!(t[3], Token::Number(4.0));
    }

    #[test]
    fn operator() {
        let t = tokens(b"1 0 0 1 100 200 cm");
        assert_eq!(t.last(), Some(&Token::Op(b"cm")));
        assert_eq!(t.len(), 7);
    }

    #[test]
    fn name() {
        let t = tokens(b"/Helvetica");
        assert_eq!(t[0], Token::Name(b"Helvetica"));
    }

    #[test]
    fn literal_string() {
        let t = tokens(b"(hello world)");
        assert_eq!(t[0], Token::String(b"hello world".to_vec()));
    }

    #[test]
    fn literal_string_nested_parens() {
        let t = tokens(b"(a(b)c)");
        assert_eq!(t[0], Token::String(b"a(b)c".to_vec()));
    }

    #[test]
    fn literal_string_octal_escape() {
        // \110 = 0o110 = 72 = 'H'
        let t = tokens(b"(\\110i)");
        assert_eq!(t[0], Token::String(b"Hi".to_vec()));
    }

    #[test]
    fn hex_string() {
        let t = tokens(b"<48656c6c6f>");
        assert_eq!(t[0], Token::String(b"Hello".to_vec()));
    }

    #[test]
    fn hex_string_trailing_nibble() {
        // <9> → high nibble 9, low nibble 0 → 0x90
        let t = tokens(b"<9>");
        assert_eq!(t[0], Token::String(vec![0x90]));
    }

    #[test]
    fn comment_skipped() {
        let t = tokens(b"% this is a comment\n1 2 m");
        assert_eq!(t.len(), 3);
        assert_eq!(t[2], Token::Op(b"m"));
    }

    #[test]
    fn array() {
        let t = tokens(b"[1 2 3]");
        assert_eq!(
            t[0],
            Token::Array(vec![Token::Number(1.0), Token::Number(2.0), Token::Number(3.0)])
        );
    }

    #[test]
    fn booleans() {
        let t = tokens(b"true false");
        assert_eq!(t[0], Token::Bool(true));
        assert_eq!(t[1], Token::Bool(false));
    }

    #[test]
    fn path_sequence() {
        // 100 200 m 300 400 l S → 4 numbers + 3 ops = 7 tokens
        let t = tokens(b"100 200 m 300 400 l S");
        assert_eq!(t.len(), 7);
        assert_eq!(t[6], Token::Op(b"S"));
    }

    #[test]
    fn inline_image() {
        // PDF spec §8.9.7: EI must be preceded by a whitespace byte.
        let src = b"BI /W 1 /H 1 /CS /G /BPC 8 ID \xFF EI";
        let t = tokens(src);
        assert_eq!(t.len(), 1);
        match &t[0] {
            Token::InlineImage { data, .. } => assert_eq!(*data, b"\xFF"),
            other => panic!("expected InlineImage, got {other:?}"),
        }
    }

    #[test]
    fn empty_stream() {
        assert_eq!(tokens(b""), vec![]);
    }

    #[test]
    fn only_comment() {
        assert_eq!(tokens(b"% nothing\n"), vec![]);
    }
}
