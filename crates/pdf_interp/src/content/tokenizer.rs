//! PDF content stream tokenizer.
//!
//! Converts raw content stream bytes into a sequence of [`Token`]s.
//! The PDF content stream grammar is simple: zero or more operand tokens
//! followed by a single operator keyword, repeated until end-of-stream.
//!
//! Inline image data (`BI … ID … EI`) is handled as a single [`Token::InlineImage`].

/// A single token from a PDF content stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    /// Integer or real number operand.
    Number(f64),
    /// Boolean literal (`true` / `false`).
    Bool(bool),
    /// Name operand (`/Foo`).
    Name(&'a [u8]),
    /// Literal string operand (`(…)` or `<…>`).
    String(Vec<u8>),
    /// Array operand (`[…]`).
    Array(Vec<Token<'a>>),
    /// Operator keyword.
    Op(&'a [u8]),
    /// Inline image: parameter dictionary bytes + raw pixel data bytes.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between BI and ID).
        params: &'a [u8],
        /// Raw image data bytes (between ID and EI).
        data: &'a [u8],
    },
}

/// Iterates over tokens in a PDF content stream byte slice.
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
                Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C') => self.advance(),
                Some(b'%') => {
                    // Comment runs to end of line.
                    while !matches!(self.peek(), None | Some(b'\r' | b'\n')) {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn read_name(&mut self) -> &'a [u8] {
        // Skip leading '/'.
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
        // Skip opening '('.
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
                b'\\' => {
                    match self.peek() {
                        Some(b'n') => { self.advance(); out.push(b'\n'); }
                        Some(b'r') => { self.advance(); out.push(b'\r'); }
                        Some(b't') => { self.advance(); out.push(b'\t'); }
                        Some(b'b') => { self.advance(); out.push(b'\x08'); }
                        Some(b'f') => { self.advance(); out.push(b'\x0C'); }
                        Some(b'(') => { self.advance(); out.push(b'('); }
                        Some(b')') => { self.advance(); out.push(b')'); }
                        Some(b'\\') => { self.advance(); out.push(b'\\'); }
                        Some(b'\r') => {
                            self.advance();
                            if self.peek() == Some(b'\n') { self.advance(); }
                        }
                        Some(b'\n') => { self.advance(); }
                        Some(d) if d.is_ascii_digit() => {
                            // Up to 3 octal digits.
                            let mut val = 0u32;
                            for _ in 0..3 {
                                match self.peek() {
                                    Some(c) if c.is_ascii_digit() && c < b'8' => {
                                        val = val * 8 + u32::from(c - b'0');
                                        self.advance();
                                    }
                                    _ => break,
                                }
                            }
                            out.push(val as u8);
                        }
                        _ => {} // ignore unrecognised escape
                    }
                }
                other => out.push(other),
            }
        }
        out
    }

    fn read_hex_string(&mut self) -> Vec<u8> {
        // Skip opening '<'.
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
        // Trailing nibble is treated as high nibble with low nibble = 0.
        if let Some(h) = hi {
            out.push(h << 4);
        }
        out
    }

    fn read_array(&mut self) -> Vec<Token<'a>> {
        // Skip opening '['.
        self.advance();
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None | Some(b']') => {
                    if self.peek() == Some(b']') { self.advance(); }
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

    /// Read the inline-image block starting just after the `BI` operator has
    /// been emitted. Consumes up to and including `EI`.
    fn read_inline_image(&mut self) -> Token<'a> {
        // Parameters run from current position to `ID` keyword.
        let param_start = self.pos;
        let param_end;
        loop {
            self.skip_whitespace_and_comments();
            // Look for `ID` followed by a single whitespace byte.
            if self.remaining().starts_with(b"ID")
                && self.remaining().get(2).map_or(true, |&b| is_whitespace(b) || is_delimiter(b))
            {
                param_end = self.pos;
                self.pos += 2; // consume `ID`
                // Consume exactly one whitespace byte after ID (PDF spec §7.3.8.1).
                if matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C')) {
                    self.advance();
                }
                break;
            }
            if self.pos >= self.src.len() {
                param_end = self.pos;
                break;
            }
            self.advance();
        }

        // Image data runs from here to `EI`.
        let data_start = self.pos;
        let data_end;
        loop {
            // Scan for `EI` preceded by whitespace (PDF spec requires it).
            if self.pos >= self.src.len() {
                data_end = self.pos;
                break;
            }
            if is_whitespace(self.src[self.pos.saturating_sub(1)])
                && self.remaining().starts_with(b"EI")
                && self.remaining().get(2).map_or(true, |&b| is_whitespace(b) || is_delimiter(b))
            {
                // data_end excludes the preceding whitespace byte.
                data_end = self.pos - 1;
                self.pos += 2; // consume `EI`
                break;
            }
            self.advance();
        }

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
                // Dictionary — treat as opaque operator for now; caller handles.
                // (Content streams rarely embed raw dicts except inside inline images.)
                // Just emit as an Op so the dispatcher can error clearly.
                let start = self.pos;
                self.pos += 2;
                let mut depth = 1usize;
                while self.pos < self.src.len() {
                    if self.src[self.pos..].starts_with(b"<<") {
                        depth += 1; self.pos += 2;
                    } else if self.src[self.pos..].starts_with(b">>") {
                        depth -= 1; self.pos += 2;
                        if depth == 0 { break; }
                    } else {
                        self.pos += 1;
                    }
                }
                Some(Token::Op(&self.src[start..self.pos]))
            }
            b'<' => Some(Token::String(self.read_hex_string())),
            b'[' => Some(Token::Array(self.read_array())),
            b']' => {
                // Stray closing bracket — skip and continue.
                self.advance();
                self.next_token()
            }
            _ => {
                // Number, boolean, or operator keyword.
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if is_whitespace(c) || is_delimiter(c) { break; }
                    self.advance();
                }
                let word = &self.src[start..self.pos];

                if word == b"true" {
                    return Some(Token::Bool(true));
                }
                if word == b"false" {
                    return Some(Token::Bool(false));
                }

                // Try parsing as a number.
                if let Ok(s) = std::str::from_utf8(word) {
                    if let Ok(n) = s.parse::<f64>() {
                        return Some(Token::Number(n));
                    }
                }

                // Operator keyword — check for inline image start.
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

fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0C' | b'\0')
}

fn is_delimiter(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

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
    fn hex_string() {
        let t = tokens(b"<48656c6c6f>");
        assert_eq!(t[0], Token::String(b"Hello".to_vec()));
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
        // 100 200 m 300 400 l S → 5 numbers + 3 ops = 8 tokens? No:
        // numbers: 100, 200, 300, 400 = 4; ops: m, l, S = 3 → 7 total
        let t = tokens(b"100 200 m 300 400 l S");
        assert_eq!(t.len(), 7);
        assert_eq!(t[6], Token::Op(b"S"));
    }
}
