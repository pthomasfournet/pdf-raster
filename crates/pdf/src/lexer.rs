//! Shared byte-level scanner primitives used by both the object parser and
//! the content-stream tokenizer.
//!
//! All functions operate on a `(src: &[u8], pos: &mut usize)` pair so callers
//! can drive parsing without wrapping state in a struct.

/// PDF whitespace characters (spec §7.2.2).
#[inline]
pub const fn is_ws(b: u8) -> bool {
    matches!(b, b'\0' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

/// PDF delimiter characters (spec §7.2.2).
#[inline]
pub const fn is_delim(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

/// Hex nibble value; non-hex bytes return 0 (Acrobat-compatible leniency).
#[inline]
pub const fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Advance past whitespace and `%`-comments.
pub fn skip_ws(src: &[u8], pos: &mut usize) {
    loop {
        match src.get(*pos).copied() {
            Some(b) if is_ws(b) => *pos += 1,
            Some(b'%') => {
                *pos += 1;
                while *pos < src.len() && !matches!(src[*pos], b'\r' | b'\n') {
                    *pos += 1;
                }
            }
            _ => break,
        }
    }
}

/// Read a `/Name` token starting at `pos` (which points at the `/`).
/// Returns the name bytes (without the leading `/`) as a slice into `src`.
/// Decodes `#xx` escape sequences into owned `Vec<u8>` when needed.
pub fn read_name(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '/'
    let start = *pos;
    while *pos < src.len() {
        let b = src[*pos];
        if is_ws(b) || is_delim(b) {
            break;
        }
        *pos += 1;
    }
    let raw = &src[start..*pos];

    // Fast path: no '#' escape sequences (the common case).
    if !raw.contains(&b'#') {
        return raw.to_vec();
    }

    // Decode #xx sequences (PDF §7.3.5).
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'#' && i + 2 < raw.len() {
            out.push((hex_nibble(raw[i + 1]) << 4) | hex_nibble(raw[i + 2]));
            i += 3;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    out
}

/// Read a literal `(string)` starting at `pos` (pointing at `(`).
/// Returns decoded bytes (escape sequences resolved, nesting handled).
pub fn read_literal_string(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '('
    let mut out = Vec::new();
    let mut depth = 1usize;
    while *pos < src.len() {
        let b = src[*pos];
        *pos += 1;
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
                if *pos >= src.len() {
                    break;
                }
                let esc = src[*pos];
                *pos += 1;
                match esc {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'b' => out.push(b'\x08'),
                    b'f' => out.push(b'\x0C'),
                    b'(' => out.push(b'('),
                    b')' => out.push(b')'),
                    b'\\' => out.push(b'\\'),
                    b'\r'
                        // Line continuation — CRLF or bare CR.
                        if src.get(*pos) == Some(&b'\n') => {
                        *pos += 1;
                    }
                    b'\n' => {} // Line continuation.
                    d if d.is_ascii_digit() && d < b'8' => {
                        // Octal escape: up to 3 digits.
                        let mut val = u16::from(d - b'0');
                        for _ in 0..2 {
                            match src.get(*pos) {
                                Some(&c) if c.is_ascii_digit() && c < b'8' => {
                                    val = val * 8 + u16::from(c - b'0');
                                    *pos += 1;
                                }
                                _ => break,
                            }
                        }
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "PDF octal values >255 are malformed; truncate per Acrobat leniency"
                        )]
                        out.push(val as u8);
                    }
                    _ => {} // Unknown escape — ignore backslash per spec.
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Read a hex `<string>` starting at `pos` (pointing at `<`).
/// Returns decoded bytes. Trailing unpaired nibble is zero-padded.
pub fn read_hex_string(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '<'
    let mut out = Vec::new();
    let mut hi: Option<u8> = None;
    while *pos < src.len() {
        let b = src[*pos];
        *pos += 1;
        if b == b'>' {
            break;
        }
        if is_ws(b) {
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
    if let Some(h) = hi {
        out.push(h << 4);
    }
    out
}

/// Scan a run of non-whitespace, non-delimiter bytes; return it as a slice.
/// Does NOT advance past the terminator.
pub fn scan_token<'a>(src: &'a [u8], pos: &mut usize) -> &'a [u8] {
    let start = *pos;
    while *pos < src.len() {
        let b = src[*pos];
        if is_ws(b) || is_delim(b) {
            break;
        }
        *pos += 1;
    }
    &src[start..*pos]
}

/// Parse a decimal non-negative integer at `pos`.  Returns `None` if no
/// digit is present.  Does not advance past the terminator.
pub fn parse_u64(src: &[u8], pos: &mut usize) -> Option<u64> {
    let start = *pos;
    while *pos < src.len() && src[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return None;
    }
    std::str::from_utf8(&src[start..*pos])
        .ok()
        .and_then(|s| s.parse().ok())
}
