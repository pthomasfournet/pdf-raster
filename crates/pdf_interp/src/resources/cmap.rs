//! PDF `CMap` parser — character code → `CID`/`GID` and `ToUnicode` mappings.
//!
//! # PDF `CMap` primer
//!
//! A Type 0 composite font maps *character codes* (1–4 bytes each) to *CIDs*
//! (Character IDs) via a `Encoding` `CMap` stream, then maps CIDs to *GIDs*
//! (glyph indices in the descendant font) via the `CIDToGIDMap` stream or
//! identity mapping.  A separate `ToUnicode` `CMap` maps character codes to
//! Unicode for text extraction (not needed for rendering, but used to resolve
//! GIDs via the Unicode charmap when no `CIDToGIDMap` is present).
//!
//! # `CMap` stream syntax (subset implemented here)
//!
//! ```text
//! begincodespacerange
//! <00> <FF>          % single-byte codes
//! <0000> <FFFF>      % two-byte codes
//! endcodespacerange
//!
//! begincidchar
//! <20> 32            % code 0x20 → CID 32
//! endcidchar
//!
//! begincidrange
//! <0020> <0080> 100  % codes 0x0020–0x0080 → CIDs 100–196
//! endcidrange
//!
//! beginbfchar          % ToUnicode variant
//! <0041> <0041>        % code 0x0041 → Unicode U+0041
//! endbfchar
//!
//! beginbfrange
//! <0020> <007E> <0020>  % code range → Unicode range
//! endbfrange
//! ```
//!
//! # Byte-width determination
//!
//! The code-space range entries determine the byte width used to read character
//! codes from a content-stream string.  We take the width of the *first*
//! codespace entry.  Real `CMaps` are consistent across all entries, or they use
//! a code-space that spans both 1- and 2-byte ranges (e.g. GB-EUC).  For the
//! common case (all-2-byte or all-1-byte), this is exact.  For mixed-width `CMaps`
//! we take the conservative minimum, which may produce wrong results for the
//! mixed-width codes — correct mixed-width parsing requires a trie/state machine
//! and is deferred.

use std::collections::HashMap;

// ── Public types ──────────────────────────────────────────────────────────────

/// A resolved `CMap`: maps character codes (as `u32`) to CIDs or Unicode
/// codepoints.
///
/// For `CIDFont` encoding `CMaps` the values are CIDs.  For `ToUnicode` `CMaps` the
/// values are Unicode codepoints.
#[derive(Debug, Clone)]
pub struct CMap {
    /// Number of bytes per character code (1–4, derived from `codespacerange`).
    pub code_bytes: u8,
    /// Map: char code → CID or Unicode codepoint.
    pub map: HashMap<u32, u32>,
}

impl CMap {
    /// Decode all character codes from `bytes`, yielding `(char_code, mapped_value)`.
    ///
    /// Reads `self.code_bytes` bytes at a time.  Codes absent from the map
    /// are yielded with `mapped_value = 0` (maps to GID 0 = `.notdef`).
    #[must_use]
    pub const fn iter_codes<'a>(&'a self, bytes: &'a [u8]) -> CMapIter<'a> {
        CMapIter {
            cmap: self,
            bytes,
            pos: 0,
        }
    }
}

/// Iterator over character codes in a byte string.
pub struct CMapIter<'a> {
    cmap: &'a CMap,
    bytes: &'a [u8],
    pos: usize,
}

impl Iterator for CMapIter<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let w = usize::from(self.cmap.code_bytes);
        if self.pos + w > self.bytes.len() {
            return None;
        }
        // Assemble code big-endian (PDF CMaps are always big-endian).
        let mut code = 0u32;
        for &b in &self.bytes[self.pos..self.pos + w] {
            code = (code << 8) | u32::from(b);
        }
        self.pos += w;
        let mapped = self.cmap.map.get(&code).copied().unwrap_or(0);
        Some((code, mapped))
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse a PDF `CMap` byte stream into a [`CMap`].
///
/// Handles both encoding `CMaps` (`begincidchar` / `begincidrange`) and
/// `ToUnicode` `CMaps` (`beginbfchar` / `beginbfrange`).  Both map char codes to
/// integer values (CIDs or Unicode codepoints respectively).
///
/// Returns `None` if the stream contains no recognised directives.  In that
/// case callers should fall back to an identity mapping or the scalar path.
#[must_use]
pub fn parse_cmap(stream: &[u8]) -> Option<CMap> {
    let text = std::str::from_utf8(stream).ok()?;
    let tokens: Vec<&str> = tokenise(text);

    let mut code_bytes: u8 = 0;
    let mut map: HashMap<u32, u32> = HashMap::new();

    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i] {
            "begincodespacerange" => {
                i += 1;
                while i < tokens.len() && tokens[i] != "endcodespacerange" {
                    // Consume pairs: <lo> <hi>
                    if let (Some(lo_bytes), Some(_hi_bytes)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_hex_string(tokens.get(i + 1).copied()),
                    ) {
                        // First codespace entry determines the byte width.
                        if code_bytes == 0 {
                            #[expect(
                                clippy::cast_possible_truncation,
                                reason = "clamped to 1..=4, always fits u8"
                            )]
                            {
                                code_bytes = lo_bytes.len().clamp(1, 4) as u8;
                            }
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            "begincidchar" | "beginbfchar" => {
                i += 1;
                while i < tokens.len() && tokens[i] != "endcidchar" && tokens[i] != "endbfchar" {
                    // <code> value
                    if let (Some(code_bytes_v), Some(val)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_value(tokens.get(i + 1).copied()),
                    ) {
                        let code = bytes_to_u32(&code_bytes_v);
                        map.insert(code, val);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            "begincidrange" | "beginbfrange" => {
                i += 1;
                while i < tokens.len() && tokens[i] != "endcidrange" && tokens[i] != "endbfrange" {
                    // <lo> <hi> base_value
                    // For bfrange, base_value may be a hex string (start Unicode).
                    if let (Some(lo_b), Some(hi_b), Some(base)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_hex_string(tokens.get(i + 1).copied()),
                        parse_value(tokens.get(i + 2).copied()),
                    ) {
                        let lo = bytes_to_u32(&lo_b);
                        let hi = bytes_to_u32(&hi_b);
                        // Guard against malformed CMaps where hi < lo or the
                        // range is implausibly large.
                        if hi >= lo && hi - lo < 0x1_0000 {
                            for (offset, code) in (lo..=hi).enumerate() {
                                #[expect(
                                    clippy::cast_possible_truncation,
                                    reason = "offset bounded by hi-lo < 0x10000, which fits u32"
                                )]
                                map.insert(code, base.saturating_add(offset as u32));
                            }
                        } else {
                            log::warn!("cmap: ignoring degenerate range {lo:04X}–{hi:04X}");
                        }
                        i += 3;
                    } else {
                        i += 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if map.is_empty() && code_bytes == 0 {
        return None;
    }

    // Default to single-byte if no codespace range was found.
    if code_bytes == 0 {
        code_bytes = 1;
    }

    Some(CMap { code_bytes, map })
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Split a `CMap` stream into tokens.
///
/// Tokens are: keyword strings, `<hexstring>` literals, decimal integers, and
/// PostScript string literals `(...)`.  Comments (`%` to end of line) are
/// stripped.  This is a minimal subset sufficient for the `CMap` syntax above.
fn tokenise(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut tokens: Vec<&str> = Vec::with_capacity(256);
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            // Skip whitespace and stray `>` (second `>` of `>>` after the
            // `<..>` arm consumed the first); without the `>` case it falls to
            // `_`, pushes an empty token, and never advances `i` → hang.
            b' ' | b'\t' | b'\r' | b'\n' | b'>' => i += 1,

            // Strip comments.
            b'%' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }

            // Hex string <...>
            b'<' => {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // consume '>'
                }
                tokens.push(&text[start..i]);
            }

            // PostScript literal string (...) — include as opaque token.
            b'(' => {
                let start = i;
                let mut depth = 0i32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'(' => {
                            depth += 1;
                            i += 1;
                        }
                        b')' => {
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        b'\\' => i += 2, // skip escaped char
                        _ => i += 1,
                    }
                }
                tokens.push(&text[start..i]);
            }

            // Array literal [...] — include as single token (not yet decoded).
            // Track nesting depth so `[[a] b]` is consumed as one token.
            b'[' => {
                let start = i;
                let mut depth = 0i32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'[' => {
                            depth += 1;
                            i += 1;
                        }
                        b']' => {
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => i += 1,
                    }
                }
                tokens.push(&text[start..i]);
            }

            // Keyword or number.
            _ => {
                let start = i;
                while i < bytes.len()
                    && !matches!(
                        bytes[i],
                        b' ' | b'\t'
                            | b'\r'
                            | b'\n'
                            | b'<'
                            | b'>'
                            | b'('
                            | b')'
                            | b'['
                            | b']'
                            | b'%'
                    )
                {
                    i += 1;
                }
                tokens.push(&text[start..i]);
            }
        }
    }

    tokens
}

// ── Token parsers ─────────────────────────────────────────────────────────────

/// Parse a `<hex>` token into raw bytes.  Returns `None` for non-hex tokens.
fn parse_hex_string(tok: Option<&str>) -> Option<Vec<u8>> {
    let tok = tok?;
    let inner = tok.strip_prefix('<')?.strip_suffix('>')?;
    // Ignore whitespace in the hex string.
    let hex: String = inner.chars().filter(|c| !c.is_whitespace()).collect();
    if hex.is_empty() {
        return None;
    }
    // Pad to even length (hex strings must have an even number of nibbles).
    let hex = if hex.len().is_multiple_of(2) {
        hex
    } else {
        format!("0{hex}")
    };
    let bytes: Option<Vec<u8>> = hex
        .as_bytes()
        .chunks(2)
        .map(|pair| {
            let s = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(s, 16).ok()
        })
        .collect();
    bytes.filter(|b| !b.is_empty())
}

/// Parse a `value` token: either a decimal integer or a hex string `<...>`.
///
/// For bfrange base values that are hex strings, the hex is interpreted as a
/// big-endian Unicode codepoint.
fn parse_value(tok: Option<&str>) -> Option<u32> {
    let tok = tok?;
    if tok.starts_with('<') {
        parse_hex_string(Some(tok)).map(|b| bytes_to_u32(&b))
    } else {
        tok.parse::<u32>().ok()
    }
}

/// Interpret `bytes` as a big-endian unsigned integer.
///
/// Only the first 4 bytes are consumed; longer inputs are silently truncated.
/// PDF char codes are at most 4 bytes (PDF spec §9.7.6), so this is safe.
fn bytes_to_u32(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(4)
        .fold(0u32, |acc, &b| (acc << 8) | u32::from(b))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cid_char_single_byte() {
        let stream = br#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def
/CMapName /Identity-H def
/CMapType 1 def
1 begincodespacerange
<00> <FF>
endcodespacerange
3 begincidchar
<20> 32
<41> 65
<61> 97
endcidchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.code_bytes, 1);
        assert_eq!(cmap.map.get(&0x20), Some(&32));
        assert_eq!(cmap.map.get(&0x41), Some(&65));
        assert_eq!(cmap.map.get(&0x61), Some(&97));
        assert_eq!(cmap.map.get(&0x00), None);
    }

    #[test]
    fn parse_cid_range_two_byte() {
        let stream = br#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0020> <0022> 100
endcidrange
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.code_bytes, 2);
        assert_eq!(cmap.map.get(&0x0020), Some(&100));
        assert_eq!(cmap.map.get(&0x0021), Some(&101));
        assert_eq!(cmap.map.get(&0x0022), Some(&102));
        assert_eq!(cmap.map.get(&0x0023), None);
    }

    #[test]
    fn parse_bfchar() {
        let stream = br#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0041> <0041>
<0042> <0042>
endbfchar
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&0x0041));
        assert_eq!(cmap.map.get(&0x0042), Some(&0x0042));
    }

    #[test]
    fn parse_bfrange() {
        let stream = br#"
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfrange
<0041> <0043> <0041>
endbfrange
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&0x0041));
        assert_eq!(cmap.map.get(&0x0042), Some(&0x0042));
        assert_eq!(cmap.map.get(&0x0043), Some(&0x0043));
    }

    #[test]
    fn degenerate_range_skipped() {
        let stream = br#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0050> <0040> 0
endcidrange
        "#;
        // hi < lo: should parse successfully but produce no entries.
        let cmap = parse_cmap(stream).expect("should parse (empty map)");
        assert!(cmap.map.is_empty());
    }

    #[test]
    fn iter_codes_two_byte() {
        let stream = br#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0041> <0043> 10
endcidrange
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        // Encode three characters as 2-byte big-endian codes.
        let bytes: &[u8] = &[0x00, 0x41, 0x00, 0x42, 0x00, 0x43];
        let decoded: Vec<(u32, u32)> = cmap.iter_codes(bytes).collect();
        assert_eq!(decoded, vec![(0x0041, 10), (0x0042, 11), (0x0043, 12)]);
    }

    #[test]
    fn iter_codes_missing_maps_to_zero() {
        let stream = br#"
1 begincodespacerange
<00> <FF>
endcodespacerange
1 begincidchar
<41> 65
endcidchar
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        let bytes: &[u8] = &[0x41, 0x42]; // 0x42 not in map
        let decoded: Vec<(u32, u32)> = cmap.iter_codes(bytes).collect();
        assert_eq!(decoded, vec![(0x41, 65), (0x42, 0)]);
    }

    #[test]
    fn empty_stream_returns_none() {
        assert!(parse_cmap(b"").is_none());
    }
}
