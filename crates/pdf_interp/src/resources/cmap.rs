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

use std::borrow::Cow;
use std::collections::HashMap;

// ── Public types ──────────────────────────────────────────────────────────────

/// A resolved `CMap`: maps character codes (as `u32`) to CIDs or Unicode
/// codepoints.
///
/// For `CIDFont` encoding `CMaps` the values are CIDs.  For `ToUnicode` `CMaps`
/// the values are Unicode codepoints.
///
/// # Invariant
///
/// `code_bytes` is always in `1..=4`.  The parser guarantees this; callers that
/// construct a `CMap` directly must uphold it.
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
    /// Reads `self.code_bytes` bytes at a time.  Codes absent from the map are
    /// yielded with `mapped_value = 0` (maps to GID 0 = `.notdef`).
    ///
    /// # Panics
    ///
    /// Panics if `self.code_bytes == 0`, which violates the type invariant.
    #[must_use]
    pub fn iter_codes<'a>(&'a self, bytes: &'a [u8]) -> CMapIter<'a> {
        assert!(self.code_bytes >= 1, "CMap::code_bytes must be ≥ 1");
        CMapIter {
            cmap: self,
            bytes,
            pos: 0,
        }
    }
}

/// Iterator over character codes in a byte string, produced by [`CMap::iter_codes`].
pub struct CMapIter<'a> {
    cmap: &'a CMap,
    bytes: &'a [u8],
    pos: usize,
}

impl Iterator for CMapIter<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let w = usize::from(self.cmap.code_bytes);
        let chunk = self.bytes.get(self.pos..self.pos + w)?;
        // Assemble code big-endian (PDF CMaps are always big-endian).
        let code = chunk.iter().fold(0u32, |acc, &b| (acc << 8) | u32::from(b));
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
#[expect(
    clippy::too_many_lines,
    reason = "flat match dispatch; extracting helpers would obscure the PDF spec mapping"
)]
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
                        // clamp(1, 4) guarantees the value fits u8; the cast
                        // is safe and try_from would add noise for no benefit.
                        if code_bytes == 0 {
                            #[expect(
                                clippy::cast_possible_truncation,
                                reason = "value is clamped to 1..=4, always fits u8"
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
                // outer i += 1 will skip "endcodespacerange"
            }
            "begincidchar" | "beginbfchar" => {
                i += 1;
                while i < tokens.len()
                    && tokens[i] != "endcidchar"
                    && tokens[i] != "endbfchar"
                    && tokens[i] != "begincidrange"
                    && tokens[i] != "beginbfrange"
                    && tokens[i] != "begincidchar"
                    && tokens[i] != "beginbfchar"
                    && tokens[i] != "endcmap"
                {
                    // <code> value
                    if let (Some(code_bytes_v), Some(val)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_value(tokens.get(i + 1).copied()),
                    ) {
                        let code = bytes_to_u32(&code_bytes_v);
                        let _ = map.insert(code, val);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                // If the loop stopped at an end-keyword, the outer i += 1 will
                // skip it.  If it stopped at a section-start sentinel (missing
                // endcidchar), i already points at the next keyword and must NOT
                // be advanced — use `continue` to bypass the outer increment.
                if tokens
                    .get(i)
                    .is_none_or(|t| *t != "endcidchar" && *t != "endbfchar")
                {
                    continue;
                }
            }
            "begincidrange" | "beginbfrange" => {
                i += 1;
                while i < tokens.len()
                    && tokens[i] != "endcidrange"
                    && tokens[i] != "endbfrange"
                    && tokens[i] != "begincidchar"
                    && tokens[i] != "beginbfchar"
                    && tokens[i] != "begincidrange"
                    && tokens[i] != "beginbfrange"
                    && tokens[i] != "endcmap"
                {
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
                        // range is implausibly large (PDF spec allows max 100
                        // entries per block; 0x10000 is a generous ceiling).
                        if hi >= lo && hi - lo < 0x1_0000 {
                            for offset in 0u32..=(hi - lo) {
                                let _ = map.insert(lo + offset, base.saturating_add(offset));
                            }
                        } else {
                            log::warn!("cmap: ignoring degenerate range {lo:04X}–{hi:04X}");
                        }
                        i += 3;
                    } else {
                        i += 1;
                    }
                }
                // Same sentinel logic as begincidchar: skip outer i += 1 when
                // we stopped at a section-start rather than an end-keyword.
                if tokens
                    .get(i)
                    .is_none_or(|t| *t != "endcidrange" && *t != "endbfrange")
                {
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if map.is_empty() && code_bytes == 0 {
        return None;
    }

    // Default to single-byte if no codespacerange was found.
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
#[expect(
    clippy::too_many_lines,
    reason = "match dispatch over 7 byte classes — extracting sub-functions would split tightly coupled index arithmetic"
)]
fn tokenise(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut tokens: Vec<&str> = Vec::with_capacity(256);
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            // Skip whitespace, stray `>` (second `>` of `>>`), PostScript
            // procedure braces `{}`, and stray `)` / `]` outside their container
            // contexts (e.g. unbalanced PS strings).  All of these must be
            // skipped explicitly: the `_` arm stops immediately on any of these
            // (they're in the keyword stop-set), pushes an empty token, and
            // never advances `i` → infinite loop.
            b' ' | b'\t' | b'\r' | b'\n' | b'>' | b')' | b']' | b'{' | b'}' => i += 1,

            // Strip comments (% to end of line).
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

            // PostScript literal string (...) — captured as an opaque token.
            b'(' => {
                let start = i;
                let mut depth = 0u32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'(' => {
                            depth += 1;
                            i += 1;
                        }
                        b')' => {
                            // Saturating prevents underflow on malformed input
                            // (more `)` than `(`) — depth stays at 0 rather
                            // than wrapping and the outer loop exits cleanly.
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        // Skip escaped character; guard against lone `\` at EOF.
                        b'\\' => i += if i + 1 < bytes.len() { 2 } else { 1 },
                        _ => i += 1,
                    }
                }
                tokens.push(&text[start..i]);
            }

            // Array literal [...] — captured as an opaque token.
            // Depth tracking handles `[[a] b]` as a single token.
            b'[' => {
                let start = i;
                let mut depth = 0u32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'[' => {
                            depth += 1;
                            i += 1;
                        }
                        b']' => {
                            if depth == 0 {
                                i += 1;
                                break;
                            }
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

            // Keyword or decimal integer.
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
                            | b'{'
                            | b'}'
                            | b'%'
                    )
                {
                    i += 1;
                }
                if i > start {
                    tokens.push(&text[start..i]);
                }
            }
        }
    }

    tokens
}

// ── Token parsers ─────────────────────────────────────────────────────────────

/// Parse a `<hex>` token into raw bytes.  Returns `None` for non-hex tokens or
/// tokens with invalid (non-hex) content.
fn parse_hex_string(tok: Option<&str>) -> Option<Vec<u8>> {
    let tok = tok?;
    let inner = tok.strip_prefix('<')?.strip_suffix('>')?;

    // PDF spec allows whitespace inside hex strings; strip it.
    // Use Cow to avoid an allocation when no whitespace is present
    // (the common case — tokeniser-produced hex strings are already compact).
    let hex: Cow<'_, str> = if inner.contains(|c: char| c.is_whitespace()) {
        Cow::Owned(inner.chars().filter(|c| !c.is_whitespace()).collect())
    } else {
        Cow::Borrowed(inner)
    };
    if hex.is_empty() {
        return None;
    }

    // Odd-length strings are left-padded with a zero nibble (PDF spec §7.3.4.3).
    let hex: Cow<'_, str> = if hex.len().is_multiple_of(2) {
        hex
    } else {
        Cow::Owned(format!("0{hex}"))
    };

    // Decode pairs of ASCII hex digits directly without going through str::from_utf8.
    // All bytes at this point are either hex digits or whitespace (already stripped),
    // so from_utf8 would never fail — but the byte arithmetic is faster and clearer.
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.bytes();
    while let (Some(hi), Some(lo)) = (iter.next(), iter.next()) {
        let h = hex_nibble(hi)?;
        let l = hex_nibble(lo)?;
        out.push((h << 4) | l);
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Parse a value token: either a decimal integer or a hex string `<...>`.
///
/// Hex strings (used as bfrange base values) are interpreted as big-endian
/// Unicode codepoints.
fn parse_value(tok: Option<&str>) -> Option<u32> {
    let tok = tok?;
    if tok.starts_with('<') {
        parse_hex_string(Some(tok)).map(|b| bytes_to_u32(&b))
    } else {
        tok.parse::<u32>().ok()
    }
}

/// Interpret `bytes` as a big-endian unsigned integer (at most 4 bytes).
///
/// Longer inputs are silently truncated to the first 4 bytes — PDF character
/// codes are at most 4 bytes wide (PDF spec §9.7.6).
fn bytes_to_u32(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(4)
        .fold(0u32, |acc, &b| (acc << 8) | u32::from(b))
}

/// Convert a single ASCII hex character to its nibble value.
/// Returns `None` for non-hex characters (invalid hex string).
const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cid_char_single_byte() {
        // Full CMap stream with a `<<...>>` dict literal — exercises the `>`
        // skip fix in the tokeniser.
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
        // hi < lo: codespace is valid but range block is empty.
        let cmap = parse_cmap(stream).expect("should parse with empty map");
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

    #[test]
    fn missing_end_keyword_does_not_consume_next_section() {
        // A truncated stream with no endcidchar — the parser must not eat the
        // begincidrange block that follows.
        let stream = br#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidchar
<0041> 10
begincidrange
<0042> <0043> 20
endcidrange
        "#;
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&10));
        assert_eq!(cmap.map.get(&0x0042), Some(&20));
        assert_eq!(cmap.map.get(&0x0043), Some(&21));
    }

    #[test]
    fn escaped_backslash_at_eof_does_not_panic() {
        // A PS string ending with a lone backslash must not advance past EOF.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<20> 32\nendcidchar\n(test\\";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }

    #[test]
    fn odd_length_hex_string_left_padded() {
        // <1> should be treated as <01> per PDF spec §7.3.4.3.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<1> 99\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x01), Some(&99));
    }

    #[test]
    fn invalid_hex_chars_rejected() {
        // A hex string containing non-hex characters must not produce garbage output.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<ZZ> 99\nendcidchar\n";
        // <ZZ> fails to parse, so the entry is skipped. The map remains empty
        // but we still get a valid CMap (codespace was found).
        let cmap = parse_cmap(stream).expect("should parse");
        assert!(cmap.map.is_empty());
    }

    #[test]
    fn ps_string_unbalanced_close_paren_does_not_hang() {
        // A PS string token with more `)` than `(` must terminate rather than
        // wrapping the depth counter.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n(unbalanced)) 1 begincidchar\n<20> 32\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }

    #[test]
    fn postscript_procedure_braces_skipped() {
        // `{...}` PostScript procedure bodies appear in Type 4 function CMaps.
        // Braces must be treated as delimiters so `{begincodespacerange}` is
        // not tokenised as a single keyword.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n{beginbfchar} 1 begincidchar\n<20> 32\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }
}
