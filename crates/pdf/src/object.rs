//! PDF object model and single-object parser.
//!
//! [`Object`] mirrors `lopdf::Object` variant-for-variant so `pdf_interp` can
//! swap the import with minimal churn.  The parser is a hand-rolled recursive
//! descent over a borrowed byte slice — no allocator-heavy parser combinators.

use crate::dictionary::Dictionary;
use crate::lexer::{
    parse_u64, read_hex_string, read_literal_string, read_name, scan_token, skip_ws,
};

/// A PDF indirect-object identifier `(object_number, generation)`.
///
/// In practice generation numbers are always 0 in modern PDFs.
pub type ObjectId = (u32, u16);

/// A PDF object value.
///
/// Variants match `lopdf::Object` so `pdf_interp` source changes are
/// mechanical substitutions.
#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    Null,
    Boolean(bool),
    Integer(i64),
    Real(f32),
    Name(Vec<u8>),
    String(Vec<u8>, StringFormat),
    Array(Vec<Object>),
    Dictionary(Dictionary),
    Stream(Stream),
    /// Indirect reference — resolved lazily by `Document::get_object`.
    Reference(ObjectId),
}

/// String encoding hint (mirrors lopdf).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringFormat {
    Literal,
    Hexadecimal,
}

/// A PDF stream: a dictionary plus raw (undecoded) bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    pub dict: Dictionary,
    /// Raw (compressed) stream bytes.
    pub content: Vec<u8>,
}

impl Stream {
    pub fn new(dict: Dictionary, content: Vec<u8>) -> Self {
        Self { dict, content }
    }
}

impl Object {
    // ── lopdf-compatible as_* accessors ──────────────────────────────────────

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            Self::Real(r) => Some(*r as i64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Real(r) => Some(*r),
            Self::Integer(n) => Some(*n as f32),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_f32().map(f64::from)
    }

    pub fn as_name(&self) -> Option<&[u8]> {
        match self {
            Self::Name(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&[u8]> {
        match self {
            Self::String(s, _) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Object]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&Dictionary> {
        match self {
            Self::Dictionary(d) => Some(d),
            Self::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }

    pub fn as_stream(&self) -> Option<&Stream> {
        match self {
            Self::Stream(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_reference(&self) -> Option<ObjectId> {
        match self {
            Self::Reference(id) => Some(*id),
            _ => None,
        }
    }

    /// Return the variant name as a static string (mirrors lopdf's
    /// `Object::enum_variant` so error messages can describe a value's kind
    /// without dumping the full Debug representation).
    #[must_use]
    pub fn enum_variant(&self) -> &'static str {
        match self {
            Self::Null => "Null",
            Self::Boolean(_) => "Boolean",
            Self::Integer(_) => "Integer",
            Self::Real(_) => "Real",
            Self::Name(_) => "Name",
            Self::String(_, _) => "String",
            Self::Array(_) => "Array",
            Self::Dictionary(_) => "Dictionary",
            Self::Stream(_) => "Stream",
            Self::Reference(_) => "Reference",
        }
    }
}

// ── Object parser ─────────────────────────────────────────────────────────────

/// Parse one PDF object starting at `*pos` in `data`.  Advances `*pos` past
/// the object.  Does NOT parse the `<id> <gen> obj … endobj` wrapper — call
/// `parse_indirect_object` for that.
///
/// Returns `None` only on genuine EOF or if the first non-whitespace token is
/// not a valid object start.
pub fn parse_object(data: &[u8], pos: &mut usize) -> Option<Object> {
    skip_ws(data, pos);

    let b = *data.get(*pos)?;

    match b {
        b'/' => {
            let name = read_name(data, pos);
            Some(Object::Name(name))
        }
        b'(' => {
            let s = read_literal_string(data, pos);
            Some(Object::String(s, StringFormat::Literal))
        }
        b'<' if data.get(*pos + 1) == Some(&b'<') => {
            // Dictionary or stream.
            let dict = parse_dict(data, pos)?;
            // Peek past whitespace to see if "stream" follows.
            let mut peek = *pos;
            skip_ws(data, &mut peek);
            if data[peek..].starts_with(b"stream") {
                peek += 6;
                // Consume exactly one newline after "stream".
                if data.get(peek) == Some(&b'\r') {
                    peek += 1;
                }
                if data.get(peek) == Some(&b'\n') {
                    peek += 1;
                }
                let stream_start = peek.min(data.len());
                // Determine stream length from /Length in the dict.
                let length = dict
                    .get(b"Length")
                    .and_then(Object::as_i64)
                    .unwrap_or(0)
                    .max(0) as usize;
                let stream_end = stream_start.saturating_add(length).min(data.len());
                let content = data[stream_start..stream_end].to_vec();
                // Advance pos past "endstream".
                *pos = stream_end;
                skip_ws(data, pos);
                if data[*pos..].starts_with(b"endstream") {
                    *pos += 9;
                }
                Some(Object::Stream(Stream::new(dict, content)))
            } else {
                Some(Object::Dictionary(dict))
            }
        }
        b'<' => {
            let s = read_hex_string(data, pos);
            Some(Object::String(s, StringFormat::Hexadecimal))
        }
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            loop {
                skip_ws(data, pos);
                match data.get(*pos) {
                    None => break,
                    Some(b']') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        if let Some(item) = parse_object(data, pos) {
                            items.push(item);
                        } else {
                            break;
                        }
                    }
                }
            }
            Some(Object::Array(items))
        }
        _ => {
            // Number, bool, null, reference, or keyword.
            parse_scalar(data, pos)
        }
    }
}

fn parse_scalar(data: &[u8], pos: &mut usize) -> Option<Object> {
    let word = scan_token(data, pos);
    match word {
        b"true" => return Some(Object::Boolean(true)),
        b"false" => return Some(Object::Boolean(false)),
        b"null" => return Some(Object::Null),
        _ => {}
    }

    // Try to parse as a number.  If it looks like an integer, peek ahead to
    // check for the "X Y R" reference pattern.
    if let Ok(s) = std::str::from_utf8(word) {
        if let Ok(n) = s.parse::<i64>() {
            // Peek for "gen R" to form an indirect reference. Only valid when
            // the object number fits in u32 and is non-negative; otherwise we
            // fall through and emit a plain Integer.
            if n >= 0 && n <= i64::from(u32::MAX) {
                let mut peek = *pos;
                skip_ws(data, &mut peek);
                let gen_tok = scan_token(data, &mut peek);
                if !gen_tok.is_empty()
                    && let Some(g) = std::str::from_utf8(gen_tok)
                        .ok()
                        .and_then(|s| s.parse::<i64>().ok())
                    && (0..=i64::from(u16::MAX)).contains(&g)
                {
                    skip_ws(data, &mut peek);
                    let r_tok = scan_token(data, &mut peek);
                    if r_tok == b"R" {
                        *pos = peek;
                        return Some(Object::Reference((n as u32, g as u16)));
                    }
                }
            }
            return Some(Object::Integer(n));
        }
        if let Ok(r) = s.parse::<f32>() {
            return Some(Object::Real(r));
        }
    }

    None
}

fn parse_dict(data: &[u8], pos: &mut usize) -> Option<Dictionary> {
    // Caller has ensured data[*pos..] starts with "<<".
    *pos += 2;
    let mut dict = Dictionary::new();

    loop {
        skip_ws(data, pos);
        if data[*pos..].starts_with(b">>") {
            *pos += 2;
            break;
        }
        if *pos >= data.len() {
            break;
        }
        // Key must be a Name.
        if *data.get(*pos)? != b'/' {
            // Malformed dict — skip one byte and retry.
            *pos += 1;
            continue;
        }
        let key = read_name(data, pos);
        // Value is any object.
        skip_ws(data, pos);
        let value = parse_object(data, pos)?;
        dict.insert(key, value);
    }

    Some(dict)
}

/// Parse a complete indirect object (`<id> <gen> obj … endobj`) at `offset`.
/// Returns `(object_id, object_value)`.
pub fn parse_indirect_object(data: &[u8], offset: usize) -> Option<(ObjectId, Object)> {
    let mut pos = offset;
    skip_ws(data, &mut pos);

    let id_raw = parse_u64(data, &mut pos)?;
    if id_raw > u64::from(u32::MAX) {
        return None;
    }
    let id_num = id_raw as u32;
    skip_ws(data, &mut pos);
    let gen_raw = parse_u64(data, &mut pos)?;
    if gen_raw > u64::from(u16::MAX) {
        return None;
    }
    let gen_num = gen_raw as u16;
    skip_ws(data, &mut pos);

    if !data[pos..].starts_with(b"obj") {
        return None;
    }
    pos += 3;

    let obj = parse_object(data, &mut pos)?;

    Some(((id_num, gen_num), obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &[u8]) -> Object {
        let mut pos = 0;
        parse_object(src, &mut pos).expect("parse failed")
    }

    #[test]
    fn integer() {
        assert_eq!(parse(b"42"), Object::Integer(42));
        assert_eq!(parse(b"-7"), Object::Integer(-7));
    }

    #[test]
    fn real() {
        assert!(matches!(parse(b"3.14"), Object::Real(_)));
    }

    #[test]
    fn boolean() {
        assert_eq!(parse(b"true"), Object::Boolean(true));
        assert_eq!(parse(b"false"), Object::Boolean(false));
    }

    #[test]
    fn null() {
        assert_eq!(parse(b"null"), Object::Null);
    }

    #[test]
    fn name() {
        assert_eq!(parse(b"/Type"), Object::Name(b"Type".to_vec()));
    }

    #[test]
    fn name_with_escape() {
        assert_eq!(parse(b"/#46oo"), Object::Name(b"Foo".to_vec()));
    }

    #[test]
    fn literal_string() {
        assert_eq!(
            parse(b"(hello)"),
            Object::String(b"hello".to_vec(), StringFormat::Literal)
        );
    }

    #[test]
    fn hex_string() {
        assert_eq!(
            parse(b"<48656c6c6f>"),
            Object::String(b"Hello".to_vec(), StringFormat::Hexadecimal)
        );
    }

    #[test]
    fn array() {
        let obj = parse(b"[1 2 3]");
        assert_eq!(
            obj,
            Object::Array(vec![
                Object::Integer(1),
                Object::Integer(2),
                Object::Integer(3)
            ])
        );
    }

    #[test]
    fn dictionary() {
        let obj = parse(b"<</Type /Page /Width 100>>");
        let dict = obj.as_dict().unwrap();
        assert_eq!(dict.get(b"Type").unwrap().as_name().unwrap(), b"Page");
        assert_eq!(dict.get(b"Width").unwrap().as_i64().unwrap(), 100);
    }

    #[test]
    fn indirect_reference() {
        assert_eq!(parse(b"5 0 R"), Object::Reference((5, 0)));
    }

    #[test]
    fn indirect_object() {
        let src = b"1 0 obj\n42\nendobj";
        let (id, obj) = parse_indirect_object(src, 0).unwrap();
        assert_eq!(id, (1, 0));
        assert_eq!(obj, Object::Integer(42));
    }

    #[test]
    fn stream_with_negative_length_does_not_panic() {
        // /Length = -1 used to overflow when cast to usize; now clamped to 0.
        let src = b"<</Length -1>>\nstream\nXY\nendstream";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert!(s.content.is_empty()),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn parse_indirect_object_rejects_oversized_id() {
        // Object ID > u32::MAX must be rejected, not silently truncated.
        let src = b"99999999999 0 obj\n42\nendobj";
        assert!(parse_indirect_object(src, 0).is_none());
    }

    #[test]
    fn reference_with_oversized_id_falls_back_to_integer() {
        // "X Y R" where X overflows u32 must NOT be recognised as a Reference.
        let src = b"99999999999 0 R";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        assert_eq!(obj, Object::Integer(99_999_999_999));
    }
}
