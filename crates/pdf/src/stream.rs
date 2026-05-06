//! Stream filter decoding.
//!
//! We support the filters encountered in real renderer workloads.
//! Image-format filters (DCT, JBIG2, JPX, CCITTFax) are intentionally
//! excluded — those bytes are passed raw to the codec layer in `pdf_interp`.

use std::collections::HashMap;

use crate::object::Object;

/// Decode a raw stream byte slice according to its /Filter chain.
///
/// `dict` is the stream dictionary; `raw` is the undecoded content bytes.
/// Returns the fully-decoded bytes, or `raw` unchanged if there is no filter.
pub fn decode_stream(raw: &[u8], dict: &HashMap<Vec<u8>, Object>) -> Result<Vec<u8>, String> {
    let filters = collect_filters(dict);
    if filters.is_empty() {
        return Ok(raw.to_vec());
    }

    let params_list = collect_decode_parms(dict, filters.len());

    let mut data = raw.to_vec();
    for (i, filter) in filters.iter().enumerate() {
        let params = params_list.get(i).and_then(|o| o.as_ref());
        data = apply_filter(filter, &data, params)?;
    }
    Ok(data)
}

fn apply_filter(filter: &[u8], data: &[u8], params: Option<&Object>) -> Result<Vec<u8>, String> {
    match filter {
        b"FlateDecode" => apply_flate(
            data,
            params
                .and_then(Object::as_dict)
                .map(|d| d as &dyn DictLookup),
        )
        .map_err(|e| e.to_string()),
        b"LZWDecode" => apply_lzw(data, params),
        b"ASCII85Decode" => decode_ascii85(data),
        b"ASCIIHexDecode" => decode_ascii_hex(data),
        // Image filters — pass through; the caller handles them.
        b"DCTDecode" | b"CCITTFaxDecode" | b"JBIG2Decode" | b"JPXDecode" => Ok(data.to_vec()),
        other => Err(format!(
            "unsupported filter: {}",
            String::from_utf8_lossy(other)
        )),
    }
}

// ── FlateDecode ───────────────────────────────────────────────────────────────

/// Internal helper used by both `stream.rs` and `xref.rs`.
pub(crate) fn apply_flate(
    data: &[u8],
    params: Option<&dyn DictLookup>,
) -> Result<Vec<u8>, std::io::Error> {
    use flate2::read::{DeflateDecoder, ZlibDecoder};
    use std::io::Read;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    // Cap the initial reservation: 3× input is a good guess for the typical
    // text/PostScript-ish payload, but capping at 16 MiB avoids reserving
    // gigabytes when called with a multi-hundred-MB compressed stream.
    let initial_cap = data.len().saturating_mul(3).min(16 * 1024 * 1024);
    let mut out = Vec::with_capacity(initial_cap);
    let mut decoder = ZlibDecoder::new(data);
    match decoder.read_to_end(&mut out) {
        Ok(_) => {}
        Err(_) if out.is_empty() && data.len() > 2 => {
            // Retry with raw deflate (skip 2-byte zlib header).
            out.clear();
            let mut raw_dec = DeflateDecoder::new(&data[2..]);
            raw_dec.read_to_end(&mut out)?;
        }
        Err(e) => {
            if out.is_empty() {
                return Err(e);
            }
            // Partial decompression — use what we have (truncated stream).
        }
    }

    if let Some(p) = params {
        out = apply_png_predictor(out, p)
            .map_err(|s| std::io::Error::new(std::io::ErrorKind::InvalidData, s))?;
    }
    Ok(out)
}

// ── PNG predictor ─────────────────────────────────────────────────────────────

fn apply_png_predictor(data: Vec<u8>, params: &dyn DictLookup) -> Result<Vec<u8>, String> {
    let predictor = params.get_i64(b"Predictor").unwrap_or(1);
    if predictor < 10 {
        // Predictor 1 = no prediction; 2 = TIFF (we don't support TIFF predictor here).
        return Ok(data);
    }

    // PNG predictors (10–15).
    // Sanity-cap each parameter to defeat malformed PDFs that would otherwise
    // request gigabyte-scale allocations.
    let colors = (params.get_i64(b"Colors").unwrap_or(1).max(1) as usize).min(32);
    let bits = (params.get_i64(b"BitsPerComponent").unwrap_or(8).max(1) as usize).min(32);
    let cols = (params.get_i64(b"Columns").unwrap_or(1).max(1) as usize).min(1_000_000);

    let stride = cols
        .checked_mul(colors)
        .and_then(|x| x.checked_mul(bits))
        .ok_or_else(|| "PNG predictor: stride overflow".to_string())?
        .div_ceil(8);
    let row_len = stride + 1; // +1 for filter byte

    // Tolerate truncated data — process as many full rows as we have.
    let n_rows = data.len() / row_len;
    let total = n_rows
        .checked_mul(stride)
        .ok_or_else(|| "PNG predictor: output size overflow".to_string())?;
    // 256 MiB ceiling — well above any realistic predictor output.
    if total > 256 * 1024 * 1024 {
        return Err(format!(
            "PNG predictor output {total} bytes exceeds 256 MiB cap"
        ));
    }
    let mut out = vec![0u8; total];
    let bytes_per_pixel = (colors * bits).div_ceil(8);

    for row in 0..n_rows {
        let src_row = &data[row * row_len..(row + 1) * row_len];
        let filter_byte = src_row[0];
        let raw = &src_row[1..];

        // Split `out` into (rows written so far, current row) to satisfy the
        // borrow checker: `prev` is in the already-written prefix, `dst` is
        // the current row being filled.
        let (done, rest) = out.split_at_mut(row * stride);
        let dst = &mut rest[..stride];
        let prev: &[u8] = if row == 0 {
            &[]
        } else {
            &done[(row - 1) * stride..row * stride]
        };

        for i in 0..stride {
            let a = if i >= bytes_per_pixel {
                dst[i - bytes_per_pixel]
            } else {
                0
            };
            let b = prev.get(i).copied().unwrap_or(0);
            let c = if i >= bytes_per_pixel {
                prev.get(i - bytes_per_pixel).copied().unwrap_or(0)
            } else {
                0
            };
            dst[i] = match filter_byte {
                0 => raw[i],
                1 => raw[i].wrapping_add(a),
                2 => raw[i].wrapping_add(b),
                3 => raw[i].wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8),
                4 => raw[i].wrapping_add(paeth(a, b, c)),
                _ => raw[i],
            };
        }
    }
    Ok(out)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (i16::from(a), i16::from(b), i16::from(c));
    let p = a + b - c;
    let pa = (p - a).unsigned_abs();
    let pb = (p - b).unsigned_abs();
    let pc = (p - c).unsigned_abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

// ── LZW ──────────────────────────────────────────────────────────────────────

fn apply_lzw(data: &[u8], params: Option<&Object>) -> Result<Vec<u8>, String> {
    let early_change = params
        .and_then(Object::as_dict)
        .and_then(|d| d.get(b"EarlyChange".as_ref()))
        .and_then(Object::as_i64)
        .map(|v| v != 0)
        .unwrap_or(true);

    let mut decoder = if early_change {
        weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
    } else {
        weezl::decode::Decoder::new(weezl::BitOrder::Msb, 8)
    };

    let mut out = Vec::new();
    let result = decoder.into_stream(&mut out).decode_all(data);
    if let Err(e) = result.status {
        if out.is_empty() {
            return Err(format!("LZWDecode failed with no output: {e}"));
        }
        log::warn!("LZWDecode partial error after {} bytes: {e}", out.len());
    }

    let pred_params = params.map(|o| o as &dyn DictLookup);
    if let Some(p) = pred_params {
        out = apply_png_predictor(out, p)?;
    }
    Ok(out)
}

// ── ASCII85 ───────────────────────────────────────────────────────────────────

fn decode_ascii85(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut count = 0usize;

    for &b in data {
        match b {
            b'z' if count == 0 => out.extend_from_slice(&[0u8; 4]),
            b'~' => break,
            b' ' | b'\t' | b'\r' | b'\n' => {}
            b'!'..=b'u' => {
                group[count] = b - b'!';
                count += 1;
                if count == 5 {
                    let n = group.iter().fold(0u32, |acc, &x| acc * 85 + u32::from(x));
                    out.extend_from_slice(&n.to_be_bytes());
                    count = 0;
                }
            }
            other => return Err(format!("ASCII85: invalid byte {other:#04x}")),
        }
    }
    if count > 0 {
        // ASCII85 final groups must be 2..=4 bytes (1 leftover char is invalid).
        if count == 1 {
            return Err("ASCII85: stray single character in final group".into());
        }
        // Pad remaining group with 'u' (84).
        for slot in group.iter_mut().skip(count) {
            *slot = 84;
        }
        let n = group.iter().fold(0u32, |acc, &x| acc * 85 + u32::from(x));
        out.extend_from_slice(&n.to_be_bytes()[..count - 1]);
    }
    Ok(out)
}

// ── ASCIIHex ──────────────────────────────────────────────────────────────────

fn decode_ascii_hex(data: &[u8]) -> Result<Vec<u8>, String> {
    use crate::lexer::hex_nibble;
    let mut out = Vec::new();
    let mut hi: Option<u8> = None;
    for &b in data {
        if b == b'>' {
            break;
        }
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
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
    Ok(out)
}

// ── Filter/DecodeParms helpers ────────────────────────────────────────────────

fn collect_filters(dict: &HashMap<Vec<u8>, Object>) -> Vec<Vec<u8>> {
    match dict.get(b"Filter".as_ref()) {
        Some(Object::Name(n)) => vec![n.clone()],
        Some(Object::Array(a)) => a
            .iter()
            .filter_map(|o| Object::as_name(o).map(<[u8]>::to_vec))
            .collect(),
        _ => vec![],
    }
}

fn collect_decode_parms(dict: &HashMap<Vec<u8>, Object>, count: usize) -> Vec<Option<Object>> {
    match dict.get(b"DecodeParms".as_ref()) {
        Some(Object::Array(a)) => a.iter().cloned().map(Some).collect(),
        Some(o) => {
            let mut v = vec![Some(o.clone())];
            v.resize(count, None);
            v
        }
        None => vec![None; count],
    }
}

// ── DictLookup trait (avoid coupling stream.rs to the full Document) ──────────

/// Minimal dictionary lookup used inside this module to avoid importing
/// `HashMap` specialisation at every call site.
pub trait DictLookup {
    fn get_i64(&self, key: &[u8]) -> Option<i64>;
}

impl DictLookup for HashMap<Vec<u8>, Object> {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        self.get(key)?.as_i64()
    }
}

// Blanket impl so we can pass `Option<&HashMap<…>>` transparently.
impl DictLookup for &dyn DictLookup {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        (*self).get_i64(key)
    }
}

// Allow apply_flate to accept Option<&dyn DictLookup> where params is a raw
// Object (e.g. from xref stream).
impl DictLookup for Object {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        self.as_dict()?.get(key)?.as_i64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii85_decode_basic() {
        // "Man " encodes to "9jqo^" in ASCII85.
        let encoded = b"9jqo^~>";
        let decoded = decode_ascii85(encoded).unwrap();
        assert_eq!(decoded, b"Man ");
    }

    #[test]
    fn ascii_hex_decode() {
        let decoded = decode_ascii_hex(b"48656c6c6f>").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn flate_roundtrip() {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let original = b"hello world, this is a test of flate encoding";
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(original).unwrap();
        let compressed = enc.finish().unwrap();
        let decompressed = apply_flate(&compressed, None).unwrap();
        assert_eq!(decompressed, original);
    }
}
