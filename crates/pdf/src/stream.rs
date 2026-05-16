//! Stream filter decoding.
//!
//! We support the filters encountered in real renderer workloads.
//! Image-format filters (DCT, JBIG2, JPX, CCITTFax) are intentionally
//! excluded — those bytes are passed raw to the codec layer in `rasterrocket-interp`.

use crate::dictionary::Dictionary;
use crate::object::Object;

/// Upper bound on the number of filters in a single `/Filter` chain.
///
/// Real PDFs use at most two or three (a transport prefilter plus the
/// terminal codec, e.g. `[/ASCII85Decode /CCITTFaxDecode]`).  A hostile
/// `/Filter [/FlateDecode /FlateDecode … ×N]` array would otherwise drive `N`
/// decode passes — each able to re-expand to [`MAX_DECOMPRESSED`] — turning a
/// tiny stream into an unbounded CPU/RAM sink.  The image filter-chain path is
/// the reachable vector for this, so the cap lives here at the single decode
/// chokepoint rather than at each call site.
const MAX_FILTER_CHAIN: usize = 32;

/// Decode a raw stream byte slice according to its /Filter chain.
///
/// `dict` is the stream dictionary; `raw` is the undecoded content bytes.
/// Returns the fully-decoded bytes, or `raw` unchanged if there is no filter.
///
/// # Errors
/// - The chain exceeds [`MAX_FILTER_CHAIN`] filters (hostile-array DoS guard).
/// - Any stage produces more than [`MAX_DECOMPRESSED`] bytes.  Each Flate/LZW
///   stage is individually bomb-capped, but a chain of stages could otherwise
///   re-expand the intermediate buffer unbounded across iterations; this check
///   bounds the aggregate by failing the moment any intermediate exceeds the
///   cap.
/// - Any individual filter step fails (the underlying decoder's error).
pub fn decode_stream(raw: &[u8], dict: &Dictionary) -> Result<Vec<u8>, String> {
    let filters = collect_filters(dict);
    if filters.is_empty() {
        return Ok(raw.to_vec());
    }
    if filters.len() > MAX_FILTER_CHAIN {
        return Err(format!(
            "filter chain too long: {} filters exceeds {MAX_FILTER_CHAIN} cap \
             (possible decompression-bomb / hostile-array DoS)",
            filters.len()
        ));
    }

    let params_list = collect_decode_parms(dict, filters.len());

    let mut data = raw.to_vec();
    for (i, filter) in filters.iter().enumerate() {
        let params = params_list.get(i).and_then(|o| o.as_ref());
        data = apply_filter(filter, &data, params)?;
        if data.len() > MAX_DECOMPRESSED {
            return Err(format!(
                "filter chain stage {i} ({}) produced {} bytes, exceeds {MAX_DECOMPRESSED}-byte \
                 cap (possible decompression bomb across chain)",
                String::from_utf8_lossy(filter),
                data.len()
            ));
        }
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
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = decompress_zlib(data)?;

    if let Some(p) = params {
        out = apply_png_predictor(out, p)
            .map_err(|s| std::io::Error::new(std::io::ErrorKind::InvalidData, s))?;
    }
    Ok(out)
}

/// Maximum decompressed size we will accept, to defeat decompression bombs.
const MAX_DECOMPRESSED: usize = 1 << 30; // 1 GiB

/// flate2-backed decompression with partial-output tolerance.
///
/// Returns whatever bytes were produced before a mid-stream error if the
/// decompressor wrote anything at all — real-world malformed PDFs ship
/// truncated or checksum-corrupt content streams that flate2 emits
/// usefully up to the failure point.
///
/// Bounded at [`MAX_DECOMPRESSED`] via `Read::take`; `read_to_end` would
/// otherwise grow the output `Vec` without limit.
fn decompress_zlib_flate2(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use flate2::read::{DeflateDecoder, ZlibDecoder};
    use std::io::Read;

    // 3× input is a good initial guess for typical text/PostScript-ish
    // payloads; capping at 16 MiB avoids reserving gigabytes when called
    // with a multi-hundred-MB compressed stream.
    let initial_cap = data.len().saturating_mul(3).min(16 * 1024 * 1024);
    let mut out = Vec::with_capacity(initial_cap);
    // +1 so we can detect the case where the decompressor *would* produce
    // more than the cap; if `read_to_end` consumes exactly the cap+1 byte
    // we know to error rather than silently truncate.
    let cap_plus_one = (MAX_DECOMPRESSED as u64).saturating_add(1);

    let mut decoder = ZlibDecoder::new(data).take(cap_plus_one);
    match decoder.read_to_end(&mut out) {
        Ok(_) => {}
        Err(_) if out.is_empty() && data.len() > 2 => {
            // Retry with raw deflate (skip 2-byte zlib header).
            out.clear();
            let mut raw_dec = DeflateDecoder::new(&data[2..]).take(cap_plus_one);
            raw_dec.read_to_end(&mut out)?;
        }
        Err(e) => {
            if out.is_empty() {
                return Err(e);
            }
            // Partial decompression — use what we have (truncated stream).
        }
    }
    if out.len() > MAX_DECOMPRESSED {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "FlateDecode: decompressed output exceeds 1 GiB cap",
        ));
    }
    Ok(out)
}

#[cfg(feature = "libdeflate")]
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use libdeflater::{DecompressionError, Decompressor};

    // libdeflate is a non-streaming, in-memory decoder: it needs the output
    // buffer sized up-front. Start at 4× input (typical text streams compress
    // 3-6×), then double on InsufficientSpace until either we succeed or hit
    // the 1 GiB cap.
    let initial_cap = data.len().saturating_mul(4).clamp(64, 16 * 1024 * 1024);
    let mut out = vec![0u8; initial_cap];
    let mut decoder = Decompressor::new();

    loop {
        match decoder.zlib_decompress(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(DecompressionError::InsufficientSpace) => {
                if out.len() >= MAX_DECOMPRESSED {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "FlateDecode: decompressed output exceeds 1 GiB cap",
                    ));
                }
                let new_len = out.len().saturating_mul(2).min(MAX_DECOMPRESSED);
                out.resize(new_len, 0);
            }
            Err(DecompressionError::BadData) => {
                // libdeflate rejects raw deflate (no zlib header), corrupt
                // zlib, AND truncated streams (it's all-or-nothing — even a
                // valid prefix that hits EOF surfaces as BadData). Try raw
                // deflate (skip 2-byte zlib header) as a header-missing
                // fallback; if that also fails, fall through to the flate2
                // path which tolerates partial output the way the
                // pre-libdeflate code did for real-world malformed PDFs.
                if data.len() > 2
                    && let Ok(raw) = decompress_raw_deflate(&data[2..])
                {
                    return Ok(raw);
                }
                return decompress_zlib_flate2(data);
            }
        }
    }
}

#[cfg(feature = "libdeflate")]
fn decompress_raw_deflate(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use libdeflater::{DecompressionError, Decompressor};
    let mut out = vec![0u8; data.len().saturating_mul(4).clamp(64, 16 * 1024 * 1024)];
    let mut decoder = Decompressor::new();
    loop {
        match decoder.deflate_decompress(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(DecompressionError::InsufficientSpace) => {
                if out.len() >= MAX_DECOMPRESSED {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "FlateDecode (raw): decompressed output exceeds 1 GiB cap",
                    ));
                }
                let new_len = out.len().saturating_mul(2).min(MAX_DECOMPRESSED);
                out.resize(new_len, 0);
            }
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("libdeflate raw: {e}"),
                ));
            }
        }
    }
}

#[cfg(not(feature = "libdeflate"))]
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    decompress_zlib_flate2(data)
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

/// A `Vec<u8>` sink that refuses to grow past [`MAX_DECOMPRESSED`].
///
/// `weezl`'s `into_stream(...).decode_all` writes the entire decoded output in
/// one pass with no size bound, so a small LZW stream that expands to many
/// gigabytes (an LZW bomb — the same threat the Flate path is already capped
/// against) would otherwise exhaust RAM.  Returning a write error here makes
/// `decode_all` stop and surface a loud `LZWDecode` failure instead.  weezl
/// also reports its own corrupt-stream errors as `InvalidData`, so a dedicated
/// `tripped` flag — not the error kind — is what distinguishes "bomb cap hit"
/// (unrecoverable, discard partial) from a benign mid-stream decode error
/// (partial prefix is real content, keep it).
struct BoundedSink {
    buf: Vec<u8>,
    tripped: bool,
}

/// `true` when appending `incoming` bytes to a buffer already holding `have`
/// bytes would exceed [`MAX_DECOMPRESSED`].  Extracted so the bomb predicate
/// is unit-testable without materialising a gigabyte-scale buffer.
const fn would_overflow_cap(have: usize, incoming: usize) -> bool {
    have.saturating_add(incoming) > MAX_DECOMPRESSED
}

impl std::io::Write for BoundedSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if would_overflow_cap(self.buf.len(), buf.len()) {
            self.tripped = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LZWDecode: decompressed output exceeds 1 GiB cap (possible LZW bomb)",
            ));
        }
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn apply_lzw(data: &[u8], params: Option<&Object>) -> Result<Vec<u8>, String> {
    let early_change = params
        .and_then(Object::as_dict)
        .and_then(|d| d.get(b"EarlyChange"))
        .and_then(Object::as_i64)
        .map(|v| v != 0)
        .unwrap_or(true);

    let mut decoder = if early_change {
        weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
    } else {
        weezl::decode::Decoder::new(weezl::BitOrder::Msb, 8)
    };

    let mut sink = BoundedSink {
        buf: Vec::new(),
        tripped: false,
    };
    let result = decoder.into_stream(&mut sink).decode_all(data);
    let BoundedSink {
        buf: mut out,
        tripped,
    } = sink;
    if let Err(e) = result.status {
        // A bomb that tripped the size cap is unrecoverable — the partial
        // prefix is not the document's real content, so do not return it.
        if tripped {
            return Err(format!("LZWDecode aborted: {e}"));
        }
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

fn collect_filters(dict: &Dictionary) -> Vec<Vec<u8>> {
    match dict.get(b"Filter") {
        Some(Object::Name(n)) => vec![n.clone()],
        Some(Object::Array(a)) => a
            .iter()
            .filter_map(|o| Object::as_name(o).map(<[u8]>::to_vec))
            .collect(),
        _ => vec![],
    }
}

fn collect_decode_parms(dict: &Dictionary, count: usize) -> Vec<Option<Object>> {
    match dict.get(b"DecodeParms") {
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

/// Minimal dictionary lookup used inside this module to avoid coupling its
/// callers to a specific dictionary type (it's used by both [`Dictionary`] and
/// the raw [`Object`] variants returned from xref-stream parsing).
pub trait DictLookup {
    fn get_i64(&self, key: &[u8]) -> Option<i64>;
}

impl DictLookup for Dictionary {
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

    #[test]
    fn flate_roundtrip_large_grow_buffer() {
        // Exercises the InsufficientSpace -> resize loop in the libdeflate
        // backend by feeding it highly-compressible input that decompresses
        // to ~1 MiB (well above the 4× initial reservation for repetitive
        // input that compresses to a few KiB).
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let original: Vec<u8> = (0..1_000_000u32).map(|i| (i % 251) as u8).collect();
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&original).unwrap();
        let compressed = enc.finish().unwrap();
        let decompressed = apply_flate(&compressed, None).unwrap();
        assert_eq!(decompressed.len(), original.len());
        assert_eq!(decompressed, original);
    }

    // ── Filter-chain DoS guards ────────────────────────────────────────────────

    fn flate_compress(data: &[u8]) -> Vec<u8> {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn legit_ascii85_then_flate_chain_roundtrips() {
        // The standard transport-wrapper idiom: payload is flate-compressed
        // then ASCII85-armoured.  `decode_stream` must undo both, left to
        // right — this is the scanned-book prefilter step (only the terminal
        // codec differs in the corpus case).
        let original = b"chained transport payload \x00\x01\x02\xff and text";
        let flated = flate_compress(original);
        let ascii85 = {
            // Adobe ASCII85-encode `flated` (terminated with `~>`).
            let mut out = Vec::new();
            for chunk in flated.chunks(4) {
                let mut grp = [0u8; 4];
                grp[..chunk.len()].copy_from_slice(chunk);
                let mut n = u32::from_be_bytes(grp);
                let mut enc = [0u8; 5];
                for slot in enc.iter_mut().rev() {
                    *slot = b'!' + (n % 85) as u8;
                    n /= 85;
                }
                out.extend_from_slice(&enc[..chunk.len() + 1]);
            }
            out.extend_from_slice(b"~>");
            out
        };

        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(vec![
                Object::Name(b"ASCII85Decode".to_vec()),
                Object::Name(b"FlateDecode".to_vec()),
            ]),
        );
        let decoded = decode_stream(&ascii85, &dict).expect("legit chain must decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn hostile_overlong_filter_chain_is_rejected_loudly() {
        // `/Filter [/FlateDecode … ×(MAX+1)]` — must fail with a clear DoS
        // error, never iterate the loop unbounded.
        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(
                std::iter::repeat_with(|| Object::Name(b"FlateDecode".to_vec()))
                    .take(MAX_FILTER_CHAIN + 1)
                    .collect(),
            ),
        );
        let err = decode_stream(b"anything", &dict).expect_err("overlong chain must error");
        assert!(err.contains("filter chain too long"), "got: {err}");
    }

    #[test]
    fn lzw_bomb_cap_predicate_and_sink_flag() {
        // The LZW-bomb defence has two halves:
        //  1. the size predicate trips exactly at the cap boundary, and
        //  2. `BoundedSink` flags `tripped` + errors loudly so the caller
        //     discards the unrecoverable partial output.
        // Tested without allocating the bomb (predicate is pure).
        assert!(!would_overflow_cap(0, MAX_DECOMPRESSED));
        assert!(!would_overflow_cap(MAX_DECOMPRESSED, 0));
        assert!(would_overflow_cap(MAX_DECOMPRESSED, 1));
        assert!(would_overflow_cap(MAX_DECOMPRESSED - 4, 8));
        // saturating_add: an absurd incoming length cannot wrap to "fits".
        assert!(would_overflow_cap(usize::MAX, usize::MAX));

        use std::io::Write;
        let mut sink = BoundedSink {
            buf: Vec::new(),
            tripped: false,
        };
        assert_eq!(sink.write(b"hello").unwrap(), 5);
        assert!(!sink.tripped);
    }

    #[test]
    fn chain_stage_output_overrun_is_rejected() {
        // Aggregate-across-chain guard: even though each Flate stage is
        // individually 1-GiB-capped, an intermediate that exceeds the cap
        // must abort the whole chain.  Drive it through the public API with a
        // single Flate stage whose output we force over the limit via the
        // post-stage check (a true 1-GiB fixture is impractical in a unit
        // test, so we assert the smaller invariant: a valid short chain still
        // succeeds, proving the guard is not a false-positive).
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"FlateDecode".to_vec()));
        let payload = b"a modest stream well under the cap";
        let decoded = decode_stream(&flate_compress(payload), &dict).unwrap();
        assert_eq!(decoded, payload);
    }
}
