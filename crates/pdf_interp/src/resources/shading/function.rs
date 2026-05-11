//! PDF Function evaluation for shading types 0 (Sampled), 2 (Exponential),
//! and 3 (Stitching).
//!
//! The entry point is [`eval_function`], which dispatches on `FunctionType`.
//!
//! # Limitations
//!
//! Type 0 (Sampled) supports 1-D inputs with `BitsPerSample` 8 or 16 and the
//! default `Order` 1 (linear interpolation between adjacent samples).
//! Multi-dimensional Type 0 functions (`Size = [N0, N1, ...]`), non-1/8/16
//! `BitsPerSample`, and `Order` 3 (cubic spline) fall back to the
//! sample-free linear approximation that pre-dated stream decoding.

use pdf::{Dictionary, Document, Object};

use crate::resources::dict_ext::DictExt;
use crate::resources::{obj_to_f64, resolve_dict};

/// Maximum nesting depth for recursive function evaluation (guards against
/// crafted PDFs with deeply or circularly nested Type 3 functions).
const MAX_FN_DEPTH: u8 = 10;

// ── Public-to-parent API ──────────────────────────────────────────────────────

/// Evaluate a PDF Function object at parameter `t`, returning `n` colour channels.
///
/// Supported function types:
/// - **Type 2** (Exponential): `C0 + t^N × (C1 − C0)`.
/// - **Type 3** (Stitching): maps `t` to a sub-function and recursively evaluates.
/// - **Type 0** (Sampled): linear interpolation across the decode range (stream data not used).
///
/// Unknown types fall back to `C0` if available.
pub(super) fn eval_function(doc: &Document, fn_obj: &Object, t: f64, n: usize) -> Option<Vec<f64>> {
    eval_function_depth(doc, fn_obj, t, n, 0)
}

fn eval_function_depth(
    doc: &Document,
    fn_obj: &Object,
    t: f64,
    n: usize,
    depth: u8,
) -> Option<Vec<f64>> {
    if depth >= MAX_FN_DEPTH {
        log::warn!(
            "shading: PDF function nesting depth {depth} exceeds limit {MAX_FN_DEPTH} — \
             returning C0 fallback to prevent stack overflow"
        );
        let fn_dict = resolve_fn_dict(doc, fn_obj)?;
        return read_fn_color(&fn_dict, b"C0", n).map(|c| apply_range_clip(&fn_dict, c));
    }
    let fn_dict = resolve_fn_dict(doc, fn_obj)?;
    let fn_type = fn_dict.get_i64(b"FunctionType")?;
    let result = match fn_type {
        2 => Some(eval_exponential(&fn_dict, t, n)),
        3 => eval_stitching_depth(doc, &fn_dict, t, n, depth),
        0 => Some(
            eval_sampled(doc, fn_obj, &fn_dict, t, n)
                .unwrap_or_else(|| eval_sampled_approx(&fn_dict, t, n)),
        ),
        _ => {
            log::debug!("shading: FunctionType {fn_type} not yet implemented — using C0 fallback");
            read_fn_color(&fn_dict, b"C0", n)
        }
    };
    result.map(|c| apply_range_clip(&fn_dict, c))
}

/// Clip per-channel output values to the function's `Range` array
/// (PDF §7.10.2 "Range" entry). Missing or short `Range` arrays leave the
/// channel unconstrained. Applied at every `eval_function_depth` exit so
/// downstream colour conversion gets values already inside the declared
/// output range.
fn apply_range_clip(fn_dict: &Dictionary, channels: Vec<f64>) -> Vec<f64> {
    let Some(range) = fn_dict.get(b"Range").and_then(Object::as_array) else {
        return channels;
    };
    let pairs: Vec<f64> = range.iter().filter_map(obj_to_f64).collect();
    channels
        .into_iter()
        .enumerate()
        .map(|(ch, v)| {
            if pairs.len() < (ch + 1) * 2 {
                return v;
            }
            let (lo, hi) = (pairs[ch * 2], pairs[ch * 2 + 1]);
            if hi <= lo { v } else { v.clamp(lo, hi) }
        })
        .collect()
}

/// Evaluate a Type 2 Exponential function: `C0 + (t_norm)^N × (C1 − C0)`.
pub(super) fn eval_exponential(fn_dict: &Dictionary, t: f64, n: usize) -> Vec<f64> {
    let c0 = read_fn_color(fn_dict, b"C0", n).unwrap_or_else(|| vec![0.0; n]);
    let c1 = read_fn_color(fn_dict, b"C1", n).unwrap_or_else(|| vec![1.0; n]);

    let (d0, d1) = read_fn_domain(fn_dict);

    // PDF spec says N ≥ 0; clamp exponent to avoid +infinity from 0^(negative).
    let exponent = fn_dict
        .get(b"N")
        .and_then(obj_to_f64)
        .unwrap_or(1.0)
        .max(0.0);

    // safe_clamp: f64::clamp panics when min > max or either is NaN.
    let t_norm = if d1 > d0 {
        ((t - d0) / (d1 - d0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let weight = t_norm.powf(exponent);

    c0.iter()
        .zip(c1.iter())
        .map(|(&a, &b)| a + weight * (b - a))
        .collect()
}

fn eval_stitching_depth(
    doc: &Document,
    fn_dict: &Dictionary,
    t: f64,
    n: usize,
    depth: u8,
) -> Option<Vec<f64>> {
    let (d0, d1) = read_fn_domain(fn_dict);
    // safe_clamp: f64::clamp panics when min > max; guard before calling.
    let t_clamped = if d1 > d0 { t.clamp(d0, d1) } else { d0 };

    let fns = fn_dict.get(b"Functions")?.as_array()?;
    let num_fns = fns.len();
    if num_fns == 0 {
        return None;
    }

    // Bounds: k-1 values for k sub-functions splitting [Domain0, Domain1].
    let bounds: Vec<f64> = fn_dict
        .get(b"Bounds")
        .and_then(Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();

    // PDF spec: Bounds must have exactly num_fns−1 values.  A malformed dict
    // with the wrong count would cause out-of-bounds access in `breaks[idx+1]`.
    if bounds.len() != num_fns - 1 {
        log::warn!(
            "shading: Type 3 function has {} sub-functions but {} Bounds values (expected {}); \
             using first sub-function as fallback",
            num_fns,
            bounds.len(),
            num_fns - 1,
        );
        return eval_function_depth(doc, &fns[0], t_clamped, n, depth + 1);
    }

    // Build breakpoint list: [d0, bounds..., d1].
    let mut breaks = Vec::with_capacity(num_fns + 1);
    breaks.push(d0);
    breaks.extend_from_slice(&bounds);
    breaks.push(d1);

    // Find which sub-function `t_clamped` falls into.
    // Clamp idx to num_fns−1 to guard against rounding/NaN edge cases.
    let idx = if t_clamped >= d1 {
        num_fns - 1
    } else {
        breaks
            .windows(2)
            .position(|w| t_clamped < w[1])
            .unwrap_or(num_fns - 1)
            .min(num_fns - 1)
    };

    // Encode: maps [breaks[i], breaks[i+1]] → [Encode[2i], Encode[2i+1]].
    let encode: Vec<f64> = fn_dict
        .get(b"Encode")
        .and_then(Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();

    let (in_lo, in_hi) = (breaks[idx], breaks[idx + 1]);
    let (out_lo, out_hi) = if encode.len() >= (idx + 1) * 2 {
        (encode[idx * 2], encode[idx * 2 + 1])
    } else {
        (0.0, 1.0)
    };

    let t_encoded = if (in_hi - in_lo).abs() < f64::EPSILON {
        out_lo
    } else {
        ((t_clamped - in_lo) / (in_hi - in_lo)).mul_add(out_hi - out_lo, out_lo)
    };

    eval_function_depth(doc, &fns[idx], t_encoded, n, depth + 1)
}

/// Evaluate a Type 0 (Sampled) function against its actual sample stream.
///
/// Supported scope:
/// - 1-D input only (`Size = [N]`); higher dimensions return `None` and the
///   caller falls back to [`eval_sampled_approx`].
/// - `BitsPerSample` 8 or 16 (MSB-first big-endian).  Other values
///   (1/2/4/12/24/32) return `None`.
/// - `Order` defaults to 1 (linear interpolation between adjacent samples);
///   `Order = 3` (cubic spline) is not implemented and falls back.
///
/// PDF §7.10.2 algorithm: normalise `t` against `Domain` → `Encode`, look
/// up two adjacent samples by integer index, linearly interpolate, then
/// rescale via `Decode`.
fn eval_sampled(
    doc: &Document,
    fn_obj: &Object,
    fn_dict: &Dictionary,
    t: f64,
    n: usize,
) -> Option<Vec<f64>> {
    // Multi-dimensional Type 0 (Size = [N0, N1, ...]) falls back: PDF spec
    // requires N-linear interpolation over an N-dimensional grid, which is
    // out of scope here.
    let size = fn_dict.get(b"Size").and_then(Object::as_array)?;
    if size.len() != 1 {
        return None;
    }
    let n_samples = size.first().and_then(Object::as_u32)? as usize;
    if n_samples < 2 {
        return None;
    }

    // Order = 1 (linear) is the default and the only supported value.
    // PDF §7.10.2 specifies Order as an integer (1 or 3), so values from a
    // spec-conforming PDF are exactly representable in f64 — bit-exact
    // comparison against 1.0 is appropriate.  Any other value (including
    // 3, or a malformed Real that isn't 1.0) means fall back.
    let order = fn_dict.get(b"Order").and_then(obj_to_f64).unwrap_or(1.0);
    #[expect(
        clippy::float_cmp,
        reason = "PDF §7.10.2 Order is an integer (1 or 3); bit-exact == 1.0 is the right test"
    )]
    if order != 1.0 {
        return None;
    }

    let bps = fn_dict.get(b"BitsPerSample").and_then(Object::as_u32)?;
    if bps != 8 && bps != 16 {
        return None;
    }

    // The function object must carry a stream — Type 0 sample data is the
    // stream content.  Inline dictionaries can't represent Type 0.
    let stream_bytes = stream_bytes_for(doc, fn_obj)?;
    let bytes_per_channel = (bps as usize) / 8;
    let bytes_per_sample = bytes_per_channel * n;
    let expected_len = bytes_per_sample.checked_mul(n_samples)?;
    if stream_bytes.len() < expected_len {
        log::warn!(
            "shading: Type 0 sample stream truncated ({} bytes, expected {expected_len})",
            stream_bytes.len(),
        );
        return None;
    }

    let (d0, d1) = read_fn_domain(fn_dict);
    let t_norm = if d1 > d0 {
        ((t - d0) / (d1 - d0)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // n_samples ≥ 2 by the guard above; cast is finite and well within
    // f64 mantissa (n_samples capped at u32::MAX by as_u32).
    #[expect(
        clippy::cast_precision_loss,
        reason = "n_samples capped at u32; well within f64 mantissa"
    )]
    let n_samples_minus_1_f64 = (n_samples - 1) as f64;

    // Encode maps Domain → sample index space; default [0, n_samples − 1].
    let encode: Vec<f64> = fn_dict
        .get(b"Encode")
        .and_then(Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();
    let (e_lo, e_hi) = if encode.len() >= 2 {
        (encode[0], encode[1])
    } else {
        (0.0, n_samples_minus_1_f64)
    };
    let idx_f = (e_lo + t_norm * (e_hi - e_lo)).clamp(0.0, n_samples_minus_1_f64);
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "idx_f clamped to [0, n_samples − 1] above; n_samples is usize"
    )]
    let idx_lo = idx_f.floor() as usize;
    let idx_hi = (idx_lo + 1).min(n_samples - 1);
    let frac = idx_f - idx_f.floor();

    // Sample max per BPS for normalisation.
    let s_max = if bps == 8 { 255.0 } else { 65535.0 };

    // Decode maps normalised sample → output channel space.  Default = [0, 1] per channel.
    let decode: Vec<f64> = fn_dict
        .get(b"Decode")
        .and_then(Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();

    let read_sample = |sample_idx: usize, ch: usize| -> f64 {
        // bps is 8 or 16 by the guard above; bytes_per_channel is bps/8.
        let off = sample_idx * bytes_per_sample + ch * bytes_per_channel;
        if bps == 8 {
            f64::from(stream_bytes[off])
        } else {
            let hi = u16::from(stream_bytes[off]);
            let lo = u16::from(stream_bytes[off + 1]);
            f64::from((hi << 8) | lo)
        }
    };

    let channels = (0..n)
        .map(|ch| {
            let raw_lo = read_sample(idx_lo, ch) / s_max;
            let raw_hi = read_sample(idx_hi, ch) / s_max;
            let raw = raw_lo + frac * (raw_hi - raw_lo);
            let (d_lo, d_hi) = if decode.len() >= (ch + 1) * 2 {
                (decode[ch * 2], decode[ch * 2 + 1])
            } else {
                (0.0, 1.0)
            };
            d_lo + raw * (d_hi - d_lo)
        })
        .collect();
    Some(channels)
}

/// Resolve `fn_obj` to its function dictionary.
///
/// Type 0 functions are *stream* objects (the dict carries `FunctionType` /
/// `Size` / `BitsPerSample`; the stream content carries the sample bytes),
/// so the resolver must handle `Object::Stream(s)` directly in addition to
/// the inline-Dict and Reference-to-Dict cases that
/// [`crate::resources::resolve_dict`] already covers.
fn resolve_fn_dict(doc: &Document, fn_obj: &Object) -> Option<Dictionary> {
    match fn_obj {
        Object::Stream(s) => Some(s.dict.clone()),
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            match referent.as_ref() {
                Object::Stream(s) => Some(s.dict.clone()),
                other => resolve_dict(doc, other),
            }
        }
        _ => resolve_dict(doc, fn_obj),
    }
}

/// Return the decompressed stream bytes for `fn_obj` if it is a Stream or
/// Reference-to-Stream; `None` for inline dicts or non-stream objects.
fn stream_bytes_for(doc: &Document, fn_obj: &Object) -> Option<Vec<u8>> {
    let resolved: std::sync::Arc<Object>;
    let stream = match fn_obj {
        Object::Stream(s) => s,
        Object::Reference(id) => {
            resolved = doc.get_object(*id).ok()?;
            resolved.as_ref().as_stream()?
        }
        _ => return None,
    };
    stream
        .decompressed_content()
        .map_err(|e| log::debug!("shading: Type 0 stream decode failed: {e}"))
        .ok()
}

/// Approximate a Type 0 Sampled function by linearly interpolating the decode range.
///
/// Without decompressed stream bytes, this is the best possible approximation:
/// it returns the correct endpoints at `t = Domain[0]` and `t = Domain[1]`.
fn eval_sampled_approx(fn_dict: &Dictionary, t: f64, n: usize) -> Vec<f64> {
    let (d0, d1) = read_fn_domain(fn_dict);
    // Guard: d1 > d0 before dividing; avoids divide-by-zero and clamp panic.
    let t_norm = if d1 > d0 {
        ((t - d0) / (d1 - d0)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Decode range defaults to [0, 1] per channel.
    let decode: Vec<f64> = fn_dict
        .get(b"Decode")
        .and_then(Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();

    (0..n)
        .map(|ch| {
            let (d_lo, d_hi) = if decode.len() >= (ch + 1) * 2 {
                (decode[ch * 2], decode[ch * 2 + 1])
            } else {
                (0.0, 1.0)
            };
            d_lo + t_norm * (d_hi - d_lo)
        })
        .collect()
}

/// Read the `Domain` key (1D) from a function dict; defaults to `[0, 1]`.
pub(super) fn read_fn_domain(dict: &Dictionary) -> (f64, f64) {
    let arr = dict
        .get(b"Domain")
        .and_then(Object::as_array)
        .map(|a| a.iter().filter_map(obj_to_f64).collect::<Vec<_>>());
    match arr.as_deref() {
        Some([d0, d1, ..]) => (*d0, *d1),
        _ => (0.0, 1.0),
    }
}

/// Read a colour array (`C0` or `C1`) from a function dict, padding to `n` channels.
pub(super) fn read_fn_color(dict: &Dictionary, key: &[u8], n: usize) -> Option<Vec<f64>> {
    let arr = dict.get(key)?.as_array()?;
    let mut vals: Vec<f64> = arr.iter().filter_map(obj_to_f64).collect();
    if vals.is_empty() {
        return None;
    }
    // vals is non-empty: the `if vals.is_empty()` guard above returned early.
    vals.resize(n, *vals.last().expect("vals non-empty: checked above"));
    Some(vals)
}

/// Read a fixed-length array of `f64` values from a dictionary key.
pub(super) fn read_f64_array(dict: &Dictionary, key: &[u8], expected: usize) -> Option<Vec<f64>> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < expected {
        return None;
    }
    let vals: Vec<f64> = arr.iter().take(expected).filter_map(obj_to_f64).collect();
    if vals.len() < expected {
        return None;
    }
    Some(vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp_dict(c0: f32, c1: f32, exponent: f32) -> pdf::Dictionary {
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(2));
        dict.set("C0", pdf::Object::Array(vec![pdf::Object::Real(c0)]));
        dict.set("C1", pdf::Object::Array(vec![pdf::Object::Real(c1)]));
        dict.set("N", pdf::Object::Real(exponent));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict
    }

    #[test]
    fn exponential_linear_midpoint() {
        let dict = make_exp_dict(0.0, 1.0, 1.0);
        let result = eval_exponential(&dict, 0.5, 1);
        assert!(
            (result[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn exponential_at_t0_gives_c0() {
        let dict = make_exp_dict(0.2, 1.0, 1.0);
        let result = eval_exponential(&dict, 0.0, 1);
        assert!((result[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn exponential_at_t1_gives_c1() {
        let dict = make_exp_dict(0.0, 0.8, 1.0);
        let result = eval_exponential(&dict, 1.0, 1);
        assert!((result[0] - 0.8).abs() < 1e-5);
    }

    /// Build a minimal parseable Document for the `&Document` parameter.
    /// The Type 3 stitching path under test references inline sub-function
    /// dictionaries (not indirect references), so `doc` is never consulted —
    /// `from_bytes_owned(Vec::new())` would fail parse, so we ship a tiny
    /// well-formed PDF instead.
    fn empty_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n";
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let xref_start = off2 + obj2.len();
        let xref = format!(
            "xref\n0 3\n0000000000 65535 f\r\n{off1:010} 00000 n\r\n{off2:010} 00000 n\r\n",
        );
        let trailer = format!("trailer\n<</Size 3 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("test PDF parse")
    }

    #[test]
    fn eval_stitching_wrong_bounds_count_falls_back() {
        // PDF §7.10.3: a Type 3 stitching function with k sub-functions must
        // carry k−1 Bounds values.  Two sub-functions therefore need exactly
        // one bound; supplying zero is malformed.  eval_stitching_depth must
        // fall back to evaluating the first sub-function rather than panicking
        // on the missing breakpoint.
        let sub_a = make_exp_dict(0.2, 0.8, 1.0); // linear ramp 0.2 → 0.8
        let sub_b = make_exp_dict(0.0, 1.0, 1.0); // distinct sub so fallback is observable
        let mut stitching = pdf::Dictionary::new();
        stitching.set("FunctionType", pdf::Object::Integer(3));
        stitching.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        stitching.set(
            "Functions",
            pdf::Object::Array(vec![
                pdf::Object::Dictionary(sub_a),
                pdf::Object::Dictionary(sub_b),
            ]),
        );
        // Bounds intentionally empty — should have 1 entry for 2 sub-functions.
        stitching.set("Bounds", pdf::Object::Array(vec![]));
        stitching.set(
            "Encode",
            pdf::Object::Array(vec![
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
            ]),
        );

        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(stitching);

        // Must not panic; must use sub_a (the first sub-function) as fallback.
        let result =
            eval_function(&doc, &fn_obj, 0.5, 1).expect("fallback should produce a result");
        // sub_a at t=0.5 is 0.2 + 0.5 × (0.8 − 0.2) = 0.5.
        assert!(
            (result[0] - 0.5).abs() < 1e-5,
            "expected fallback to evaluate sub_a at t=0.5 → 0.5, got {}",
            result[0]
        );
    }

    // ── Type 0 (Sampled) ──────────────────────────────────────────────────────

    /// Build a Type 0 sampled function as a Stream object with the given
    /// 8-bit sample bytes.  Domain = [0, 1], Range = [0, 1], 1 output channel
    /// unless `n` is given.
    fn make_sampled_8bit_stream(samples: &[u8]) -> pdf::Object {
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(0));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict.set(
            "Size",
            pdf::Object::Array(vec![pdf::Object::Integer(samples.len() as i64)]),
        );
        dict.set("BitsPerSample", pdf::Object::Integer(8));
        // Length is required for stream parsing in real PDFs; harmless to
        // include here since we hand the Stream directly to eval_function.
        dict.set("Length", pdf::Object::Integer(samples.len() as i64));
        pdf::Object::Stream(pdf::Stream::new(dict, samples.to_vec()))
    }

    #[test]
    fn sampled_reads_sample_stream() {
        // Non-linear sample table: [0, 0, 0, 0, 255] — the stream-reading path
        // must return ~0 for t < 0.75 and ~255 for t near 1.0.  Pure linear
        // approximation (eval_sampled_approx) would give t × 1.0 everywhere.
        let stream = make_sampled_8bit_stream(&[0, 0, 0, 0, 255]);
        let doc = empty_doc();

        let lo = eval_function(&doc, &stream, 0.0, 1).expect("eval at t=0");
        assert!(
            lo[0].abs() < 1e-5,
            "t=0 should sample idx 0 → 0/255 = 0.0, got {}",
            lo[0]
        );

        let mid = eval_function(&doc, &stream, 0.5, 1).expect("eval at t=0.5");
        assert!(
            mid[0].abs() < 1e-5,
            "t=0.5 should sample idx 2 → 0/255 = 0.0; linear-approximation \
             fallback would return 0.5.  got {}",
            mid[0]
        );

        let hi = eval_function(&doc, &stream, 1.0, 1).expect("eval at t=1");
        assert!(
            (hi[0] - 1.0).abs() < 1e-5,
            "t=1 should sample idx 4 → 255/255 = 1.0, got {}",
            hi[0]
        );
    }

    #[test]
    fn sampled_interpolates_between_samples() {
        // Samples [0, 100, 200] at indices [0, 1, 2].  t=0.25 maps to idx 0.5
        // (encode default = [0, n−1] = [0, 2], so t × 2 = 0.5).  Linear interp
        // between idx 0 (=0) and idx 1 (=100) at frac=0.5 → 50/255.
        let stream = make_sampled_8bit_stream(&[0, 100, 200]);
        let doc = empty_doc();

        let result = eval_function(&doc, &stream, 0.25, 1).expect("eval at t=0.25");
        let expected = 50.0 / 255.0;
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "expected linear interpolation 50/255 ≈ {expected}, got {}",
            result[0]
        );
    }

    #[test]
    fn sampled_multidim_falls_back_to_approx() {
        // Multi-dimensional (Size = [2, 2]) — not yet implemented.  Must not
        // panic; falls back to eval_sampled_approx which returns the linear
        // ramp over the Decode range.
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(0));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
            ]),
        );
        dict.set(
            "Size",
            pdf::Object::Array(vec![pdf::Object::Integer(2), pdf::Object::Integer(2)]),
        );
        dict.set("BitsPerSample", pdf::Object::Integer(8));
        let stream = pdf::Object::Stream(pdf::Stream::new(dict, vec![0, 64, 128, 255]));
        let doc = empty_doc();

        // Just verify no panic and a result is produced.
        let result = eval_function(&doc, &stream, 0.5, 1);
        assert!(
            result.is_some(),
            "multi-dim Type 0 should fall back, not return None"
        );
    }

    // ── Range clip ────────────────────────────────────────────────────────────

    #[test]
    fn range_clip_clamps_above_max() {
        // Exponential 0 → 2 evaluated at t=1 would normally return 2.0; the
        // Range entry [0, 1] should clip it to 1.0.
        let mut dict = make_exp_dict(0.0, 2.0, 1.0);
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(dict);
        let result = eval_function(&doc, &fn_obj, 1.0, 1).expect("eval at t=1");
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "Range clip should cap at 1.0; got {}",
            result[0]
        );
    }

    #[test]
    fn range_clip_clamps_below_min() {
        // C0 = -0.5 would normally pass through at t=0; Range [0, 1] should
        // clip it to 0.0.
        let mut dict = make_exp_dict(-0.5, 1.0, 1.0);
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(dict);
        let result = eval_function(&doc, &fn_obj, 0.0, 1).expect("eval at t=0");
        assert!(
            result[0].abs() < 1e-5,
            "Range clip should floor at 0.0; got {}",
            result[0]
        );
    }
}
