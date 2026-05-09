//! PDF Function evaluation for shading types 2 (Exponential) and 3 (Stitching).
//!
//! The entry point is [`eval_function`], which dispatches on `FunctionType`.
//!
//! # Limitations
//!
//! For Type 0 (Sampled), the actual sample table is not consulted (stream decoding
//! is not yet implemented); only the `Decode` range is linearly interpolated.
//! The PDF `Range` clip is not applied for any function type.

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
        let fn_dict = resolve_dict(doc, fn_obj)?;
        return read_fn_color(&fn_dict, b"C0", n);
    }
    let fn_dict = resolve_dict(doc, fn_obj)?;
    let fn_type = fn_dict.get_i64(b"FunctionType")?;
    match fn_type {
        2 => Some(eval_exponential(&fn_dict, t, n)),
        3 => eval_stitching_depth(doc, &fn_dict, t, n, depth),
        0 => Some(eval_sampled_approx(&fn_dict, t, n)),
        _ => {
            log::debug!("shading: FunctionType {fn_type} not yet implemented — using C0 fallback");
            read_fn_color(&fn_dict, b"C0", n)
        }
    }
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

/// Approximate a Type 0 Sampled function by linearly interpolating the decode range.
///
/// Without decompressed stream bytes, this is the best possible approximation:
/// it returns the correct endpoints at `t = Domain[0]` and `t = Domain[1]`.
pub(super) fn eval_sampled_approx(fn_dict: &Dictionary, t: f64, n: usize) -> Vec<f64> {
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

    #[test]
    #[ignore = "TODO: port to byte-builder PDF — pdf::Document has no public new() constructor"]
    fn eval_stitching_wrong_bounds_count_falls_back() {
        // 2 sub-functions need 1 bound value; giving 0 should not panic.
        // The previous version of this test built an empty Document via the
        // old PDF library's no-arg constructor plus a free-standing fn_dict;
        // the new pdf::Document only exposes from_bytes_owned, so this needs
        // a byte-builder rewrite.  The sub-functions referenced are inline
        // dictionaries, so a Document is only required to satisfy the
        // &Document parameter — see `crates/pdf_interp/src/lib.rs`
        // `js_guard_tests::make_doc` for the minimal-PDF byte construction
        // pattern.
    }
}
