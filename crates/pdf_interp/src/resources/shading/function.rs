//! PDF Function evaluation for shading types 0 (Sampled), 2 (Exponential),
//! 3 (Stitching), and 4 (PostScript calculator).
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
//!
//! Type 4 (PostScript calculator) implements the bounded operator subset of
//! PDF §7.10.5 — arithmetic, boolean/relational, stack, and `if`/`ifelse`
//! conditionals only. There are no loops, defs, strings, or arrays beyond
//! the operand stack; a malformed program or stack underflow returns `None`
//! so the caller falls back loudly.

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
/// - **Type 4** (PostScript calculator): evaluates the bounded §7.10.5 operator subset.
///
/// Unknown types fall back to `C0` if available.
pub fn eval_function(doc: &Document, fn_obj: &Object, t: f64, n: usize) -> Option<Vec<f64>> {
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
        4 => eval_postscript(doc, fn_obj, &fn_dict, t, n).or_else(|| {
            log::warn!(
                "shading: Type 4 PostScript function failed to evaluate \
                 (malformed program or stack underflow) — colour unresolved"
            );
            None
        }),
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

// ── Type 4 (PostScript calculator) ────────────────────────────────────────────

/// Cap on total operators executed by a single Type-4 invocation.  PDF
/// §7.10.5 forbids loops, so a well-formed program executes a fixed bounded
/// number of operators; this guards against pathological hand-crafted input
/// that nests `if`/`ifelse` to blow the budget.
const MAX_PS_STEPS: u32 = 100_000;

/// Cap on `{ }` procedure nesting.  Both the parser and the executor recurse
/// per nested block, so unbounded depth can overflow the native stack on a
/// hand-crafted `{{{{…}}}}` program (the step budget does not bound parse
/// recursion, which runs before any step is charged, nor the executor's
/// native frame depth on conditional branches).  Real tint transforms nest a
/// handful of levels at most; 64 is far beyond any legitimate program.
const MAX_PS_DEPTH: usize = 64;

/// Cap on the operand stack.  A single `copy`/`dup`/`index` can request a
/// huge push while charging only one step against [`MAX_PS_STEPS`], so the
/// step budget alone does not bound memory; `{ 0 0 2000000000 copy }` would
/// otherwise allocate billions of slots.  §7.10.5 calculator programs operate
/// on a tiny working set (tint in, a few colorants out); 100k is generous.
const MAX_PS_STACK: usize = 100_000;

/// A value on the Type-4 operand stack: a real number or a boolean.  PDF
/// §7.10.5 booleans arise from relational/boolean operators and are consumed
/// only by `if`/`ifelse`.
#[derive(Clone, Copy, Debug)]
enum PsVal {
    Num(f64),
    Bool(bool),
}

/// A parsed Type-4 token: a literal, an operator, or a `{ ... }` procedure
/// block (only ever an operand to `if`/`ifelse`).
#[derive(Clone, Debug)]
enum PsTok {
    Num(f64),
    Op(PsOp),
    Proc(Vec<Self>),
}

#[derive(Clone, Copy, Debug)]
enum PsOp {
    Abs,
    Add,
    Atan,
    Ceiling,
    Cos,
    Cvi,
    Cvr,
    Div,
    Exp,
    Floor,
    Idiv,
    Ln,
    Log,
    Mod,
    Mul,
    Neg,
    Round,
    Sin,
    Sqrt,
    Sub,
    Truncate,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    And,
    Or,
    Xor,
    Not,
    Bitshift,
    True,
    False,
    Copy,
    Dup,
    Exch,
    Index,
    Pop,
    Roll,
    If,
    Ifelse,
}

fn parse_ps_op(word: &str) -> Option<PsOp> {
    Some(match word {
        "abs" => PsOp::Abs,
        "add" => PsOp::Add,
        "atan" => PsOp::Atan,
        "ceiling" => PsOp::Ceiling,
        "cos" => PsOp::Cos,
        "cvi" => PsOp::Cvi,
        "cvr" => PsOp::Cvr,
        "div" => PsOp::Div,
        "exp" => PsOp::Exp,
        "floor" => PsOp::Floor,
        "idiv" => PsOp::Idiv,
        "ln" => PsOp::Ln,
        "log" => PsOp::Log,
        "mod" => PsOp::Mod,
        "mul" => PsOp::Mul,
        "neg" => PsOp::Neg,
        "round" => PsOp::Round,
        "sin" => PsOp::Sin,
        "sqrt" => PsOp::Sqrt,
        "sub" => PsOp::Sub,
        "truncate" => PsOp::Truncate,
        "eq" => PsOp::Eq,
        "ne" => PsOp::Ne,
        "gt" => PsOp::Gt,
        "ge" => PsOp::Ge,
        "lt" => PsOp::Lt,
        "le" => PsOp::Le,
        "and" => PsOp::And,
        "or" => PsOp::Or,
        "xor" => PsOp::Xor,
        "not" => PsOp::Not,
        "bitshift" => PsOp::Bitshift,
        "true" => PsOp::True,
        "false" => PsOp::False,
        "copy" => PsOp::Copy,
        "dup" => PsOp::Dup,
        "exch" => PsOp::Exch,
        "index" => PsOp::Index,
        "pop" => PsOp::Pop,
        "roll" => PsOp::Roll,
        "if" => PsOp::If,
        "ifelse" => PsOp::Ifelse,
        _ => return None,
    })
}

/// Split the decoded program text into whitespace/brace-delimited lexemes.
/// `{` and `}` are returned as standalone tokens; everything else is a word.
///
/// PDF §7.10.5 calculator functions are PostScript, which treats `%` as a
/// line comment to end-of-line.  Comments are stripped here so an otherwise
/// well-formed commented program is not rejected wholesale (a rejected tint
/// transform silently falls back to the gray heuristic, blanking
/// spot-colour pages).
fn ps_lex(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_comment = false;
    for ch in src.chars() {
        if in_comment {
            // A PostScript comment runs to the next end-of-line.
            if ch == '\n' || ch == '\r' {
                in_comment = false;
            }
            continue;
        }
        match ch {
            '%' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                in_comment = true;
            }
            '{' | '}' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Parse a lexeme slice starting just after a `{` into a token list, stopping
/// at the matching `}`.  Returns the tokens and the index just past the `}`.
///
/// `depth` is the current `{ }` nesting level; recursion is refused past
/// [`MAX_PS_DEPTH`] so a `{{{{…}}}}` bomb cannot overflow the native stack
/// during parsing (it returns `None`, which the caller logs and falls back
/// on, rather than aborting the process).
fn parse_ps_block(lexemes: &[String], mut i: usize, depth: usize) -> Option<(Vec<PsTok>, usize)> {
    if depth > MAX_PS_DEPTH {
        return None;
    }
    let mut toks = Vec::new();
    while i < lexemes.len() {
        let lx = lexemes[i].as_str();
        match lx {
            "}" => return Some((toks, i + 1)),
            "{" => {
                let (inner, next) = parse_ps_block(lexemes, i + 1, depth + 1)?;
                toks.push(PsTok::Proc(inner));
                i = next;
            }
            _ => {
                if let Ok(num) = lx.parse::<f64>() {
                    toks.push(PsTok::Num(num));
                } else {
                    toks.push(PsTok::Op(parse_ps_op(lx)?));
                }
                i += 1;
            }
        }
    }
    // Reached end of input without a closing brace.
    None
}

/// Parse the full Type-4 program (a single top-level `{ ... }` block).
fn parse_ps_program(src: &str) -> Option<Vec<PsTok>> {
    let lexemes = ps_lex(src);
    let first = lexemes.first()?;
    if first != "{" {
        return None;
    }
    let (toks, end) = parse_ps_block(&lexemes, 1, 0)?;
    // The top-level block must be the entire program; trailing lexemes after
    // the matching `}` mean the input is malformed.
    if end != lexemes.len() {
        return None;
    }
    Some(toks)
}

fn ps_pop(stack: &mut Vec<PsVal>) -> Option<PsVal> {
    stack.pop()
}

fn ps_pop_num(stack: &mut Vec<PsVal>) -> Option<f64> {
    match stack.pop()? {
        PsVal::Num(n) => Some(n),
        PsVal::Bool(_) => None,
    }
}

fn ps_pop_bool(stack: &mut Vec<PsVal>) -> Option<bool> {
    match stack.pop()? {
        PsVal::Bool(b) => Some(b),
        PsVal::Num(_) => None,
    }
}

/// Execute a single operator.  `If`/`Ifelse` never reach here — [`exec_ps`]
/// resolves them by consuming the preceding `Proc` token(s) as branches.
fn run_ps_op(op: PsOp, stack: &mut Vec<PsVal>) -> Option<()> {
    match op {
        PsOp::Abs => unary(stack, f64::abs),
        PsOp::Neg => unary(stack, |a| -a),
        PsOp::Ceiling => unary(stack, f64::ceil),
        PsOp::Floor => unary(stack, f64::floor),
        PsOp::Round => unary(stack, f64::round),
        // `truncate` and `cvi` are both truncate-toward-zero; `cvr` is a
        // no-op since the stack is already real-valued.
        PsOp::Truncate | PsOp::Cvi => unary(stack, f64::trunc),
        PsOp::Cvr => unary(stack, |a| a),
        PsOp::Sqrt => unary(stack, f64::sqrt),
        PsOp::Sin => unary(stack, |a| a.to_radians().sin()),
        PsOp::Cos => unary(stack, |a| a.to_radians().cos()),
        PsOp::Ln => unary(stack, f64::ln),
        PsOp::Log => unary(stack, f64::log10),
        PsOp::Add => binary(stack, |a, b| a + b),
        PsOp::Sub => binary(stack, |a, b| a - b),
        PsOp::Mul => binary(stack, |a, b| a * b),
        PsOp::Div => binary(stack, |a, b| a / b),
        PsOp::Exp => binary(stack, f64::powf),
        PsOp::Atan => binary(stack, |num, den| {
            let deg = num.atan2(den).to_degrees();
            if deg < 0.0 { deg + 360.0 } else { deg }
        }),
        PsOp::Idiv => int_binary(stack, |a, b| (a / b).trunc()),
        PsOp::Mod => int_binary(stack, |a, b| a % b),
        PsOp::Eq => rel_eq(stack, true),
        PsOp::Ne => rel_eq(stack, false),
        PsOp::Gt => rel_ord(stack, |o| o == std::cmp::Ordering::Greater),
        PsOp::Ge => rel_ord(stack, |o| o != std::cmp::Ordering::Less),
        PsOp::Lt => rel_ord(stack, |o| o == std::cmp::Ordering::Less),
        PsOp::Le => rel_ord(stack, |o| o != std::cmp::Ordering::Greater),
        PsOp::And => logic(stack, |a, b| a & b, |a, b| a & b),
        PsOp::Or => logic(stack, |a, b| a | b, |a, b| a | b),
        PsOp::Xor => logic(stack, |a, b| a ^ b, |a, b| a ^ b),
        PsOp::Not => ps_not(stack),
        PsOp::Bitshift => ps_bitshift(stack),
        PsOp::True => {
            stack.push(PsVal::Bool(true));
            Some(())
        }
        PsOp::False => {
            stack.push(PsVal::Bool(false));
            Some(())
        }
        PsOp::Pop => ps_pop(stack).map(|_| ()),
        PsOp::Dup => {
            let v = *stack.last()?;
            stack.push(v);
            Some(())
        }
        PsOp::Exch => {
            let len = stack.len();
            if len < 2 {
                return None;
            }
            stack.swap(len - 1, len - 2);
            Some(())
        }
        PsOp::Index => ps_index(stack),
        PsOp::Copy => ps_copy(stack),
        PsOp::Roll => ps_roll(stack),
        // If/Ifelse are resolved in exec_ps; reaching here is malformed.
        PsOp::If | PsOp::Ifelse => None,
    }
}

/// Pop two numbers, truncate both toward zero, apply `f`, push the result.
/// Used by `idiv` and `mod`, which operate on integer values.
fn int_binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = ps_pop_num(stack)?.trunc();
    let a = ps_pop_num(stack)?.trunc();
    if b == 0.0 {
        return None;
    }
    let r = f(a, b);
    // §7.10.5 leaves a non-finite result undefined; refuse it rather than
    // let an Inf/NaN flow into the tint→colour conversion (silent-wrong).
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

/// `not`: logical complement for booleans, bitwise complement for integers.
fn ps_not(stack: &mut Vec<PsVal>) -> Option<()> {
    match ps_pop(stack)? {
        PsVal::Bool(b) => stack.push(PsVal::Bool(!b)),
        PsVal::Num(n) => stack.push(PsVal::Num(f64::from(!ps_to_i32(n)?))),
    }
    Some(())
}

/// `bitshift`: arithmetic shift of an integer by a signed shift count
/// (positive = left, negative = right).
fn ps_bitshift(stack: &mut Vec<PsVal>) -> Option<()> {
    let shift = ps_to_i32(ps_pop_num(stack)?)?;
    let val = ps_to_i32(ps_pop_num(stack)?)?;
    let r = if shift >= 0 {
        val.wrapping_shl(shift.unsigned_abs())
    } else {
        val.wrapping_shr(shift.unsigned_abs())
    };
    stack.push(PsVal::Num(f64::from(r)));
    Some(())
}

/// `n index`: push a copy of the element `n` positions below the top.
fn ps_index(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    let idx = stack.len().checked_sub(1 + n)?;
    stack.push(stack[idx]);
    Some(())
}

/// `n copy`: duplicate the top `n` elements, preserving order.
fn ps_copy(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    let start = stack.len().checked_sub(n)?;
    for i in 0..n {
        stack.push(stack[start + i]);
    }
    Some(())
}

/// `n j roll`: circularly shift the top `n` elements by `j` (positive `j`
/// rotates toward the top of the stack).
fn ps_roll(stack: &mut Vec<PsVal>) -> Option<()> {
    let j = ps_to_i32(ps_pop_num(stack)?)?;
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    if n == 0 {
        return Some(());
    }
    let start = stack.len().checked_sub(n)?;
    let slice = &mut stack[start..];
    // rem_euclid keeps the shift in [0, n) for negative j as well.  i64
    // avoids any 32-bit wrap; n came from a stack number bounded by ps_to_i32
    // so it fits i64.
    let n_i64 = i64::try_from(n).ok()?;
    let shift = usize::try_from(i64::from(j).rem_euclid(n_i64)).ok()?;
    slice.rotate_right(shift);
    Some(())
}

/// Pop one number, apply `f`, push the result.  A non-finite result (e.g.
/// `sqrt` of a negative, `ln`/`log` of a non-positive) is undefined per
/// §7.10.5; refuse it loud-graceful so no Inf/NaN reaches colour conversion.
fn unary(stack: &mut Vec<PsVal>, f: impl Fn(f64) -> f64) -> Option<()> {
    let a = ps_pop_num(stack)?;
    let r = f(a);
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

/// Pop two numbers, apply `f`, push the result.  A non-finite result (e.g.
/// `1 0 div` → Inf, `exp` overflow) is undefined per §7.10.5; refuse it
/// loud-graceful so no Inf/NaN reaches colour conversion.
fn binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = ps_pop_num(stack)?;
    let a = ps_pop_num(stack)?;
    let r = f(a, b);
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

fn rel_eq(stack: &mut Vec<PsVal>, want_eq: bool) -> Option<()> {
    let b = ps_pop(stack)?;
    let a = ps_pop(stack)?;
    let eq = match (a, b) {
        (PsVal::Num(x), PsVal::Num(y)) => x == y,
        (PsVal::Bool(x), PsVal::Bool(y)) => x == y,
        _ => return None,
    };
    stack.push(PsVal::Bool(eq == want_eq));
    Some(())
}

fn rel_ord(stack: &mut Vec<PsVal>, pred: impl Fn(std::cmp::Ordering) -> bool) -> Option<()> {
    let b = ps_pop_num(stack)?;
    let a = ps_pop_num(stack)?;
    let ord = a.partial_cmp(&b)?;
    stack.push(PsVal::Bool(pred(ord)));
    Some(())
}

fn logic(
    stack: &mut Vec<PsVal>,
    bf: impl Fn(bool, bool) -> bool,
    nf: impl Fn(i32, i32) -> i32,
) -> Option<()> {
    let b = ps_pop(stack)?;
    let a = ps_pop(stack)?;
    match (a, b) {
        (PsVal::Bool(x), PsVal::Bool(y)) => stack.push(PsVal::Bool(bf(x, y))),
        (PsVal::Num(x), PsVal::Num(y)) => {
            stack.push(PsVal::Num(f64::from(nf(ps_to_i32(x)?, ps_to_i32(y)?))));
        }
        _ => return None,
    }
    Some(())
}

/// Convert a stack number to an i32 for bitwise ops; reject non-integral or
/// out-of-range values (PostScript integers are 32-bit).
fn ps_to_i32(n: f64) -> Option<i32> {
    if !n.is_finite() || n.fract() != 0.0 || n < f64::from(i32::MIN) || n > f64::from(i32::MAX) {
        return None;
    }
    // Bounds + integrality checked above.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value verified finite, integral, and within i32 range above"
    )]
    Some(n as i32)
}

/// Convert a stack number to a non-negative count for `index`/`copy`/`roll`.
/// Rejects negative, non-integral, or out-of-i32-range values.
fn ps_to_usize(n: f64) -> Option<usize> {
    let i = ps_to_i32(n)?;
    if i < 0 {
        return None;
    }
    usize::try_from(i).ok()
}

/// Walk a token list, executing it while resolving `if`/`ifelse`.  The parser
/// places procedure blocks as `Proc` tokens immediately before the `if` /
/// `ifelse` operator; this pass consumes them as the conditional branches.
///
/// `depth` mirrors the `{ }` nesting and bounds this function's own native
/// recursion: a deeply nested conditional chain (each `if` consuming only one
/// step) could otherwise recurse far enough to overflow the native stack
/// before [`MAX_PS_STEPS`] is reached.  The operand stack is also capped at
/// [`MAX_PS_STACK`] because a single `copy`/`dup`/`index` can push many
/// values while charging only one step.
fn exec_ps(toks: &[PsTok], stack: &mut Vec<PsVal>, steps: &mut u32, depth: usize) -> Option<()> {
    if depth > MAX_PS_DEPTH {
        return None;
    }
    let mut i = 0;
    while i < toks.len() {
        if *steps == 0 {
            return None;
        }
        match &toks[i] {
            PsTok::Num(n) => {
                *steps -= 1;
                stack.push(PsVal::Num(*n));
                i += 1;
            }
            PsTok::Op(PsOp::If) => {
                // Preceding token must be the procedure block.
                let proc = match toks.get(i.wrapping_sub(1)) {
                    Some(PsTok::Proc(p)) if i > 0 => p,
                    _ => return None,
                };
                *steps -= 1;
                let cond = ps_pop_bool(stack)?;
                if cond {
                    exec_ps(proc, stack, steps, depth + 1)?;
                }
                i += 1;
            }
            PsTok::Op(PsOp::Ifelse) => {
                // Preceding two tokens must be the two procedure blocks.
                let (p1, p2) = match (toks.get(i.wrapping_sub(2)), toks.get(i.wrapping_sub(1))) {
                    (Some(PsTok::Proc(a)), Some(PsTok::Proc(b))) if i >= 2 => (a, b),
                    _ => return None,
                };
                *steps -= 1;
                let cond = ps_pop_bool(stack)?;
                if cond {
                    exec_ps(p1, stack, steps, depth + 1)?;
                } else {
                    exec_ps(p2, stack, steps, depth + 1)?;
                }
                i += 1;
            }
            PsTok::Proc(_) => {
                // A procedure block is only valid as an operand to a
                // following if/ifelse; skip it here (it is consumed by the
                // If/Ifelse arms by looking back).  But verify it IS followed
                // by if/ifelse, otherwise the program is malformed.
                let followed_by_cond =
                    matches!(toks.get(i + 1), Some(PsTok::Op(PsOp::If | PsOp::Ifelse)))
                        || matches!(
                            (toks.get(i + 1), toks.get(i + 2)),
                            (Some(PsTok::Proc(_)), Some(PsTok::Op(PsOp::Ifelse)))
                        );
                if !followed_by_cond {
                    return None;
                }
                i += 1;
            }
            PsTok::Op(op) => {
                *steps -= 1;
                run_ps_op(*op, stack)?;
                i += 1;
            }
        }
        // Bound operand-stack growth.  `copy`/`dup`/`index` and bare literals
        // can grow the stack while charging at most one step, so the step
        // budget alone is not a memory bound; refuse loud-graceful past
        // MAX_PS_STACK rather than let a `copy` bomb exhaust memory.
        if stack.len() > MAX_PS_STACK {
            return None;
        }
    }
    Some(())
}

/// Evaluate a Type 4 (PostScript calculator) function.
///
/// `t` is the single input value (Separation tint or shading parameter).
/// PDF §7.10.5: clip the input to `Domain`, run the bounded operator program,
/// then return the top `n` stack values (deepest → channel 0).  The caller
/// applies `Range` clipping.
fn eval_postscript(
    doc: &Document,
    fn_obj: &Object,
    fn_dict: &Dictionary,
    t: f64,
    n: usize,
) -> Option<Vec<f64>> {
    let src_bytes = stream_bytes_for(doc, fn_obj)?;
    let src = std::str::from_utf8(&src_bytes).ok()?;
    let program = parse_ps_program(src)?;

    // Clip the input to Domain (consistent with the other function types,
    // which all normalise/clamp against Domain before evaluating).
    let (d0, d1) = read_fn_domain(fn_dict);
    let t_in = if d1 > d0 { t.clamp(d0, d1) } else { t };

    let mut stack: Vec<PsVal> = vec![PsVal::Num(t_in)];
    let mut steps = MAX_PS_STEPS;
    exec_ps(&program, &mut stack, &mut steps, 0)?;

    if stack.len() < n {
        return None;
    }
    let out: Vec<f64> = stack[stack.len() - n..]
        .iter()
        .map(|v| match v {
            PsVal::Num(x) => Some(*x),
            PsVal::Bool(_) => None,
        })
        .collect::<Option<Vec<f64>>>()?;
    // Reject any non-finite output (NaN or ±Inf — e.g. a literal `1e400`
    // parses to Inf).  A non-finite colour component is silent-wrong garbage
    // once it reaches the rasterizer, so fail loud-graceful here instead.
    if out.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(out)
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

    use crate::test_helpers::empty_doc;

    #[test]
    fn eval_stitching_wrong_bounds_count_falls_back() {
        // PDF §7.10.3: a Type 3 stitching function with k sub-functions must
        // carry k−1 Bounds values.  Two sub-functions therefore need exactly
        // one bound; supplying zero is malformed.  eval_stitching_depth must
        // fall back to evaluating the first sub-function rather than panicking
        // on the missing breakpoint.
        let sub_a = make_exp_dict(0.2, 0.8, 1.0); // sub_a(0.5) = 0.5
        // sub_b chosen so sub_b(0.5) = 0.95, distinct from sub_a(0.5) = 0.5;
        // if the fallback used the wrong sub-function the assertion below
        // would fire.
        let sub_b = make_exp_dict(0.9, 1.0, 1.0);
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
        let len = i64::try_from(samples.len()).expect("test fixture < i64::MAX");
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
        dict.set("Size", pdf::Object::Array(vec![pdf::Object::Integer(len)]));
        dict.set("BitsPerSample", pdf::Object::Integer(8));
        // Length is required for stream parsing in real PDFs; harmless to
        // include here since we hand the Stream directly to eval_function.
        dict.set("Length", pdf::Object::Integer(len));
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

    // ── Type 4 (PostScript calculator) ────────────────────────────────────────

    /// Build a Type 4 function as a Stream whose content is `program`.
    /// Domain = [0, 1]; `range_pairs` flattened into the `Range` array.
    fn make_ps_stream(program: &str, range_pairs: &[(f64, f64)]) -> pdf::Object {
        let bytes = program.as_bytes().to_vec();
        let len = i64::try_from(bytes.len()).expect("test fixture < i64::MAX");
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(4));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let mut range = Vec::new();
        for (lo, hi) in range_pairs {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "test fixture values are small and exactly representable in f32"
            )]
            {
                range.push(pdf::Object::Real(*lo as f32));
                range.push(pdf::Object::Real(*hi as f32));
            }
        }
        dict.set("Range", pdf::Object::Array(range));
        dict.set("Length", pdf::Object::Integer(len));
        pdf::Object::Stream(pdf::Stream::new(dict, bytes))
    }

    #[test]
    fn ps_simple_arithmetic() {
        // { 2 mul } at input 0.5 → 1.0
        let f = make_ps_stream("{ 2 mul }", &[(0.0, 10.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.5, 1).expect("eval { 2 mul }");
        assert!((r[0] - 1.0).abs() < 1e-9, "expected 1.0, got {}", r[0]);
    }

    #[test]
    fn ps_ifelse_branches() {
        // { dup 0.5 lt { pop 0 } { pop 1 } ifelse }
        let prog = "{ dup 0.5 lt { pop 0 } { pop 1 } ifelse }";
        let f = make_ps_stream(prog, &[(0.0, 1.0)]);
        let doc = empty_doc();
        let lo = eval_function(&doc, &f, 0.3, 1).expect("eval at 0.3");
        assert!(lo[0].abs() < 1e-9, "0.3 < 0.5 → 0, got {}", lo[0]);
        let hi = eval_function(&doc, &f, 0.7, 1).expect("eval at 0.7");
        assert!((hi[0] - 1.0).abs() < 1e-9, "0.7 ≥ 0.5 → 1, got {}", hi[0]);
    }

    #[test]
    fn ps_separation_to_cmyk_shape() {
        // obj-12-style: 1 input tint → 4 CMYK outputs.  This program maps a
        // "Black" separation tint t to DeviceCMYK [0 0 0 t]: push three zeros
        // then duplicate-free copy the tint through.
        let prog = "{ 0 0 0 4 -1 roll }";
        let f = make_ps_stream(prog, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.6, 4).expect("eval 1→4");
        assert_eq!(r.len(), 4);
        assert!(r[0].abs() < 1e-9 && r[1].abs() < 1e-9 && r[2].abs() < 1e-9);
        assert!(
            (r[3] - 0.6).abs() < 1e-9,
            "K channel should be tint, got {}",
            r[3]
        );
    }

    #[test]
    fn ps_stack_operators() {
        // exch: { 1 exch sub } at t=0.25 → 0.75
        let f = make_ps_stream("{ 1 exch sub }", &[(0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.25, 1).expect("exch");
        assert!((r[0] - 0.75).abs() < 1e-9, "expected 0.75, got {}", r[0]);

        // index: { 2 3 1 index } leaves [2 3 2]; pop top two via pop pop → 2.
        let f2 = make_ps_stream("{ pop 2 3 1 index add add }", &[(0.0, 100.0)]);
        let r2 = eval_function(&doc, &f2, 0.0, 1).expect("index");
        // 2 + 3 + 2 = 7
        assert!((r2[0] - 7.0).abs() < 1e-9, "expected 7, got {}", r2[0]);

        // copy: { pop 1 2 2 copy add add add } → 1 2 1 2 → 1+2+1+2 = 6
        let f3 = make_ps_stream("{ pop 1 2 2 copy add add add }", &[(0.0, 100.0)]);
        let r3 = eval_function(&doc, &f3, 0.0, 1).expect("copy");
        assert!((r3[0] - 6.0).abs() < 1e-9, "expected 6, got {}", r3[0]);

        // roll: { pop 1 2 3 3 1 roll add add } → roll right by 1 → 3 1 2 → sum 6
        let f4 = make_ps_stream("{ pop 1 2 3 3 1 roll add add }", &[(0.0, 100.0)]);
        let r4 = eval_function(&doc, &f4, 0.0, 1).expect("roll");
        assert!((r4[0] - 6.0).abs() < 1e-9, "expected 6, got {}", r4[0]);
    }

    #[test]
    fn ps_malformed_returns_none() {
        let doc = empty_doc();
        // Missing closing brace.
        let bad = make_ps_stream("{ 2 mul", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad, 0.5, 1).is_none());
        // Unknown operator.
        let bad2 = make_ps_stream("{ frobnicate }", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad2, 0.5, 1).is_none());
        // Stack underflow (add with one operand).
        let bad3 = make_ps_stream("{ add }", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad3, 0.5, 1).is_none());
        // Not enough outputs requested vs produced.
        let bad4 = make_ps_stream("{ }", &[(0.0, 1.0), (0.0, 1.0)]);
        assert!(eval_function(&doc, &bad4, 0.5, 2).is_none());
    }

    #[test]
    fn ps_recursion_bomb_is_bounded() {
        // Deeply nested `{{{{…}}}}` must terminate loud-graceful (None), not
        // overflow the native stack during parse or exec.  Spawn on a thread
        // with a small stack so an unbounded recursion would actually abort
        // the test rather than silently pass on a large default stack.
        let depth = MAX_PS_DEPTH + 5_000;
        let mut prog = String::with_capacity(depth * 2 + 8);
        prog.push('{');
        for _ in 0..depth {
            prog.push('{');
        }
        for _ in 0..depth {
            prog.push('}');
        }
        prog.push_str(" if }");
        let handle = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let doc = empty_doc();
                let f = make_ps_stream(&prog, &[(0.0, 1.0)]);
                eval_function(&doc, &f, 0.5, 1)
            })
            .expect("spawn bounded-stack thread");
        let result = handle.join().expect("recursion-bomb must not overflow");
        assert!(
            result.is_none(),
            "over-deep program must fail loud-graceful"
        );
    }

    #[test]
    fn ps_stack_bomb_is_bounded() {
        // `n copy` duplicates the top n elements; feeding it the exact
        // current stack length doubles the stack with a single step.  A
        // chain of doublings reaches MAX_PS_STACK in a handful of ops while
        // charging almost no step budget, so only the operand-stack cap
        // stops it.  Build the doubling chain statically (length is known
        // after each `copy`): start at 1, then `1 copy`→2, `2 copy`→4, …
        let mut prog = String::from("{");
        let mut len: usize = 1; // the single input tint already on the stack
        while len <= MAX_PS_STACK {
            prog.push(' ');
            prog.push_str(&len.to_string());
            prog.push_str(" copy");
            len *= 2;
        }
        prog.push_str(" }");
        let doc = empty_doc();
        let f = make_ps_stream(&prog, &[(0.0, 1.0)]);
        assert!(
            eval_function(&doc, &f, 0.5, 1).is_none(),
            "operand-stack bomb must fail loud-graceful, not OOM"
        );
    }

    #[test]
    fn ps_div_zero_and_domain_errors_fail_graceful() {
        let doc = empty_doc();
        // 1 0 div → +Inf: must not leak Inf into the colour.
        let f = make_ps_stream("{ pop 1 0 div }", &[(-1e9, 1e9)]);
        assert!(
            eval_function(&doc, &f, 0.5, 1).is_none(),
            "div-by-zero must fail loud-graceful, not push Inf"
        );
        // -1 sqrt → NaN.
        let f2 = make_ps_stream("{ pop -1 sqrt }", &[(-1e9, 1e9)]);
        assert!(
            eval_function(&doc, &f2, 0.5, 1).is_none(),
            "sqrt(-1) → None"
        );
        // 0 ln → -Inf.
        let f3 = make_ps_stream("{ pop 0 ln }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f3, 0.5, 1).is_none(), "ln(0) → None");
        // idiv by zero.
        let f4 = make_ps_stream("{ pop 5 0 idiv }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f4, 0.5, 1).is_none(), "idiv 0 → None");
        // A literal that parses to Inf must not reach the output.
        let f5 = make_ps_stream("{ pop 1e400 }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f5, 0.5, 1).is_none(), "1e400 → None");
    }

    #[test]
    fn ps_comments_are_stripped() {
        // A PostScript `%` comment to end-of-line must not break parsing.
        let prog = "{ % map tint through\n 2 mul % double it\n }";
        let f = make_ps_stream(prog, &[(0.0, 10.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.5, 1).expect("commented program evaluates");
        assert!((r[0] - 1.0).abs() < 1e-9, "expected 1.0, got {}", r[0]);
    }

    #[test]
    fn ps_sot_apprenti_tint_program() {
        // The exact obj-12 tint transform from sot-apprenti-1774:
        // Separation /Black → DeviceCMYK, mapping tint t to [0 0 0 t].
        // A regression here re-blanks the page.
        let prog = "{dup 0 mul exch dup 0 mul exch dup 0 mul exch 1 mul }";
        let f = make_ps_stream(prog, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 1.0, 4).expect("sot-apprenti tint at t=1");
        assert_eq!(r.len(), 4);
        assert!(
            r[0].abs() < 1e-9 && r[1].abs() < 1e-9 && r[2].abs() < 1e-9,
            "C/M/Y must be 0, got {r:?}"
        );
        assert!(
            (r[3] - 1.0).abs() < 1e-9,
            "K must equal the tint (1.0), got {}",
            r[3]
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
