//! PDF shading resource resolution (Types 2–5 — axial, radial, Gouraud mesh).
//!
//! [`resolve_shading`] looks up a named `Shading` resource and returns a
//! [`ShadingResult`] which is either a [`raster::pipe::Pattern`] (for smooth
//! gradient types 2–3) or a pre-decoded triangle mesh (for mesh types 4–5).
//!
//! # Supported shading types
//!
//! | PDF type | Name | Support |
//! |---|---|---|
//! | 2 | Axial (linear) | yes |
//! | 3 | Radial | yes |
//! | 4 | Free-form Gouraud triangle mesh | yes |
//! | 5 | Lattice-form Gouraud mesh | yes |
//! | 1, 6–7 | Function / Coons / tensor patch | stub (logged + skipped) |
//!
//! # Colour spaces
//!
//! Shading colours are converted to sRGB via the same colour-space resolution
//! used for images.  Only the `N`-channel output of the function is used; ICC
//! profiles and Indexed spaces fall back to Gray/RGB as in the image path.
//!
//! # PDF Function objects
//!
//! The `Function` key maps `t → [c₀, c₁, …, cₙ]`.  Only Type 2 (Exponential)
//! and Type 3 (Stitching) functions are implemented; other types are treated
//! as a solid mid-point colour.

use lopdf::{Dictionary, Document, Object, Stream};
use raster::pipe::Pattern;
use raster::shading::axial::AxialPattern;
use raster::shading::gouraud::GouraudVertex;
use raster::shading::radial::RadialPattern;

use super::dict_ext::DictExt;
use super::image::{ImageColorSpace, cs_to_image_color_space};

// ── Public types ──────────────────────────────────────────────────────────────

/// Result of resolving a named shading resource.
///
/// - `Pattern`: smooth gradient — use with [`raster::shading::shaded_fill`].
/// - `Mesh`: pre-decoded Gouraud triangle list — call
///   [`raster::shading::gouraud::gouraud_triangle_fill`] for each triangle.
pub enum ShadingResult {
    /// Smooth gradient — use with [`raster::shading::shaded_fill`].
    /// The `[f64; 4]` is an approximate bounding box `[xmin, ymin, xmax, ymax]`.
    Pattern(Box<dyn Pattern + Send + Sync>, [f64; 4]),
    /// Pre-decoded Gouraud triangle list — call
    /// [`raster::shading::gouraud::gouraud_triangle_fill`] for each element.
    Mesh(Vec<[GouraudVertex; 3]>),
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Resolve the named shading from the page/form resource dict.
///
/// Returns `None` if the name is absent, the type is unsupported, or any
/// required key is missing.  A warning is logged for unsupported types so the
/// caller can fall back gracefully.
#[must_use]
pub fn resolve_shading(
    doc: &Document,
    resource_context_dict: &Dictionary,
    name: &[u8],
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<ShadingResult> {
    let res = super::image::resolve_dict(doc, resource_context_dict.get(b"Resources").ok()?)?;
    let sh_res = super::image::resolve_dict(doc, res.get(b"Shading").ok()?)?;
    let sh_obj = sh_res.get(name).ok()?;

    // Shading types 2–3 are plain dicts; types 4–7 are streams.
    // Try stream first (stream has a dict too), then fall back to plain dict.
    let (sh_dict, stream_data): (&Dictionary, Option<Vec<u8>>) = match sh_obj {
        Object::Stream(s) => (&s.dict, stream_content(s)),
        Object::Reference(id) => {
            match doc.get_object(*id).ok()? {
                Object::Stream(s) => (&s.dict, stream_content(s)),
                obj => (super::image::resolve_dict(doc, obj)?, None),
            }
        }
        obj => (super::image::resolve_dict(doc, obj)?, None),
    };

    let shading_type = sh_dict.get_i64(b"ShadingType")?;

    let cs_obj = sh_dict.get(b"ColorSpace").ok()?;
    let cs = cs_to_image_color_space(doc, cs_obj);
    let n_channels = cs_channel_count(cs);

    match shading_type {
        2 => resolve_axial(doc, sh_dict, cs, n_channels, ctm, page_h)
            .map(|(p, bb)| ShadingResult::Pattern(p, bb)),
        3 => resolve_radial(doc, sh_dict, cs, n_channels, ctm, page_h)
            .map(|(p, bb)| ShadingResult::Pattern(p, bb)),
        4 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type4_mesh(
                sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        5 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type5_mesh(
                sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        other => {
            log::warn!("shading: ShadingType {other} not yet implemented — skipping sh operator");
            None
        }
    }
}

/// Decompress a stream, returning `None` on failure (logged as a debug message).
fn stream_content(s: &Stream) -> Option<Vec<u8>> {
    s.decompressed_content().ok()
}

// ── Axial (Type 2) ────────────────────────────────────────────────────────────

fn resolve_axial(
    doc: &Document,
    sh: &Dictionary,
    cs: ImageColorSpace,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<(Box<dyn Pattern + Send + Sync>, [f64; 4])> {
    // Coords: [x0, y0, x1, y1] in user space.
    let coords = read_f64_array(sh, b"Coords", 4)?;

    // Domain: [t0, t1] — defaults to [0, 1].
    let (t0, t1) = read_domain(sh);

    // Extend: [extend_start, extend_end].
    let (ext_s, ext_e) = read_extend(sh);

    // Evaluate the function at t0 and t1 to get the two colours.
    let fn_obj = sh.get(b"Function").ok()?;
    let c0_user = eval_function(doc, fn_obj, t0, n)?;
    let c1_user = eval_function(doc, fn_obj, t1, n)?;

    let rgb0 = cs_to_rgb(cs, &c0_user);
    let rgb1 = cs_to_rgb(cs, &c1_user);

    // Reject non-finite user-space coordinates before transforming.
    if !coords.iter().all(|v| v.is_finite()) {
        log::warn!("shading/axial: non-finite Coords — skipping");
        return None;
    }

    // Transform Coords from user space to device space (y-flip included).
    let (dx0, dy0) = transform_point(ctm, coords[0], coords[1], page_h);
    let (dx1, dy1) = transform_point(ctm, coords[2], coords[3], page_h);

    if !dx0.is_finite() || !dy0.is_finite() || !dx1.is_finite() || !dy1.is_finite() {
        log::warn!("shading/axial: non-finite device coords after CTM transform — skipping");
        return None;
    }

    let bbox = bbox_from_coords(&[dx0, dy0, dx1, dy1]);

    let pattern = AxialPattern::new(rgb0, rgb1, dx0, dy0, dx1, dy1, t0, t1, ext_s, ext_e);
    Some((Box::new(pattern), bbox))
}

// ── Radial (Type 3) ───────────────────────────────────────────────────────────

fn resolve_radial(
    doc: &Document,
    sh: &Dictionary,
    cs: ImageColorSpace,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<(Box<dyn Pattern + Send + Sync>, [f64; 4])> {
    // Coords: [x0, y0, r0, x1, y1, r1] in user space.
    let coords = read_f64_array(sh, b"Coords", 6)?;

    let (t0, t1) = read_domain(sh);
    let (ext_s, ext_e) = read_extend(sh);

    let fn_obj = sh.get(b"Function").ok()?;
    let c0_user = eval_function(doc, fn_obj, t0, n)?;
    let c1_user = eval_function(doc, fn_obj, t1, n)?;

    let rgb0 = cs_to_rgb(cs, &c0_user);
    let rgb1 = cs_to_rgb(cs, &c1_user);

    // Reject non-finite user-space coordinates before transforming.
    if !coords.iter().all(|v| v.is_finite()) {
        log::warn!("shading/radial: non-finite Coords — skipping");
        return None;
    }

    // Transform centres; scale radius by the geometric mean of the CTM scale.
    let (dx0, dy0) = transform_point(ctm, coords[0], coords[1], page_h);
    let (dx1, dy1) = transform_point(ctm, coords[3], coords[4], page_h);

    if !dx0.is_finite() || !dy0.is_finite() || !dx1.is_finite() || !dy1.is_finite() {
        log::warn!("shading/radial: non-finite device coords after CTM transform — skipping");
        return None;
    }

    let scale = ctm_scale(ctm);
    let dr0 = coords[2] * scale;
    let dr1 = coords[5] * scale;

    // Bounding box: union of both circles.
    let outer_r = dr0.max(dr1);
    let bbox = [
        dx0.min(dx1) - outer_r,
        dy0.min(dy1) - outer_r,
        dx0.max(dx1) + outer_r,
        dy0.max(dy1) + outer_r,
    ];

    let pattern = RadialPattern::new(
        rgb0, rgb1, dx0, dy0, dr0, dx1, dy1, dr1, t0, t1, ext_s, ext_e,
    );
    Some((Box::new(pattern), bbox))
}

// ── Type 4 — Free-form Gouraud triangle mesh ──────────────────────────────────

/// Bit-stream reader — reads `bits` bits at a time from a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bits_in_buf: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Read `n` bits (1–32) and return as `u32`. Returns `None` on EOF.
    fn read_bits(&mut self, n: u8) -> Option<u32> {
        while self.bits_in_buf < n {
            if self.byte_pos >= self.data.len() {
                return None;
            }
            self.bit_buf = (self.bit_buf << 8) | u32::from(self.data[self.byte_pos]);
            self.byte_pos += 1;
            self.bits_in_buf += 8;
        }
        self.bits_in_buf -= n;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        Some((self.bit_buf >> self.bits_in_buf) & mask)
    }
}

/// Decode a Type 4 (free-form Gouraud triangle mesh) shading stream.
///
/// Records: `flag(8 bits)`, `x(BitsPerCoordinate bits)`, `y(BitsPerCoordinate bits)`,
/// then `n_channels × BitsPerComponent bits` per colour channel.
/// Flag 0 = new triangle (3 vertices); flag 1 = shared edge v[1]–v[2]; flag 2 = shared edge v[0]–v[2].
fn decode_type4_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bits_per_coord) = sh.get_i64(b"BitsPerCoordinate").map(|v| v as u8) else {
        log::warn!("shading/type4: missing BitsPerCoordinate");
        return vec![];
    };
    let Some(bits_per_comp) = sh.get_i64(b"BitsPerComponent").map(|v| v as u8) else {
        log::warn!("shading/type4: missing BitsPerComponent");
        return vec![];
    };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    let mut prev: Vec<GouraudVertex> = Vec::new();

    loop {
        let Some(flag) = reader.read_bits(8) else { break };
        let n_new = if flag == 0 { 3 } else { 1 };
        let mut new_verts = Vec::with_capacity(n_new);

        for _ in 0..n_new {
            let Some(raw_x) = reader.read_bits(bits_per_coord) else { break };
            let Some(raw_y) = reader.read_bits(bits_per_coord) else { break };
            let mut channels = Vec::with_capacity(n_channels);
            let mut ok = true;
            for _ in 0..n_channels {
                if let Some(raw_c) = reader.read_bits(bits_per_comp) {
                    channels.push(raw_c);
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
            let v = decode_vertex(
                raw_x,
                raw_y,
                &channels,
                bits_per_coord,
                bits_per_comp,
                &decode,
                cs,
                ctm,
                page_h,
            );
            new_verts.push(v);
        }

        let tri_opt: Option<[GouraudVertex; 3]> = match (flag, new_verts.as_slice()) {
            (0, [a, b, c]) => Some([*a, *b, *c]),
            (1, [c]) if prev.len() >= 2 => Some([prev[prev.len() - 2], prev[prev.len() - 1], *c]),
            (2, [c]) if prev.len() >= 2 => Some([prev[prev.len() - 3 + 2], prev[prev.len() - 1], *c]),
            _ => None,
        };

        if let Some(tri) = tri_opt {
            prev = tri.to_vec();
            triangles.push(tri);
        }
    }

    triangles
}

// ── Type 5 — Lattice-form Gouraud mesh ───────────────────────────────────────

/// Decode a Type 5 (lattice-form Gouraud) shading stream.
///
/// `VerticesPerRow` vertices wide, rows of vertices form a grid.
/// Adjacent 2×2 quads are split into two triangles each.
fn decode_type5_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bits_per_coord) = sh.get_i64(b"BitsPerCoordinate").map(|v| v as u8) else {
        log::warn!("shading/type5: missing BitsPerCoordinate");
        return vec![];
    };
    let Some(bits_per_comp) = sh.get_i64(b"BitsPerComponent").map(|v| v as u8) else {
        log::warn!("shading/type5: missing BitsPerComponent");
        return vec![];
    };
    let Some(verts_per_row) = sh.get_i64(b"VerticesPerRow") else {
        log::warn!("shading/type5: missing VerticesPerRow");
        return vec![];
    };
    if verts_per_row < 2 {
        log::warn!("shading/type5: VerticesPerRow < 2 ({verts_per_row}) — skipping");
        return vec![];
    }
    #[expect(clippy::cast_sign_loss, reason = "guarded >= 2 above")]
    let vpr = verts_per_row as usize;
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut all_rows: Vec<Vec<GouraudVertex>> = Vec::new();

    // Read all rows of vertices.
    'outer: loop {
        let mut row = Vec::with_capacity(vpr);
        for _ in 0..vpr {
            let Some(raw_x) = reader.read_bits(bits_per_coord) else { break 'outer };
            let Some(raw_y) = reader.read_bits(bits_per_coord) else { break 'outer };
            let mut channels = Vec::with_capacity(n_channels);
            let mut ok = true;
            for _ in 0..n_channels {
                if let Some(raw_c) = reader.read_bits(bits_per_comp) {
                    channels.push(raw_c);
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break 'outer;
            }
            row.push(decode_vertex(
                raw_x,
                raw_y,
                &channels,
                bits_per_coord,
                bits_per_comp,
                &decode,
                cs,
                ctm,
                page_h,
            ));
        }
        all_rows.push(row);
    }

    // Tessellate adjacent row pairs into triangles.
    let mut triangles = Vec::new();
    for rows in all_rows.windows(2) {
        let (top, bot) = (&rows[0], &rows[1]);
        for col in 0..vpr.saturating_sub(1) {
            // Quad: top[col], top[col+1], bot[col], bot[col+1]
            // Split into two triangles.
            triangles.push([top[col], top[col + 1], bot[col]]);
            triangles.push([top[col + 1], bot[col + 1], bot[col]]);
        }
    }

    triangles
}

// ── Mesh vertex helpers ───────────────────────────────────────────────────────

/// Decode arrays for Type 4/5: `Decode` = `[xmin, xmax, ymin, ymax, c0min, c0max, …]`.
fn read_decode_array(sh: &Dictionary, n_channels: usize) -> Vec<f64> {
    let expected = 4 + n_channels * 2; // x,y + per-channel
    sh.get(b"Decode")
        .ok()
        .and_then(|o| o.as_array().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|o| match o {
                    Object::Real(r) => Some(f64::from(*r)),
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "decode values are small PDF numbers"
                    )]
                    Object::Integer(i) => Some(*i as f64),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            // Default: x,y in [0,1], each colour channel in [0,1].
            let mut d = vec![0.0_f64, 1.0, 0.0, 1.0];
            d.extend(std::iter::repeat_n([0.0_f64, 1.0], n_channels).flatten());
            d
        })
        .into_iter()
        .take(expected)
        .collect()
}

/// Decode one raw vertex from bit-fields to a [`GouraudVertex`] in device space.
#[expect(
    clippy::cast_precision_loss,
    reason = "raw bit fields are at most 32 bits; f64 has 53-bit mantissa, so values up to 2^32 are exact"
)]
fn decode_vertex(
    raw_x: u32,
    raw_y: u32,
    raw_channels: &[u32],
    bits_per_coord: u8,
    bits_per_comp: u8,
    decode: &[f64],
    cs: ImageColorSpace,
    ctm: &[f64; 6],
    page_h: f64,
) -> GouraudVertex {
    let max_coord = ((1u64 << bits_per_coord) - 1) as f64;
    let max_comp = if bits_per_comp == 0 {
        1.0
    } else {
        ((1u64 << bits_per_comp) - 1) as f64
    };

    let x_min = decode.first().copied().unwrap_or(0.0);
    let x_max = decode.get(1).copied().unwrap_or(1.0);
    let y_min = decode.get(2).copied().unwrap_or(0.0);
    let y_max = decode.get(3).copied().unwrap_or(1.0);

    let ux = x_min + (raw_x as f64 / max_coord) * (x_max - x_min);
    let uy = y_min + (raw_y as f64 / max_coord) * (y_max - y_min);

    let channels: Vec<f64> = raw_channels
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            let c_min = decode.get(4 + i * 2).copied().unwrap_or(0.0);
            let c_max = decode.get(4 + i * 2 + 1).copied().unwrap_or(1.0);
            c_min + (raw as f64 / max_comp) * (c_max - c_min)
        })
        .collect();

    let color = cs_to_rgb(cs, &channels);
    let (dx, dy) = transform_point(ctm, ux, uy, page_h);

    GouraudVertex { x: dx, y: dy, color }
}

// ── PDF Function evaluation ───────────────────────────────────────────────────

/// Evaluate a PDF Function object at parameter `t`, returning `n` colour channels.
///
/// Supported function types:
/// - **Type 2** (Exponential): `C0 + t^N × (C1 − C0)`.
/// - **Type 3** (Stitching): maps `t` to a sub-function and recursively evaluates.
/// - **Type 0** (Sampled): linear interpolation across the decode range (no stream data).
///
/// Unknown types fall back to `C0` if available.
fn eval_function(doc: &Document, fn_obj: &Object, t: f64, n: usize) -> Option<Vec<f64>> {
    let fn_dict = resolve_fn_dict(doc, fn_obj)?;
    let fn_type = fn_dict.get_i64(b"FunctionType")?;
    match fn_type {
        2 => Some(eval_exponential(fn_dict, t, n)),
        3 => eval_stitching(doc, fn_dict, t, n),
        0 => Some(eval_sampled_approx(fn_dict, t, n)),
        _ => {
            log::debug!("shading: FunctionType {fn_type} not yet implemented — using C0 fallback");
            read_fn_color(fn_dict, b"C0", n)
        }
    }
}

/// Evaluate a Type 2 Exponential function: `C0 + (t_norm)^N × (C1 − C0)`.
fn eval_exponential(fn_dict: &Dictionary, t: f64, n: usize) -> Vec<f64> {
    let c0 = read_fn_color(fn_dict, b"C0", n).unwrap_or_else(|| vec![0.0; n]);
    let c1 = read_fn_color(fn_dict, b"C1", n).unwrap_or_else(|| vec![1.0; n]);

    let (d0, d1) = read_fn_domain(fn_dict);
    let t_clamped = t.clamp(d0, d1);

    let exponent = fn_dict.get(b"N").ok().and_then(obj_to_f64).unwrap_or(1.0);
    let t_norm = if (d1 - d0).abs() < f64::EPSILON {
        0.0
    } else {
        (t_clamped - d0) / (d1 - d0)
    };
    let weight = t_norm.max(0.0).powf(exponent);

    c0.iter()
        .zip(c1.iter())
        .map(|(&a, &b)| a + weight * (b - a))
        .collect()
}

/// Evaluate a Type 3 Stitching function.
///
/// Maps `t` to the appropriate sub-function via `Bounds` and `Encode`,
/// then recursively evaluates that sub-function.
fn eval_stitching(doc: &Document, fn_dict: &Dictionary, t: f64, n: usize) -> Option<Vec<f64>> {
    let (d0, d1) = read_fn_domain(fn_dict);
    let t_clamped = t.clamp(d0, d1);

    let fns = fn_dict.get(b"Functions").ok()?.as_array().ok()?;
    let num_fns = fns.len();
    if num_fns == 0 {
        return None;
    }

    // Bounds: k-1 values for k sub-functions splitting [Domain0, Domain1].
    let bounds: Vec<f64> = fn_dict
        .get(b"Bounds")
        .ok()
        .and_then(|o| o.as_array().ok())
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
        return eval_function(doc, &fns[0], t_clamped, n);
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
        .ok()
        .and_then(|o| o.as_array().ok())
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

    eval_function(doc, &fns[idx], t_encoded, n)
}

/// Approximate a Type 0 Sampled function by linearly interpolating the decode range.
///
/// Without decompressed stream bytes, this is the best possible approximation:
/// it returns the correct endpoints at `t = Domain[0]` and `t = Domain[1]`.
fn eval_sampled_approx(fn_dict: &Dictionary, t: f64, n: usize) -> Vec<f64> {
    let (d0, d1) = read_fn_domain(fn_dict);
    let t_norm = if (d1 - d0).abs() < f64::EPSILON {
        0.0
    } else {
        ((t - d0) / (d1 - d0)).clamp(0.0, 1.0)
    };

    // Decode range defaults to [0, 1] per channel.
    let decode: Vec<f64> = fn_dict
        .get(b"Decode")
        .ok()
        .and_then(|o| o.as_array().ok())
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

// ── Colour conversion helpers ─────────────────────────────────────────────────

/// Convert a function output (values in `[0, 1]`) to an sRGB triple `[u8; 3]`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to [0,1] then scaled to [0,255]; round() fits in u8"
)]
fn cs_to_rgb(cs: ImageColorSpace, channels: &[f64]) -> [u8; 3] {
    let scale =
        |i: usize| (channels.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0).round() as u8;
    match cs {
        ImageColorSpace::Gray | ImageColorSpace::Mask => {
            let g = scale(0);
            [g, g, g]
        }
        ImageColorSpace::Rgb => [scale(0), scale(1), scale(2)],
    }
}

/// Return the number of colour channels for a colour space.
const fn cs_channel_count(cs: ImageColorSpace) -> usize {
    match cs {
        ImageColorSpace::Gray | ImageColorSpace::Mask => 1,
        ImageColorSpace::Rgb => 3,
    }
}

// ── Dictionary / function helpers ─────────────────────────────────────────────

/// Resolve a `Function` object (direct dict or indirect reference) to a dict.
fn resolve_fn_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d),
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        _ => None,
    }
}

/// Read a fixed-length array of `f64` values from a dictionary key.
fn read_f64_array(dict: &Dictionary, key: &[u8], expected: usize) -> Option<Vec<f64>> {
    let arr = dict.get(key).ok()?.as_array().ok()?;
    if arr.len() < expected {
        return None;
    }
    let vals: Vec<f64> = arr.iter().take(expected).filter_map(obj_to_f64).collect();
    if vals.len() < expected {
        return None;
    }
    Some(vals)
}

/// Read the `Domain` key (1D) from a function dict; defaults to `[0, 1]`.
fn read_fn_domain(dict: &Dictionary) -> (f64, f64) {
    let arr = dict
        .get(b"Domain")
        .ok()
        .and_then(|o| o.as_array().ok())
        .map(|a| a.iter().filter_map(obj_to_f64).collect::<Vec<_>>());
    match arr.as_deref() {
        Some([d0, d1, ..]) => (*d0, *d1),
        _ => (0.0, 1.0),
    }
}

/// Read the shading `Domain` key; defaults to `[0, 1]`.
fn read_domain(sh: &Dictionary) -> (f64, f64) {
    read_fn_domain(sh)
}

/// Read the `Extend` array `[extend_start, extend_end]`; both default to `false`.
fn read_extend(sh: &Dictionary) -> (bool, bool) {
    let arr = sh.get(b"Extend").ok().and_then(|o| o.as_array().ok());
    match arr {
        Some(arr) if arr.len() >= 2 => {
            let s = arr[0].as_bool().unwrap_or(false);
            let e = arr[1].as_bool().unwrap_or(false);
            (s, e)
        }
        _ => (false, false),
    }
}

/// Read a colour array (`C0` or `C1`) from a function dict, padding to `n` channels.
fn read_fn_color(dict: &Dictionary, key: &[u8], n: usize) -> Option<Vec<f64>> {
    let arr = dict.get(key).ok()?.as_array().ok()?;
    let mut vals: Vec<f64> = arr.iter().filter_map(obj_to_f64).collect();
    if vals.is_empty() {
        return None;
    }
    vals.resize(n, *vals.last().unwrap_or(&0.0));
    Some(vals)
}

/// Convert a `lopdf::Object` (Real or Integer) to `f64`.
fn obj_to_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "PDF numeric values in function dicts are small; precision loss is negligible"
        )]
        Object::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

// ── Coordinate transforms ─────────────────────────────────────────────────────

/// Transform a user-space point through the CTM and y-flip to device space.
fn transform_point(ctm: &[f64; 6], x: f64, y: f64, page_h: f64) -> (f64, f64) {
    let dx = ctm[0].mul_add(x, ctm[2] * y) + ctm[4];
    let dy = ctm[1].mul_add(x, ctm[3] * y) + ctm[5];
    (dx, page_h - dy)
}

/// Compute the uniform scale factor of the CTM (geometric mean of x and y scales).
///
/// Falls back to `1.0` when the CTM contains non-finite entries or produces a
/// non-finite scale (e.g., from an extreme anisotropic or degenerate matrix).
fn ctm_scale(ctm: &[f64; 6]) -> f64 {
    let sx = ctm[0].hypot(ctm[1]);
    let sy = ctm[2].hypot(ctm[3]);
    let result = (sx * sy).sqrt();
    if result.is_finite() { result } else { 1.0 }
}

/// Compute a bounding box from a flat list of interleaved `[x, y, x, y, …]` values.
fn bbox_from_coords(pts: &[f64]) -> [f64; 4] {
    debug_assert!(pts.len() >= 4 && pts.len().is_multiple_of(2));
    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut i = 0;
    while i + 1 < pts.len() {
        xmin = xmin.min(pts[i]);
        xmax = xmax.max(pts[i]);
        ymin = ymin.min(pts[i + 1]);
        ymax = ymax.max(pts[i + 1]);
        i += 2;
    }
    [xmin, ymin, xmax, ymax]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cs_to_rgb_gray_midpoint() {
        let rgb = cs_to_rgb(ImageColorSpace::Gray, &[0.5]);
        // round(0.5 * 255) = 128
        assert_eq!(rgb, [128, 128, 128]);
    }

    #[test]
    fn cs_to_rgb_gray_white() {
        assert_eq!(cs_to_rgb(ImageColorSpace::Gray, &[1.0]), [255, 255, 255]);
    }

    #[test]
    fn cs_to_rgb_rgb_passthrough() {
        let rgb = cs_to_rgb(ImageColorSpace::Rgb, &[1.0, 0.0, 0.5]);
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 0);
        assert!(rgb[2] > 120 && rgb[2] < 135);
    }

    #[test]
    fn cs_channel_count_values() {
        assert_eq!(cs_channel_count(ImageColorSpace::Gray), 1);
        assert_eq!(cs_channel_count(ImageColorSpace::Rgb), 3);
    }

    fn make_exp_dict(c0: f32, c1: f32, exponent: f32) -> lopdf::Dictionary {
        let mut dict = lopdf::Dictionary::new();
        dict.set("FunctionType", lopdf::Object::Integer(2));
        dict.set("C0", lopdf::Object::Array(vec![lopdf::Object::Real(c0)]));
        dict.set("C1", lopdf::Object::Array(vec![lopdf::Object::Real(c1)]));
        dict.set("N", lopdf::Object::Real(exponent));
        dict.set(
            "Domain",
            lopdf::Object::Array(vec![lopdf::Object::Real(0.0), lopdf::Object::Real(1.0)]),
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
    fn transform_point_identity_with_yflip() {
        let ctm = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (dx, dy) = transform_point(&ctm, 10.0, 20.0, 100.0);
        assert!((dx - 10.0).abs() < 1e-9);
        assert!((dy - 80.0).abs() < 1e-9); // y-flip: 100 - 20
    }

    #[test]
    fn bbox_from_two_points() {
        let b = bbox_from_coords(&[3.0, 7.0, 10.0, 2.0]);
        assert_eq!(b, [3.0, 2.0, 10.0, 7.0]);
    }

    #[test]
    fn ctm_scale_identity_is_one() {
        let ctm = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((ctm_scale(&ctm) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ctm_scale_uniform_2x() {
        let ctm = [2.0f64, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!((ctm_scale(&ctm) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ctm_scale_nan_falls_back_to_one() {
        let ctm = [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(ctm_scale(&ctm), 1.0);
    }

    #[test]
    fn ctm_scale_inf_falls_back_to_one() {
        let ctm = [f64::INFINITY, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(ctm_scale(&ctm), 1.0);
    }

    #[test]
    fn eval_stitching_wrong_bounds_count_falls_back() {
        // 2 sub-functions need 1 bound value; giving 0 should not panic.
        let doc = lopdf::Document::new();
        let mut fn_dict = lopdf::Dictionary::new();
        fn_dict.set("FunctionType", lopdf::Object::Integer(3));
        fn_dict.set(
            "Domain",
            lopdf::Object::Array(vec![lopdf::Object::Real(0.0), lopdf::Object::Real(1.0)]),
        );
        // Two sub-functions.
        let sub = {
            let mut d = lopdf::Dictionary::new();
            d.set("FunctionType", lopdf::Object::Integer(2));
            d.set("C0", lopdf::Object::Array(vec![lopdf::Object::Real(0.0)]));
            d.set("C1", lopdf::Object::Array(vec![lopdf::Object::Real(1.0)]));
            d.set("N", lopdf::Object::Real(1.0));
            d.set(
                "Domain",
                lopdf::Object::Array(vec![lopdf::Object::Real(0.0), lopdf::Object::Real(1.0)]),
            );
            lopdf::Object::Dictionary(d)
        };
        fn_dict.set("Functions", lopdf::Object::Array(vec![sub.clone(), sub]));
        // Wrong: 0 bounds values for 2 sub-functions.
        fn_dict.set("Bounds", lopdf::Object::Array(vec![]));
        fn_dict.set(
            "Encode",
            lopdf::Object::Array(vec![
                lopdf::Object::Real(0.0),
                lopdf::Object::Real(1.0),
                lopdf::Object::Real(0.0),
                lopdf::Object::Real(1.0),
            ]),
        );
        // Must not panic; result is Some (fallback to first sub-function).
        let result = eval_stitching(&doc, &fn_dict, 0.5, 1);
        assert!(result.is_some());
    }
}
