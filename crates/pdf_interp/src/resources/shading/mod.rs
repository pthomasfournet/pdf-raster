//! PDF shading resource resolution (Types 1–7).
//!
//! [`resolve_shading`] looks up a named `Shading` resource and returns a
//! [`ShadingResult`] which is either a [`raster::pipe::Pattern`] (for smooth
//! gradient types 1–3) or a pre-decoded triangle mesh (for mesh types 4–7).
//!
//! # Supported shading types
//!
//! | PDF type | Name | Support |
//! |---|---|---|
//! | 1 | Function-based | yes (pre-sampled 64×64 grid) |
//! | 2 | Axial (linear) | yes |
//! | 3 | Radial | yes |
//! | 4 | Free-form Gouraud triangle mesh | yes |
//! | 5 | Lattice-form Gouraud mesh | yes |
//! | 6 | Coons patch mesh | yes |
//! | 7 | Tensor-product patch mesh | yes |
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

mod function;
mod patch;

use std::sync::Arc;

use color::convert::gray_to_u8;
use pdf::{Dictionary, Document, Object, Stream};
use raster::pipe::Pattern;
use raster::shading::axial::AxialPattern;
use raster::shading::function::FunctionPattern;
use raster::shading::gouraud::GouraudVertex;
use raster::shading::radial::RadialPattern;

use super::dict_ext::DictExt;
use super::image::{ImageColorSpace, cs_to_image_color_space};
use super::resolve_dict;
use function::{eval_function, read_f64_array, read_fn_domain};
use patch::{decode_type6_mesh, decode_type7_mesh};

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
    let res = resolve_dict(doc, resource_context_dict.get(b"Resources")?)?;
    let sh_res = resolve_dict(doc, res.get(b"Shading")?)?;
    let sh_obj = sh_res.get(name)?;

    // Shading types 2–3 are plain dicts; types 4–7 are streams.
    // Try stream first (stream has a dict too), then fall back to plain dict.
    // Owned Dictionary because the new pdf crate hands back Arc<Object> from
    // get_object — we can't keep a borrow into a temporary Arc.
    let (sh_dict, stream_data): (Dictionary, Option<Vec<u8>>) = match sh_obj {
        Object::Stream(s) => (s.dict.clone(), stream_content(s)),
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            match referent.as_ref() {
                Object::Stream(s) => (s.dict.clone(), stream_content(s)),
                obj => (resolve_dict(doc, obj)?, None),
            }
        }
        obj => (resolve_dict(doc, obj)?, None),
    };

    let shading_type = sh_dict.get_i64(b"ShadingType")?;

    let cs_obj = sh_dict.get(b"ColorSpace")?;
    let cs = cs_to_image_color_space(doc, cs_obj);
    let n_channels = cs_channel_count(cs);

    match shading_type {
        1 => resolve_function_based(doc, &sh_dict, cs, n_channels, ctm, page_h)
            .map(|(p, bb)| ShadingResult::Pattern(p, bb)),
        2 => resolve_axial(doc, &sh_dict, cs, n_channels, ctm, page_h)
            .map(|(p, bb)| ShadingResult::Pattern(p, bb)),
        3 => resolve_radial(doc, &sh_dict, cs, n_channels, ctm, page_h)
            .map(|(p, bb)| ShadingResult::Pattern(p, bb)),
        4 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type4_mesh(
                &sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        5 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type5_mesh(
                &sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        6 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type6_mesh(
                &sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        7 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type7_mesh(
                &sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        other => {
            log::warn!("shading: ShadingType {other} not yet implemented — skipping sh operator");
            None
        }
    }
}

/// Decompress a stream, logging a warning on failure.
fn stream_content(s: &Stream) -> Option<Vec<u8>> {
    match s.decompressed_content() {
        Ok(data) => Some(data),
        Err(e) => {
            log::warn!("shading: failed to decompress stream — {e}");
            None
        }
    }
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
    let (t0, t1) = read_fn_domain(sh);

    // Extend: [extend_start, extend_end].
    let (ext_s, ext_e) = read_extend(sh);

    // Evaluate the function at t0 and t1 to get the two colours.
    let fn_obj = sh.get(b"Function")?;
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

    let (t0, t1) = read_fn_domain(sh);
    let (ext_s, ext_e) = read_extend(sh);

    let fn_obj = sh.get(b"Function")?;
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
    // PDF spec requires r ≥ 0; take abs to handle malformed input gracefully.
    let dr0 = coords[2].abs() * scale;
    let dr1 = coords[5].abs() * scale;

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

// ── Function-based (Type 1) ───────────────────────────────────────────────────

/// Number of grid samples per axis for the pre-sampled Type 1 shading grid.
const TYPE1_GRID: usize = 64;

#[expect(
    clippy::too_many_lines,
    reason = "linear resolver: parse → sample grid → build closure; not decomposable without splitting state"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "CTM components a–f and grid dims g, n follow PDF spec notation"
)]
fn resolve_function_based(
    doc: &Document,
    sh: &Dictionary,
    cs: ImageColorSpace,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<(Box<dyn Pattern + Send + Sync>, [f64; 4])> {
    // Domain: [xmin, xmax, ymin, ymax] in user space. PDF spec default is [0,1,0,1].
    let domain = read_f64_array(sh, b"Domain", 4).unwrap_or_else(|| vec![0.0, 1.0, 0.0, 1.0]);
    let (xd0, xd1, yd0, yd1) = (domain[0], domain[1], domain[2], domain[3]);

    if !xd0.is_finite() || !xd1.is_finite() || !yd0.is_finite() || !yd1.is_finite() {
        log::warn!("shading/type1: non-finite Domain — skipping");
        return None;
    }
    if xd0 >= xd1 || yd0 >= yd1 {
        log::warn!("shading/type1: degenerate Domain [{xd0},{xd1}]×[{yd0},{yd1}] — skipping");
        return None;
    }

    let fn_obj = sh.get(b"Function")?;

    // Invert the CTM so fill_span can map device pixels → user space.
    let [a, b, c, d, e, f] = *ctm;
    let det = a.mul_add(d, -(b * c));
    if !det.is_finite() || det.abs() < f64::EPSILON {
        log::warn!("shading/type1: degenerate CTM (det={det:.3e}) — skipping");
        return None;
    }
    let inv: [f64; 6] = [
        d / det,
        -b / det,
        -c / det,
        a / det,
        c.mul_add(f, -(d * e)) / det,
        b.mul_add(e, -(a * f)) / det,
    ];

    // Pre-sample the function on a TYPE1_GRID×TYPE1_GRID grid in user space.
    // All document references are resolved here; the closure captures only the
    // finished grid (Arc<[…]>) and scalar domain bounds — no lifetime issues.
    let g = TYPE1_GRID;
    #[expect(
        clippy::cast_precision_loss,
        reason = "g=64; g-1=63 fits exactly in f64 mantissa"
    )]
    let gf = (g - 1) as f64;
    let mut grid: Vec<[u8; 3]> = Vec::with_capacity(g * g);
    // Type 1 functions take a 2D input [ux, uy]. eval_function handles only
    // 1D types (0, 2, 3) and takes a single scalar; we pass ux (the x
    // coordinate). The y coordinate is unused until full 2D sampled-function
    // support is added as follow-on work. Every row gets the same colours.
    let row_colors: Vec<[u8; 3]> = (0..g)
        .map(|col| {
            #[expect(clippy::cast_precision_loss, reason = "col < 64; exact in f64")]
            let ux = (col as f64 / gf).mul_add(xd1 - xd0, xd0);
            let channels = eval_function(doc, fn_obj, ux, n).unwrap_or_else(|| vec![0.0; n]);
            cs_to_rgb(cs, &channels)
        })
        .collect();
    for _ in 0..g {
        grid.extend_from_slice(&row_colors);
    }
    let grid: Arc<[[u8; 3]]> = Arc::from(grid.into_boxed_slice());

    // Device-space bbox: transform the four domain corners.
    // If BBox is present it clips the shading in user space, so intersect.
    let (mut bx0, mut bx1, mut by0, mut by1) = (xd0, xd1, yd0, yd1);
    if let Some(bb) = read_f64_array(sh, b"BBox", 4) {
        if bb.iter().all(|v| v.is_finite()) {
            bx0 = bx0.max(bb[0].min(bb[2]));
            bx1 = bx1.min(bb[0].max(bb[2]));
            by0 = by0.max(bb[1].min(bb[3]));
            by1 = by1.min(bb[1].max(bb[3]));
        } else {
            log::warn!("shading/type1: non-finite BBox — ignoring BBox");
        }
    }
    // After intersection the clip region may be empty; that's fine — the
    // rasterizer will simply paint nothing inside an empty bbox.
    let mut dev_pts = [0f64; 8];
    let corners = [(bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)];
    for (i, (ux, uy)) in corners.into_iter().enumerate() {
        let (dx, dy) = transform_point(ctm, ux, uy, page_h);
        dev_pts[i * 2] = dx;
        dev_pts[i * 2 + 1] = dy;
    }
    let bbox = bbox_from_coords(&dev_pts);

    let pattern = FunctionPattern::new(move |px, py| {
        // Undo the y-flip applied by transform_point, then apply inverse CTM.
        let py_pdf = page_h - py;
        let ux = inv[0].mul_add(px, inv[2] * py_pdf) + inv[4];
        let uy = inv[1].mul_add(px, inv[3] * py_pdf) + inv[5];

        // Normalise to [0, 1] within the domain; clamp to handle rounding at edges.
        let tx = ((ux - xd0) / (xd1 - xd0)).clamp(0.0, 1.0);
        let ty = ((uy - yd0) / (yd1 - yd0)).clamp(0.0, 1.0);

        // Bilinear interpolation over the pre-sampled grid.
        // tx/ty ∈ [0,1] and gf > 0, so fx/fy ∈ [0, gf]; floor ∈ [0, g-1].
        let fx = tx * gf;
        let fy = ty * gf;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fx/fy clamped to [0,gf=63]; floor fits in usize"
        )]
        let (c0, r0) = (fx.floor() as usize, fy.floor() as usize);
        let c1 = (c0 + 1).min(g - 1);
        let r1 = (r0 + 1).min(g - 1);
        #[expect(clippy::cast_precision_loss, reason = "c0/r0 ≤ g-1=63; exact in f64")]
        let (wc, wr) = (fx - c0 as f64, fy - r0 as f64); // weights ∈ [0,1)

        let s00 = grid[r0 * g + c0];
        let s10 = grid[r0 * g + c1];
        let s01 = grid[r1 * g + c0];
        let s11 = grid[r1 * g + c1];

        // Lerp a single u8 channel; t ∈ [0,1), result ∈ [0,255].
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a,b ∈ [0,255] and t ∈ [0,1); result rounds into [0,255]"
        )]
        let lerp_ch = |a: u8, b: u8, t: f64| -> u8 {
            t.mul_add(f64::from(b) - f64::from(a), f64::from(a)).round() as u8
        };

        let top = [
            lerp_ch(s00[0], s10[0], wc),
            lerp_ch(s00[1], s10[1], wc),
            lerp_ch(s00[2], s10[2], wc),
        ];
        let bot = [
            lerp_ch(s01[0], s11[0], wc),
            lerp_ch(s01[1], s11[1], wc),
            lerp_ch(s01[2], s11[2], wc),
        ];
        [
            lerp_ch(top[0], bot[0], wr),
            lerp_ch(top[1], bot[1], wr),
            lerp_ch(top[2], bot[2], wr),
        ]
    });

    Some((Box::new(pattern), bbox))
}

// ── Type 4 — Free-form Gouraud triangle mesh ──────────────────────────────────

/// Bit-stream reader for packed mesh vertex data (PDF §8.7.4.5).
///
/// PDF mesh shading streams pack coordinates and colour components into
/// consecutive bit fields (MSB first, byte-aligned at the record level only
/// for the flag byte; the rest are truly packed).  `read_bits` pulls `n`
/// bits (1–32) from the stream, returning `None` on EOF.
pub(super) struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bits already fetched from `data` but not yet consumed.
    /// Right-justified (LSB-aligned): live bits occupy `bit_buf[bits_in_buf-1..0]`.
    /// Extraction uses `(bit_buf >> bits_in_buf) & mask` to pull from the top of
    /// the live region.  64-bit so `n=32` never overflows the accumulator.
    bit_buf: u64,
    bits_in_buf: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over the given byte slice.
    pub(super) const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Read exactly `n` bits (1–32) and return as a right-justified `u32`.
    ///
    /// Returns `None` when the stream is exhausted before `n` bits are available.
    /// Callers must not pass `n == 0` (asserted in debug builds; release returns 0).
    pub(super) fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!(
            (1..=32).contains(&n),
            "read_bits: n={n} out of range 1..=32"
        );
        if n == 0 || n > 32 {
            return Some(0);
        }
        while self.bits_in_buf < n {
            if self.byte_pos >= self.data.len() {
                return None;
            }
            // u64 accumulator: safe for n=32 even when bits_in_buf > 0 on entry,
            // since 32 + 32 (max pre-existing) = 64 ≤ u64::BITS.
            self.bit_buf = (self.bit_buf << 8) | u64::from(self.data[self.byte_pos]);
            self.byte_pos += 1;
            self.bits_in_buf += 8;
        }
        self.bits_in_buf -= n;
        let mask = if n == 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << n) - 1
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "mask is at most u32::MAX; the extracted value fits u32"
        )]
        Some(((self.bit_buf >> self.bits_in_buf) & mask) as u32)
    }
}

/// PDF-legal values for `BitsPerCoordinate` and `BitsPerComponent` (Table 84/85).
pub(super) const VALID_BITS: &[u8] = &[1, 2, 4, 8, 12, 16, 24, 32];

/// Validate a PDF integer field against a set of legal bit-width values.
///
/// Shared by `parse_bits_per_coord`, `parse_bits_per_comp`, and `parse_bits_per_flag`.
/// Logs a warning and returns `None` if the value is absent or not in `valid_set`.
pub(super) fn parse_bits_field(
    sh: &Dictionary,
    key: &[u8],
    valid_set: &[u8],
    field_name: &str,
    tag: &str,
) -> Option<u8> {
    let v = sh.get_i64(key)?;
    #[expect(
        clippy::option_if_let_else,
        reason = "else branch has a side-effect (log::warn); map_or_else would be less clear"
    )]
    if let Some(bits) = u8::try_from(v).ok().filter(|b| valid_set.contains(b)) {
        Some(bits)
    } else {
        log::warn!(
            "shading/{tag}: {field_name}={v} is not a legal PDF value \
             (must be one of {valid_set:?}) — skipping"
        );
        None
    }
}

/// Validate and extract `BitsPerCoordinate` from a mesh shading dictionary.
///
/// Logs a warning and returns `None` if the key is absent or the value is not
/// one of the PDF-legal values {1, 2, 4, 8, 12, 16, 24, 32}.
pub(super) fn parse_bits_per_coord(sh: &Dictionary, tag: &str) -> Option<u8> {
    parse_bits_field(
        sh,
        b"BitsPerCoordinate",
        VALID_BITS,
        "BitsPerCoordinate",
        tag,
    )
}

/// Validate and extract `BitsPerComponent` from a mesh shading dictionary.
///
/// Logs a warning and returns `None` if the key is absent or the value is not
/// one of the PDF-legal values {1, 2, 4, 8, 12, 16, 24, 32}.
pub(super) fn parse_bits_per_comp(sh: &Dictionary, tag: &str) -> Option<u8> {
    parse_bits_field(sh, b"BitsPerComponent", VALID_BITS, "BitsPerComponent", tag)
}

/// Decode a Type 4 (free-form Gouraud triangle mesh) shading stream.
///
/// Stream layout: for each record, 8-bit flag followed by `BitsPerCoordinate`
/// bits each for X and Y, then `BitsPerComponent` bits per colour channel.
///
/// Flag semantics (PDF §8.7.4.5.3):
/// - 0 → new triangle: 3 new vertices A, B, C
/// - 1 → extend from edge (B, C) of previous triangle: 1 new vertex D → triangle (B, C, D)
/// - 2 → extend from edge (C, A) of previous triangle: 1 new vertex D → triangle (C, A, D)
fn decode_type4_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type4") else {
        return vec![];
    };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type4") else {
        return vec![];
    };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    // The three vertices of the most recently emitted triangle, for fan continuation.
    let mut prev: Option<[GouraudVertex; 3]> = None;

    // Flag is 8 bits; read_bits(8) guarantees value ≤ 255.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "read_bits(8) returns at most 255"
    )]
    while let Some(flag_raw) = reader.read_bits(8) {
        let flag = flag_raw as u8;

        let n_new: usize = if flag == 0 { 3 } else { 1 };

        // Read new vertices; bail out of the entire mesh on any truncation.
        let mut new_verts = [GouraudVertex {
            x: 0.0,
            y: 0.0,
            color: [0; 3],
        }; 3];
        for slot in new_verts.iter_mut().take(n_new) {
            let Some(raw_x) = reader.read_bits(bpc) else {
                return triangles;
            };
            let Some(raw_y) = reader.read_bits(bpc) else {
                return triangles;
            };
            let mut channels = [0u32; 4]; // max 4 channels (CMYK)
            for ch in channels.iter_mut().take(n_channels) {
                let Some(raw_c) = reader.read_bits(bpcomp) else {
                    return triangles;
                };
                *ch = raw_c;
            }
            *slot = decode_vertex(
                raw_x,
                raw_y,
                &channels[..n_channels],
                bpc,
                bpcomp,
                &decode,
                cs,
                ctm,
                page_h,
            );
        }

        // Build the triangle from flag + previous vertices.
        let tri: Option<[GouraudVertex; 3]> = match (flag, prev) {
            // Flag 0: independent triangle from 3 fresh vertices.
            (0, _) => Some([new_verts[0], new_verts[1], new_verts[2]]),
            // Flag 1: extend from edge (B=prev[1], C=prev[2]) with new vertex D.
            (1, Some(p)) => Some([p[1], p[2], new_verts[0]]),
            // Flag 2: extend from edge (C=prev[2], A=prev[0]) with new vertex D.
            (2, Some(p)) => Some([p[2], p[0], new_verts[0]]),
            // Flag 1/2 with no prior triangle, or unknown flag: skip record.
            (f, _) => {
                if f > 2 {
                    log::debug!("shading/type4: unknown flag {f} — skipping record");
                }
                None
            }
        };

        if let Some(t) = tri {
            prev = Some(t);
            triangles.push(t);
        }
    }

    triangles
}

// ── Type 5 — Lattice-form Gouraud mesh ───────────────────────────────────────

/// Decode a Type 5 (lattice-form Gouraud) shading stream.
///
/// `VerticesPerRow` vertices wide; vertices are laid out in row-major order.
/// Adjacent rows of vertices form a grid; each 2×2 quad is split into two
/// triangles (top-left + bottom-right of the quad diagonal).
fn decode_type5_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type5") else {
        return vec![];
    };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type5") else {
        return vec![];
    };
    let Some(verts_per_row) = sh.get_i64(b"VerticesPerRow") else {
        log::warn!("shading/type5: missing VerticesPerRow — skipping");
        return vec![];
    };
    if verts_per_row < 2 {
        log::warn!("shading/type5: VerticesPerRow={verts_per_row} < 2 — skipping");
        return vec![];
    }
    let Ok(vpr) = usize::try_from(verts_per_row) else {
        log::warn!("shading/type5: VerticesPerRow={verts_per_row} overflows usize — skipping");
        return vec![];
    };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    // Keep only two rows at a time; tessellate as we go to avoid storing the entire mesh.
    let mut prev_row: Vec<GouraudVertex> = Vec::with_capacity(vpr);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();

    'rows: loop {
        let mut row = Vec::with_capacity(vpr);
        for _ in 0..vpr {
            let Some(raw_x) = reader.read_bits(bpc) else {
                break 'rows;
            };
            let Some(raw_y) = reader.read_bits(bpc) else {
                break 'rows;
            };
            let mut channels = [0u32; 4];
            for ch in channels.iter_mut().take(n_channels) {
                let Some(raw_c) = reader.read_bits(bpcomp) else {
                    break 'rows;
                };
                *ch = raw_c;
            }
            row.push(decode_vertex(
                raw_x,
                raw_y,
                &channels[..n_channels],
                bpc,
                bpcomp,
                &decode,
                cs,
                ctm,
                page_h,
            ));
        }

        if row.len() < vpr {
            // Truncated row at EOF — discard partial row.
            break;
        }

        if !prev_row.is_empty() {
            // Tessellate the strip between prev_row and row.
            for col in 0..vpr - 1 {
                // Quad corners: TL=prev[col], TR=prev[col+1], BL=row[col], BR=row[col+1].
                // Two triangles: (TL, TR, BL) and (TR, BR, BL).
                triangles.push([prev_row[col], prev_row[col + 1], row[col]]);
                triangles.push([prev_row[col + 1], row[col + 1], row[col]]);
            }
        }

        prev_row = row;
    }

    triangles
}

// ── Mesh vertex helpers ───────────────────────────────────────────────────────

/// Build the Decode range array for Type 4/5 streams.
///
/// Layout: `[xmin, xmax, ymin, ymax, c0min, c0max, …]` (PDF Table 84).
/// If the dict lacks `Decode` or has the wrong number of entries, a
/// warning is logged and the identity range `[0, 1]` is used for each slot.
pub(super) fn read_decode_array(sh: &Dictionary, n_channels: usize) -> Vec<f64> {
    let expected = 4 + n_channels * 2; // coords (x,y) + per-channel pairs

    let parsed: Option<Vec<f64>> = sh
        .get(b"Decode")
        .and_then(|o| o.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|o| match o {
                    Object::Real(r) => Some(f64::from(*r)),
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "Decode values are small PDF numbers; i64→f64 precision loss is negligible"
                    )]
                    Object::Integer(i) => Some(*i as f64),
                    _ => None,
                })
                .collect::<Vec<_>>()
        });

    match parsed {
        Some(v) if v.len() == expected => v,
        Some(v) => {
            log::warn!(
                "shading: Decode array has {} entries, expected {expected} — using [0,1] defaults",
                v.len()
            );
            default_decode(n_channels)
        }
        None => {
            log::warn!(
                "shading: Decode array missing (required for mesh shadings) — using [0,1] defaults"
            );
            default_decode(n_channels)
        }
    }
}

/// Identity decode range: `[0,1]` for x, `[0,1]` for y, `[0,1]` per channel.
fn default_decode(n_channels: usize) -> Vec<f64> {
    let mut d = vec![0.0_f64, 1.0, 0.0, 1.0];
    d.extend(std::iter::repeat_n([0.0_f64, 1.0], n_channels).flatten());
    d
}

/// Decode one packed vertex into a [`GouraudVertex`] in device space.
///
/// Raw bit values are mapped from `[0, 2^bits − 1]` to the user-space range
/// given by the `Decode` array, then transformed through the CTM to device space.
#[expect(
    clippy::too_many_arguments,
    reason = "all args are required by the PDF mesh vertex decode formula; no sensible grouping"
)]
pub(super) fn decode_vertex(
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
    // bits_per_coord/comp ∈ VALID_BITS ⊆ [1,32]; shift is safe.
    // Cast from u64 to f64: max_coord ≤ 2^32−1 < 2^53, so conversion is exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "max_coord ≤ 2^32−1; f64 mantissa is 52 bits"
    )]
    let max_coord = ((1u64 << bits_per_coord) - 1) as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "max_comp ≤ 2^32−1; f64 mantissa is 52 bits"
    )]
    let max_comp = ((1u64 << bits_per_comp) - 1) as f64;

    let x_min = decode.first().copied().unwrap_or(0.0);
    let x_max = decode.get(1).copied().unwrap_or(1.0);
    let y_min = decode.get(2).copied().unwrap_or(0.0);
    let y_max = decode.get(3).copied().unwrap_or(1.0);

    // u32 → f64 is always lossless (u32 < 2^32 ≤ 2^53 mantissa bits).
    let ux = (f64::from(raw_x) / max_coord).mul_add(x_max - x_min, x_min);
    let uy = (f64::from(raw_y) / max_coord).mul_add(y_max - y_min, y_min);

    let channels: Vec<f64> = raw_channels
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            let c_min = decode.get(4 + i * 2).copied().unwrap_or(0.0);
            let c_max = decode.get(4 + i * 2 + 1).copied().unwrap_or(1.0);
            (f64::from(raw) / max_comp).mul_add(c_max - c_min, c_min)
        })
        .collect();

    let color = cs_to_rgb(cs, &channels);
    let (dx, dy) = transform_point(ctm, ux, uy, page_h);

    GouraudVertex {
        x: dx,
        y: dy,
        color,
    }
}

// ── Colour conversion helpers ─────────────────────────────────────────────────

/// Convert a function output (values in `[0, 1]`) to an sRGB triple `[u8; 3]`.
///
/// # Tolerance for malformed inputs
///
/// - Missing channels (`channels.len() < cs_channel_count(cs)`) default to `0.0`
///   — encountered with malformed shading functions that return short arrays.
/// - Values outside `[0, 1]` are clamped by [`gray_to_u8`].
/// - `NaN` channel values map to `0` (via [`gray_to_u8`]'s float-to-int
///   saturation).
pub(super) fn cs_to_rgb(cs: ImageColorSpace, channels: &[f64]) -> [u8; 3] {
    let scale = |i: usize| gray_to_u8(channels.get(i).copied().unwrap_or(0.0));
    match cs {
        ImageColorSpace::Gray | ImageColorSpace::Mask => {
            let g = scale(0);
            [g, g, g]
        }
        ImageColorSpace::Rgb => [scale(0), scale(1), scale(2)],
    }
}

/// Return the number of colour channels for a colour space.
pub(super) const fn cs_channel_count(cs: ImageColorSpace) -> usize {
    match cs {
        ImageColorSpace::Gray | ImageColorSpace::Mask => 1,
        ImageColorSpace::Rgb => 3,
    }
}

// ── Coordinate transforms ─────────────────────────────────────────────────────

/// Transform a user-space point through the CTM and y-flip to device space.
///
/// Applies the affine CTM `[a b c d e f]` to `(x, y)` then flips the y axis
/// with `page_h − dy`.  This differs from `gstate::ctm_transform` which does
/// not include the y-flip.
pub(super) fn transform_point(ctm: &[f64; 6], x: f64, y: f64, page_h: f64) -> (f64, f64) {
    let dx = ctm[0].mul_add(x, ctm[2] * y) + ctm[4];
    let dy = ctm[1].mul_add(x, ctm[3] * y) + ctm[5];
    (dx, page_h - dy)
}

/// Compute the uniform scale factor of the CTM (geometric mean of x and y scales).
///
/// Falls back to `1.0` when the CTM contains non-finite entries or produces a
/// non-finite scale (e.g., from an extreme anisotropic or degenerate matrix).
pub(super) fn ctm_scale(ctm: &[f64; 6]) -> f64 {
    let sx = ctm[0].hypot(ctm[1]);
    let sy = ctm[2].hypot(ctm[3]);
    let result = (sx * sy).sqrt();
    if result.is_finite() { result } else { 1.0 }
}

/// Compute a bounding box from a flat list of interleaved `[x, y, x, y, …]` values.
pub(super) fn bbox_from_coords(pts: &[f64]) -> [f64; 4] {
    debug_assert!(pts.len() >= 4 && pts.len().is_multiple_of(2));
    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for chunk in pts.chunks_exact(2) {
        xmin = xmin.min(chunk[0]);
        xmax = xmax.max(chunk[0]);
        ymin = ymin.min(chunk[1]);
        ymax = ymax.max(chunk[1]);
    }
    [xmin, ymin, xmax, ymax]
}

/// Read the `Extend` array `[extend_start, extend_end]`; both default to `false`.
fn read_extend(sh: &Dictionary) -> (bool, bool) {
    let arr = sh.get(b"Extend").and_then(|o| o.as_array());
    let Some(a) = arr.filter(|a| a.len() >= 2) else {
        return (false, false);
    };
    let b = |i: usize| a.get(i).and_then(pdf::Object::as_bool).unwrap_or(false);
    (b(0), b(1))
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

    /// Short channel slices must default missing channels to 0, not panic —
    /// malformed PDF shading functions sometimes return undersized arrays.
    #[test]
    fn cs_to_rgb_short_channel_slice_defaults_to_zero() {
        assert_eq!(cs_to_rgb(ImageColorSpace::Rgb, &[]), [0, 0, 0]);
        assert_eq!(cs_to_rgb(ImageColorSpace::Rgb, &[1.0]), [255, 0, 0]);
        assert_eq!(cs_to_rgb(ImageColorSpace::Gray, &[]), [0, 0, 0]);
    }

    /// Over-range values are clamped via `gray_to_u8`; NaN saturates to 0.
    #[test]
    fn cs_to_rgb_out_of_range_and_nan() {
        assert_eq!(
            cs_to_rgb(ImageColorSpace::Rgb, &[2.0, -0.5, 0.5]),
            [255, 0, 128]
        );
        assert_eq!(
            cs_to_rgb(ImageColorSpace::Rgb, &[f64::NAN, 1.0, 1.0]),
            [0, 255, 255]
        );
    }

    #[test]
    fn cs_channel_count_values() {
        assert_eq!(cs_channel_count(ImageColorSpace::Gray), 1);
        assert_eq!(cs_channel_count(ImageColorSpace::Rgb), 3);
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
}
