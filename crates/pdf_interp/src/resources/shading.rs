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
        6 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type6_mesh(
                sh_dict, &data, cs, n_channels, ctm, page_h,
            )))
        }
        7 => {
            let data = stream_data?;
            Some(ShadingResult::Mesh(decode_type7_mesh(
                sh_dict, &data, cs, n_channels, ctm, page_h,
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

/// Bit-stream reader for packed mesh vertex data (PDF §8.7.4.5).
///
/// PDF mesh shading streams pack coordinates and colour components into
/// consecutive bit fields (MSB first, byte-aligned at the record level only
/// for the flag byte; the rest are truly packed).  `read_bits` pulls `n`
/// bits (1–32) from the stream, returning `None` on EOF.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bits already fetched from `data` but not yet consumed (MSB-justified).
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

    /// Read exactly `n` bits (1–32) and return as a right-justified `u32`.
    ///
    /// Returns `None` when the stream is exhausted before `n` bits are available.
    /// Callers must not pass `n == 0` (asserted in debug builds; release returns 0).
    fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!(n >= 1 && n <= 32, "read_bits: n={n} out of range 1..=32");
        if n == 0 || n > 32 {
            return Some(0);
        }
        while self.bits_in_buf < n {
            if self.byte_pos >= self.data.len() {
                return None;
            }
            self.bit_buf = (self.bit_buf << 8) | u32::from(self.data[self.byte_pos]);
            self.byte_pos += 1;
            self.bits_in_buf += 8;
        }
        self.bits_in_buf -= n;
        // Safe: n ∈ [1,32], bits_in_buf ≥ 0 after subtraction.
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        Some((self.bit_buf >> self.bits_in_buf) & mask)
    }
}

/// PDF-legal values for `BitsPerCoordinate` and `BitsPerComponent` (Table 84/85).
const VALID_BITS: &[u8] = &[1, 2, 4, 8, 12, 16, 24, 32];

/// Validate and extract `BitsPerCoordinate` from a mesh shading dictionary.
///
/// Logs a warning and returns `None` if the key is absent or the value is not
/// one of the PDF-legal values {1, 2, 4, 8, 12, 16, 24, 32}.
fn parse_bits_per_coord(sh: &Dictionary, tag: &str) -> Option<u8> {
    let v = sh.get_i64(b"BitsPerCoordinate")?;
    match u8::try_from(v).ok().filter(|b| VALID_BITS.contains(b)) {
        Some(bits) => Some(bits),
        None => {
            log::warn!("shading/{tag}: BitsPerCoordinate={v} is not a legal PDF value (must be one of {VALID_BITS:?}) — skipping");
            None
        }
    }
}

/// Validate and extract `BitsPerComponent` from a mesh shading dictionary.
///
/// Logs a warning and returns `None` if the key is absent or the value is not
/// one of the PDF-legal values {1, 2, 4, 8, 12, 16, 24, 32}.
fn parse_bits_per_comp(sh: &Dictionary, tag: &str) -> Option<u8> {
    let v = sh.get_i64(b"BitsPerComponent")?;
    match u8::try_from(v).ok().filter(|b| VALID_BITS.contains(b)) {
        Some(bits) => Some(bits),
        None => {
            log::warn!("shading/{tag}: BitsPerComponent={v} is not a legal PDF value (must be one of {VALID_BITS:?}) — skipping");
            None
        }
    }
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
    let Some(bpc) = parse_bits_per_coord(sh, "type4") else { return vec![] };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type4") else { return vec![] };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    // The three vertices of the most recently emitted triangle, for fan continuation.
    let mut prev: Option<[GouraudVertex; 3]> = None;

    loop {
        // Flag is 8 bits; read_bits(8) guarantees value ≤ 255.
        let Some(flag_raw) = reader.read_bits(8) else { break };
        #[expect(clippy::cast_possible_truncation, reason = "read_bits(8) returns at most 255")]
        let flag = flag_raw as u8;

        let n_new: usize = if flag == 0 { 3 } else { 1 };

        // Read new vertices; bail out of the entire mesh on any truncation.
        let mut new_verts = [GouraudVertex { x: 0.0, y: 0.0, color: [0; 3] }; 3];
        for slot in new_verts.iter_mut().take(n_new) {
            let Some(raw_x) = reader.read_bits(bpc) else { return triangles };
            let Some(raw_y) = reader.read_bits(bpc) else { return triangles };
            let mut channels = [0u32; 4]; // max 4 channels (CMYK)
            for ch in channels.iter_mut().take(n_channels) {
                let Some(raw_c) = reader.read_bits(bpcomp) else { return triangles };
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
    let Some(bpc) = parse_bits_per_coord(sh, "type5") else { return vec![] };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type5") else { return vec![] };
    let Some(verts_per_row) = sh.get_i64(b"VerticesPerRow") else {
        log::warn!("shading/type5: missing VerticesPerRow — skipping");
        return vec![];
    };
    if verts_per_row < 2 {
        log::warn!("shading/type5: VerticesPerRow={verts_per_row} < 2 — skipping");
        return vec![];
    }
    #[expect(clippy::cast_sign_loss, reason = "guarded ≥ 2 above")]
    let vpr = verts_per_row as usize;
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    // Keep only two rows at a time; tessellate as we go to avoid storing the entire mesh.
    let mut prev_row: Vec<GouraudVertex> = Vec::with_capacity(vpr);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();

    loop {
        let mut row = Vec::with_capacity(vpr);
        for _ in 0..vpr {
            let Some(raw_x) = reader.read_bits(bpc) else { break };
            let Some(raw_y) = reader.read_bits(bpc) else { break };
            let mut channels = [0u32; 4];
            let mut ok = true;
            for ch in channels.iter_mut().take(n_channels) {
                if let Some(raw_c) = reader.read_bits(bpcomp) {
                    *ch = raw_c;
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
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

// ── Type 6 — Coons patch mesh ─────────────────────────────────────────────────
// ── Type 7 — Tensor-product patch mesh ───────────────────────────────────────

/// A bicubic patch: 4×4 control points in device space + 2×2 corner colours.
///
/// Grid convention (matches poppler GfxPatch / PDF §8.7.4.5.6):
/// - `xy[row][col]` where row 0 = "first" edge (u=0), row 3 = "last" edge (u=1).
/// - `color[u_corner][v_corner]` where 0 = min, 1 = max.
///
/// Coordinates are already in device space (CTM applied + y-flipped).
#[derive(Clone, Copy)]
struct Patch {
    xy: [[[f64; 2]; 4]; 4],
    color: [[[u8; 3]; 2]; 2],
}

impl Patch {
    fn zero() -> Self {
        Self {
            xy: [[[0.0; 2]; 4]; 4],
            color: [[[0u8; 3]; 2]; 2],
        }
    }
}

/// Colour mixing threshold for subdivision: stop when all corner pairs differ
/// by less than this per component (matches poppler `patchColorDelta`).
const COLOR_DELTA: f64 = 3.0 / 255.0;
/// Maximum adaptive subdivision depth (matches poppler `patchMaxDepth`).
const MAX_PATCH_DEPTH: u8 = 6;

/// PDF-legal values for `BitsPerFlag` in patch mesh shadings (Table 86).
const VALID_FLAG_BITS: &[u8] = &[2, 4, 8];

// ── Patch math helpers ────────────────────────────────────────────────────────

/// Componentwise midpoint of two device-space points — exact, never overflows.
#[inline]
fn mid2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])]
}

/// Componentwise midpoint of two sRGB corner colours.
#[inline]
fn mid_color(a: [u8; 3], b: [u8; 3]) -> [u8; 3] {
    // lerp_u8(a, b, 128) = round((a + b) / 2) — matches poppler's bilinear interp.
    use color::convert::lerp_u8;
    [lerp_u8(a[0], b[0], 128), lerp_u8(a[1], b[1], 128), lerp_u8(a[2], b[2], 128)]
}

/// Maximum per-component colour difference across all 4 corner pairs of the patch.
/// Used as the subdivision-stop criterion.
fn color_delta(p: &Patch) -> f64 {
    let corners = [
        p.color[0][0],
        p.color[0][1],
        p.color[1][0],
        p.color[1][1],
    ];
    let mut max_d = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            for ch in 0..3 {
                let d = f64::from(corners[i][ch].abs_diff(corners[j][ch])) / 255.0;
                if d > max_d {
                    max_d = d;
                }
            }
        }
    }
    max_d
}

// ── Patch subdivision ─────────────────────────────────────────────────────────

/// Apply one step of de Casteljau to a row of 4 points, splitting at t=0.5.
///
/// Returns `(left, right)` where each is a 4-point row.
#[inline]
fn casteljau_row(row: [[f64; 2]; 4]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let m01 = mid2(row[0], row[1]);
    let m12 = mid2(row[1], row[2]);
    let m23 = mid2(row[2], row[3]);
    let m012 = mid2(m01, m12);
    let m123 = mid2(m12, m23);
    let m0123 = mid2(m012, m123);
    ([row[0], m01, m012, m0123], [m0123, m123, m23, row[3]])
}

/// Split `p` at u=0.5 — applies de Casteljau along columns (u direction).
///
/// Returns `(lo_u, hi_u)` where `lo_u` covers u∈[0,0.5] and `hi_u` u∈[0.5,1].
fn split_patch_u(p: &Patch) -> (Patch, Patch) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for col in 0..4 {
        let col_pts = [p.xy[0][col], p.xy[1][col], p.xy[2][col], p.xy[3][col]];
        let (l, r) = casteljau_row(col_pts);
        for row in 0..4 {
            lo.xy[row][col] = l[row];
            hi.xy[row][col] = r[row];
        }
    }
    // Interpolate corner colours along u.
    let c_mid_v0 = mid_color(p.color[0][0], p.color[1][0]);
    let c_mid_v1 = mid_color(p.color[0][1], p.color[1][1]);
    lo.color[0] = p.color[0];
    lo.color[1] = [c_mid_v0, c_mid_v1];
    hi.color[0] = [c_mid_v0, c_mid_v1];
    hi.color[1] = p.color[1];
    (lo, hi)
}

/// Split `p` at v=0.5 — applies de Casteljau along rows (v direction).
///
/// Returns `(lo_v, hi_v)` where `lo_v` covers v∈[0,0.5] and `hi_v` v∈[0.5,1].
fn split_patch_v(p: &Patch) -> (Patch, Patch) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for row in 0..4 {
        let (l, r) = casteljau_row(p.xy[row]);
        lo.xy[row] = l;
        hi.xy[row] = r;
    }
    // Interpolate corner colours along v.
    let c_mid_u0 = mid_color(p.color[0][0], p.color[0][1]);
    let c_mid_u1 = mid_color(p.color[1][0], p.color[1][1]);
    lo.color[0][0] = p.color[0][0];
    lo.color[0][1] = c_mid_u0;
    lo.color[1][0] = p.color[1][0];
    lo.color[1][1] = c_mid_u1;
    hi.color[0][0] = c_mid_u0;
    hi.color[0][1] = p.color[0][1];
    hi.color[1][0] = c_mid_u1;
    hi.color[1][1] = p.color[1][1];
    (lo, hi)
}

// ── Tessellation ──────────────────────────────────────────────────────────────

/// Extract a corner device-space vertex from a patch.
///
/// `u=0` → row 0; `u=1` → row 3. `v=0` → col 0; `v=1` → col 3.
#[inline]
fn corner_vertex(p: &Patch, u: usize, v: usize) -> GouraudVertex {
    let row = u * 3;
    let col = v * 3;
    GouraudVertex {
        x: p.xy[row][col][0],
        y: p.xy[row][col][1],
        color: p.color[u][v],
    }
}

/// Adaptively subdivide `patch` into Gouraud triangles and append them to `out`.
///
/// Subdivision stops when all corner colours are within [`COLOR_DELTA`] of each
/// other or [`MAX_PATCH_DEPTH`] is reached, matching poppler's strategy.
/// Uses an explicit work stack (no recursion) to avoid stack overflow.
fn tessellate_patch(patch: Patch, out: &mut Vec<[GouraudVertex; 3]>) {
    // Pre-allocate the work stack; worst case is 4 * MAX_PATCH_DEPTH live entries.
    let mut stack: Vec<(Patch, u8)> = Vec::with_capacity(4 * MAX_PATCH_DEPTH as usize);
    stack.push((patch, 0));

    while let Some((p, depth)) = stack.pop() {
        if depth >= MAX_PATCH_DEPTH || color_delta(&p) <= COLOR_DELTA {
            // Leaf: emit 2 triangles covering the patch quad from its 4 corners.
            let v00 = corner_vertex(&p, 0, 0);
            let v01 = corner_vertex(&p, 0, 1);
            let v10 = corner_vertex(&p, 1, 0);
            let v11 = corner_vertex(&p, 1, 1);
            out.push([v00, v01, v10]);
            out.push([v01, v11, v10]);
        } else {
            // Split into 4 sub-patches and push them.
            let (pu0, pu1) = split_patch_u(&p);
            let (p00, p01) = split_patch_v(&pu0);
            let (p10, p11) = split_patch_v(&pu1);
            let nd = depth + 1;
            stack.push((p00, nd));
            stack.push((p01, nd));
            stack.push((p10, nd));
            stack.push((p11, nd));
        }
    }
}

// ── Stream I/O helpers ────────────────────────────────────────────────────────

/// Read `n_pts` coordinate pairs from the bit stream and convert to device space.
///
/// Returns `None` on stream truncation (all previously decoded triangles are
/// returned by the caller rather than discarding the whole shading).
fn read_patch_points(
    reader: &mut BitReader<'_>,
    n_pts: usize,
    bpc: u8,
    decode: &[f64],
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Vec<[f64; 2]>> {
    let max_coord = ((1u64 << bpc) - 1) as f64;
    let x_min = decode.first().copied().unwrap_or(0.0);
    let x_max = decode.get(1).copied().unwrap_or(1.0);
    let y_min = decode.get(2).copied().unwrap_or(0.0);
    let y_max = decode.get(3).copied().unwrap_or(1.0);

    let mut pts = Vec::with_capacity(n_pts);
    for _ in 0..n_pts {
        let raw_x = reader.read_bits(bpc)?;
        let raw_y = reader.read_bits(bpc)?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "raw bit fields ≤ 32 bits; f64 mantissa exact for u32 values"
        )]
        let ux = x_min + (raw_x as f64 / max_coord) * (x_max - x_min);
        #[expect(
            clippy::cast_precision_loss,
            reason = "same as ux — u32 → f64 exact up to 2^53"
        )]
        let uy = y_min + (raw_y as f64 / max_coord) * (y_max - y_min);
        let (dx, dy) = transform_point(ctm, ux, uy, page_h);
        pts.push([dx, dy]);
    }
    Some(pts)
}

/// Read `n_colors` corner colours from the bit stream.
///
/// Returns `None` on stream truncation.
fn read_patch_colors(
    reader: &mut BitReader<'_>,
    n_colors: usize,
    n_channels: usize,
    bpcomp: u8,
    decode: &[f64],
    cs: ImageColorSpace,
) -> Option<Vec<[u8; 3]>> {
    let max_comp = ((1u64 << bpcomp) - 1) as f64;
    let mut colors = Vec::with_capacity(n_colors);
    for _ in 0..n_colors {
        let mut raw = [0u32; 4];
        for ch in raw.iter_mut().take(n_channels) {
            *ch = reader.read_bits(bpcomp)?;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "u32 → f64 exact up to 2^53; colour component values are small"
        )]
        let channels: Vec<f64> = raw[..n_channels]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let c_min = decode.get(4 + i * 2).copied().unwrap_or(0.0);
                let c_max = decode.get(4 + i * 2 + 1).copied().unwrap_or(1.0);
                c_min + (v as f64 / max_comp) * (c_max - c_min)
            })
            .collect();
        colors.push(cs_to_rgb(cs, &channels));
    }
    Some(colors)
}

/// Validate and extract `BitsPerFlag` from a patch mesh shading dictionary.
fn parse_bits_per_flag(sh: &Dictionary, tag: &str) -> Option<u8> {
    let v = sh.get_i64(b"BitsPerFlag")?;
    match u8::try_from(v).ok().filter(|b| VALID_FLAG_BITS.contains(b)) {
        Some(bits) => Some(bits),
        None => {
            log::warn!(
                "shading/{tag}: BitsPerFlag={v} is not a legal PDF value \
                 (must be one of {VALID_FLAG_BITS:?}) — skipping"
            );
            None
        }
    }
}

// ── Patch assembly — Type 6 ───────────────────────────────────────────────────

/// Assign the 12 boundary stream points into the Type 6 4×4 grid for flag=0.
///
/// Stream order traces the boundary clockwise:
/// top row (left→right), right col (down), bottom row (right→left), left col (up).
/// Interior points `[1][1]`, `[1][2]`, `[2][1]`, `[2][2]` are computed from the boundary.
fn build_patch_type6(pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 12);
    debug_assert_eq!(colors.len(), 4);
    let mut p = Patch::zero();

    // Boundary point assignment (PDF §8.7.4.5.6 Table 87, flag=0).
    p.xy[0][0] = pts[0];  p.xy[0][1] = pts[1];  p.xy[0][2] = pts[2];  p.xy[0][3] = pts[3];
    p.xy[1][3] = pts[4];  p.xy[2][3] = pts[5];
    p.xy[3][3] = pts[6];  p.xy[3][2] = pts[7];  p.xy[3][1] = pts[8];  p.xy[3][0] = pts[9];
    p.xy[2][0] = pts[10]; p.xy[1][0] = pts[11];

    // Corner colours: [0][0]=c0, [0][1]=c1, [1][1]=c2, [1][0]=c3.
    p.color[0][0] = colors[0];
    p.color[0][1] = colors[1];
    p.color[1][1] = colors[2];
    p.color[1][0] = colors[3];

    fill_type6_interior(&mut p);
    p
}

/// Derive the 4 interior control points of a Coons patch from its boundary.
///
/// Formula from poppler `GfxState.cc:5374–5385`, which implements the standard
/// Coons patch interior derivation (same formula for x and y independently).
fn fill_type6_interior(p: &mut Patch) {
    for k in 0..2usize {
        // Snapshot boundary values before writing interior — avoids borrow conflict.
        let [p00, p01, p02, p03] = [p.xy[0][0][k], p.xy[0][1][k], p.xy[0][2][k], p.xy[0][3][k]];
        let [p10, p13]            = [p.xy[1][0][k], p.xy[1][3][k]];
        let [p20, p23]            = [p.xy[2][0][k], p.xy[2][3][k]];
        let [p30, p31, p32, p33] = [p.xy[3][0][k], p.xy[3][1][k], p.xy[3][2][k], p.xy[3][3][k]];
        p.xy[1][1][k] = (-4.0*p00 + 6.0*(p01+p10) - 2.0*(p03+p30) + 3.0*(p31+p13) - p33) / 9.0;
        p.xy[1][2][k] = (-4.0*p03 + 6.0*(p02+p13) - 2.0*(p00+p33) + 3.0*(p32+p10) - p30) / 9.0;
        p.xy[2][1][k] = (-4.0*p30 + 6.0*(p31+p20) - 2.0*(p33+p00) + 3.0*(p01+p23) - p03) / 9.0;
        p.xy[2][2][k] = (-4.0*p33 + 6.0*(p32+p23) - 2.0*(p30+p03) + 3.0*(p02+p20) - p00) / 9.0;
    }
}

/// Apply shared-edge continuation for Type 6, flags 1–3.
///
/// Copies 4 control points and 2 corner colours from `prev`, then places the
/// 8 new boundary points from `pts` and 2 new colours from `colors`.
/// Interior points are derived after all boundary points are placed.
fn apply_flag_type6(flag: u8, prev: &Patch, pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 8);
    debug_assert_eq!(colors.len(), 2);
    let mut p = Patch::zero();

    match flag {
        1 => {
            // Shared: prev right edge → new left edge (col 0 of new patch = col 3 of prev).
            for row in 0..4 { p.xy[row][0] = prev.xy[row][3]; }
            p.color[0][0] = prev.color[0][1];
            p.color[1][0] = prev.color[1][1];
            // New boundary: right edge (col 3), then bottom (row 3) and top (row 0) partial.
            p.xy[0][1] = pts[0]; p.xy[0][2] = pts[1]; p.xy[0][3] = pts[2];
            p.xy[1][3] = pts[3]; p.xy[2][3] = pts[4];
            p.xy[3][3] = pts[5]; p.xy[3][2] = pts[6]; p.xy[3][1] = pts[7];
            p.color[0][1] = colors[0];
            p.color[1][1] = colors[1];
        }
        2 => {
            // Shared: prev bottom edge → new top edge (row 0 of new = row 3 of prev).
            for col in 0..4 { p.xy[0][col] = prev.xy[3][col]; }
            p.color[0][0] = prev.color[1][0];
            p.color[0][1] = prev.color[1][1];
            // New boundary.
            p.xy[1][3] = pts[0]; p.xy[2][3] = pts[1];
            p.xy[3][3] = pts[2]; p.xy[3][2] = pts[3]; p.xy[3][1] = pts[4]; p.xy[3][0] = pts[5];
            p.xy[2][0] = pts[6]; p.xy[1][0] = pts[7];
            p.color[1][1] = colors[0];
            p.color[1][0] = colors[1];
        }
        3 => {
            // Shared: prev left edge → new right edge (col 3 of new = col 0 of prev).
            for row in 0..4 { p.xy[row][3] = prev.xy[row][0]; }
            p.color[0][1] = prev.color[0][0];
            p.color[1][1] = prev.color[1][0];
            // New boundary: left edge (col 0), then top (row 0) and bottom (row 3) partial.
            p.xy[3][2] = pts[0]; p.xy[3][1] = pts[1]; p.xy[3][0] = pts[2];
            p.xy[2][0] = pts[3]; p.xy[1][0] = pts[4];
            p.xy[0][0] = pts[5]; p.xy[0][1] = pts[6]; p.xy[0][2] = pts[7];
            p.color[0][0] = colors[0];
            p.color[1][0] = colors[1];
        }
        _ => unreachable!("apply_flag_type6 called with flag={flag}; only 1–3 are valid"),
    }

    fill_type6_interior(&mut p);
    p
}

// ── Patch assembly — Type 7 ───────────────────────────────────────────────────

/// Assign the 16 stream points into the Type 7 4×4 grid for flag=0.
///
/// Unlike Type 6, all 16 points (12 boundary + 4 interior) are read from the stream.
/// Stream order: top row, right col, bottom row (reversed), left col (reversed),
/// then 4 interior points in a specific order.
fn build_patch_type7(pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 16);
    debug_assert_eq!(colors.len(), 4);
    let mut p = Patch::zero();

    // Boundary (same layout as Type 6).
    p.xy[0][0] = pts[0];  p.xy[0][1] = pts[1];  p.xy[0][2] = pts[2];  p.xy[0][3] = pts[3];
    p.xy[1][3] = pts[4];  p.xy[2][3] = pts[5];
    p.xy[3][3] = pts[6];  p.xy[3][2] = pts[7];  p.xy[3][1] = pts[8];  p.xy[3][0] = pts[9];
    p.xy[2][0] = pts[10]; p.xy[1][0] = pts[11];

    // Interior points (PDF §8.7.4.5.7 Table 89, flag=0, pts 12–15).
    p.xy[1][1] = pts[12]; p.xy[1][2] = pts[13];
    p.xy[2][2] = pts[14]; p.xy[2][1] = pts[15];

    // Corner colours.
    p.color[0][0] = colors[0];
    p.color[0][1] = colors[1];
    p.color[1][1] = colors[2];
    p.color[1][0] = colors[3];
    p
}

/// Apply shared-edge continuation for Type 7, flags 1–3.
///
/// For Type 7, the 4 interior control points are always read from the stream
/// (as part of the 12 new points), never copied from the previous patch.
fn apply_flag_type7(flag: u8, prev: &Patch, pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 12);
    debug_assert_eq!(colors.len(), 2);
    let mut p = Patch::zero();

    match flag {
        1 => {
            // Shared: prev right edge → new left edge.
            for row in 0..4 { p.xy[row][0] = prev.xy[row][3]; }
            p.color[0][0] = prev.color[0][1];
            p.color[1][0] = prev.color[1][1];
            // 8 new boundary points + 4 interior.
            p.xy[0][1] = pts[0]; p.xy[0][2] = pts[1]; p.xy[0][3] = pts[2];
            p.xy[1][3] = pts[3]; p.xy[2][3] = pts[4];
            p.xy[3][3] = pts[5]; p.xy[3][2] = pts[6]; p.xy[3][1] = pts[7];
            p.xy[1][1] = pts[8]; p.xy[1][2] = pts[9]; p.xy[2][2] = pts[10]; p.xy[2][1] = pts[11];
            p.color[0][1] = colors[0];
            p.color[1][1] = colors[1];
        }
        2 => {
            // Shared: prev bottom edge → new top edge.
            for col in 0..4 { p.xy[0][col] = prev.xy[3][col]; }
            p.color[0][0] = prev.color[1][0];
            p.color[0][1] = prev.color[1][1];
            p.xy[1][3] = pts[0]; p.xy[2][3] = pts[1];
            p.xy[3][3] = pts[2]; p.xy[3][2] = pts[3]; p.xy[3][1] = pts[4]; p.xy[3][0] = pts[5];
            p.xy[2][0] = pts[6]; p.xy[1][0] = pts[7];
            p.xy[1][1] = pts[8]; p.xy[1][2] = pts[9]; p.xy[2][2] = pts[10]; p.xy[2][1] = pts[11];
            p.color[1][1] = colors[0];
            p.color[1][0] = colors[1];
        }
        3 => {
            // Shared: prev left edge → new right edge.
            for row in 0..4 { p.xy[row][3] = prev.xy[row][0]; }
            p.color[0][1] = prev.color[0][0];
            p.color[1][1] = prev.color[1][0];
            p.xy[3][2] = pts[0]; p.xy[3][1] = pts[1]; p.xy[3][0] = pts[2];
            p.xy[2][0] = pts[3]; p.xy[1][0] = pts[4];
            p.xy[0][0] = pts[5]; p.xy[0][1] = pts[6]; p.xy[0][2] = pts[7];
            p.xy[1][1] = pts[8]; p.xy[1][2] = pts[9]; p.xy[2][2] = pts[10]; p.xy[2][1] = pts[11];
            p.color[0][0] = colors[0];
            p.color[1][0] = colors[1];
        }
        _ => unreachable!("apply_flag_type7 called with flag={flag}; only 1–3 are valid"),
    }
    p
}

// ── Top-level mesh decoders ───────────────────────────────────────────────────

/// Decode a Type 6 (Coons patch mesh) shading stream.
///
/// Each record: `BitsPerFlag`-bit flag, then 12 (flag=0) or 8 (flag≥1) coordinate
/// pairs at `BitsPerCoordinate` bits each, then 4 (flag=0) or 2 (flag≥1) colours
/// at `BitsPerComponent` bits per channel. Interior control points are derived from
/// the boundary using the standard Coons formula. Each patch is tessellated into
/// Gouraud triangles via adaptive subdivision.
fn decode_type6_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type6") else { return vec![] };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type6") else { return vec![] };
    let Some(bpf) = parse_bits_per_flag(sh, "type6") else { return vec![] };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    let mut prev: Option<Patch> = None;

    loop {
        let Some(flag_raw) = reader.read_bits(bpf) else { break };
        #[expect(clippy::cast_possible_truncation, reason = "bpf ∈ {2,4,8}; value fits u8")]
        let flag = flag_raw as u8;

        let (n_pts, n_colors): (usize, usize) = if flag == 0 { (12, 4) } else { (8, 2) };

        let Some(pts) = read_patch_points(&mut reader, n_pts, bpc, &decode, ctm, page_h)
        else { break };
        let Some(colors) = read_patch_colors(&mut reader, n_colors, n_channels, bpcomp, &decode, cs)
        else { break };

        let patch = match (flag, &prev) {
            (0, _) => build_patch_type6(&pts, &colors),
            (1..=3, Some(p)) => apply_flag_type6(flag, p, &pts, &colors),
            (f, None) => {
                log::debug!("shading/type6: flag={f} with no previous patch — skipping record");
                continue;
            }
            (f, _) => {
                log::debug!("shading/type6: unknown flag {f} — skipping record");
                continue;
            }
        };

        tessellate_patch(patch, &mut triangles);
        prev = Some(patch);
    }

    triangles
}

/// Decode a Type 7 (tensor-product patch mesh) shading stream.
///
/// Like Type 6 but all 16 control points per patch are read directly from the
/// stream (4 interior points are not derived). Flag semantics and colour handling
/// are identical to Type 6.
fn decode_type7_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type7") else { return vec![] };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type7") else { return vec![] };
    let Some(bpf) = parse_bits_per_flag(sh, "type7") else { return vec![] };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    let mut prev: Option<Patch> = None;

    loop {
        let Some(flag_raw) = reader.read_bits(bpf) else { break };
        #[expect(clippy::cast_possible_truncation, reason = "bpf ∈ {2,4,8}; value fits u8")]
        let flag = flag_raw as u8;

        let (n_pts, n_colors): (usize, usize) = if flag == 0 { (16, 4) } else { (12, 2) };

        let Some(pts) = read_patch_points(&mut reader, n_pts, bpc, &decode, ctm, page_h)
        else { break };
        let Some(colors) = read_patch_colors(&mut reader, n_colors, n_channels, bpcomp, &decode, cs)
        else { break };

        let patch = match (flag, &prev) {
            (0, _) => build_patch_type7(&pts, &colors),
            (1..=3, Some(p)) => apply_flag_type7(flag, p, &pts, &colors),
            (f, None) => {
                log::debug!("shading/type7: flag={f} with no previous patch — skipping record");
                continue;
            }
            (f, _) => {
                log::debug!("shading/type7: unknown flag {f} — skipping record");
                continue;
            }
        };

        tessellate_patch(patch, &mut triangles);
        prev = Some(patch);
    }

    triangles
}

// ── Mesh vertex helpers ───────────────────────────────────────────────────────

/// Build the Decode range array for Type 4/5 streams.
///
/// Layout: `[xmin, xmax, ymin, ymax, c0min, c0max, …]` (PDF Table 84).
/// If the dict lacks `Decode` or has the wrong number of entries, a
/// warning is logged and the identity range `[0, 1]` is used for each slot.
fn read_decode_array(sh: &Dictionary, n_channels: usize) -> Vec<f64> {
    let expected = 4 + n_channels * 2; // coords (x,y) + per-channel pairs

    let parsed: Option<Vec<f64>> = sh
        .get(b"Decode")
        .ok()
        .and_then(|o| o.as_array().ok())
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
                "shading: Decode array has {} entries, expected {expected} — using defaults",
                v.len()
            );
            default_decode(n_channels)
        }
        None => default_decode(n_channels),
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
    // bits_per_coord and bits_per_comp are validated ∈ VALID_BITS before this
    // function is called, so both are in [1, 32] — the shift is safe.
    #[expect(
        clippy::cast_precision_loss,
        reason = "bit fields are at most 32 bits; f64 has 53-bit mantissa, exact for all u32 values"
    )]
    let max_coord = ((1u64 << bits_per_coord) - 1) as f64; // always > 0 (bits ≥ 1)
    #[expect(
        clippy::cast_precision_loss,
        reason = "same as max_coord — bits_per_comp ∈ [1,32]"
    )]
    let max_comp = ((1u64 << bits_per_comp) - 1) as f64; // always > 0

    let x_min = decode.first().copied().unwrap_or(0.0);
    let x_max = decode.get(1).copied().unwrap_or(1.0);
    let y_min = decode.get(2).copied().unwrap_or(0.0);
    let y_max = decode.get(3).copied().unwrap_or(1.0);

    #[expect(clippy::cast_precision_loss, reason = "u32 → f64 exact up to 2^53")]
    let ux = x_min + (raw_x as f64 / max_coord) * (x_max - x_min);
    #[expect(clippy::cast_precision_loss, reason = "u32 → f64 exact up to 2^53")]
    let uy = y_min + (raw_y as f64 / max_coord) * (y_max - y_min);

    #[expect(clippy::cast_precision_loss, reason = "u32 → f64 exact up to 2^53")]
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

    // ── Patch subdivision tests ───────────────────────────────────────────────

    fn flat_patch(x0: f64, y0: f64, x1: f64, y1: f64, c00: [u8;3], c01: [u8;3], c10: [u8;3], c11: [u8;3]) -> super::Patch {
        // Bilinear interpolation of control points across the 4×4 grid.
        let mut p = super::Patch::zero();
        for r in 0..4usize {
            let tr = r as f64 / 3.0;
            for c in 0..4usize {
                let tc = c as f64 / 3.0;
                p.xy[r][c][0] = x0 + (x1 - x0) * tc;
                p.xy[r][c][1] = y0 + (y1 - y0) * tr;
            }
        }
        p.color[0][0] = c00; p.color[0][1] = c01;
        p.color[1][0] = c10; p.color[1][1] = c11;
        p
    }

    #[test]
    fn split_patch_u_flat_corners_unchanged() {
        let p = flat_patch(0.0, 0.0, 6.0, 4.0,
            [0,0,0], [255,0,0], [0,255,0], [0,0,255]);
        let (left, right) = super::split_patch_u(&p);
        // Top-left corner should be exactly preserved in both halves.
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[3][0], p.xy[3][0]);
        // Split point on left side should be at midpoint in y.
        let mid_y = (p.xy[0][0][1] + p.xy[3][0][1]) / 2.0;
        assert!((left.xy[3][0][1] - mid_y).abs() < 1e-9,
            "left bottom y={} expected={}", left.xy[3][0][1], mid_y);
        assert!((right.xy[0][0][1] - mid_y).abs() < 1e-9,
            "right top y={} expected={}", right.xy[0][0][1], mid_y);
        // Colors: left's u=1 edge should be midpoint of original u edges.
        for ch in 0..3 {
            let expected = ((p.color[0][0][ch] as u16 + p.color[1][0][ch] as u16) / 2) as u8;
            assert!((left.color[1][0][ch] as i16 - expected as i16).abs() <= 1,
                "color mismatch ch={ch}");
        }
    }

    #[test]
    fn split_patch_v_flat_corners_unchanged() {
        let p = flat_patch(0.0, 0.0, 8.0, 6.0,
            [10,20,30], [200,100,50], [30,60,90], [180,90,45]);
        let (left, right) = super::split_patch_v(&p);
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[0][3], p.xy[0][3]);
        let mid_x = (p.xy[0][0][0] + p.xy[0][3][0]) / 2.0;
        assert!((left.xy[0][3][0] - mid_x).abs() < 1e-9);
        assert!((right.xy[0][0][0] - mid_x).abs() < 1e-9);
    }

    #[test]
    fn color_delta_uniform_is_zero() {
        let mut p = super::Patch::zero();
        let c = [128u8, 64, 32];
        p.color[0][0] = c; p.color[0][1] = c; p.color[1][0] = c; p.color[1][1] = c;
        assert_eq!(super::color_delta(&p), 0.0);
    }

    #[test]
    fn color_delta_max_diff_is_one() {
        let mut p = super::Patch::zero();
        p.color[0][0] = [0,0,0]; p.color[0][1] = [255,0,0];
        p.color[1][0] = [0,0,0]; p.color[1][1] = [0,0,0];
        let d = super::color_delta(&p);
        assert!((d - 1.0).abs() < 1e-9, "expected 1.0, got {d}");
    }

    #[test]
    fn tessellate_patch_uniform_emits_two_triangles() {
        // A patch where all corners have the same colour → depth=0 leaf immediately.
        let p = flat_patch(0.0, 0.0, 1.0, 1.0,
            [128,128,128], [128,128,128], [128,128,128], [128,128,128]);
        let mut out = Vec::new();
        super::tessellate_patch(p, &mut out);
        assert_eq!(out.len(), 2, "expected exactly 2 triangles for uniform-colour patch");
    }

    #[test]
    fn build_patch_type6_interior_formula() {
        // Use a simple square boundary with known analytic interior.
        // p[0][0]=(0,0)  p[0][1]=(1,0) p[0][2]=(2,0) p[0][3]=(3,0)
        // p[1][3]=(3,1)  p[2][3]=(3,2)
        // p[3][3]=(3,3)  p[3][2]=(2,3) p[3][1]=(1,3) p[3][0]=(0,3)
        // p[2][0]=(0,2)  p[1][0]=(0,1)
        //
        // For a uniform-spaced rectilinear grid the Coons interior formula collapses to
        // simple bilinear: p[1][1]=(1,1), p[1][2]=(2,1), p[2][1]=(1,2), p[2][2]=(2,2).
        let pts: Vec<[f64;2]> = vec![
            [0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0], // p[0][0..3]
            [3.0,1.0],[3.0,2.0],                      // p[1][3], p[2][3]
            [3.0,3.0],[2.0,3.0],[1.0,3.0],[0.0,3.0], // p[3][3..0]
            [0.0,2.0],[0.0,1.0],                       // p[2][0], p[1][0]
        ];
        let colors: Vec<[u8;3]> = vec![[0,0,0],[255,0,0],[0,255,0],[0,0,255]];
        let p = super::build_patch_type6(&pts, &colors);
        for (r, c, ex, ey) in [(1,1,1.0,1.0),(1,2,2.0,1.0),(2,1,1.0,2.0),(2,2,2.0,2.0)] {
            assert!((p.xy[r][c][0]-ex).abs() < 1e-9 && (p.xy[r][c][1]-ey).abs() < 1e-9,
                "interior[{r}][{c}] = {:?} expected ({ex},{ey})", p.xy[r][c]);
        }
    }
}
