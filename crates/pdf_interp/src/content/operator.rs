//! Typed PDF content stream operator representation.
//!
//! [`decode`] consumes the pending operand stack and produces a strongly-typed
//! [`Operator`] value ready for dispatch to the renderer.

use super::operands::{
    drain_numbers, pop_f64, pop_i32, pop_matrix, pop_name, pop_number_array, pop_string, pop2,
    pop3, pop4,
};
use super::tokenizer::Token;

/// A decoded PDF content stream operator with its operands.
///
/// Variants follow the PDF 1.7 / ISO 32000-2 operator table (§8.2).
/// Operand order matches the PDF spec (left-to-right as written in the stream).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Operator {
    // ── Graphics state ────────────────────────────────────────────────────────
    /// `q` — push graphics state.
    Save,
    /// `Q` — pop graphics state.
    Restore,
    /// `cm a b c d e f` — concatenate matrix to CTM.
    ConcatMatrix([f64; 6]),
    /// `w` — set line width.
    SetLineWidth(f64),
    /// `J` — set line cap style (0 = butt, 1 = round, 2 = projecting square).
    SetLineCap(i32),
    /// `j` — set line join style (0 = miter, 1 = round, 2 = bevel).
    SetLineJoin(i32),
    /// `M` — set miter limit.
    SetMiterLimit(f64),
    /// `d [array] phase` — set dash pattern.
    SetDash {
        /// Dash array lengths in user space units.
        dashes: Vec<f64>,
        /// Phase offset into the dash array.
        phase: f64,
    },
    /// `ri name` — set colour rendering intent.
    SetRenderingIntent(Vec<u8>),
    /// `i` — set flatness tolerance (0–100).
    SetFlatness(f64),
    /// `gs name` — apply named extended graphics state dictionary entry.
    SetExtGState(Vec<u8>),

    // ── Path construction ─────────────────────────────────────────────────────
    /// `m x y` — begin new subpath at (x, y).
    MoveTo(f64, f64),
    /// `l x y` — append straight line to (x, y).
    LineTo(f64, f64),
    /// `c x1 y1 x2 y2 x3 y3` — append cubic Bézier (both control points explicit).
    CurveTo(f64, f64, f64, f64, f64, f64),
    /// `v x2 y2 x3 y3` — cubic Bézier; first control point = current point.
    CurveToV(f64, f64, f64, f64),
    /// `y x1 y1 x3 y3` — cubic Bézier; second control point = endpoint.
    CurveToY(f64, f64, f64, f64),
    /// `h` — close current subpath with a straight line to the start point.
    ClosePath,
    /// `re x y w h` — append rectangle as a complete subpath.
    Rectangle(f64, f64, f64, f64),

    // ── Path painting ─────────────────────────────────────────────────────────
    /// `S` — stroke path; clear path.
    Stroke,
    /// `s` — close and stroke; clear path.
    CloseStroke,
    /// `f` / `F` — fill path (non-zero winding rule); clear path.
    Fill,
    /// `f*` — fill path (even-odd rule); clear path.
    FillEvenOdd,
    /// `B` — fill then stroke (non-zero); clear path.
    FillStroke,
    /// `B*` — fill then stroke (even-odd); clear path.
    FillStrokeEvenOdd,
    /// `b` — close, fill, stroke (non-zero); clear path.
    CloseFillStroke,
    /// `b*` — close, fill, stroke (even-odd); clear path.
    CloseFillStrokeEvenOdd,
    /// `n` — end path without painting (used after clipping operators).
    EndPath,

    // ── Clipping ─────────────────────────────────────────────────────────────
    /// `W` — intersect clip path with current path (non-zero winding rule).
    Clip,
    /// `W*` — intersect clip path with current path (even-odd rule).
    ClipEvenOdd,

    // ── Text objects ──────────────────────────────────────────────────────────
    /// `BT` — begin text object; initialise text matrix to identity.
    BeginText,
    /// `ET` — end text object; discard text matrix.
    EndText,

    // ── Text state ────────────────────────────────────────────────────────────
    /// `Tf name size` — set font resource and size.
    SetFont {
        /// Font resource name (key into the page Font resource dict).
        name: Vec<u8>,
        /// Font size in text space units.
        size: f64,
    },
    /// `Tc` — set character spacing (added to advance after each glyph).
    SetCharSpacing(f64),
    /// `Tw` — set word spacing (added to advance after ASCII SPACE, 0x20).
    SetWordSpacing(f64),
    /// `Tz` — set horizontal text scaling (percentage of normal width).
    SetHorizScaling(f64),
    /// `TL` — set text leading (vertical advance for `T*`, `'`, `"`).
    SetLeading(f64),
    /// `Ts` — set text rise (vertical offset from the baseline).
    SetTextRise(f64),
    /// `Tr` — set text rendering mode (0–7; controls fill/stroke/clip).
    SetTextRenderMode(i32),

    // ── Text positioning ──────────────────────────────────────────────────────
    /// `Td tx ty` — move text position by (tx, ty).
    TextMove(f64, f64),
    /// `TD tx ty` — move text position by (tx, ty) and set leading to −ty.
    TextMoveSetLeading(f64, f64),
    /// `Tm a b c d e f` — set the text matrix and text line matrix directly.
    SetTextMatrix([f64; 6]),
    /// `T*` — move to start of next line (equivalent to `0 −TL Td`).
    NextLine,

    // ── Text showing ──────────────────────────────────────────────────────────
    /// `Tj string` — show a string of glyphs.
    ShowText(Vec<u8>),
    /// `TJ array` — show glyphs with individual horizontal adjustments.
    ShowTextArray(Vec<TextArrayElement>),
    /// `' string` — move to next line, then show string (equivalent to `T* Tj`).
    MoveNextLineShow(Vec<u8>),
    /// `" aw ac string` — set word/char spacing, move to next line, show string.
    MoveNextLineShowSpaced {
        /// Word spacing to set before showing.
        aw: f64,
        /// Character spacing to set before showing.
        ac: f64,
        /// String to show.
        text: Vec<u8>,
    },

    // ── Colour ────────────────────────────────────────────────────────────────
    /// `cs name` — set fill colour space.
    SetFillColorSpace(Vec<u8>),
    /// `CS name` — set stroke colour space.
    SetStrokeColorSpace(Vec<u8>),
    /// `sc` / `scn` — set fill colour (components in current fill space).
    SetFillColor(Vec<f64>),
    /// `SC` / `SCN` — set stroke colour (components in current stroke space).
    SetStrokeColor(Vec<f64>),
    /// `g` — set fill colour to gray level (DeviceGray shorthand).
    SetFillGray(f64),
    /// `G` — set stroke colour to gray level (DeviceGray shorthand).
    SetStrokeGray(f64),
    /// `rg r g b` — set fill colour (DeviceRGB shorthand).
    SetFillRgb(f64, f64, f64),
    /// `RG r g b` — set stroke colour (DeviceRGB shorthand).
    SetStrokeRgb(f64, f64, f64),
    /// `k c m y k` — set fill colour (DeviceCMYK shorthand).
    SetFillCmyk(f64, f64, f64, f64),
    /// `K c m y k` — set stroke colour (DeviceCMYK shorthand).
    SetStrokeCmyk(f64, f64, f64, f64),

    // ── XObjects & images ─────────────────────────────────────────────────────
    /// `Do name` — paint an XObject (image or form XObject).
    PaintXObject(Vec<u8>),
    /// `BI … ID … EI` — paint an inline image.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between `BI` and `ID`).
        params: Vec<u8>,
        /// Raw image data bytes (between `ID` and `EI`).
        data: Vec<u8>,
    },

    // ── Shading ───────────────────────────────────────────────────────────────
    /// `sh name` — paint a shading pattern.
    PaintShading(Vec<u8>),

    // ── Marked content (no rendering effect) ─────────────────────────────────
    /// `BMC` / `BDC` / `EMC` / `MP` / `DP` — marked-content operators.
    MarkedContent,

    // ── Compatibility sections ────────────────────────────────────────────────
    /// `BX` / `EX` — begin/end compatibility section (contents ignored).
    CompatibilitySection,

    // ── Unknown / unimplemented ───────────────────────────────────────────────
    /// An operator keyword the decoder does not recognise.
    Unknown(Vec<u8>),
}

/// Element of a `TJ` text-showing array.
#[derive(Debug, Clone)]
pub enum TextArrayElement {
    /// A string of glyph codes to render.
    Text(Vec<u8>),
    /// Horizontal adjustment in thousandths of a text-space unit.
    /// Negative values move the text origin to the right.
    Offset(f64),
}

/// Decode a pending operand stack and operator keyword into an [`Operator`].
///
/// `operands` is always cleared on return, regardless of success or failure,
/// so the stack is always empty and ready for the next operator.
#[expect(
    clippy::too_many_lines,
    reason = "operator dispatch table — splitting adds no clarity"
)]
pub fn decode(op: &[u8], operands: &mut Vec<Token<'_>>) -> Operator {
    let result = match op {
        // ── Graphics state ────────────────────────────────────────────────────
        b"q" => Operator::Save,
        b"Q" => Operator::Restore,
        b"cm" => Operator::ConcatMatrix(pop_matrix(operands)),
        b"w" => Operator::SetLineWidth(pop_f64(operands)),
        b"J" => Operator::SetLineCap(pop_i32(operands)),
        b"j" => Operator::SetLineJoin(pop_i32(operands)),
        b"M" => Operator::SetMiterLimit(pop_f64(operands)),
        b"d" => {
            // Stream order: `[dash array] phase d`
            // Stack order (LIFO): phase is on top, array below.
            let phase = pop_f64(operands);
            let dashes = pop_number_array(operands);
            Operator::SetDash { dashes, phase }
        }
        b"ri" => Operator::SetRenderingIntent(pop_name(operands)),
        b"i" => Operator::SetFlatness(pop_f64(operands)),
        b"gs" => Operator::SetExtGState(pop_name(operands)),

        // ── Path construction ─────────────────────────────────────────────────
        b"m" => {
            let (x, y) = pop2(operands);
            Operator::MoveTo(x, y)
        }
        b"l" => {
            let (x, y) = pop2(operands);
            Operator::LineTo(x, y)
        }
        b"c" => {
            let (x1, y1, x2, y2, x3, y3) = {
                let f = pop_f64(operands);
                let e = pop_f64(operands);
                let d = pop_f64(operands);
                let c = pop_f64(operands);
                let b = pop_f64(operands);
                let a = pop_f64(operands);
                (a, b, c, d, e, f)
            };
            Operator::CurveTo(x1, y1, x2, y2, x3, y3)
        }
        b"v" => {
            let (x2, y2, x3, y3) = pop4(operands);
            Operator::CurveToV(x2, y2, x3, y3)
        }
        b"y" => {
            let (x1, y1, x3, y3) = pop4(operands);
            Operator::CurveToY(x1, y1, x3, y3)
        }
        b"h" => Operator::ClosePath,
        b"re" => {
            let (x, y, w, h) = pop4(operands);
            Operator::Rectangle(x, y, w, h)
        }

        // ── Path painting ─────────────────────────────────────────────────────
        b"S" => Operator::Stroke,
        b"s" => Operator::CloseStroke,
        b"f" | b"F" => Operator::Fill,
        b"f*" => Operator::FillEvenOdd,
        b"B" => Operator::FillStroke,
        b"B*" => Operator::FillStrokeEvenOdd,
        b"b" => Operator::CloseFillStroke,
        b"b*" => Operator::CloseFillStrokeEvenOdd,
        b"n" => Operator::EndPath,

        // ── Clipping ─────────────────────────────────────────────────────────
        b"W" => Operator::Clip,
        b"W*" => Operator::ClipEvenOdd,

        // ── Text objects ──────────────────────────────────────────────────────
        b"BT" => Operator::BeginText,
        b"ET" => Operator::EndText,

        // ── Text state ────────────────────────────────────────────────────────
        b"Tf" => {
            // Stream order: `name size Tf`
            let size = pop_f64(operands);
            let name = pop_name(operands);
            Operator::SetFont { name, size }
        }
        b"Tc" => Operator::SetCharSpacing(pop_f64(operands)),
        b"Tw" => Operator::SetWordSpacing(pop_f64(operands)),
        b"Tz" => Operator::SetHorizScaling(pop_f64(operands)),
        b"TL" => Operator::SetLeading(pop_f64(operands)),
        b"Ts" => Operator::SetTextRise(pop_f64(operands)),
        b"Tr" => Operator::SetTextRenderMode(pop_i32(operands)),

        // ── Text positioning ──────────────────────────────────────────────────
        b"Td" => {
            let (tx, ty) = pop2(operands);
            Operator::TextMove(tx, ty)
        }
        b"TD" => {
            let (tx, ty) = pop2(operands);
            Operator::TextMoveSetLeading(tx, ty)
        }
        b"Tm" => Operator::SetTextMatrix(pop_matrix(operands)),
        b"T*" => Operator::NextLine,

        // ── Text showing ──────────────────────────────────────────────────────
        b"Tj" => Operator::ShowText(pop_string(operands)),
        b"TJ" => Operator::ShowTextArray(pop_text_array(operands)),
        b"'" => Operator::MoveNextLineShow(pop_string(operands)),
        b"\"" => {
            // Stream order: `aw ac string "`
            let text = pop_string(operands);
            let ac = pop_f64(operands);
            let aw = pop_f64(operands);
            Operator::MoveNextLineShowSpaced { aw, ac, text }
        }

        // ── Colour ────────────────────────────────────────────────────────────
        b"cs" => Operator::SetFillColorSpace(pop_name(operands)),
        b"CS" => Operator::SetStrokeColorSpace(pop_name(operands)),
        b"sc" | b"scn" => Operator::SetFillColor(drain_numbers(operands)),
        b"SC" | b"SCN" => Operator::SetStrokeColor(drain_numbers(operands)),
        b"g" => Operator::SetFillGray(pop_f64(operands)),
        b"G" => Operator::SetStrokeGray(pop_f64(operands)),
        b"rg" => {
            let (r, g, b) = pop3(operands);
            Operator::SetFillRgb(r, g, b)
        }
        b"RG" => {
            let (r, g, b) = pop3(operands);
            Operator::SetStrokeRgb(r, g, b)
        }
        b"k" => {
            let (c, m, y, k) = pop4(operands);
            Operator::SetFillCmyk(c, m, y, k)
        }
        b"K" => {
            let (c, m, y, k) = pop4(operands);
            Operator::SetStrokeCmyk(c, m, y, k)
        }

        // ── XObjects & images ─────────────────────────────────────────────────
        b"Do" => Operator::PaintXObject(pop_name(operands)),

        // ── Shading ───────────────────────────────────────────────────────────
        b"sh" => Operator::PaintShading(pop_name(operands)),

        // ── Marked content (no rendering effect) ─────────────────────────────
        b"BMC" | b"BDC" | b"EMC" | b"MP" | b"DP" => Operator::MarkedContent,

        // ── PDF compatibility sections (contents opaque to conforming readers) ─
        b"BX" | b"EX" => Operator::CompatibilitySection,

        _ => Operator::Unknown(op.to_vec()),
    };

    operands.clear();
    result
}

/// Pop the top of the stack as a `TJ` text array.
fn pop_text_array(stack: &mut Vec<Token<'_>>) -> Vec<TextArrayElement> {
    match stack.pop() {
        Some(Token::Array(arr)) => arr
            .into_iter()
            .map(|t| match t {
                Token::String(s) => TextArrayElement::Text(s),
                Token::Number(n) => TextArrayElement::Offset(n),
                Token::Bool(b) => TextArrayElement::Offset(f64::from(u8::from(b))),
                _ => TextArrayElement::Offset(0.0),
            })
            .collect(),
        _ => Vec::new(),
    }
}
