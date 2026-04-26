//! Typed PDF content stream operator representation.
//!
//! After tokenization, operand stacks are consumed by [`Operator::decode`] to
//! produce strongly-typed operator values ready for the dispatcher.

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
    /// `J` — set line cap (0=butt, 1=round, 2=square).
    SetLineCap(i32),
    /// `j` — set line join (0=miter, 1=round, 2=bevel).
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
    /// `ri` — set rendering intent.
    SetRenderingIntent(Vec<u8>),
    /// `i` — set flatness tolerance.
    SetFlatness(f64),
    /// `gs` — apply extended graphics state dictionary.
    SetExtGState(Vec<u8>),

    // ── Path construction ─────────────────────────────────────────────────────
    /// `m x y` — begin new subpath.
    MoveTo(f64, f64),
    /// `l x y` — append straight line.
    LineTo(f64, f64),
    /// `c x1 y1 x2 y2 x3 y3` — append cubic Bézier (full).
    CurveTo(f64, f64, f64, f64, f64, f64),
    /// `v x2 y2 x3 y3` — cubic Bézier, first control = current point.
    CurveToV(f64, f64, f64, f64),
    /// `y x1 y1 x3 y3` — cubic Bézier, last control = endpoint.
    CurveToY(f64, f64, f64, f64),
    /// `h` — close current subpath.
    ClosePath,
    /// `re x y w h` — append rectangle.
    Rectangle(f64, f64, f64, f64),

    // ── Path painting ─────────────────────────────────────────────────────────
    /// `S` — stroke path.
    Stroke,
    /// `s` — close and stroke.
    CloseStroke,
    /// `f` / `F` — fill (non-zero winding rule).
    Fill,
    /// `f*` — fill (even-odd rule).
    FillEvenOdd,
    /// `B` — fill then stroke (non-zero).
    FillStroke,
    /// `B*` — fill then stroke (even-odd).
    FillStrokeEvenOdd,
    /// `b` — close, fill, stroke (non-zero).
    CloseFillStroke,
    /// `b*` — close, fill, stroke (even-odd).
    CloseFillStrokeEvenOdd,
    /// `n` — end path without painting (used for clipping).
    EndPath,

    // ── Clipping ─────────────────────────────────────────────────────────────
    /// `W` — set clipping path (non-zero).
    Clip,
    /// `W*` — set clipping path (even-odd).
    ClipEvenOdd,

    // ── Text ──────────────────────────────────────────────────────────────────
    /// `BT` — begin text object.
    BeginText,
    /// `ET` — end text object.
    EndText,
    /// `Tf name size` — set font and size.
    SetFont {
        /// Font resource name (key into the page's Font resource dict).
        name: Vec<u8>,
        /// Font size in text space units.
        size: f64,
    },
    /// `Td tx ty` — move text position.
    TextMove(f64, f64),
    /// `TD tx ty` — move text position and set leading.
    TextMoveSetLeading(f64, f64),
    /// `Tm a b c d e f` — set text matrix.
    SetTextMatrix([f64; 6]),
    /// `T*` — move to start of next line.
    NextLine,
    /// `Tj string` — show text string.
    ShowText(Vec<u8>),
    /// `TJ array` — show text with glyph spacing.
    ShowTextArray(Vec<TextArrayElement>),
    /// `'  string` — move to next line and show text.
    MoveNextLineShow(Vec<u8>),
    /// `" aw ac string` — set spacing, move to next line, show text.
    MoveNextLineShowSpaced {
        /// Word spacing.
        aw: f64,
        /// Character spacing.
        ac: f64,
        /// Text to show.
        text: Vec<u8>,
    },
    /// `Tc` — set character spacing.
    SetCharSpacing(f64),
    /// `Tw` — set word spacing.
    SetWordSpacing(f64),
    /// `Tz` — set horizontal scaling (percentage).
    SetHorizScaling(f64),
    /// `TL` — set text leading.
    SetLeading(f64),
    /// `Ts` — set text rise.
    SetTextRise(f64),
    /// `Tr` — set text rendering mode.
    SetTextRenderMode(i32),

    // ── Color ─────────────────────────────────────────────────────────────────
    /// `cs` — set fill color space.
    SetFillColorSpace(Vec<u8>),
    /// `CS` — set stroke color space.
    SetStrokeColorSpace(Vec<u8>),
    /// `sc` / `scn` — set fill color (components).
    SetFillColor(Vec<f64>),
    /// `SC` / `SCN` — set stroke color (components).
    SetStrokeColor(Vec<f64>),
    /// `g` — set fill gray.
    SetFillGray(f64),
    /// `G` — set stroke gray.
    SetStrokeGray(f64),
    /// `rg r g b` — set fill RGB.
    SetFillRgb(f64, f64, f64),
    /// `RG r g b` — set stroke RGB.
    SetStrokeRgb(f64, f64, f64),
    /// `k c m y k` — set fill CMYK.
    SetFillCmyk(f64, f64, f64, f64),
    /// `K c m y k` — set stroke CMYK.
    SetStrokeCmyk(f64, f64, f64, f64),

    // ── Images & XObjects ─────────────────────────────────────────────────────
    /// `Do name` — paint XObject (image or form).
    PaintXObject(Vec<u8>),
    /// `BI … ID … EI` — inline image.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between BI and ID).
        params: Vec<u8>,
        /// Raw image data bytes (between ID and EI).
        data: Vec<u8>,
    },

    // ── Shading ───────────────────────────────────────────────────────────────
    /// `sh name` — paint shading pattern.
    PaintShading(Vec<u8>),

    // ── Marked content (treated as no-ops for rendering) ─────────────────────
    /// `BMC` / `BDC` / `EMC` / `MP` / `DP`.
    MarkedContent,

    // ── Catch-all for unrecognised or unimplemented operators ─────────────────
    /// Unknown operator with raw keyword bytes.
    Unknown(Vec<u8>),
}

/// Element of a `TJ` text array: either a string chunk or a glyph offset.
#[derive(Debug, Clone)]
pub enum TextArrayElement {
    /// String to render.
    Text(Vec<u8>),
    /// Horizontal adjustment in thousandths of a text-space unit (negative = right).
    Offset(f64),
}

/// Decode a pending operand stack and operator keyword into an [`Operator`].
///
/// `operands` is consumed (drained) regardless of the result, so the stack is
/// always empty after this call.
///
/// Returns `None` for operators that are known no-ops (marked content etc.) so
/// the dispatcher can skip them cheaply, and `Operator::Unknown` for anything
/// not recognised.
#[expect(clippy::too_many_lines, reason = "large match on PDF operator table — no meaningful way to split")]
pub fn decode<'a>(op: &[u8], operands: &mut Vec<Token<'a>>) -> Option<Operator> {
    let result = match op {
        // ── Graphics state ────────────────────────────────────────────────────
        b"q"  => Operator::Save,
        b"Q"  => Operator::Restore,
        b"cm" => Operator::ConcatMatrix(take_matrix(operands)),
        b"w"  => Operator::SetLineWidth(take_f64(operands, 0)),
        b"J"  => Operator::SetLineCap(take_i32(operands, 0)),
        b"j"  => Operator::SetLineJoin(take_i32(operands, 0)),
        b"M"  => Operator::SetMiterLimit(take_f64(operands, 0)),
        b"d"  => {
            let phase = take_f64(operands, 0);
            let dashes = take_number_array(operands);
            Operator::SetDash { dashes, phase }
        }
        b"ri" => Operator::SetRenderingIntent(take_name(operands)),
        b"i"  => Operator::SetFlatness(take_f64(operands, 0)),
        b"gs" => Operator::SetExtGState(take_name(operands)),

        // ── Path construction ─────────────────────────────────────────────────
        b"m"  => Operator::MoveTo(take_f64(operands, 1), take_f64(operands, 0)),
        b"l"  => Operator::LineTo(take_f64(operands, 1), take_f64(operands, 0)),
        b"c"  => {
            let (a, b, c, d, e, f) = take6(operands);
            Operator::CurveTo(a, b, c, d, e, f)
        }
        b"v"  => {
            let (a, b, c, d) = take4(operands);
            Operator::CurveToV(a, b, c, d)
        }
        b"y"  => {
            let (a, b, c, d) = take4(operands);
            Operator::CurveToY(a, b, c, d)
        }
        b"h"  => Operator::ClosePath,
        b"re" => {
            let (a, b, c, d) = take4(operands);
            Operator::Rectangle(a, b, c, d)
        }

        // ── Path painting ─────────────────────────────────────────────────────
        b"S"  => Operator::Stroke,
        b"s"  => Operator::CloseStroke,
        b"f" | b"F" => Operator::Fill,
        b"f*" => Operator::FillEvenOdd,
        b"B"  => Operator::FillStroke,
        b"B*" => Operator::FillStrokeEvenOdd,
        b"b"  => Operator::CloseFillStroke,
        b"b*" => Operator::CloseFillStrokeEvenOdd,
        b"n"  => Operator::EndPath,

        // ── Clipping ─────────────────────────────────────────────────────────
        b"W"  => Operator::Clip,
        b"W*" => Operator::ClipEvenOdd,

        // ── Text ──────────────────────────────────────────────────────────────
        b"BT" => Operator::BeginText,
        b"ET" => Operator::EndText,
        b"Tf" => {
            let size = take_f64(operands, 0);
            let name = take_name(operands);
            Operator::SetFont { name, size }
        }
        b"Td" => Operator::TextMove(take_f64(operands, 1), take_f64(operands, 0)),
        b"TD" => Operator::TextMoveSetLeading(take_f64(operands, 1), take_f64(operands, 0)),
        b"Tm" => Operator::SetTextMatrix(take_matrix(operands)),
        b"T*" => Operator::NextLine,
        b"Tj" => Operator::ShowText(take_string(operands)),
        b"TJ" => Operator::ShowTextArray(take_text_array(operands)),
        b"'"  => Operator::MoveNextLineShow(take_string(operands)),
        b"\"" => {
            let text = take_string(operands);
            let ac   = take_f64(operands, 0);
            let aw   = take_f64(operands, 0);
            Operator::MoveNextLineShowSpaced { aw, ac, text }
        }
        b"Tc" => Operator::SetCharSpacing(take_f64(operands, 0)),
        b"Tw" => Operator::SetWordSpacing(take_f64(operands, 0)),
        b"Tz" => Operator::SetHorizScaling(take_f64(operands, 0)),
        b"TL" => Operator::SetLeading(take_f64(operands, 0)),
        b"Ts" => Operator::SetTextRise(take_f64(operands, 0)),
        b"Tr" => Operator::SetTextRenderMode(take_i32(operands, 0)),

        // ── Color ─────────────────────────────────────────────────────────────
        b"cs"  => Operator::SetFillColorSpace(take_name(operands)),
        b"CS"  => Operator::SetStrokeColorSpace(take_name(operands)),
        b"sc" | b"scn" => Operator::SetFillColor(take_numbers(operands)),
        b"SC" | b"SCN" => Operator::SetStrokeColor(take_numbers(operands)),
        b"g"   => Operator::SetFillGray(take_f64(operands, 0)),
        b"G"   => Operator::SetStrokeGray(take_f64(operands, 0)),
        b"rg"  => {
            let (r, g, b) = take3(operands);
            Operator::SetFillRgb(r, g, b)
        }
        b"RG"  => {
            let (r, g, b) = take3(operands);
            Operator::SetStrokeRgb(r, g, b)
        }
        b"k"   => {
            let (c, m, y, k) = take4(operands);
            Operator::SetFillCmyk(c, m, y, k)
        }
        b"K"   => {
            let (c, m, y, k) = take4(operands);
            Operator::SetStrokeCmyk(c, m, y, k)
        }

        // ── XObjects & images ─────────────────────────────────────────────────
        b"Do"  => Operator::PaintXObject(take_name(operands)),

        // ── Shading ───────────────────────────────────────────────────────────
        b"sh"  => Operator::PaintShading(take_name(operands)),

        // ── Marked content (no-ops for rendering) ────────────────────────────
        b"BMC" | b"BDC" | b"EMC" | b"MP" | b"DP" => {
            operands.clear();
            return Some(Operator::MarkedContent);
        }

        // ── Compatibility sections (ignore contents) ──────────────────────────
        b"BX" | b"EX" => {
            operands.clear();
            return Some(Operator::MarkedContent);
        }

        _ => Operator::Unknown(op.to_vec()),
    };

    operands.clear();
    Some(result)
}

// ── Operand extraction helpers ────────────────────────────────────────────────

fn pop_number(stack: &mut Vec<Token<'_>>) -> f64 {
    match stack.pop() {
        Some(Token::Number(n)) => n,
        Some(Token::Bool(b)) => if b { 1.0 } else { 0.0 },
        _ => 0.0,
    }
}

/// Take the Nth number from the bottom (0 = bottom of remaining stack).
/// For single-operand operators just call with `idx = 0`.
fn take_f64(stack: &mut Vec<Token<'_>>, _idx: usize) -> f64 {
    pop_number(stack)
}

fn take_i32(stack: &mut Vec<Token<'_>>, _idx: usize) -> i32 {
    pop_number(stack) as i32
}

fn take_name(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::Name(n)) => n.to_vec(),
        Some(Token::String(s)) => s,
        _ => Vec::new(),
    }
}

fn take_string(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::String(s)) => s,
        Some(Token::Name(n)) => n.to_vec(),
        _ => Vec::new(),
    }
}

fn take_numbers(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    let mut out = Vec::new();
    while let Some(Token::Number(_) | Token::Bool(_)) = stack.last() {
        out.push(pop_number(stack));
    }
    out.reverse();
    out
}

fn take_number_array(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    match stack.pop() {
        Some(Token::Array(arr)) => arr
            .into_iter()
            .filter_map(|t| if let Token::Number(n) = t { Some(n) } else { None })
            .collect(),
        _ => Vec::new(),
    }
}

fn take_matrix(stack: &mut Vec<Token<'_>>) -> [f64; 6] {
    let f = pop_number(stack);
    let e = pop_number(stack);
    let d = pop_number(stack);
    let c = pop_number(stack);
    let b = pop_number(stack);
    let a = pop_number(stack);
    [a, b, c, d, e, f]
}

fn take3(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64) {
    let c = pop_number(stack);
    let b = pop_number(stack);
    let a = pop_number(stack);
    (a, b, c)
}

fn take4(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64, f64) {
    let d = pop_number(stack);
    let c = pop_number(stack);
    let b = pop_number(stack);
    let a = pop_number(stack);
    (a, b, c, d)
}

fn take6(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64, f64, f64, f64) {
    let f = pop_number(stack);
    let e = pop_number(stack);
    let d = pop_number(stack);
    let c = pop_number(stack);
    let b = pop_number(stack);
    let a = pop_number(stack);
    (a, b, c, d, e, f)
}

fn take_text_array(stack: &mut Vec<Token<'_>>) -> Vec<TextArrayElement> {
    match stack.pop() {
        Some(Token::Array(arr)) => arr
            .into_iter()
            .map(|t| match t {
                Token::String(s) => TextArrayElement::Text(s),
                Token::Number(n) => TextArrayElement::Offset(n),
                _ => TextArrayElement::Offset(0.0),
            })
            .collect(),
        _ => Vec::new(),
    }
}
