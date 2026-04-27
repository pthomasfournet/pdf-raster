//! Operand stack extraction helpers.
//!
//! PDF content stream operands accumulate on a stack left-to-right, so the
//! last operand pushed is the rightmost value written in the stream.
//! All `pop_*` helpers consume the topmost (rightmost) element.

use super::tokenizer::Token;

/// Pop the topmost token as a number. Returns 0.0 on underflow or wrong type.
pub fn pop_f64(stack: &mut Vec<Token<'_>>) -> f64 {
    match stack.pop() {
        Some(Token::Number(n)) => n,
        Some(Token::Bool(b)) => f64::from(u8::from(b)),
        _ => 0.0,
    }
}

/// Pop the topmost token as a truncated integer. Returns 0 on underflow.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional truncating cast: PDF integers are used as small values (cap style, render mode, etc.)"
)]
pub fn pop_i32(stack: &mut Vec<Token<'_>>) -> i32 {
    pop_f64(stack) as i32
}

/// Pop the topmost token as a name or string byte vector.
/// Returns an empty `Vec` on underflow or wrong type.
pub fn pop_name(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::Name(n)) => n.to_vec(),
        Some(Token::String(s)) => s,
        _ => Vec::new(),
    }
}

/// Pop the topmost token as a string byte vector.
/// Returns an empty `Vec` on underflow or wrong type.
pub fn pop_string(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::String(s)) => s,
        Some(Token::Name(n)) => n.to_vec(),
        _ => Vec::new(),
    }
}

/// Drain all contiguous numbers (and booleans) from the top of the stack,
/// returning them in stream order (leftmost first).
///
/// Stops at the first non-numeric token, leaving it on the stack.
pub fn drain_numbers(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    // Find how many leading numeric tokens sit at the bottom of the stack
    // (they were pushed left-to-right, so the first stream operand is at
    // index 0 and the last is at the top).
    let first_non_num = stack
        .iter()
        .position(|t| !matches!(t, Token::Number(_) | Token::Bool(_)))
        .unwrap_or(stack.len());
    stack
        .drain(..first_non_num)
        .map(|t| match t {
            Token::Number(n) => n,
            Token::Bool(b) => f64::from(u8::from(b)),
            _ => unreachable!(),
        })
        .collect()
}

/// Result of parsing `scn` / `SCN` operands: either a pattern name or numeric
/// colour components.
///
/// PDF §8.6.8: `scn` operands are zero or more numbers followed by an optional
/// name.  When the last token is a name the colour space is Pattern and the name
/// selects the pattern resource.  For all other colour spaces only numbers appear.
pub enum ScnOperands {
    /// `/PatternName [c1 … cn]` — Pattern colour space; name is the resource key,
    /// components are the underlying non-Pattern colour (may be empty for
    /// `PaintType` 1 / coloured patterns).
    Pattern {
        /// Pattern resource name (key into the page `Pattern` resource dict).
        name: Vec<u8>,
        /// Optional tint components for uncoloured (`PaintType` 2) patterns.
        components: Vec<f64>,
    },
    /// Plain numeric components for non-pattern colour spaces.
    Components(Vec<f64>),
}

/// Parse the operand stack for `scn` / `SCN`.
///
/// The PDF grammar is `c1 … cn name?`, where the optional trailing `name` selects
/// a Pattern resource.  Numbers are consumed left-to-right; if the topmost token
/// is a Name the result is `ScnOperands::Pattern`.
pub fn pop_scn(stack: &mut Vec<Token<'_>>) -> ScnOperands {
    // Check whether the top token is a name — if so it is the pattern key.
    let name = match stack.last() {
        Some(Token::Name(_)) => {
            if let Some(Token::Name(n)) = stack.pop() {
                Some(n.to_vec())
            } else {
                unreachable!()
            }
        }
        _ => None,
    };
    let components = drain_numbers(stack);
    match name {
        Some(n) => ScnOperands::Pattern {
            name: n,
            components,
        },
        None => ScnOperands::Components(components),
    }
}

/// Pop the topmost token as a flat array of numbers.
///
/// If the top is a `Token::Array`, extract numbers from it.
/// If the top is a bare number, treat the entire numeric prefix as the array.
/// Returns an empty `Vec` on underflow.
pub fn pop_number_array(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    match stack.last() {
        Some(Token::Array(_)) => {
            if let Some(Token::Array(arr)) = stack.pop() {
                arr.into_iter()
                    .filter_map(|t| match t {
                        Token::Number(n) => Some(n),
                        Token::Bool(b) => Some(f64::from(u8::from(b))),
                        _ => None,
                    })
                    .collect()
            } else {
                unreachable!()
            }
        }
        _ => Vec::new(),
    }
}

/// Pop the top 2 numbers as `(first, second)` in stream order.
pub fn pop2(stack: &mut Vec<Token<'_>>) -> (f64, f64) {
    let b = pop_f64(stack);
    let a = pop_f64(stack);
    (a, b)
}

/// Pop the top 3 numbers as `(first, second, third)` in stream order.
pub fn pop3(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64) {
    let c = pop_f64(stack);
    let b = pop_f64(stack);
    let a = pop_f64(stack);
    (a, b, c)
}

/// Pop the top 4 numbers as `(first, second, third, fourth)` in stream order.
pub fn pop4(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64, f64) {
    let d = pop_f64(stack);
    let c = pop_f64(stack);
    let b = pop_f64(stack);
    let a = pop_f64(stack);
    (a, b, c, d)
}

/// Pop the top 6 numbers as a PDF matrix `[a b c d e f]` in stream order.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
pub fn pop_matrix(stack: &mut Vec<Token<'_>>) -> [f64; 6] {
    let f = pop_f64(stack);
    let e = pop_f64(stack);
    let d = pop_f64(stack);
    let c = pop_f64(stack);
    let b = pop_f64(stack);
    let a = pop_f64(stack);
    [a, b, c, d, e, f]
}
