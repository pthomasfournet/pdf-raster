//! PDF content stream parsing: tokenizer + operator decoder.

pub mod operator;
pub mod tokenizer;

pub use operator::{Operator, TextArrayElement, decode};
pub use tokenizer::Tokenizer;

/// Parse a raw content stream byte slice into a `Vec` of decoded operators.
///
/// Unrecognised operand types are silently skipped; unrecognised operator
/// keywords become [`Operator::Unknown`].
#[must_use]
pub fn parse(src: &[u8]) -> Vec<Operator> {
    let mut operands = Vec::new();
    let mut ops = Vec::new();

    for token in Tokenizer::new(src) {
        match token {
            tokenizer::Token::Op(kw) => {
                if let Some(op) = decode(kw, &mut operands) {
                    // Handle inline image: the tokenizer emits BI as InlineImage
                    // directly and never produces a separate Op("BI").
                    ops.push(op);
                }
            }
            tokenizer::Token::InlineImage { params, data } => {
                operands.clear();
                ops.push(Operator::InlineImage {
                    params: params.to_vec(),
                    data: data.to_vec(),
                });
            }
            other => operands.push(other),
        }
    }

    ops
}
