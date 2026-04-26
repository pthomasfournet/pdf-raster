//! PDF content stream parsing: tokenizer → operand stack → typed operators.

pub mod operands;
pub mod operator;
pub mod tokenizer;

pub use operator::{Operator, TextArrayElement};
pub use tokenizer::Tokenizer;

/// Parse a raw content stream byte slice into a `Vec` of decoded operators.
///
/// - Unrecognised operator keywords become [`Operator::Unknown`].
/// - Operand tokens that do not match the expected type for an operator are
///   treated as zero/empty (lenient, matching Acrobat's behaviour).
/// - Stray operands with no following operator are silently discarded.
#[must_use]
pub fn parse(src: &[u8]) -> Vec<Operator> {
    let mut operands: Vec<tokenizer::Token<'_>> = Vec::new();
    let mut ops = Vec::new();

    for token in Tokenizer::new(src) {
        match token {
            tokenizer::Token::Op(kw) => {
                ops.push(operator::decode(kw, &mut operands));
            }
            tokenizer::Token::InlineImage { params, data } => {
                // The tokenizer consumed BI..ID..EI and emits a single token;
                // any preceding operands are stray — clear them.
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
