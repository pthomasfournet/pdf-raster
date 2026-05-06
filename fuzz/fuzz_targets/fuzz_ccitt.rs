#![no_main]

use libfuzzer_sys::fuzz_target;

// Minimum input: 1 byte for the path selector + 4 bytes for dims + payload.
// Layout: [selector(1)] [width_lo(1) width_hi(1) height_lo(1) height_hi(1)] [payload...]
//
// selector & 0b11:
//   0 → K=0  (Group 3 1D, default when parms=None)
//   1 → K<0  (Group 4 / T.6 2D)
//   2 → K>0  (Group 3 mixed 1D/2D, hayro-ccitt path)
//   3 → K>0  (same path, different K value from selector bits)
fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    let selector = data[0];
    let (dims, payload) = data[1..].split_at(4);

    // Width and height from the next 4 bytes; avoid zero and cap to prevent OOM.
    // saturating_add(1) ensures neither dimension is zero (0+1=1 minimum).
    let width = u32::from(
        u16::from_le_bytes([dims[0], dims[1]])
            .saturating_add(1)
            .min(4096),
    );
    let height = u32::from(
        u16::from_le_bytes([dims[2], dims[3]])
            .saturating_add(1)
            .min(4096),
    );

    // Exercise the three K-dispatch paths explicitly.  parms=None only reaches K=0,
    // so we must build synthetic lopdf Objects for the other two paths.
    match selector & 0b11 {
        // K=0: Group 3 1D — no parms needed (decode_ccitt defaults K=0).
        0 => {
            let _ =
                pdf_interp::fuzz_helpers::decode_ccitt(payload, width, height, false, None);
        }
        // K<0: Group 4 (T.6) — requires K=-1 in DecodeParms.
        1 => {
            let parms = make_ccitt_parms(-1, selector);
            let _ = pdf_interp::fuzz_helpers::decode_ccitt(
                payload,
                width,
                height,
                false,
                Some(&parms),
            );
        }
        // K>0: Group 3 mixed 1D/2D — use K=1 (minimum T.4 2D parameter).
        2 => {
            let parms = make_ccitt_parms(1, selector);
            let _ = pdf_interp::fuzz_helpers::decode_ccitt(
                payload,
                width,
                height,
                false,
                Some(&parms),
            );
        }
        // K>0 with is_mask=true: exercises the mask colour-space path.
        _ => {
            let parms = make_ccitt_parms(1, selector);
            let _ = pdf_interp::fuzz_helpers::decode_ccitt(
                payload,
                width,
                height,
                true,
                Some(&parms),
            );
        }
    }
});

/// Build a minimal `DecodeParms` dict for CCITT with the given `K` value.
///
/// `BlackIs1` is derived from the low bit of `extra` so the fuzzer can
/// flip it without needing a separate byte.
fn make_ccitt_parms(k: i64, extra: u8) -> pdf::Object {
    let black_is_1 = extra & 0b1000_0000 != 0;
    pdf::Object::Dictionary(pdf::Dictionary::from_iter([
        (b"K".to_vec(), pdf::Object::Integer(k)),
        (b"BlackIs1".to_vec(), pdf::Object::Boolean(black_is_1)),
    ]))
}
