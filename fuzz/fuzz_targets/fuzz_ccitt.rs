#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes to extract synthetic width/height.
    if data.len() < 4 {
        return;
    }
    let (dims, payload) = data.split_at(4);

    // Clamp dimensions to something that won't OOM the process.
    let width = u32::from(u16::from_le_bytes([dims[0], dims[1]]).saturating_add(1).min(4096));
    let height = u32::from(u16::from_le_bytes([dims[2], dims[3]]).saturating_add(1).min(4096));

    // Exercise all three K paths: G4 (K<0), G3-1D (K=0), G3-2D (K>0).
    // parms=None → K defaults to 0 (G3-1D) inside decode_ccitt.
    let _ = pdf_interp::fuzz_helpers::decode_ccitt(payload, width, height, false, None);
    let _ = pdf_interp::fuzz_helpers::decode_ccitt(payload, width, height, true, None);
});
