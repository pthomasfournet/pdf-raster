#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes for synthetic width/height.
    if data.len() < 4 {
        return;
    }
    let (dims, payload) = data.split_at(4);

    let width = u32::from(u16::from_le_bytes([dims[0], dims[1]]).saturating_add(1).min(4096));
    let height = u32::from(u16::from_le_bytes([dims[2], dims[3]]).saturating_add(1).min(4096));

    // No-globals path: use an empty document so JBIG2Globals lookup returns None.
    // This exercises the most common embedded-page-data path in hayro-jbig2.
    let doc = lopdf::Document::new();
    let _ = pdf_interp::fuzz_helpers::decode_jbig2(&doc, payload, width, height, false, None);

    // Also fuzz hayro-jbig2 directly (raw segment data, no PDF wrapper).
    // This reaches the parser sooner without going through the PDF layer.
    let _ = hayro_jbig2::Image::new_embedded(payload, None);
});
