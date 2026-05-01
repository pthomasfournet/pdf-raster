#![no_main]

use libfuzzer_sys::fuzz_target;

// Layout: [width_lo(1) width_hi(1) height_lo(1) height_hi(1)] [payload...]
//
// We always pass parms=None (no JBIG2Globals), which is the overwhelmingly
// common case in real PDFs and gives hayro-jbig2 the most direct exposure.
//
// The `hayro_jbig2::Image::new_embedded` call goes through the same parser
// path that `decode_jbig2` uses internally (it is called unconditionally before
// any dimension checking).  We therefore call `decode_jbig2` only — one call
// to the full pipeline is sufficient to cover both the PDF wrapper logic and the
// parser.  A second direct call would duplicate coverage without adding new paths.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let (dims, payload) = data.split_at(4);

    // saturating_add(1): prevents zero dimensions without changing the range
    // for non-zero inputs.  Cap at 4096 to avoid OOM on large valid streams.
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

    // An empty document satisfies the `&Document` requirement; no globals stream
    // is looked up because parms=None short-circuits before any document access.
    // Constructing the document once per input (not once per process) is the
    // cheapest correct approach — lopdf::Document::new() is a plain allocation.
    let doc = lopdf::Document::new();
    let _ =
        pdf_interp::fuzz_helpers::decode_jbig2(&doc, payload, width, height, false, None);
});
