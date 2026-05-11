//! SIMD-accelerated hot paths for the rasterizer.
//!
//! All SIMD code is gated behind `#[cfg(target_arch = "x86_64")]` and uses
//! runtime feature detection (`is_x86_feature_detected!`) so the binary runs
//! correctly on machines that lack the required extensions.
//!
//! # Sub-modules
//!
//! - [`aa_coverage`]   — per-pixel AA coverage counts (`aa_coverage_span`)
//! - [`blend`]         — solid-colour fill (`blend_solid_rgb8`, `blend_solid_gray8`)
//! - [`composite`]     — AA per-pixel blend (`composite_aa_rgb8_opaque`)
//! - [`glyph_unpack`]  — 1-bit-per-pixel mono glyph expansion (`unpack_mono_row`)

// SIMD functions are inherently unsafe; unsafe_code is required throughout this module tree.
#![expect(
    unsafe_code,
    reason = "SIMD intrinsics require unsafe throughout this module tree"
)]

pub mod aa_coverage;
pub mod blend;
pub mod composite;
pub mod glyph_unpack;

pub use aa_coverage::aa_coverage_span;
pub use blend::{blend_solid_gray8, blend_solid_rgb8};
pub use composite::composite_aa_rgb8_opaque;
pub use glyph_unpack::unpack_mono_row;
