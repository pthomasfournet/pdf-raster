//! # color
//!
//! Pixel types, color mode definitions, shared arithmetic primitives, and
//! transfer function LUTs for the rasterrocket workspace.
//!
//! ## Design rule
//! Every compositing primitive (`div255`, `lerp_u8`, `cmyk_to_rgb`,
//! `cmyk_to_rgb_reflectance`, `splash_floor`, …) lives in [`convert`].
//! Downstream crates import from here and never duplicate the logic.
//! This is the primary enforcement point of the workspace's shared-helper policy.
//!
//! ## Module layout
//! - [`mode`] — [`PixelMode`] enum + bytes-per-pixel table
//! - [`pixel`] — [`Pixel`] trait and concrete types: [`Rgb8`], [`Rgba8`], [`Gray8`], [`Cmyk8`], [`DeviceN8`]
//! - [`convert`] — all shared math (div255, lerp, Porter-Duff, color conversion, floor/ceil/round)
//! - [`transfer`] — [`TransferLut`] newtype `([u8; 256])`

pub mod convert;
pub mod mode;
pub mod pixel;
pub mod transfer;

pub use mode::{NCOMPS, PixelMode};
pub use pixel::{AnyColor, Cmyk8, DeviceN8, Gray8, Pixel, Rgb8, Rgba8};
pub use transfer::TransferLut;
