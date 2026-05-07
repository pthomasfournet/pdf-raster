//! Cache-side helpers that need access to the PDF object graph and
//! the image decode pipeline (so they live in `pdf_interp` rather
//! than `gpu`).
//!
//! Currently exposes the [`prefetch`] module — a background worker
//! that walks every page's `/XObject` resource dict, decodes each
//! `/DCTDecode`-filtered image once, and primes
//! [`gpu::cache::DeviceImageCache`] so the renderer thread sees
//! cache hits on first touch.

#![cfg(feature = "cache")]

pub mod prefetch;

pub use prefetch::{PrefetchConfig, PrefetchHandle, PrefetchStats, spawn_prefetch};
