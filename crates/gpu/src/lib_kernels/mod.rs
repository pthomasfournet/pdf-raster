//! Per-kernel implementations split out of `lib.rs`.
//!
//! Each module owns a single GPU kernel's host-side launch logic. The CUDA
//! backend's `record_*` methods call into these modules; tests call the
//! same `pub fn` entry points directly.

pub mod aa;
pub mod composite;
pub mod soft_mask;
