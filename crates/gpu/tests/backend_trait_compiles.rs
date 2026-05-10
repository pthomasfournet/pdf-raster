//! Smoke-test that the `GpuBackend` trait can be used as a generic bound.
use gpu::backend::GpuBackend;

const fn _accepts_backend<B: GpuBackend>(_b: &B) {}
