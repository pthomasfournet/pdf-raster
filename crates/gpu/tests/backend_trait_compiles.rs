//! Smoke-test that the GpuBackend trait exists and is object-safe-via-generics.
use gpu::backend::GpuBackend;

fn _accepts_backend<B: GpuBackend>(_b: &B) {}
