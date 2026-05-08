//! Smoke test: CudaBackend implements GpuBackend.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   cargo test -p gpu --features gpu-validation --test cuda_backend_smoke
use gpu::backend::GpuBackend;
use gpu::backend::cuda::CudaBackend;

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_initializes() {
    let backend = CudaBackend::new().expect("init");
    let budget = backend.detect_vram_budget().expect("budget");
    assert!(budget.total_bytes > 0);
    assert!(budget.usable_bytes <= budget.total_bytes);
}
