//! Smoke test: CudaBackend implements GpuBackend.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   cargo test -p gpu --features gpu-validation --test cuda_backend_smoke

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_initializes() {
    use gpu::backend::GpuBackend;
    use gpu::backend::cuda::CudaBackend;

    let backend = CudaBackend::new()
        .expect("CudaBackend::new failed — is a CUDA device available and the driver loaded?");
    let budget = backend
        .detect_vram_budget()
        .expect("detect_vram_budget failed — driver query returned an error");
    assert!(
        budget.total_bytes > 0,
        "expected non-zero total VRAM, got {}",
        budget.total_bytes
    );
    assert!(
        budget.usable_bytes <= budget.total_bytes,
        "VramBudget invariant violated at runtime: usable {} > total {}",
        budget.usable_bytes,
        budget.total_bytes
    );
}
