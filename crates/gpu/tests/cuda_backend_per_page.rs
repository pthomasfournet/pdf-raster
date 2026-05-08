//! End-to-end: alloc, record two ops, submit, wait, verify the path.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   cargo test -p gpu --features gpu-validation --test cuda_backend_per_page

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_per_page_composite_then_softmask() {
    use gpu::backend::cuda::CudaBackend;
    use gpu::backend::{GpuBackend, params};

    let backend = CudaBackend::new().expect("init");
    let n_pixels: u32 = 1024;

    let src = backend
        .alloc_device((n_pixels * 4) as usize)
        .expect("alloc src");
    let dst = backend
        .alloc_device((n_pixels * 4) as usize)
        .expect("alloc dst");
    let mask = backend.alloc_device(n_pixels as usize).expect("alloc mask");

    backend.begin_page().expect("begin");
    backend
        .record_composite(params::CompositeParams {
            src: &src,
            dst: &dst,
            n_pixels,
        })
        .expect("record composite");
    backend
        .record_apply_soft_mask(params::SoftMaskParams {
            pixels: &dst,
            mask: &mask,
            n_pixels,
        })
        .expect("record softmask");
    let fence = backend.submit_page().expect("submit");
    backend.wait_page(fence).expect("wait");

    // Output buffer is on device; readback would happen in a real
    // renderer's download path.  This test asserts that recording two
    // sequential ops, submitting once, and waiting on the page fence
    // all succeed end-to-end without a panic, an `unimplemented!`, or
    // a CUDA error.
}
