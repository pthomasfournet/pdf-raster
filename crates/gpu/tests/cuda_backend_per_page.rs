//! End-to-end: alloc, record two ops, submit, wait, verify the path.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   `cargo test -p gpu --features gpu-validation --test cuda_backend_per_page`

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_per_page_composite_then_softmask() {
    use gpu::backend::cuda::CudaBackend;
    use gpu::backend::{GpuBackend, params};

    let backend = CudaBackend::new()
        .expect("CudaBackend::new failed — is a CUDA device available and the driver loaded?");
    let n_pixels: u32 = 1024;
    let rgba_bytes = (n_pixels as usize)
        .checked_mul(4)
        .expect("n_pixels * 4 fits usize");

    let src = backend
        .alloc_device(rgba_bytes)
        .expect("alloc_device(src) failed — VRAM exhausted or driver error");
    let dst = backend
        .alloc_device(rgba_bytes)
        .expect("alloc_device(dst) failed — VRAM exhausted or driver error");
    let mask = backend
        .alloc_device(n_pixels as usize)
        .expect("alloc_device(mask) failed — VRAM exhausted or driver error");

    backend
        .begin_page()
        .expect("begin_page failed — recorder did not enter page-recording state");
    backend
        .record_composite(params::CompositeParams {
            src: &src,
            dst: &dst,
            n_pixels,
        })
        .expect("record_composite failed — kernel launch error or arg mismatch");
    backend
        .record_apply_soft_mask(params::SoftMaskParams {
            pixels: &dst,
            mask: &mask,
            n_pixels,
        })
        .expect("record_apply_soft_mask failed — kernel launch error or arg mismatch");
    let fence = backend
        .submit_page()
        .expect("submit_page failed — could not record CudaEvent on the stream");
    backend
        .wait_page(fence)
        .expect("wait_page failed — event.synchronize() reported a CUDA error");

    // Output buffer is on device; readback would happen in a real
    // renderer's download path.  This test asserts that recording two
    // sequential ops, submitting once, and waiting on the page fence
    // all succeed end-to-end without a panic, an `unimplemented!`, or
    // a CUDA error.
}
