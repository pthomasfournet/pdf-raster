//! Host-side dispatcher for the Blelloch exclusive scan.
//!
//! Wraps the three-phase trait dispatch into a single
//! `dispatch_blelloch_scan` call that takes a u32 slice and returns
//! the exclusive prefix sum. Used by the GPU JPEG decoder to compute
//! output offsets from the per-subsequence symbol counts.
//!
//! Generic over `B: GpuBackend` rather than `dyn`-trait: the
//! associated-type shape of `GpuBackend` rules out a dyn-trait
//! signature.

use crate::backend::params::{SCAN_WORKGROUP_SIZE, ScanParams, ScanPhase};
use crate::backend::{BackendError, GpuBackend, Result};

/// Exclusive prefix sum of `input` via the multi-workgroup Blelloch
/// scan kernel.
///
/// Caller-owned arena: allocates a `len_elems * 4`-byte `data` buffer
/// and a `block_count * 4`-byte `block_sums` buffer, runs the three
/// phases, downloads, and frees. For repeated scans against the same
/// `len_elems`, callers should keep their own buffers and use the
/// trait methods directly — this entry point is for one-shot use
/// (tests, ad-hoc dispatch, and the JPEG decoder's per-page scan).
///
/// # Errors
/// Returns the underlying `BackendError` if any allocation, upload,
/// dispatch, or download fails.
pub fn dispatch_blelloch_scan<B: GpuBackend>(backend: &B, input: &[u32]) -> Result<Vec<u32>> {
    let len_elems = u32::try_from(input.len()).map_err(|_| {
        BackendError::msg(format!(
            "dispatch_blelloch_scan: input length {} exceeds u32::MAX",
            input.len()
        ))
    })?;

    // Trivial empty case — no kernel work, return immediately.
    if len_elems == 0 {
        return Ok(Vec::new());
    }

    let data_bytes = (len_elems as usize) * 4;
    let tile = 2 * SCAN_WORKGROUP_SIZE;
    let block_count = len_elems.div_ceil(tile);
    let block_sums_bytes = (block_count as usize) * 4;

    let data = backend.alloc_device(data_bytes)?;
    let block_sums = backend.alloc_device(block_sums_bytes)?;

    let input_bytes = bytemuck::cast_slice::<u32, u8>(input);
    let upload_fence = backend.upload_async(&data, input_bytes)?;
    backend.wait_transfer(upload_fence)?;

    backend.begin_page()?;
    for phase in [
        ScanPhase::PerWorkgroup,
        ScanPhase::BlockSums,
        ScanPhase::ScatterBlockSums,
    ] {
        backend.record_scan(ScanParams {
            data: &data,
            block_sums: &block_sums,
            len_elems,
            phase,
        })?;
    }
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    let mut out_bytes = vec![0u8; data_bytes];
    let handle = backend.download_async(&data, &mut out_bytes)?;
    backend.wait_download(handle)?;

    backend.free_device(data);
    backend.free_device(block_sums);

    // Reinterpret the byte buffer as u32. The host is little-endian
    // x86-64; the GPU kernel writes u32 in the same byte order; no
    // swap needed.
    let out = bytemuck::cast_slice::<u8, u32>(&out_bytes).to_vec();
    Ok(out)
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;

    fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(input.len());
        let mut acc = 0u32;
        for &v in input {
            out.push(acc);
            acc = acc.wrapping_add(v);
        }
        out
    }

    fn try_backend() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    #[test]
    fn empty_input_returns_empty() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let out = dispatch_blelloch_scan(&b, &[]).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn single_element_yields_zero() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let out = dispatch_blelloch_scan(&b, &[42]).expect("ok");
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn within_one_tile_matches_cpu() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let input: Vec<u32> = (1..=512).collect(); // 512 < tile (1024)
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        let want = cpu_exclusive_scan(&input);
        assert_eq!(got, want);
    }

    #[test]
    fn exactly_one_tile_matches_cpu() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let input: Vec<u32> = (1..=1024).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        let want = cpu_exclusive_scan(&input);
        assert_eq!(got, want);
    }

    #[test]
    fn multiple_tiles_match_cpu() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // 4096 elements = 4 tiles; exercises every phase.
        let input: Vec<u32> = (1..=4096).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        let want = cpu_exclusive_scan(&input);
        assert_eq!(got, want);
    }

    #[test]
    fn partial_final_tile_matches_cpu() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // 3000 elements = 2 full tiles + a partial tile of 952.
        let input: Vec<u32> = (1..=3000).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        let want = cpu_exclusive_scan(&input);
        assert_eq!(got, want);
    }

    #[test]
    fn uniform_zeros_yield_zeros() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let input = vec![0u32; 2048];
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        assert!(got.iter().all(|&v| v == 0));
    }

    #[test]
    fn uniform_ones_yield_ramp() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let input = vec![1u32; 1024];
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        let want: Vec<u32> = (0..1024).collect();
        assert_eq!(got, want);
    }
}

#[cfg(all(test, feature = "vulkan", feature = "gpu-validation"))]
mod vulkan_tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::backend::vulkan::VulkanBackend;

    fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(input.len());
        let mut acc = 0u32;
        for &v in input {
            out.push(acc);
            acc = acc.wrapping_add(v);
        }
        out
    }

    fn try_vulkan() -> Option<VulkanBackend> {
        VulkanBackend::new().ok()
    }

    fn try_cuda() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    /// Same input through both backends; outputs must be byte-identical.
    /// Skipped if either backend can't initialise on this host.
    fn cross_backend_parity(input: &[u32]) {
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let cuda_out = dispatch_blelloch_scan(&cuda, input).expect("cuda ok");
        let vk_out = dispatch_blelloch_scan(&vk, input).expect("vulkan ok");
        let cpu = cpu_exclusive_scan(input);
        assert_eq!(cuda_out, cpu, "CUDA diverges from CPU oracle");
        assert_eq!(vk_out, cpu, "Vulkan diverges from CPU oracle");
        assert_eq!(cuda_out, vk_out, "CUDA and Vulkan disagree");
    }

    #[test]
    fn vulkan_within_one_tile_matches_cpu() {
        let Some(b) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let input: Vec<u32> = (1..=512).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        assert_eq!(got, cpu_exclusive_scan(&input));
    }

    #[test]
    fn vulkan_multiple_tiles_match_cpu() {
        let Some(b) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let input: Vec<u32> = (1..=4096).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        assert_eq!(got, cpu_exclusive_scan(&input));
    }

    #[test]
    fn vulkan_partial_final_tile_matches_cpu() {
        let Some(b) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let input: Vec<u32> = (1..=3000).collect();
        let got = dispatch_blelloch_scan(&b, &input).expect("ok");
        assert_eq!(got, cpu_exclusive_scan(&input));
    }

    #[test]
    fn cuda_vs_vulkan_within_one_tile() {
        cross_backend_parity(&(1..=512).collect::<Vec<_>>());
    }

    #[test]
    fn cuda_vs_vulkan_multiple_tiles() {
        cross_backend_parity(&(1..=4096).collect::<Vec<_>>());
    }

    #[test]
    fn cuda_vs_vulkan_partial_final_tile() {
        cross_backend_parity(&(1..=3000).collect::<Vec<_>>());
    }

    #[test]
    fn cuda_vs_vulkan_uniform_zeros() {
        cross_backend_parity(&vec![0u32; 2048]);
    }
}
