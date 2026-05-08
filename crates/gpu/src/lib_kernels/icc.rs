//! ICC CMYK→RGB kernel dispatch.

use cudarc::driver::PushKernelArg;

use crate::{GPU_ICC_CLUT_THRESHOLD, GpuCtx, cmyk::icc_cmyk_to_rgb_cpu, launch_cfg};

impl GpuCtx {
    /// Convert CMYK pixels to RGB using a GPU kernel.
    ///
    /// `cmyk` is interleaved CMYK, 4 bytes per pixel (PDF convention: 0 = no ink,
    /// 255 = full ink).  Returns interleaved RGB, 3 bytes per pixel.
    ///
    /// Two dispatch paths:
    /// - `clut` is `None` — uses the fast matrix kernel (subtractive complement
    ///   formula, identical to the CPU fallback).
    /// - `clut` is `Some((table, grid_n))` — uses the 4D quadrilinear CLUT kernel.
    ///   `table` must be `grid_n^4 * 3` bytes, ordered
    ///   `(k * G³ + c * G² + m * G + y) * 3` (RGB output values, u8).
    ///   `grid_n` is typically 17 (83 521 nodes) or 33 (1 185 921 nodes).
    ///
    /// Falls back to [`icc_cmyk_to_rgb_cpu`] when `n_pixels < GPU_ICC_CLUT_THRESHOLD`
    /// or `cmyk` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4, or if `clut` is `Some` and
    /// `table.len() != grid_n^4 * 3`.
    pub fn icc_cmyk_to_rgb(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );

        // Early-out before any CLUT validation: empty input always produces empty output.
        let n = cmyk.len() / 4;
        if n == 0 {
            return Ok(Vec::new());
        }

        if let Some((table, grid_n)) = clut {
            // grid_n ≤ 255 is enforced by the baking API; checked_pow guards future misuse.
            let expected = (grid_n as usize)
                .checked_pow(4)
                .and_then(|n| n.checked_mul(3))
                .unwrap_or_else(|| {
                    panic!("grid_n({grid_n})^4*3 overflows usize — grid_n must be ≤ 255")
                });
            assert_eq!(
                table.len(),
                expected,
                "CLUT table length {got} ≠ grid_n({grid_n})^4*3={expected}",
                got = table.len(),
            );
        }
        // Matrix path (clut=None): CPU AVX-512 always beats GPU on this machine —
        // threshold_bench showed the PCIe round-trip cost exceeds the compute cost
        // at all measured sizes (256–4M pixels).  Always use the CPU path here.
        if clut.is_none() {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, None));
        }
        if n < GPU_ICC_CLUT_THRESHOLD {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, clut));
        }

        self.icc_cmyk_to_rgb_gpu(cmyk, clut)
    }

    /// Unconditional GPU dispatch for CMYK→RGB (skips threshold check).
    ///
    /// Use this when the caller has already decided GPU is appropriate
    /// (e.g. benchmarking or when the pixel count is known to be large).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4 or if the pixel count
    /// overflows `u32::MAX`.
    pub fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );
        let n = cmyk.len() / 4;
        let n_u32 = u32::try_from(n).expect("pixel count exceeds u32::MAX");
        let stream = &self.stream;

        let d_cmyk = stream.clone_htod(cmyk)?;
        let rgb_init = vec![0u8; n * 3];
        let mut d_rgb = stream.clone_htod(&rgb_init)?;

        let cfg = launch_cfg(n);

        match clut {
            None => {
                let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_matrix);
                let _ = builder.arg(&d_cmyk);
                let _ = builder.arg(&mut d_rgb);
                let _ = builder.arg(&n_u32);
                // SAFETY: 3 args match icc_cmyk_matrix PTX signature exactly.
                let _ = unsafe { builder.launch(cfg) }?;
            }
            Some((table, grid_n)) => {
                let d_clut = stream.clone_htod(table)?;
                let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_clut);
                let _ = builder.arg(&d_cmyk);
                let _ = builder.arg(&mut d_rgb);
                let _ = builder.arg(&d_clut);
                let _ = builder.arg(&grid_n);
                let _ = builder.arg(&n_u32);
                // SAFETY: 5 args match icc_cmyk_clut PTX signature exactly.
                let _ = unsafe { builder.launch(cfg) }?;
            }
        }

        stream.synchronize()?;
        let mut rgb = vec![0u8; n * 3];
        stream.memcpy_dtoh(&d_rgb, &mut rgb)?;
        Ok(rgb)
    }
}
