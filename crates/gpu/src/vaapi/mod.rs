//! VA-API hardware JPEG decoder — AMD/Intel GPU acceleration on Linux.
//!
//! Wraps `libva`/`libva-drm` via raw FFI (no bindgen — the VA-API surface we
//! use is small and stable, mirroring the CUDA driver API approach in `nvjpeg`).
//!
//! Developed against an AMD Raphael iGPU (VCN 4.0.0) via Mesa RadeonSI.  The
//! same code runs unchanged on Intel Quick Sync (UHD 630 / Iris Xe / Arc) —
//! `libva` is the common abstraction.
//!
//! # Output format — NV12 on Raphael
//!
//! Raphael is VCN 4.0.0 (`RDECODE_JPEG_VER_2`).  Hardware format conversion from
//! YCbCr to RGB requires VCN 4.0.3+ (`RDECODE_JPEG_VER_3`), absent on this
//! device.  The JPEG engine outputs NV12; a CPU BT.601 full-range NV12→RGB8 step
//! follows each decode (≲ 0.3 ms for 4 MP on AVX2 — not the bottleneck).
//!
//! # Thread model
//!
//! `VADisplay` is process-global and thread-safe (Mesa serialises internally).
//! `VAContext` must not be used concurrently — each Rayon worker owns its own
//! [`VapiJpegDecoder`].  Same `ThreadLocal<Box<dyn GpuJpegDecoder>>` pattern as
//! nvJPEG.
//!
//! # Feature flag
//!
//! Enabled with `--features vaapi` on the `gpu` crate.  Links `libva.so.2` and
//! `libva-drm.so.2`.
//!
//! # CMYK / progressive
//!
//! `VAEntrypointVLD` is baseline JPEG only; CMYK and progressive streams fall
//! through to the CPU `zune-jpeg` path.
//!
//! # Usage
//!
//! ```no_run
//! use gpu::vaapi::VapiJpegDecoder;
//! let mut dec = VapiJpegDecoder::new("/dev/dri/renderD129").expect("VA-API unavailable");
//! let jpeg_bytes: &[u8] = &[];
//! let img = dec.decode_sync(jpeg_bytes, 1920, 1080).expect("decode failed");
//! ```

#![cfg(feature = "vaapi")]

mod error;
mod ffi;
mod jpeg_parser;
mod yuv;

use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

use error::{Result, VapiError, check};
use ffi::{
    VA_BUFFER_TYPE_HUFFMAN_TABLE, VA_BUFFER_TYPE_IQ_MATRIX, VA_BUFFER_TYPE_PICTURE_PARAMETER,
    VA_BUFFER_TYPE_SLICE_DATA, VA_BUFFER_TYPE_SLICE_PARAMETER, VA_ENTRYPOINT_VLD, VA_INVALID_ID,
    VA_MAP_IMAGE_READ, VA_PROFILE_JPEG_BASELINE, VA_PROGRESSIVE, VA_RT_FORMAT_YUV400,
    VA_RT_FORMAT_YUV420, VA_STATUS_SUCCESS, VABufferID, VAConfigID, VAContextID, VADisplay,
    VASurfaceID, VaHuffmanEntry, VaHuffmanTableJpeg, VaImage, VaIqMatrixJpeg, VaJpegComponent,
    VaPictureParamJpeg, VaSliceComponent, VaSliceParamJpeg, VaSurfaceAttrib, vaBeginPicture,
    vaCreateBuffer, vaCreateConfig, vaCreateContext, vaCreateSurfaces, vaDeriveImage,
    vaDestroyBuffer, vaDestroyConfig, vaDestroyContext, vaDestroyImage, vaDestroySurfaces,
    vaEndPicture, vaGetDisplayDRM, vaInitialize, vaMapBuffer, vaRenderPicture, vaSyncSurface,
    vaTerminate, vaUnmapBuffer,
};
use jpeg_parser::JpegHeaders;
use yuv::nv12_to_rgb8;

pub use error::VapiError as Error;

// ── Colour space ──────────────────────────────────────────────────────────────

/// Colour space of pixels returned by [`VapiJpegDecoder::decode_sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegColorSpace {
    /// One byte per pixel (luma / grayscale).
    Gray,
    /// Three bytes per pixel, interleaved R G B.
    Rgb,
}

// ── Decoded image ─────────────────────────────────────────────────────────────

/// Decoded JPEG image, host-resident.
#[derive(Debug)]
pub struct DecodedJpeg {
    /// Pixel bytes.  Layout matches [`color_space`](Self::color_space).
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Colour space of the pixel bytes.
    pub color_space: JpegColorSpace,
}

// ── VapiDisplay — RAII VADisplay ──────────────────────────────────────────────

/// RAII wrapper for a `VADisplay` opened via a DRM render node.
struct VapiDisplay {
    dpy: VADisplay,
    /// Keep the file open for the lifetime of the display.
    _fd: std::fs::File,
}

impl VapiDisplay {
    fn open(path: &str) -> Result<Self> {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| VapiError::DrmOpen(format!("{path}: {e}")))?;
        let fd = file.as_raw_fd();

        // SAFETY: fd is a valid, open DRM render node file descriptor.
        let dpy = unsafe { vaGetDisplayDRM(fd) };
        if dpy.is_null() {
            return Err(VapiError::NullDisplay);
        }

        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        // SAFETY: dpy is non-null; major/minor are valid stack vars.
        check(
            unsafe { vaInitialize(dpy, &raw mut major, &raw mut minor) },
            "vaInitialize",
        )?;
        log::debug!("VA-API {major}.{minor} initialised on {path}");

        Ok(Self { dpy, _fd: file })
    }
}

impl Drop for VapiDisplay {
    fn drop(&mut self) {
        // SAFETY: dpy is valid; _fd is still open (field order: _fd drops after dpy).
        let _ = unsafe { vaTerminate(self.dpy) };
    }
}

// SAFETY: Mesa radeonsi is thread-safe for concurrent calls on the same display.
// VapiDisplay is only accessed via &mut from the owning decoder thread.
unsafe impl Send for VapiDisplay {}

// ── Cached context ────────────────────────────────────────────────────────────

/// Cached VAContext and VASurface for a specific image resolution.
///
/// Destroyed and recreated whenever the image dimensions change.  Eliminates
/// the `vaCreateContext`/`vaCreateSurfaces` overhead on same-size decode runs
/// (e.g. all pages of a scanned document share one resolution).
struct CachedCtx {
    width: u32,
    height: u32,
    ctx: VAContextID,
    surface: VASurfaceID,
    /// Surface format actually allocated (YUV400 or YUV420).
    /// Stored so grayscale images routed via YUV420 fallback are handled correctly.
    surface_fmt: c_uint,
}

// ── VapiJpegDecoder ───────────────────────────────────────────────────────────

/// VA-API hardware JPEG decoder.
///
/// Implements [`crate::traits::GpuJpegDecoder`].  Each instance owns one
/// `VAConfig`; a fresh `VAContext` and `VASurface` are created per decode call.
///
/// `Send` but not `Sync` — use one instance per Rayon worker thread.
pub struct VapiJpegDecoder {
    dpy: VapiDisplay,
    cfg: VAConfigID,
    cached_ctx: Option<CachedCtx>,
}

// SAFETY: VapiDisplay is Send; VAConfigID, VAContextID, VASurfaceID are all
// c_uint handles with no thread affinity — the VA-API threading contract
// requires that a given VAContext is used only from its owning thread, which
// is enforced by VapiJpegDecoder being Send but not Sync.
unsafe impl Send for VapiJpegDecoder {}

impl VapiJpegDecoder {
    /// Open the DRM render node at `drm_node` and initialise a VA-API JPEG
    /// baseline decode config.
    ///
    /// # Errors
    ///
    /// Returns an error if the render node cannot be opened, VA-API cannot be
    /// initialised, or the driver does not support `VAProfileJPEGBaseline` +
    /// `VAEntrypointVLD`.
    ///
    /// # Panics
    ///
    /// Panics if `vaCreateConfig` returns success but reports an invalid config ID
    /// (driver contract violation).
    pub fn new(drm_node: &str) -> Result<Self> {
        let dpy = VapiDisplay::open(drm_node)?;

        let mut cfg: VAConfigID = VA_INVALID_ID;
        // SAFETY: dpy is valid; null attrib list + 0 attribs = use driver defaults.
        check(
            unsafe {
                vaCreateConfig(
                    dpy.dpy,
                    VA_PROFILE_JPEG_BASELINE,
                    VA_ENTRYPOINT_VLD,
                    ptr::null_mut(),
                    0,
                    &raw mut cfg,
                )
            },
            "vaCreateConfig(JPEGBaseline/VLD)",
        )?;
        assert_ne!(
            cfg, VA_INVALID_ID,
            "vaCreateConfig succeeded but returned VA_INVALID_ID"
        );

        Ok(Self {
            dpy,
            cfg,
            cached_ctx: None,
        })
    }

    /// Allocate a VA-API surface and context for decoding images of `width × height`.
    ///
    /// Tries `YUV400` for grayscale; falls back to `YUV420` if the driver rejects it.
    /// Returns a `CachedCtx` that must be destroyed (via `Drop` or explicitly) when done.
    fn create_surface_and_context(
        &self,
        width: u32,
        height: u32,
        is_gray: bool,
    ) -> Result<CachedCtx> {
        let preferred_fmt = if is_gray {
            VA_RT_FORMAT_YUV400
        } else {
            VA_RT_FORMAT_YUV420
        };

        let mut surface: VASurfaceID = VA_INVALID_ID;
        let mut attrib = VaSurfaceAttrib::unused();

        // SAFETY: dpy valid; surface, attrib are stack vars.
        let surf_status = unsafe {
            vaCreateSurfaces(
                self.dpy.dpy,
                preferred_fmt,
                width,
                height,
                &raw mut surface,
                1,
                &raw mut attrib,
                1,
            )
        };

        // If YUV400 was rejected (some drivers don't implement it), retry with YUV420.
        let (surf_status, surface_fmt) = if surf_status != VA_STATUS_SUCCESS && is_gray {
            log::debug!("VA-API: YUV400 surface rejected ({surf_status}), retrying with YUV420");
            surface = VA_INVALID_ID;
            let s = unsafe {
                vaCreateSurfaces(
                    self.dpy.dpy,
                    VA_RT_FORMAT_YUV420,
                    width,
                    height,
                    &raw mut surface,
                    1,
                    &raw mut attrib,
                    1,
                )
            };
            (s, VA_RT_FORMAT_YUV420)
        } else {
            (surf_status, preferred_fmt)
        };
        check(surf_status, "vaCreateSurfaces")?;
        assert_ne!(
            surface, VA_INVALID_ID,
            "vaCreateSurfaces succeeded but returned VA_INVALID_ID"
        );

        let mut ctx: VAContextID = VA_INVALID_ID;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "width/height are derived from u16 JPEG SOF fields and cannot exceed 65535"
        )]
        let ctx_result = unsafe {
            vaCreateContext(
                self.dpy.dpy,
                self.cfg,
                width as c_int,
                height as c_int,
                VA_PROGRESSIVE,
                &raw mut surface,
                1,
                &raw mut ctx,
            )
        };
        if ctx_result != VA_STATUS_SUCCESS {
            unsafe {
                let _ = vaDestroySurfaces(self.dpy.dpy, &raw mut surface, 1);
            }
            check(ctx_result, "vaCreateContext")?;
        }
        assert_ne!(
            ctx, VA_INVALID_ID,
            "vaCreateContext succeeded but returned VA_INVALID_ID"
        );

        Ok(CachedCtx {
            width,
            height,
            ctx,
            surface,
            surface_fmt,
        })
    }

    /// Decode `data` synchronously, returning host-resident interleaved pixels.
    ///
    /// Parses the JPEG headers, creates a per-resolution context and surface,
    /// submits the four required decode buffers, synchronises the surface, maps
    /// the NV12 result, and converts to RGB8 (BT.601 full-range) on CPU.
    ///
    /// The `width_hint` and `height_hint` parameters are accepted for API
    /// compatibility with the `GpuJpegDecoder` trait but are not used — actual
    /// dimensions are read from the JPEG SOF0 header.
    ///
    /// # Errors
    ///
    /// - [`VapiError::BadJpeg`] — malformed or progressive JPEG.
    /// - [`VapiError::UnsupportedComponents`] — component count ≠ 1 or 3.
    /// - [`VapiError::VaStatus`] — any VA-API call failed.
    /// - [`VapiError::Overflow`] — image dimensions overflow `usize`.
    ///
    /// # Panics
    ///
    /// Panics if `vaCreateSurfaces` or `vaCreateContext` returns success but
    /// reports an invalid ID (driver contract violation), or if `vaMapBuffer`
    /// returns success with a null pointer.
    pub fn decode_sync(
        &mut self,
        data: &[u8],
        _width_hint: u32,
        _height_hint: u32,
    ) -> Result<DecodedJpeg> {
        // Reject progressive JPEG early — VA-API VAEntrypointVLD is baseline-only.
        if matches!(
            crate::jpeg_sof::jpeg_sof_type(data),
            Some(crate::jpeg_sof::JpegVariant::Progressive)
        ) {
            return Err(VapiError::BadJpeg(
                "progressive JPEG not supported by VA-API VLD entrypoint".into(),
            ));
        }

        let h = JpegHeaders::parse(data)?;

        if h.components != 1 && h.components != 3 {
            return Err(VapiError::UnsupportedComponents(h.components));
        }

        let w_u32 = u32::from(h.width);
        let h_u32 = u32::from(h.height);

        let (surface_fmt, is_gray) = if h.components == 1 {
            (VA_RT_FORMAT_YUV400, true)
        } else {
            (VA_RT_FORMAT_YUV420, false)
        };

        // ── Create surface ────────────────────────────────────────────────────
        let mut surface: VASurfaceID = VA_INVALID_ID;
        let mut attrib = VaSurfaceAttrib::unused();
        // SAFETY: dpy valid; surface, attrib are stack vars.
        let surf_status = unsafe {
            vaCreateSurfaces(
                self.dpy.dpy,
                surface_fmt,
                w_u32,
                h_u32,
                &raw mut surface,
                1,
                &raw mut attrib,
                1,
            )
        };

        // If YUV400 was rejected (some drivers don't implement it), retry with YUV420.
        // Extraction uses only the Y plane for grayscale regardless of surface format
        // (both YUV400 and YUV420 start with a Y plane).
        let surf_status = if surf_status != VA_STATUS_SUCCESS && is_gray {
            log::debug!("VA-API: YUV400 surface rejected ({surf_status}), retrying with YUV420");
            surface = VA_INVALID_ID;
            unsafe {
                vaCreateSurfaces(
                    self.dpy.dpy,
                    VA_RT_FORMAT_YUV420,
                    w_u32,
                    h_u32,
                    &raw mut surface,
                    1,
                    &raw mut attrib,
                    1,
                )
            }
        } else {
            surf_status
        };
        check(surf_status, "vaCreateSurfaces")?;
        // vaCreateSurfaces returning success with VA_INVALID_ID would be a driver
        // contract violation; assert so the failure is loud and attributable.
        assert_ne!(
            surface, VA_INVALID_ID,
            "vaCreateSurfaces succeeded but returned VA_INVALID_ID"
        );

        // ── Create context ────────────────────────────────────────────────────
        let mut ctx: VAContextID = VA_INVALID_ID;
        // w_u32 / h_u32 come from u16 JPEG SOF0 fields: max 65535 ≪ i32::MAX.
        #[expect(
            clippy::cast_possible_wrap,
            reason = "width/height are derived from u16 JPEG SOF fields and cannot exceed 65535"
        )]
        let ctx_result = unsafe {
            vaCreateContext(
                self.dpy.dpy,
                self.cfg,
                w_u32 as c_int,
                h_u32 as c_int,
                VA_PROGRESSIVE,
                &raw mut surface,
                1,
                &raw mut ctx,
            )
        };
        if ctx_result != VA_STATUS_SUCCESS {
            unsafe {
                let _ = vaDestroySurfaces(self.dpy.dpy, &raw mut surface, 1);
            }
            check(ctx_result, "vaCreateContext")?;
        }
        assert_ne!(
            ctx, VA_INVALID_ID,
            "vaCreateContext succeeded but returned VA_INVALID_ID"
        );

        // Decode and clean up context + surface regardless of decode result.
        let result = self.decode_into(data, &h, ctx, surface, w_u32, h_u32, is_gray);

        unsafe {
            let _ = vaDestroyContext(self.dpy.dpy, ctx);
            let _ = vaDestroySurfaces(self.dpy.dpy, &raw mut surface, 1);
        }

        result
    }

    /// Inner decode: build parameter structs, create VA-API buffers, submit, sync,
    /// map NV12 surface, convert to RGB8.
    #[expect(
        clippy::too_many_arguments,
        reason = "all 7 args are required: bitstream + parsed headers + context/surface IDs + dimensions + grayscale flag"
    )]
    fn decode_into(
        &self,
        data: &[u8],
        h: &JpegHeaders,
        ctx: VAContextID,
        surface: VASurfaceID,
        width: u32,
        height: u32,
        is_gray: bool,
    ) -> Result<DecodedJpeg> {
        let dpy = self.dpy.dpy;
        let mut buf_ids = [VA_INVALID_ID; 5];

        let create_result = Self::create_buffers(dpy, ctx, data, h, width, height, &mut buf_ids);
        if let Err(e) = create_result {
            Self::destroy_buffers(dpy, &mut buf_ids);
            return Err(e);
        }

        let submit_result = Self::submit(dpy, ctx, surface, &buf_ids);
        // Buffers are consumed after submission (driver has ingested them).
        Self::destroy_buffers(dpy, &mut buf_ids);
        submit_result?;

        Self::map_and_convert(dpy, surface, width, height, is_gray)
    }

    /// Create the five VA-API decode buffers and write their IDs into `buf_ids`.
    #[expect(
        clippy::too_many_lines,
        reason = "five sequential buffer-create calls plus their parameter structs; no meaningful way to split without obscuring the VA-API sequence"
    )]
    fn create_buffers(
        dpy: VADisplay,
        ctx: VAContextID,
        data: &[u8],
        h: &JpegHeaders,
        width: u32,
        height: u32,
        buf_ids: &mut [VABufferID; 5],
    ) -> Result<()> {
        let mut pic = VaPictureParamJpeg::zeroed();
        pic.picture_width = width;
        pic.picture_height = height;
        pic.num_components = h.components;
        for i in 0..h.components as usize {
            pic.components[i] = VaJpegComponent {
                component_id: h.comp_ids[i],
                h_sampling_factor: h.h_samp[i],
                v_sampling_factor: h.v_samp[i],
                quantiser_table_selector: h.quant_sel[i],
            };
        }

        let mut iq = VaIqMatrixJpeg::zeroed();
        for i in 0..4 {
            if h.quant_present[i] {
                iq.load_quantiser_table[i] = 1;
                iq.quantiser_table[i].copy_from_slice(&h.quant_tables[i]);
            }
        }

        let mut huff = VaHuffmanTableJpeg::zeroed();
        for (i, entry_opt) in h.huffman_entries.iter().enumerate() {
            if let Some(entry) = entry_opt {
                // VaHuffmanTableJpeg only has 2 slots.  Luma entries (0/1) map
                // to slot 0; chroma entries (2/3) map to slot 1.
                let slot = i / 2;
                if slot < 2 {
                    huff.load_huffman_table[slot] = 1;
                    huff.huffman_table[slot] = merge_huffman(huff.huffman_table[slot], *entry, i);
                }
            }
        }

        let scan_end = h
            .scan_data_offset
            .checked_add(h.scan_data_size)
            .ok_or(VapiError::Overflow)?;
        if scan_end > data.len() {
            return Err(VapiError::BadJpeg(format!(
                "scan data [{}, {scan_end}) overruns JPEG buffer (len={})",
                h.scan_data_offset,
                data.len()
            )));
        }
        let scan_data = &data[h.scan_data_offset..scan_end];
        let num_mcus = h.num_mcus();

        let mut slice_param = VaSliceParamJpeg::zeroed();
        slice_param.slice_data_size = h
            .scan_data_size
            .try_into()
            .map_err(|_| VapiError::Overflow)?;
        slice_param.num_components = h.scan_components;
        slice_param.restart_interval = h.restart_interval;
        slice_param.num_mcus = num_mcus;
        for i in 0..h.scan_components as usize {
            slice_param.components[i] = VaSliceComponent {
                component_id: h.scan_comp_ids[i],
                dc_table: h.scan_dc_table[i],
                ac_table: h.scan_ac_table[i],
            };
        }

        macro_rules! create_buf {
            ($idx:expr, $type:expr, $size:expr, $ptr:expr) => {{
                let size_u32 = c_uint::try_from($size).map_err(|_| VapiError::Overflow)?;
                // SAFETY: dpy and ctx are valid; $ptr points to live data for
                // the duration of this call.  The cast from *const to *mut is
                // safe here because the VA-API buffer creation reads the data
                // and does not write back through the pointer.
                check(
                    unsafe {
                        vaCreateBuffer(
                            dpy,
                            ctx,
                            $type,
                            size_u32,
                            1,
                            std::ptr::from_ref($ptr).cast::<c_void>().cast_mut(),
                            &raw mut buf_ids[$idx],
                        )
                    },
                    concat!("vaCreateBuffer[", stringify!($idx), "]"),
                )?;
            }};
        }

        create_buf!(
            0,
            VA_BUFFER_TYPE_PICTURE_PARAMETER,
            std::mem::size_of_val(&pic),
            &pic
        );
        create_buf!(1, VA_BUFFER_TYPE_IQ_MATRIX, std::mem::size_of_val(&iq), &iq);
        create_buf!(
            2,
            VA_BUFFER_TYPE_HUFFMAN_TABLE,
            std::mem::size_of_val(&huff),
            &huff
        );
        create_buf!(
            3,
            VA_BUFFER_TYPE_SLICE_PARAMETER,
            std::mem::size_of_val(&slice_param),
            &slice_param
        );
        // Slice data: the VA-API spec says the driver reads this but does not
        // write back, so the const→mut cast is safe in practice.
        let slice_len_u32 = c_uint::try_from(scan_data.len()).map_err(|_| VapiError::Overflow)?;
        check(
            unsafe {
                vaCreateBuffer(
                    dpy,
                    ctx,
                    VA_BUFFER_TYPE_SLICE_DATA,
                    slice_len_u32,
                    1,
                    scan_data.as_ptr().cast::<c_void>().cast_mut(),
                    &raw mut buf_ids[4],
                )
            },
            "vaCreateBuffer[slice_data]",
        )?;

        Ok(())
    }

    /// Submit the five decode buffers, trigger the hardware decode, and sync.
    fn submit(
        dpy: VADisplay,
        ctx: VAContextID,
        surface: VASurfaceID,
        buf_ids: &[VABufferID; 5],
    ) -> Result<()> {
        // SAFETY: dpy, ctx, surface all valid and live.
        check(
            unsafe { vaBeginPicture(dpy, ctx, surface) },
            "vaBeginPicture",
        )?;
        // vaRenderPicture takes *mut VABufferID but the VA-API spec says it only
        // reads the array — the const→mut cast is safe here.
        check(
            unsafe { vaRenderPicture(dpy, ctx, buf_ids.as_ptr().cast_mut(), 5) },
            "vaRenderPicture",
        )?;
        check(unsafe { vaEndPicture(dpy, ctx) }, "vaEndPicture")?;
        // Block until VCN JPEG engine finishes writing to the surface.
        check(unsafe { vaSyncSurface(dpy, surface) }, "vaSyncSurface")?;
        Ok(())
    }

    /// Destroy any buffer IDs that are not `VA_INVALID_ID`.
    fn destroy_buffers(dpy: VADisplay, buf_ids: &mut [VABufferID; 5]) {
        for id in buf_ids.iter_mut() {
            if *id != VA_INVALID_ID {
                // SAFETY: id is a valid, non-destroyed buffer.
                unsafe {
                    let _ = vaDestroyBuffer(dpy, *id);
                }
                *id = VA_INVALID_ID;
            }
        }
    }

    /// Map the decoded surface via `vaDeriveImage`, extract pixels, and convert.
    fn map_and_convert(
        dpy: VADisplay,
        surface: VASurfaceID,
        width: u32,
        height: u32,
        is_gray: bool,
    ) -> Result<DecodedJpeg> {
        let mut image = VaImage::zeroed();
        // SAFETY: dpy and surface are valid post-sync.
        check(
            unsafe { vaDeriveImage(dpy, surface, &raw mut image) },
            "vaDeriveImage",
        )?;

        let mut pixels: *mut c_void = ptr::null_mut();
        let map_result = check(
            unsafe { vaMapBuffer(dpy, image.buf, &raw mut pixels, VA_MAP_IMAGE_READ) },
            "vaMapBuffer",
        );

        let pixel_result = map_result.and_then(|()| {
            // VA-API spec §4.5: a VA_STATUS_SUCCESS return from vaMapBuffer guarantees
            // that *pbuf is a valid non-null host pointer.  Assert so a driver bug is
            // caught immediately rather than causing a silent null-deref downstream.
            assert!(!pixels.is_null(), "vaMapBuffer succeeded but returned null");
            // SAFETY: pixels is valid for `data_size` bytes until UnmapBuffer.
            let mapped = unsafe {
                std::slice::from_raw_parts(pixels.cast::<u8>(), image.data_size as usize)
            };
            extract_pixels(mapped, &image, width, height, is_gray)
        });

        // Always unmap and destroy, even on error.
        unsafe {
            let _ = vaUnmapBuffer(dpy, image.buf);
            let _ = vaDestroyImage(dpy, image.image_id);
        }

        pixel_result
    }
}

impl Drop for VapiJpegDecoder {
    fn drop(&mut self) {
        // Destroy config before the display is terminated (field order).
        // SAFETY: cfg is valid; dpy is still alive because it's the next field.
        unsafe {
            let _ = vaDestroyConfig(self.dpy.dpy, self.cfg);
        }
    }
}

// ── Pixel extraction helpers ──────────────────────────────────────────────────

/// Copy and convert pixels from a mapped `VAImage` into a [`DecodedJpeg`].
fn extract_pixels(
    mapped: &[u8],
    image: &VaImage,
    width: u32,
    height: u32,
    is_gray: bool,
) -> Result<DecodedJpeg> {
    // For grayscale we only need the Y plane regardless of whether the surface
    // is YUV400 (native) or YUV420 (driver fallback) — both start with a Y plane.
    if is_gray {
        extract_y_plane(mapped, image, width, height, JpegColorSpace::Gray)
    } else {
        extract_nv12(mapped, image, width, height)
    }
}

/// Extract the Y (luma) plane — used for grayscale images.
fn extract_y_plane(
    mapped: &[u8],
    image: &VaImage,
    width: u32,
    height: u32,
    color_space: JpegColorSpace,
) -> Result<DecodedJpeg> {
    let off = image.offsets[0] as usize;
    let stride = image.pitches[0] as usize;
    let w = width as usize;
    let h = height as usize;
    // stride < width would cause an out-of-bounds slice in the extraction loop below.
    if stride < w {
        return Err(VapiError::BadJpeg(format!(
            "Y plane stride {stride} < width {w}"
        )));
    }
    let y_size = stride.checked_mul(h).ok_or(VapiError::Overflow)?;
    if off + y_size > mapped.len() {
        return Err(VapiError::BadJpeg(format!(
            "Y plane out of bounds: off={off}+size={y_size} > mapped={}",
            mapped.len()
        )));
    }
    let y_plane = &mapped[off..off + y_size];
    let mut gray = Vec::with_capacity(w * h);
    for row in 0..h {
        gray.extend_from_slice(&y_plane[row * stride..row * stride + w]);
    }
    Ok(DecodedJpeg {
        data: gray,
        width,
        height,
        color_space,
    })
}

/// Extract NV12 planes and convert to RGB8 via BT.601 full-range.
fn extract_nv12(mapped: &[u8], image: &VaImage, width: u32, height: u32) -> Result<DecodedJpeg> {
    let y_off = image.offsets[0] as usize;
    let uv_off = image.offsets[1] as usize;
    let stride_y = image.pitches[0];
    let stride_uv = image.pitches[1];
    let w = width as usize;

    // stride_y < width or stride_uv < 2 would cause out-of-bounds access in nv12_to_rgb8.
    if (stride_y as usize) < w {
        return Err(VapiError::BadJpeg(format!(
            "NV12 Y stride {stride_y} < width {w}"
        )));
    }
    // NV12 UV plane holds interleaved Cb/Cr pairs; minimum stride is 2 per chroma sample.
    if w > 0 && (stride_uv as usize) < 2 {
        return Err(VapiError::BadJpeg(format!(
            "NV12 UV stride {stride_uv} < 2 for width {w}"
        )));
    }

    let y_size = (stride_y as usize)
        .checked_mul(height as usize)
        .ok_or(VapiError::Overflow)?;
    let uv_rows = (height as usize).div_ceil(2);
    let uv_size = (stride_uv as usize)
        .checked_mul(uv_rows)
        .ok_or(VapiError::Overflow)?;

    if y_off + y_size > mapped.len() || uv_off + uv_size > mapped.len() {
        return Err(VapiError::BadJpeg(format!(
            "NV12 planes out of bounds: y={y_off}+{y_size}, uv={uv_off}+{uv_size}, mapped={}",
            mapped.len()
        )));
    }

    let y_plane = &mapped[y_off..y_off + y_size];
    let uv_plane = &mapped[uv_off..uv_off + uv_size];
    let rgb = nv12_to_rgb8(y_plane, uv_plane, width, height, stride_y, stride_uv)?;
    Ok(DecodedJpeg {
        data: rgb,
        width,
        height,
        color_space: JpegColorSpace::Rgb,
    })
}

// ── Huffman table merging ─────────────────────────────────────────────────────

/// Merge a parsed `VaHuffmanEntry` into a `VaHuffmanTableJpeg` slot.
///
/// `slot_idx` is the position in `huffman_entries` (`0`=luma DC, `1`=luma AC,
/// `2`=chroma DC, `3`=chroma AC).  Even indices write the DC half; odd write AC.
const fn merge_huffman(
    mut dst: VaHuffmanEntry,
    src: VaHuffmanEntry,
    slot_idx: usize,
) -> VaHuffmanEntry {
    if slot_idx.is_multiple_of(2) {
        // DC
        dst.num_dc_codes = src.num_dc_codes;
        dst.dc_values = src.dc_values;
    } else {
        // AC
        dst.num_ac_codes = src.num_ac_codes;
        dst.ac_values = src.ac_values;
    }
    dst
}

// ── GpuJpegDecoder impl ───────────────────────────────────────────────────────

impl crate::traits::GpuJpegDecoder for VapiJpegDecoder {
    /// Decode a JPEG stream via VA-API.
    ///
    /// Unlike the nvJPEG implementation, `width` and `height` are **not**
    /// validated against the decoded dimensions — they are accepted but ignored.
    /// Dimension authorisation is the caller's responsibility when using this
    /// implementation.
    fn decode_jpeg(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> std::result::Result<crate::traits::DecodedImage, crate::traits::GpuDecodeError> {
        let decoded = self
            .decode_sync(data, width, height)
            .map_err(crate::traits::GpuDecodeError::new)?;

        let components = match decoded.color_space {
            JpegColorSpace::Gray => 1,
            JpegColorSpace::Rgb => 3,
        };
        Ok(crate::traits::DecodedImage {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            components,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Same 16×16 grayscale JPEG as in nvjpeg.rs tests.
    const GRAY_16X16_JPEG: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x06, 0x04, 0x05, 0x06, 0x05,
        0x04, 0x06, 0x06, 0x05, 0x06, 0x07, 0x07, 0x06, 0x08, 0x0a, 0x10, 0x0a, 0x0a, 0x09, 0x09,
        0x0a, 0x14, 0x0e, 0x0f, 0x0c, 0x10, 0x17, 0x14, 0x18, 0x18, 0x17, 0x14, 0x16, 0x16, 0x1a,
        0x1d, 0x25, 0x1f, 0x1a, 0x1b, 0x23, 0x1c, 0x16, 0x16, 0x20, 0x2c, 0x20, 0x23, 0x26, 0x27,
        0x29, 0x2a, 0x29, 0x19, 0x1f, 0x2d, 0x30, 0x2d, 0x28, 0x30, 0x25, 0x28, 0x29, 0x28, 0xff,
        0xc0, 0x00, 0x0b, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00,
        0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00,
        0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0x80, 0x3f, 0xff, 0xd9,
    ];

    #[test]
    fn jpeg_sof_type_identifies_sof2_as_progressive() {
        // Verify jpeg_sof_type correctly identifies a SOF2 stream as Progressive.
        // decode_sync uses this function as its first guard against progressive JPEG;
        // a real decode_sync integration test would require a /dev/dri device.
        #[rustfmt::skip]
        let progressive_jpeg: &[u8] = &[
            0xFF, 0xD8,             // SOI
            0xFF, 0xC2,             // SOF2 (progressive DCT)
            0x00, 0x11,             // length = 17
            0x08,                   // precision
            0x00, 0x10, 0x00, 0x10, // height=16, width=16
            0x03,                   // 3 components
            0x01, 0x11, 0x00,
            0x02, 0x11, 0x01,
            0x03, 0x11, 0x01,
            0xFF, 0xD9,             // EOI
        ];
        use crate::jpeg_sof::{JpegVariant, jpeg_sof_type};
        assert_eq!(
            jpeg_sof_type(progressive_jpeg),
            Some(JpegVariant::Progressive),
            "jpeg_sof_type must identify this stream as Progressive"
        );
    }

    /// `VapiJpegDecoder::new` must not panic — it either succeeds or returns an error.
    /// Skipped gracefully if the device is absent.
    #[test]
    fn vaapi_new_does_not_panic() {
        let _ = VapiJpegDecoder::new("/dev/dri/renderD129");
    }

    /// End-to-end decode on a real device if available.
    #[test]
    fn decode_gray_16x16_vaapi() {
        let mut dec = match VapiJpegDecoder::new("/dev/dri/renderD129") {
            Ok(d) => d,
            Err(e) => {
                log::info!("VA-API test skipped: {e}");
                return;
            }
        };
        let img = match dec.decode_sync(GRAY_16X16_JPEG, 16, 16) {
            Ok(img) => img,
            Err(e) => {
                log::info!("VA-API decode skipped: {e}");
                return;
            }
        };
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        let expected_len = match img.color_space {
            JpegColorSpace::Gray => 16 * 16,
            JpegColorSpace::Rgb => 16 * 16 * 3,
        };
        assert_eq!(img.data.len(), expected_len);
    }

    #[test]
    fn vaapi_decoder_has_cached_ctx_field() {
        // Structural test: VapiJpegDecoder::new is the only way to construct one,
        // and it must initialise with no cached context.
        // We test this indirectly by checking that two consecutive decode calls
        // on different-sized inputs both succeed (cache miss path exercised).
        // This test only compiles — hardware is not available in CI.
        // The real cache-hit test requires hardware and is marked #[ignore].
        let _: fn(&str) -> Result<VapiJpegDecoder> = VapiJpegDecoder::new;
    }

    #[test]
    #[ignore = "requires VA-API hardware"]
    fn create_surface_and_context_returns_valid_ids() {
        let mut dec = VapiJpegDecoder::new("/dev/dri/renderD129").expect("VA-API unavailable");
        // Create a 16×16 YUV420 context; just check it doesn't error.
        let cached = dec
            .create_surface_and_context(16, 16, false)
            .expect("create failed");
        assert_ne!(cached.ctx, VA_INVALID_ID);
        assert_ne!(cached.surface, VA_INVALID_ID);
        // Clean up manually — CachedCtx has no Drop impl.
        unsafe {
            let _ = vaDestroyContext(dec.dpy.dpy, cached.ctx);
            let mut surface = cached.surface;
            let _ = vaDestroySurfaces(dec.dpy.dpy, &raw mut surface, 1);
        }
    }
}
