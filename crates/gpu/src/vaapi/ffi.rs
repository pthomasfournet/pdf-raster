//! Raw FFI declarations for `libva.so.2` and `libva-drm.so.2`.
//!
//! We declare only the VA-API surface we use — the JPEG baseline decode path.
//! This avoids bindgen at build time while keeping the binding surface minimal
//! and auditable.  All types mirror `<va/va.h>` and `<va/va_drm.h>` exactly.

#![cfg(feature = "vaapi")]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ── Type aliases ──────────────────────────────────────────────────────────────

pub(super) type VADisplay = *mut c_void;
pub(super) type VAStatus = c_int;
pub(super) type VAConfigID = c_uint;
pub(super) type VAContextID = c_uint;
pub(super) type VASurfaceID = c_uint;
pub(super) type VABufferID = c_uint;

/// `VAProfileJPEGBaseline` — JPEG baseline profile identifier.
pub(super) const VA_PROFILE_JPEG_BASELINE: c_int = 6;
/// `VAEntrypointVLD` — variable-length decode entrypoint.
pub(super) const VA_ENTRYPOINT_VLD: c_int = 1;

/// `VA_RT_FORMAT_YUV420` — NV12 output (Y plane + interleaved UV half-plane).
pub(super) const VA_RT_FORMAT_YUV420: c_uint = 0x0000_0001;
/// `VA_RT_FORMAT_YUV400` — grayscale (Y plane only).
pub(super) const VA_RT_FORMAT_YUV400: c_uint = 0x0000_0008;

/// `VA_PROGRESSIVE` flag for `vaCreateContext`.
pub(super) const VA_PROGRESSIVE: c_int = 0x1;

/// VA-API success status code.
pub(super) const VA_STATUS_SUCCESS: VAStatus = 0;

/// Sentinel value meaning "no valid buffer / surface / config".
pub(super) const VA_INVALID_ID: c_uint = 0xFFFF_FFFF;

/// `VAPictureParameterBufferType` — JPEG picture-level parameters.
pub(super) const VA_BUFFER_TYPE_PICTURE_PARAMETER: c_uint = 0;
/// `VAIQMatrixBufferType` — quantisation matrix.
pub(super) const VA_BUFFER_TYPE_IQ_MATRIX: c_uint = 2;
/// `VAHuffmanTableBufferType` — Huffman tables (JPEG-specific, value 12).
pub(super) const VA_BUFFER_TYPE_HUFFMAN_TABLE: c_uint = 12;
/// `VASliceParameterBufferType` — per-scan decode parameters.
pub(super) const VA_BUFFER_TYPE_SLICE_PARAMETER: c_uint = 8;
/// `VASliceDataBufferType` — raw encoded bitstream data.
pub(super) const VA_BUFFER_TYPE_SLICE_DATA: c_uint = 9;

/// `VA_MAP_IMAGE_READ` — map image buffer as read-only.
pub(super) const VA_MAP_IMAGE_READ: c_uint = 0x01;

// ── Surface attribute ─────────────────────────────────────────────────────────

/// `VASurfaceAttrib` — passed to `vaCreateSurfaces`.
///
/// For a default (unconstrained) surface, use `VaSurfaceAttrib::unused()`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VaSurfaceAttrib {
    type_: c_uint,
    flags: c_uint,
    value: VaSurfaceAttribValue,
}

#[repr(C)]
#[derive(Clone, Copy)]
union VaSurfaceAttribValue {
    _i: c_int,
    _ui: c_uint,
    _p: *mut c_void,
}

impl VaSurfaceAttrib {
    /// A no-op attribute: tells the driver to apply defaults.
    pub(super) const fn unused() -> Self {
        Self {
            type_: 0, // VA_SURFACE_ATTRIB_TYPE_NONE
            flags: 0,
            value: VaSurfaceAttribValue { _ui: 0 },
        }
    }
}

// SAFETY: the union pointer field is only ever null (unused sentinel value).
unsafe impl Send for VaSurfaceAttrib {}

// ── VAImage ───────────────────────────────────────────────────────────────────

/// `VAImage` — describes a mapped surface image (pixel layout, pitches, offsets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaImage {
    pub(super) image_id: VABufferID,
    pub(super) format: VaImageFormat,
    pub(super) buf: VABufferID,
    pub(super) width: c_uint,
    pub(super) height: c_uint,
    pub(super) data_size: c_uint,
    pub(super) num_planes: c_uint,
    pub(super) offsets: [c_uint; 3],
    pub(super) pitches: [c_uint; 3],
    _reserved: [c_uint; 8],
}

impl VaImage {
    pub(super) const fn zeroed() -> Self {
        Self {
            image_id: VA_INVALID_ID,
            format: VaImageFormat::zeroed(),
            buf: VA_INVALID_ID,
            width: 0,
            height: 0,
            data_size: 0,
            num_planes: 0,
            offsets: [0; 3],
            pitches: [0; 3],
            _reserved: [0; 8],
        }
    }
}

/// `VAImageFormat` — pixel format descriptor embedded in `VAImage`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaImageFormat {
    pub(super) fourcc: c_uint,
    _byte_order: c_uint,
    _bits_per_pixel: c_uint,
    _depth: c_uint,
    _red_mask: c_uint,
    _green_mask: c_uint,
    _blue_mask: c_uint,
    _alpha_mask: c_uint,
    _reserved: [c_uint; 4],
}

impl VaImageFormat {
    pub(super) const fn zeroed() -> Self {
        Self {
            fourcc: 0,
            _byte_order: 0,
            _bits_per_pixel: 0,
            _depth: 0,
            _red_mask: 0,
            _green_mask: 0,
            _blue_mask: 0,
            _alpha_mask: 0,
            _reserved: [0; 4],
        }
    }
}

// ── JPEG parameter buffer types ───────────────────────────────────────────────

/// `VAPictureParameterBufferJPEGBaseline` — top-level picture parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaPictureParamJpeg {
    pub(super) picture_width: c_uint,
    pub(super) picture_height: c_uint,
    pub(super) components: [VaJpegComponent; 4],
    pub(super) num_components: u8,
    pub(super) color_space: u8,
    pub(super) rotation: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct VaJpegComponent {
    pub(super) component_id: u8,
    pub(super) h_sampling_factor: u8,
    pub(super) v_sampling_factor: u8,
    pub(super) quantiser_table_selector: u8,
}

impl VaPictureParamJpeg {
    pub(super) const fn zeroed() -> Self {
        const ZERO_COMP: VaJpegComponent = VaJpegComponent {
            component_id: 0,
            h_sampling_factor: 0,
            v_sampling_factor: 0,
            quantiser_table_selector: 0,
        };
        Self {
            picture_width: 0,
            picture_height: 0,
            components: [ZERO_COMP; 4],
            num_components: 0,
            color_space: 0,
            rotation: 0,
        }
    }
}

/// `VAIQMatrixBufferJPEGBaseline` — up to four 64-entry quantisation tables.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaIqMatrixJpeg {
    pub(super) load_quantiser_table: [u8; 4],
    pub(super) quantiser_table: [[u8; 64]; 4],
}

impl VaIqMatrixJpeg {
    pub(super) const fn zeroed() -> Self {
        Self {
            load_quantiser_table: [0; 4],
            quantiser_table: [[0; 64]; 4],
        }
    }
}

/// `VAHuffmanTableBufferJPEGBaseline` — up to two DC+AC Huffman table pairs.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaHuffmanTableJpeg {
    pub(super) load_huffman_table: [u8; 2],
    pub(super) huffman_table: [VaHuffmanEntry; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaHuffmanEntry {
    pub(super) num_dc_codes: [u8; 16],
    pub(super) dc_values: [u8; 12],
    pub(super) num_ac_codes: [u8; 16],
    pub(super) ac_values: [u8; 162],
    pub(super) _pad: [u8; 2],
}

impl VaHuffmanTableJpeg {
    pub(super) const fn zeroed() -> Self {
        const ZERO_ENTRY: VaHuffmanEntry = VaHuffmanEntry {
            num_dc_codes: [0; 16],
            dc_values: [0; 12],
            num_ac_codes: [0; 16],
            ac_values: [0; 162],
            _pad: [0; 2],
        };
        Self {
            load_huffman_table: [0; 2],
            huffman_table: [ZERO_ENTRY; 2],
        }
    }
}

/// `VASliceParameterBufferJPEGBaseline` — per-scan decode parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct VaSliceParamJpeg {
    pub(super) slice_data_size: c_uint,
    pub(super) slice_data_offset: c_uint,
    pub(super) slice_data_flag: c_uint,
    pub(super) slice_horizontal_position: c_uint,
    pub(super) slice_vertical_position: c_uint,
    pub(super) components: [VaSliceComponent; 4],
    pub(super) num_components: u8,
    pub(super) restart_interval: u16,
    pub(super) num_mcus: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct VaSliceComponent {
    pub(super) component_id: u8,
    pub(super) dc_table: u8,
    pub(super) ac_table: u8,
}

impl VaSliceParamJpeg {
    pub(super) const fn zeroed() -> Self {
        const ZERO_COMP: VaSliceComponent = VaSliceComponent {
            component_id: 0,
            dc_table: 0,
            ac_table: 0,
        };
        Self {
            slice_data_size: 0,
            slice_data_offset: 0,
            slice_data_flag: 0,
            slice_horizontal_position: 0,
            slice_vertical_position: 0,
            components: [ZERO_COMP; 4],
            num_components: 0,
            restart_interval: 0,
            num_mcus: 0,
        }
    }
}

// ── Extern functions ──────────────────────────────────────────────────────────

unsafe extern "C" {
    // libva-drm.so.2
    pub(super) fn vaGetDisplayDRM(fd: c_int) -> VADisplay;

    // libva.so.2
    pub(super) fn vaInitialize(
        dpy: VADisplay,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> VAStatus;
    pub(super) fn vaTerminate(dpy: VADisplay) -> VAStatus;
    pub(super) fn vaErrorStr(error_status: VAStatus) -> *const c_char;
    pub(super) fn vaCreateConfig(
        dpy: VADisplay,
        profile: c_int,
        entrypoint: c_int,
        attrib_list: *mut c_void,
        num_attribs: c_int,
        config_id: *mut VAConfigID,
    ) -> VAStatus;
    pub(super) fn vaDestroyConfig(dpy: VADisplay, config_id: VAConfigID) -> VAStatus;
    pub(super) fn vaCreateContext(
        dpy: VADisplay,
        config_id: VAConfigID,
        picture_width: c_int,
        picture_height: c_int,
        flag: c_int,
        render_targets: *mut VASurfaceID,
        num_render_targets: c_int,
        context: *mut VAContextID,
    ) -> VAStatus;
    pub(super) fn vaDestroyContext(dpy: VADisplay, context: VAContextID) -> VAStatus;
    pub(super) fn vaCreateSurfaces(
        dpy: VADisplay,
        format: c_uint,
        width: c_uint,
        height: c_uint,
        surfaces: *mut VASurfaceID,
        num_surfaces: c_uint,
        attrib_list: *mut VaSurfaceAttrib,
        num_attribs: c_uint,
    ) -> VAStatus;
    pub(super) fn vaDestroySurfaces(
        dpy: VADisplay,
        surfaces: *mut VASurfaceID,
        num_surfaces: c_int,
    ) -> VAStatus;
    pub(super) fn vaCreateBuffer(
        dpy: VADisplay,
        context: VAContextID,
        type_: c_uint,
        size: c_uint,
        num_elements: c_uint,
        data: *mut c_void,
        buf_id: *mut VABufferID,
    ) -> VAStatus;
    pub(super) fn vaDestroyBuffer(dpy: VADisplay, buffer_id: VABufferID) -> VAStatus;
    pub(super) fn vaBeginPicture(
        dpy: VADisplay,
        context: VAContextID,
        render_target: VASurfaceID,
    ) -> VAStatus;
    pub(super) fn vaRenderPicture(
        dpy: VADisplay,
        context: VAContextID,
        buffers: *mut VABufferID,
        num_buffers: c_int,
    ) -> VAStatus;
    pub(super) fn vaEndPicture(dpy: VADisplay, context: VAContextID) -> VAStatus;
    pub(super) fn vaSyncSurface(dpy: VADisplay, render_target: VASurfaceID) -> VAStatus;
    pub(super) fn vaDeriveImage(
        dpy: VADisplay,
        surface: VASurfaceID,
        image: *mut VaImage,
    ) -> VAStatus;
    pub(super) fn vaDestroyImage(dpy: VADisplay, image: VABufferID) -> VAStatus;
    pub(super) fn vaMapBuffer(
        dpy: VADisplay,
        buf_id: VABufferID,
        pbuf: *mut *mut c_void,
        flags: c_uint,
    ) -> VAStatus;
    pub(super) fn vaUnmapBuffer(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;
}
