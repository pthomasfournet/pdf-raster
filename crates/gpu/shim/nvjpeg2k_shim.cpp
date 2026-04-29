// C++ shim: wraps nvjpeg2kStreamParse and nvjpeg2kDecode in try/catch so that
// C++ exceptions thrown by the library cannot propagate through Rust FFI
// (which is undefined behaviour).  Any exception is caught and converted to
// NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED (9), causing the Rust caller
// to fall back to the CPU OpenJPEG path.

#include <nvjpeg2k.h>

extern "C" {

nvjpeg2kStatus_t
nvjpeg2k_shim_stream_parse(nvjpeg2kHandle_t      handle,
                            const unsigned char  *data,
                            size_t                length,
                            int                   save_metadata,
                            int                   save_stream,
                            nvjpeg2kStream_t      stream)
{
    try {
        return nvjpeg2kStreamParse(handle, data, length,
                                   save_metadata, save_stream, stream);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_decode(nvjpeg2kHandle_t       handle,
                     nvjpeg2kDecodeState_t  decode_state,
                     nvjpeg2kStream_t       j2k_stream,
                     nvjpeg2kImage_t       *output,
                     cudaStream_t           cuda_stream)
{
    try {
        return nvjpeg2kDecode(handle, decode_state, j2k_stream,
                              output, cuda_stream);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_create_simple(nvjpeg2kHandle_t *handle)
{
    try {
        return nvjpeg2kCreateSimple(handle);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_decode_state_create(nvjpeg2kHandle_t handle,
                                   nvjpeg2kDecodeState_t *decode_state)
{
    try {
        return nvjpeg2kDecodeStateCreate(handle, decode_state);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_stream_create(nvjpeg2kStream_t *stream)
{
    try {
        return nvjpeg2kStreamCreate(stream);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_stream_get_image_info(nvjpeg2kStream_t stream,
                                     nvjpeg2kImageInfo_t *image_info)
{
    try {
        return nvjpeg2kStreamGetImageInfo(stream, image_info);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

nvjpeg2kStatus_t
nvjpeg2k_shim_stream_get_image_component_info(nvjpeg2kStream_t stream,
                                               nvjpeg2kImageComponentInfo_t *comp_info,
                                               uint32_t component_id)
{
    try {
        return nvjpeg2kStreamGetImageComponentInfo(stream, comp_info, component_id);
    } catch (...) {
        return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    }
}

} // extern "C"
