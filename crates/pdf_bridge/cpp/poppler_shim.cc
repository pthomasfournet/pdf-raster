// C shim over poppler-cpp so Rust can call it without C++ name mangling.
//
// All entry points are `extern "C"` to give stable symbol names.  Ownership
// semantics: opaque pointers returned from `_new`/`_load` functions must be
// freed by the matching `_free` function.  No exceptions cross the boundary —
// errors are signalled by returning null or a sentinel value.

#include "poppler-document.h"
#include "poppler-page.h"
#include "poppler-page-renderer.h"
#include "poppler-image.h"
#include "poppler-global.h"

// poppler-version.h is cmake-generated; declare the symbols directly.
namespace poppler {
    unsigned int version_major();
    unsigned int version_minor();
    unsigned int version_micro();
}

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Opaque wrapper types
// ---------------------------------------------------------------------------

struct PopplerShimDocument {
    std::unique_ptr<poppler::document> doc;
};

struct PopplerShimPage {
    std::unique_ptr<poppler::page> page;
};

struct PopplerShimImage {
    poppler::image img;
};

// ---------------------------------------------------------------------------
// Library initialisation
// ---------------------------------------------------------------------------

// Rust-provided callback: receives a null-terminated message string.
// Function pointer type uses extern "C" calling convention so Rust can supply
// a plain fn pointer without name mangling.
using RustLogFn = void (*)(const char *msg);

// Atomic so concurrent reads from poppler's error callback (which may fire on
// any thread) are safe against a concurrent write from set_log_callback.
// relaxed store/load is sufficient: we only need to see either null or a valid
// stable function pointer, never a partially-written value.
static std::atomic<RustLogFn> g_rust_log{nullptr};

static void poppler_log_bridge(const std::string &msg, void * /*closure*/) {
    // Load once; if null between check and call that is impossible (fn ptrs
    // are pointer-sized and atomically loaded).
    RustLogFn fn = g_rust_log.load(std::memory_order_relaxed);
    if (fn) fn(msg.c_str());
}

/// Install a Rust log callback and redirect poppler's stderr through it.
///
/// Safe to call from any thread.  Must be called before any document is opened
/// to guarantee all messages are captured; later calls update the pointer
/// atomically.  Pass null to revert to silent discard.
extern "C" void poppler_shim_set_log_callback(RustLogFn fn) {
    g_rust_log.store(fn, std::memory_order_relaxed);
    // Wire poppler's output through our bridge unconditionally.  When fn is
    // null the bridge still runs but discards messages — better than the
    // default which writes directly to stderr.
    poppler::set_debug_error_function(poppler_log_bridge, nullptr);
}

extern "C" void poppler_shim_set_data_dir(const char *path) {
    if (path) poppler::set_data_dir(std::string(path));
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// Open a PDF from the filesystem.  Returns null on failure.
extern "C" PopplerShimDocument *poppler_shim_document_load_from_file(
        const char *filename,
        const char *owner_password,
        const char *user_password) {
    std::string fn(filename);
    std::string op(owner_password ? owner_password : "");
    std::string up(user_password  ? user_password  : "");
    auto doc = poppler::document::load_from_file(fn, op, up);
    if (!doc || doc->is_locked()) {
        return nullptr;
    }
    auto *shim = new PopplerShimDocument{};
    shim->doc.reset(doc);
    return shim;
}

/// Open a PDF from an in-memory buffer.  Returns null on failure.
/// The caller retains ownership of `data`; this function copies it.
extern "C" PopplerShimDocument *poppler_shim_document_load_from_data(
        const char *data, int len,
        const char *owner_password,
        const char *user_password) {
    std::string op(owner_password ? owner_password : "");
    std::string up(user_password  ? user_password  : "");
    auto doc = poppler::document::load_from_raw_data(data, len, op, up);
    if (!doc || doc->is_locked()) {
        return nullptr;
    }
    auto *shim = new PopplerShimDocument{};
    shim->doc.reset(doc);
    return shim;
}

extern "C" void poppler_shim_document_free(PopplerShimDocument *d) {
    delete d;
}

/// Returns the number of pages, or -1 on error.
extern "C" int poppler_shim_document_pages(const PopplerShimDocument *d) {
    if (!d) return -1;
    return d->doc->pages();
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/// 0-indexed page access.  Returns null if index is out of range.
extern "C" PopplerShimPage *poppler_shim_document_create_page(
        const PopplerShimDocument *d, int index) {
    if (!d) return nullptr;
    auto *p = d->doc->create_page(index);
    if (!p) return nullptr;
    auto *shim = new PopplerShimPage{};
    shim->page.reset(p);
    return shim;
}

extern "C" void poppler_shim_page_free(PopplerShimPage *p) {
    delete p;
}

/// Page width in points (1/72 inch).
extern "C" double poppler_shim_page_width(const PopplerShimPage *p) {
    if (!p) return 0.0;
    return p->page->page_rect().width();
}

/// Page height in points (1/72 inch).
extern "C" double poppler_shim_page_height(const PopplerShimPage *p) {
    if (!p) return 0.0;
    return p->page->page_rect().height();
}

/// Page rotation in degrees (0, 90, 180, 270).
extern "C" int poppler_shim_page_rotation(const PopplerShimPage *p) {
    if (!p) return 0;
    switch (p->page->orientation()) {
        case poppler::page::landscape:  return 90;
        case poppler::page::upside_down: return 180;
        case poppler::page::seascape:   return 270;
        default:                        return 0;
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Image format constants matching poppler::image::format_enum.
/// Must stay in sync with ImageFormat in the Rust bridge.
#define SHIM_FORMAT_MONO   1
#define SHIM_FORMAT_RGB24  2
#define SHIM_FORMAT_ARGB32 3
#define SHIM_FORMAT_GRAY8  4
#define SHIM_FORMAT_BGR24  5

/// Render a page to an image.
///
/// Parameters:
///   - `page`    : page to render
///   - `xres`    : horizontal resolution in DPI
///   - `yres`    : vertical resolution in DPI
///   - `format`  : one of the SHIM_FORMAT_* constants (default: SHIM_FORMAT_RGB24)
///   - `hints`   : bitfield — bit 0: antialiasing, bit 1: text-AA, bit 2: text-hinting
///
/// Returns: allocated PopplerShimImage, or null on failure.
/// Free with `poppler_shim_image_free`.
extern "C" PopplerShimImage *poppler_shim_page_render(
        const PopplerShimPage *page,
        double xres, double yres,
        int format, unsigned int hints) {
    if (!page) return nullptr;

    poppler::page_renderer renderer;

    // Map format constant.  Unknown values fall back to RGB24 (safe default;
    // the Rust layer validates the format before calling, so this path is only
    // reachable if the shim constants fall out of sync).
    switch (format) {
        case SHIM_FORMAT_MONO:   renderer.set_image_format(poppler::image::format_mono);   break;
        case SHIM_FORMAT_GRAY8:  renderer.set_image_format(poppler::image::format_gray8);  break;
        case SHIM_FORMAT_ARGB32: renderer.set_image_format(poppler::image::format_argb32); break;
        case SHIM_FORMAT_BGR24:  renderer.set_image_format(poppler::image::format_bgr24);  break;
        case SHIM_FORMAT_RGB24:
        default:                 renderer.set_image_format(poppler::image::format_rgb24);  break;
    }

    // Map hint bitfield.
    if (hints & 0x01) renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    if (hints & 0x02) renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);
    if (hints & 0x04) renderer.set_render_hint(poppler::page_renderer::text_hinting, true);

    // White paper.
    renderer.set_paper_color(0xFFFFFFFF);

    poppler::image img = renderer.render_page(page->page.get(), xres, yres);
    if (!img.is_valid()) return nullptr;

    auto *shim = new PopplerShimImage{};
    shim->img = img;
    return shim;
}

extern "C" void poppler_shim_image_free(PopplerShimImage *img) {
    delete img;
}

extern "C" int poppler_shim_image_width(const PopplerShimImage *img) {
    return img ? img->img.width() : 0;
}

extern "C" int poppler_shim_image_height(const PopplerShimImage *img) {
    return img ? img->img.height() : 0;
}

/// Bytes per row (may include padding).
extern "C" int poppler_shim_image_bytes_per_row(const PopplerShimImage *img) {
    return img ? img->img.bytes_per_row() : 0;
}

/// Pointer to the raw pixel data.  Valid until `poppler_shim_image_free`.
extern "C" const char *poppler_shim_image_data(const PopplerShimImage *img) {
    return img ? img->img.const_data() : nullptr;
}

/// Format of the image data (one of the SHIM_FORMAT_* constants).
extern "C" int poppler_shim_image_format(const PopplerShimImage *img) {
    if (!img) return 0;
    switch (img->img.format()) {
        case poppler::image::format_mono:   return SHIM_FORMAT_MONO;
        case poppler::image::format_rgb24:  return SHIM_FORMAT_RGB24;
        case poppler::image::format_argb32: return SHIM_FORMAT_ARGB32;
        case poppler::image::format_gray8:  return SHIM_FORMAT_GRAY8;
        case poppler::image::format_bgr24:  return SHIM_FORMAT_BGR24;
        default:                            return 0;
    }
}

// ---------------------------------------------------------------------------
// Library version
// ---------------------------------------------------------------------------

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" int poppler_shim_version_major(void) { return static_cast<int>(poppler::version_major()); }
extern "C" int poppler_shim_version_minor(void) { return static_cast<int>(poppler::version_minor()); }
extern "C" int poppler_shim_version_micro(void) { return static_cast<int>(poppler::version_micro()); }
#pragma GCC diagnostic pop
