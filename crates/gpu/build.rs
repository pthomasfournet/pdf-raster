//! Build script.
//!
//! Compiles compute kernels and emits link directives:
//! - `.cu` → PTX via `nvcc` for the CUDA backend (when any CUDA feature is on).
//! - `.slang` → SPIR-V via `slangc` for the Vulkan backend (when `vulkan` is on).
//! - Emits `cargo:rustc-link-{search,lib}` directives per active GPU feature.
//!
//! All artifacts land in `OUT_DIR` and are consumed by `include_str!` /
//! `include_bytes!` in `src/lib.rs` and the backend modules.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Emit a `cargo:rustc-link-search` directive for the first of `dirs` that
/// exists on the filesystem, then emit `cargo:rustc-link-lib=dylib={lib}`.
///
/// If none of the directories exist, emit a build warning and let the linker
/// search its default paths (handles non-standard install locations).
fn link_lib_in_dir(dirs: &[&str], lib: &str, warn_context: &str) {
    let mut found = false;
    for dir in dirs {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
            found = true;
            break;
        }
    }
    if !found {
        println!(
            "cargo:warning={warn_context} directory not found; linker will search default paths."
        );
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

/// Compute kernels compiled to PTX by nvcc and (optionally) to SPIR-V by slangc.
///
/// Listed once here to ensure the placeholder-write loops, the nvcc-compile loop,
/// and the slangc-compile loop stay in sync — adding a new kernel only requires
/// updating this list.
const KERNELS: &[&str] = &[
    "composite_rgba8",
    "apply_soft_mask",
    "aa_fill",
    "tile_fill",
    "icc_clut",
    "blit_image",
    "blelloch_scan",
    "parallel_huffman",
];

/// Slang profile per kernel.  `aa_fill` uses subgroup ops (`WaveActiveSum`,
/// `WaveIsFirstLane`) so it needs the `GroupNonUniform*` capability strings;
/// the rest use only baseline `spirv_1_5`.
fn slang_profile(kernel: &str) -> &'static str {
    match kernel {
        "aa_fill" => "spirv_1_5+spvGroupNonUniformBallot+spvGroupNonUniformArithmetic",
        _ => "spirv_1_5",
    }
}

/// Entry-point selector for the slangc invocation.
///
/// - `Some(name)` → pass `-entry name` so slangc compiles exactly that
///   entry. Used when a `.slang` file declares multiple entries but
///   only one is consumed (e.g. `icc_clut.slang` has both matrix and
///   CLUT entries; we only need CLUT on the GPU).
/// - `None` → pass no `-entry` flag; slangc uses every
///   `[shader("compute")]` attribute in the file as an entry point and
///   emits them into a single multi-entry SPIR-V module. Used by the
///   Blelloch scan, which has 3 cooperating entries sharing helper
///   logic and is naturally one .slang file.
fn slang_entry(kernel: &str) -> Option<&'static str> {
    match kernel {
        "icc_clut" => Some("icc_cmyk_clut"),
        // Multi-entry: slangc picks all [shader("compute")] entries
        // when no -entry is passed. blelloch_scan has 3 entries
        // (per-workgroup / block-sums / scatter); slangc preserves
        // their source names in the SPIR-V OpEntryPoint op.
        "blelloch_scan" => None,
        // parallel_huffman is single-entry today (phase1_intra_sync)
        // but expects to grow to multi-entry when Phase 2 / Phase 4
        // land. While it's single-entry, slangc renames the entry
        // to "main"; the moment a second [shader] entry is added,
        // slangc switches to name-preservation. To keep the SPIR-V
        // entry name stable in the host lookup, force the rename
        // explicitly via -entry today; switch to None when the
        // second entry lands (the change is symmetric on the
        // Vulkan side: KernelId::entry_point() returns c"main" for
        // single-entry slangc-renamed kernels, source name for
        // multi-entry).
        "parallel_huffman" => Some("phase1_intra_sync"),
        // Default: single entry point named after the file.
        "composite_rgba8" => Some("composite_rgba8"),
        "apply_soft_mask" => Some("apply_soft_mask"),
        "aa_fill" => Some("aa_fill"),
        "tile_fill" => Some("tile_fill"),
        "blit_image" => Some("blit_image"),
        _ => panic!("unknown kernel name: {kernel}"),
    }
}

/// Candidate directories for CUDA toolkit libraries, in preference order.
///
/// First the generic symlink (`/usr/local/cuda`, which on Ubuntu is managed
/// by `update-alternatives` and points at whichever toolkit is selected),
/// then the legacy flat layout. Versioned trees like `/usr/local/cuda-13`
/// or `/usr/local/cuda-12.8` are reachable through the same symlink, so we
/// don't enumerate them here — that just creates stale entries every time
/// a new toolkit ships.
/// Note: paths are x86-64-specific; GPU features are only supported on x86-64.
const CUDA_LIB_DIRS: &[&str] = &[
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/lib64",
];

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = PathBuf::from("kernels");

    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=shim/nvjpeg2k_shim.cpp");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVJPEG2K_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=SLANGC");

    let nvcc_path = env::var("NVCC").unwrap_or_else(|_| {
        let candidate = "/usr/local/cuda/bin/nvcc";
        if PathBuf::from(candidate).exists() {
            return candidate.to_owned();
        }
        "nvcc".to_owned()
    });
    let nvcc = &nvcc_path;

    // nvJPEG: libnvjpeg.so ships with the CUDA toolkit.
    // On Linux this is provided by the NVIDIA driver, typically at
    // /usr/lib/x86_64-linux-gnu/libcuda.so.1 (driver-managed symlink).
    if env::var("CARGO_FEATURE_NVJPEG").is_ok() {
        link_lib_in_dir(
            CUDA_LIB_DIRS,
            "nvjpeg",
            "nvjpeg feature enabled but no CUDA lib",
        );
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // nvJPEG2000: libnvjpeg2k.so lives in a non-standard path (not in the CUDA
    // toolkit tree) and requires cudart for cudaMalloc / cudaMemcpy2D.
    if env::var("CARGO_FEATURE_NVJPEG2K").is_ok() {
        // Path layout depends on which CUDA major version the libnvjpeg2k
        // package was built for. CUDA 13 → /libnvjpeg2k/13, CUDA 12 →
        // /libnvjpeg2k/12. Probe newest first.
        link_lib_in_dir(
            &[
                "/usr/lib/x86_64-linux-gnu/libnvjpeg2k/13",
                "/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12",
                "/usr/lib/x86_64-linux-gnu/libnvjpeg2k",
            ],
            "nvjpeg2k",
            "nvjpeg2k feature enabled but libnvjpeg2k not found",
        );
        // cudart provides cudaMalloc / cudaFree / cudaMemcpy2D (runtime API).
        link_lib_in_dir(
            CUDA_LIB_DIRS,
            "cudart",
            "nvjpeg2k feature enabled but no CUDA lib for cudart",
        );
        // cuStreamSynchronize / cuCtxSetCurrent live in libcuda.so (driver).
        println!("cargo:rustc-link-lib=dylib=cuda");

        // Compile the C++ exception-boundary shim.  nvjpeg2k throws C++
        // exceptions on malformed codestreams; those cannot propagate through
        // Rust extern "C" FFI (undefined behaviour).  The shim wraps all
        // nvjpeg2k entry points that may throw in try/catch and maps any
        // exception to NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED (9).
        compile_nvjpeg2k_shim(&out_dir, nvcc);
        println!("cargo:rustc-link-lib=static=nvjpeg2k_shim");
        // The shim uses C++ exception handling; link the C++ runtime.
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // VA-API: libva.so.2 (core API) + libva-drm.so.2 (headless DRM connection).
    // Ships with Mesa (mesa-va-drivers) or intel-media-driver.
    //
    // The runtime libraries are in /usr/lib/x86_64-linux-gnu on Debian/Ubuntu but
    // the dev package (libva-dev) is not required — we link directly to the .so.2
    // versioned file using the `filename:` link syntax, which works without an
    // unversioned symlink.  This avoids a hard build-time dependency on libva-dev.
    if env::var("CARGO_FEATURE_VAAPI").is_ok() {
        const VA_DIRS: &[&str] = &["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib"];
        for dir in VA_DIRS {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
                break;
            }
        }
        // Link with explicit versioned filename so no unversioned symlink is required.
        println!("cargo:rustc-link-lib=dylib:+verbatim=libva.so.2");
        println!("cargo:rustc-link-lib=dylib:+verbatim=libva-drm.so.2");
    }

    // GPU deskew: CUDA NPP geometry library (nppiRotate_8u_C1R_Ctx lives in
    // libnppig; the stream context helpers nppSetStream/nppGetStreamContext live
    // in libnppc).  cudart and the driver are also required for cudaMalloc /
    // cudaMemcpy / cuStreamSynchronize.
    if env::var("CARGO_FEATURE_GPU_DESKEW").is_ok() {
        link_lib_in_dir(CUDA_LIB_DIRS, "nppig", "gpu-deskew: libnppig not found");
        link_lib_in_dir(CUDA_LIB_DIRS, "nppc", "gpu-deskew: libnppc not found");
        link_lib_in_dir(CUDA_LIB_DIRS, "cudart", "gpu-deskew: libcudart not found");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // Allow `cfg(ptx_placeholder)` so `GpuCtx::init` can short-circuit
    // with a clear error when the build had no NVCC available and only
    // empty placeholders were written.
    println!("cargo:rustc-check-cfg=cfg(ptx_placeholder)");

    // PTX policy: compile real kernels whenever NVCC is available; write
    // empty placeholders only as a last-resort fallback for CPU-only CI
    // runners that don't have the CUDA toolkit.  Keying on a feature-flag
    // heuristic is wrong because pdf_interp's `gpu-aa` / `gpu-icc` features
    // (which cause `GpuCtx::init` to be called at runtime) don't propagate
    // to the gpu crate's build environment, so a build with `gpu-aa` alone
    // would emit placeholders and produce a binary that crashes with
    // `CUDA_ERROR_INVALID_IMAGE` the first time `GpuCtx::init` runs.
    if nvcc_works(nvcc) {
        compile_cuda_kernels(&kernels_dir, &out_dir, nvcc);
    } else {
        println!(
            "cargo:warning=NVCC unavailable ({nvcc}); writing placeholder PTX. \
             GpuCtx::init() will fail with a clear error if invoked at runtime."
        );
        println!("cargo:rustc-cfg=ptx_placeholder");
        write_placeholder_ptx(&out_dir);
    }

    // Slang→SPIR-V kernel compile, gated on the `vulkan` feature.  Independent
    // of the PTX path: a vulkan-only build (no CUDA features) still wants the
    // .spv artifacts.  Output is one .spv per kernel in OUT_DIR; the Vulkan
    // backend (Phase 10 Task 3) consumes them via include_bytes!.
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        compile_slang_kernels(&kernels_dir, &out_dir);
    }
}

/// Probe whether `nvcc` is invokable.  Returns `false` when the binary
/// can't be executed (missing toolkit) so the build can fall back to
/// placeholder PTX rather than panic.
///
/// We intentionally invoke `--version` rather than just checking
/// existence: a stale symlink, a wrapper that requires unavailable
/// libraries, or a build-host without CUDA installed should all be
/// treated as "no NVCC".  On probe failure we emit a `cargo:warning`
/// with the diagnostic so the user can see *why* nvcc was rejected
/// (spawn error vs. non-zero exit) without `--verbose` builds.
fn nvcc_works(nvcc: &str) -> bool {
    match Command::new(nvcc)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let stderr = stderr.trim();
            if stderr.is_empty() {
                println!(
                    "cargo:warning=nvcc probe `{nvcc} --version` exited with {} (no stderr output)",
                    o.status,
                );
            } else {
                println!(
                    "cargo:warning=nvcc probe `{nvcc} --version` exited with {}: {stderr}",
                    o.status,
                );
            }
            false
        }
        Err(e) => {
            println!("cargo:warning=nvcc probe `{nvcc} --version` could not be spawned: {e}");
            false
        }
    }
}

/// Compile every kernel's `.cu` source to PTX via `nvcc`.
///
/// `CUDA_ARCH` env var overrides the target architecture (e.g. `sm_75` for
/// Turing, `sm_120` for Blackwell).  Default `sm_80` covers Ampere through
/// Blackwell.
fn compile_cuda_kernels(kernels_dir: &Path, out_dir: &Path, nvcc: &str) {
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_owned());

    for kernel in KERNELS {
        let src = kernels_dir.join(format!("{kernel}.cu"));
        let ptx = out_dir.join(format!("{kernel}.ptx"));

        // Capture stderr so a kernel-compile failure shows nvcc's actual
        // diagnostic (e.g. "unsupported arch", missing intrinsic) rather
        // than a bare exit-status panic.
        let output = Command::new(nvcc)
            .args([
                "--ptx",
                &format!("-arch={arch}"),
                "-O3",
                "--use_fast_math",
                "-o",
                ptx.to_str().expect("OUT_DIR path contains non-UTF-8"),
                src.to_str().expect("kernel source path contains non-UTF-8"),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to run nvcc ({nvcc}): {e}"));

        assert!(
            output.status.success(),
            "nvcc failed for {kernel}.cu (arch={arch}, status={}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        );
    }
}

/// Write empty placeholder PTX files so the unconditional `include_str!` macros
/// in `src/lib.rs` compile successfully on CPU-only builds (Intel without CUDA,
/// ARM, CI runners without a GPU).  The placeholders are never loaded;
/// `GpuCtx::init()` fails with a CUDA error before any kernel is invoked.
///
/// Always overwritten (no existence check) because `OUT_DIR` is fresh on each
/// clean build and cargo guarantees this script only runs when inputs change.
fn write_placeholder_ptx(out_dir: &Path) {
    for kernel in KERNELS {
        let ptx = out_dir.join(format!("{kernel}.ptx"));
        std::fs::write(&ptx, "")
            .unwrap_or_else(|e| panic!("failed to write placeholder {kernel}.ptx: {e}"));
    }
}

/// Compile every kernel's `.slang` source to SPIR-V via `slangc`.
///
/// Targets `spvGroupNonUniformBallot+spvGroupNonUniformArithmetic` for
/// `aa_fill` (the only kernel using `WaveActiveSum`/`WaveIsFirstLane`);
/// the remaining five compile against vanilla `spirv_1_5`.
///
/// `slangc` is bundled with the `LunarG` Vulkan SDK and ships in the
/// Ubuntu `slang` package; falls back to PATH lookup.  Override with
/// the `SLANGC` env var.
fn compile_slang_kernels(kernels_dir: &Path, out_dir: &Path) {
    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_owned());

    for kernel in KERNELS {
        let src = kernels_dir.join(format!("{kernel}.slang"));
        assert!(
            src.exists(),
            "slang source missing: {} (did you forget to add it alongside the .cu kernel?)",
            src.display()
        );
        let spv = out_dir.join(format!("{kernel}.spv"));
        let profile = slang_profile(kernel);

        // -entry is omitted for multi-entry kernels; slangc then uses
        // the [shader(...)] attributes in the source to pick entries.
        let entry_args: &[&str] = match slang_entry(kernel) {
            Some(entry) => &["-entry", entry],
            None => &[],
        };
        let output = Command::new(&slangc)
            .args([
                "-target",
                "spirv",
                "-profile",
                profile,
                "-o",
                spv.to_str().expect("OUT_DIR path contains non-UTF-8"),
            ])
            .args(entry_args)
            .arg(src.to_str().expect("slang source path contains non-UTF-8"))
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to run slangc ({slangc}): {e} — install the Vulkan SDK or `apt install slang`"
                )
            });

        assert!(
            output.status.success(),
            "slangc failed for {kernel}.slang (profile={profile}, status={}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        );
    }
}

/// Compile `shim/nvjpeg2k_shim.cpp` into a static library `libnvjpeg2k_shim.a`
/// placed in `OUT_DIR`.  Uses nvcc as the C++ compiler so the include path for
/// nvjpeg2k headers is automatically available, and links with -lstdc++.
///
/// The include directory is auto-probed (preferring `/usr/include/libnvjpeg2k/13`
/// over `/12` to match the link-time library), and can be overridden via the
/// `NVJPEG2K_INCLUDE_DIR` environment variable for non-standard installations.
fn compile_nvjpeg2k_shim(out_dir: &Path, nvcc: &str) {
    let shim_src = PathBuf::from("shim/nvjpeg2k_shim.cpp");
    let shim_obj = out_dir.join("nvjpeg2k_shim.o");
    let shim_lib = out_dir.join("libnvjpeg2k_shim.a");
    let out_dir_s = out_dir.to_str().expect("OUT_DIR path non-UTF-8");

    let include_dir = env::var("NVJPEG2K_INCLUDE_DIR").unwrap_or_else(|_| {
        for candidate in ["/usr/include/libnvjpeg2k/13", "/usr/include/libnvjpeg2k/12"] {
            if PathBuf::from(candidate).exists() {
                return candidate.to_owned();
            }
        }
        "/usr/include/libnvjpeg2k/13".to_owned()
    });

    // Compile the C++ source to an object file.
    let status = Command::new(nvcc)
        .args([
            "-x",
            "c++", // treat as C++ (nvcc default for .cpp)
            "-O2",
            &format!("-I{include_dir}"),
            "-c",
            "-o",
            shim_obj.to_str().expect("OUT_DIR path non-UTF-8"),
            shim_src.to_str().expect("shim source path non-UTF-8"),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run nvcc for nvjpeg2k shim ({nvcc}): {e}"));
    assert!(status.success(), "nvcc failed to compile nvjpeg2k_shim.cpp");

    // Archive into a static library so cargo can link it.
    let status = Command::new("ar")
        .args([
            "rcs",
            shim_lib.to_str().expect("OUT_DIR path non-UTF-8"),
            shim_obj.to_str().expect("OUT_DIR path non-UTF-8"),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run ar for nvjpeg2k shim: {e}"));
    assert!(
        status.success(),
        "ar failed to archive nvjpeg2k_shim.o into libnvjpeg2k_shim.a"
    );

    println!("cargo:rustc-link-search=native={out_dir_s}");
}
