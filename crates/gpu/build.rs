//! Build script: compiles CUDA kernels to PTX via nvcc and places them in `OUT_DIR`.

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

/// CUDA compute kernels compiled to PTX by nvcc.
///
/// Listed once here to ensure the placeholder-write and the nvcc-compile loops
/// stay in sync — adding a new kernel only requires updating this list.
const KERNELS: &[&str] = &[
    "composite_rgba8",
    "apply_soft_mask",
    "aa_fill",
    "tile_fill",
    "icc_clut",
    "blit_image",
];

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

    // PTX kernels are needed whenever any CUDA feature is active.
    // CARGO_FEATURE_GPU_AA / GPU_ICC are features of pdf_interp, not gpu, so
    // they are never set in the gpu crate's build script. Use the gpu-crate
    // features that are actually propagated here instead.
    let need_ptx = [
        "CARGO_FEATURE_NVJPEG",
        "CARGO_FEATURE_NVJPEG2K",
        "CARGO_FEATURE_GPU_DESKEW",
        // Phase 9 image cache: ships its own PTX (blit_image) and
        // also loads the existing kernels via GpuCtx::init.
        "CARGO_FEATURE_CACHE",
        // gpu-validation tests touch every kernel, including those
        // not gated by a specific feature.
        "CARGO_FEATURE_GPU_VALIDATION",
    ]
    .iter()
    .any(|f| env::var(f).is_ok());

    if !need_ptx {
        // Write empty placeholder PTX files so that the unconditional `include_str!`
        // macros in src/lib.rs compile successfully on CPU-only builds (Intel without
        // CUDA, ARM, CI runners without a GPU).  The placeholders are never loaded;
        // GpuCtx::init() fails with a CUDA error before any kernel is invoked.
        //
        // Always written (no existence check) because OUT_DIR is fresh on each
        // clean build and cargo guarantees this script only runs when inputs change.
        for kernel in KERNELS {
            let ptx = out_dir.join(format!("{kernel}.ptx"));
            std::fs::write(&ptx, "")
                .unwrap_or_else(|e| panic!("failed to write placeholder {kernel}.ptx: {e}"));
        }
        return;
    }

    // Allow overriding the PTX target arch (e.g. CUDA_ARCH=sm_75 for Turing,
    // CUDA_ARCH=sm_120 for Blackwell).  Default sm_80 covers Ampere through Blackwell.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_owned());

    for kernel in KERNELS {
        let src = kernels_dir.join(format!("{kernel}.cu"));
        let ptx = out_dir.join(format!("{kernel}.ptx"));

        let status = Command::new(nvcc)
            .args([
                "--ptx",
                &format!("-arch={arch}"),
                "-O3",
                "--use_fast_math",
                "-o",
                ptx.to_str().expect("OUT_DIR path contains non-UTF-8"),
                src.to_str().expect("kernel source path contains non-UTF-8"),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc ({nvcc}): {e}"));

        assert!(
            status.success(),
            "nvcc failed for {kernel}.cu (arch={arch})"
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
