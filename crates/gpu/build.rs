//! Build script: compiles CUDA kernels to PTX via nvcc and places them in `OUT_DIR`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // When the nvjpeg feature is enabled, emit the linker directive so that
    // rustc links against libnvjpeg.so from the CUDA toolkit.  The library is
    // available on any machine with CUDA 12 installed; it ships at
    // /usr/local/cuda-12/targets/x86_64-linux/lib/libnvjpeg.so.
    if env::var("CARGO_FEATURE_NVJPEG").is_ok() {
        // Prefer the versioned CUDA 12 install directory so the exact .so is
        // found even when /usr/local/cuda is a symlink to a different version.
        let mut found = false;
        for dir in [
            "/usr/local/cuda-12/targets/x86_64-linux/lib",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
        ] {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
                found = true;
                break;
            }
        }
        if !found {
            println!(
                "cargo:warning=nvjpeg feature enabled but no CUDA lib directory found; linker will search default paths. Set CUDA_LIB_DIR or install CUDA 12."
            );
        }
        println!("cargo:rustc-link-lib=dylib=nvjpeg");
        // cuStreamSynchronize lives in the CUDA driver library (libcuda.so).
        // On Linux this is provided by the NVIDIA driver, typically at
        // /usr/lib/x86_64-linux-gnu/libcuda.so.1 (driver-managed symlink).
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = PathBuf::from("kernels");

    // Tell cargo to rerun if kernels or CUDA_ARCH env changes.
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        for path in ["/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda/bin/nvcc"] {
            if PathBuf::from(path).exists() {
                return path.to_owned();
            }
        }
        "nvcc".to_owned()
    });

    // Allow overriding the PTX target arch (e.g. CUDA_ARCH=sm_86 for Ampere).
    // Default is sm_80 which runs on Ampere, Ada, Hopper, and Blackwell.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_owned());

    for kernel in ["composite_rgba8", "apply_soft_mask", "aa_fill", "tile_fill"] {
        let src = kernels_dir.join(format!("{kernel}.cu"));
        let ptx = out_dir.join(format!("{kernel}.ptx"));

        let status = Command::new(&nvcc)
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
