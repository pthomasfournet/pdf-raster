//! Build script: compiles CUDA kernels to PTX via nvcc and places them in `OUT_DIR`.
#![allow(missing_docs)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = PathBuf::from("kernels");

    // Tell cargo to rerun if kernels change.
    println!("cargo:rerun-if-changed=kernels/");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        // Try standard CUDA install paths.
        for path in ["/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda/bin/nvcc"] {
            if PathBuf::from(path).exists() {
                return path.to_owned();
            }
        }
        "nvcc".to_owned()
    });

    for kernel in ["composite_rgba8", "apply_soft_mask"] {
        let src = kernels_dir.join(format!("{kernel}.cu"));
        let ptx = out_dir.join(format!("{kernel}.ptx"));

        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "-arch=sm_120", // Blackwell / sm_120 (RTX 5070)
                "-O3",
                "--use_fast_math",
                "-o",
                ptx.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc: {e}"));

        assert!(status.success(), "nvcc failed for {kernel}.cu");
    }
}
