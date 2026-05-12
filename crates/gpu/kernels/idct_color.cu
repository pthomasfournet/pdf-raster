// IDCT + dequant + colour kernel — CUDA stub.
//
// The IDCT/colour path runs on Vulkan via idct_color.slang. This file
// exists only to satisfy the build.rs KERNELS list (which compiles every
// .cu alongside its .slang counterpart).  The PTX artifact is never
// loaded by GpuCtx; the Slang/SPIR-V artifact is the active path.
//
// A future CUDA mirror can be added here if throughput profiling shows
// the Vulkan dispatch is a bottleneck on NVIDIA hardware.
