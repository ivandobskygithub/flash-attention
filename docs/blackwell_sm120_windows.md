# FlashAttention-3 on SM120 (Windows + Ninja)

This document summarizes the SM120 enablement for FlashAttention-3 on Windows hosts. The SM120 path mirrors the SM90 flow while forcing 128x128x128 tiles, TN layouts, and TMA+mbarrier pipelines.

## Build notes
- Use the `windows-ninja` preset in `CMakeUserPresets.json` to pin the Ninja generator and request `CMAKE_CUDA_ARCHITECTURES=120;90`.
- When installing from the `hopper` package, ensure `TORCH_CUDA_ARCH_LIST=12.0` is set.
- CUDA 12.8+ and MSVC 2022 are required for local builds.

## Architectural scope
- SM120 targets GeForce Blackwell (RTX 5090 class) and uses warp-specialized TMA flows.
- Datacenter Blackwell (SM100) requires TMEM/UMMA paths that remain out of scope for this plan.
- Only TN MMA forms are permitted and the epilogue schedule stays on `Auto`.

## Troubleshooting
- Validate Ninja is being picked up by checking `CMAKE_GENERATOR` in the CMake cache.
- Confirm that generated cubins contain `sm_120` by running `cuobjdump --list-elf` on the built Python extension.
