
# FlashAttention‑3 (Hopper → Blackwell GeForce)  
## SM120 / RTX 5090 Implementation Plan (Windows + Ninja)

**Goal (Phase 1).** Add a **working SM120** (compute capability **12.0**) FlashAttention‑3 path under the repo’s **`hopper/`** subtree, preserving SM90 behavior, with Windows+Ninja build support. **NVFP4** comes later (Phase 2).  
**Why SM120?** RTX 5090 is **CC 12.0 (SM120)**. Datacenter Blackwell is SM100 (10.x) with UMMA/TMEM/cluster features not present on GeForce. SM120 uses **TMA+mbarrier** pipelines with **SM120‑specific MMA forms** and constraints (**TN‑only**, specific tiles, **EpilogueScheduleAuto**).

---

## Assumptions & environment

- You’ll **fork** `Dao-AILab/flash-attention` and work on a feature branch.  
- **No heavy build** runs in the cloud agent: we do **syntax/lint checks** and text‑level validation, compilation and tests where possible.  
- **Local build** (after agent is done): Windows 10/11, CUDA **12.8+**, **MSVC 2022**, **Ninja** generator.
- This plan edits only the FA‑3 Hopper subtree (`hopper/…`) and doc/scripts.
- We need to be able to run a build for a minimal number of architectures, meaning we should be able to build FA-3 for SM120 only, not all the SM80 and SM90 archs.
---

## Deliverables (what should exist after the agent finishes)

1. **New SM120 kernels** (forward + backward) under `hopper/instantiations/*sm120*.cu` with **CUTLASS SM120** collectives.  
2. **Launch template gating** added for SM120 (`__CUDA_ARCH__ >= 1200`).  
3. **Windows+Ninja** build support maintained.  
4. **Lint/syntax‑only** scripts to validate C++/CUDA. Validate references to any Cutlass/CUDA resources are valid. 
5. **Docs** for Windows build.  
6. **Optional CI lint workflow**.

---

## Task list for the web agent (one‑shot edit pass)

### Task 1 — Create branch & scaffolding

- Create `feature/fa3-sm120-windows` branch.  
- Add directories:  
  - `hopper/sm120/`  
  - `hopper/instantiations/`  
  - `tools/lint/`  
  - `docs/`  
- Place this plan in `docs/IMPLEMENTATION_PLAN_SM120_WINDOWS.md`.

---

### Task 2 — Pin Ninja on Windows

- Add `CMakeUserPresets.json` with `"generator": "Ninja"`, binaryDir, preset.  
- Ensure `setup.py`/CMake respects Ninja generator.  
- Mention Ninja usage in docs.

---

### Task 3 — Add **sm_120** build flags

- Ensure build args include:
  ```
  -gencode arch=compute_120,code=sm_120
  ```
- Keep `sm_90`.  
- Prefer `CMAKE_CUDA_ARCHITECTURES=120;90`.

---

### Task 4 — Add SM120 compile‑time guards

Modify in:
- `hopper/flash_fwd_launch_template.h`
- `hopper/flash_bwd_launch_template.h`

Add:
```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
// SM120
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
// SM90
#else
#error "FlashAttention-3: unsupported compute capability"
#endif
```

---

### Task 5 — Create `sm120_traits.hpp`

Includes:

- **TN‑only** layout.  
- Allowed SM120 tile shapes (**128×128×128**).  
- Valid schedules:  
  - `KernelTmaWarpSpecializedCooperative`  
  - `KernelTmaWarpSpecializedPingpong`  
- Cluster size = **1×1×1**.  
- Epilogue = **Auto**.  
- `static_assert` checks.

---

### Task 6 — Implement SM120 forward mainloop

Create `hopper/sm120/flash_fwd_mainloop_sm120.hpp`:

- Use CUTLASS SM120 MMA (`mma.sync.aligned.kind::…`).  
- Use traits from Task 5.  
- Use TMA+mbarrier (no SM100 TMEM/UMMA).  
- Reuse FA3 softmax utilities.

---

### Task 7 — Forward instantiations

Add:

- `hopper/instantiations/flash_fwd_hdim64_fp16_sm120.cu`  
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm120.cu`

Include mainloop header + traits.
Update generate_kernels.py to provide these.

---

### Task 8 — Implement SM120 backward mainloop

Create `hopper/sm120/flash_bwd_mainloop_sm120.hpp`:

- mirror forward mainloop  
- same schedules/tiles

---

### Task 9 — Backward instantiations

Add:

- `hopper/instantiations/flash_bwd_hdim64_fp16_sm120.cu`  
- `hopper/instantiations/flash_bwd_hdim128_bf16_sm120.cu`
Update generate_kernels.py to provide these.
---

### Task 10 — Runtime dispatch

In Python interface:

- Detect architecture via:
  ```python
  torch.cuda.get_device_capability()
  ```
- Use SM120 kernels when device cc == (12, 0).  
- Fallback to SM90 or FA2.

---

### Task 11 — Documentation

Add `docs/blackwell_sm120_windows.md`:

- Prereqs (CUDA 12.8+, MSVC, Ninja).  
- Build instructions:  
  ```
  $env:CMAKE_GENERATOR="Ninja"
  $env:TORCH_CUDA_ARCH_LIST="12.0"
  cd hopper
  python setup.py develop
  ```
- Troubleshooting.  
- Mention CC mapping.

---

### Task 12 — Add lint/syntax-only tooling

Create:

- `tools/lint/requirements.txt`: ruff, black, cpplint, clang-format.  
- `tools/lint/run_all.ps1` running:
  - `ruff hopper`
  - `black --check hopper`
  - `clang-format -n hopper/**/*.cu`
  - `python tools/lint/check_tokens.py`
  - `python tools/lint/check_includes.py`
  - `python tools/lint/grep_arch_flags.py`
- `tools/lint/check_tokens.py`: balanced braces, parentheses, CUDA <<< >>>.  
- `tools/lint/check_includes.py`: validate includes.  
- `tools/lint/grep_arch_flags.py`: check sm_120 flags.

Use linting tools to validate changes. If linting fails, resolve issues then re-run until successful.
---

### Task 13 — Optional CI lint workflow

`.github/workflows/lint.yml`:

- Windows runner  
- Install lint requirements  
- Run lint scripts  
- No CUDA builds

---

### Task 14 — SM120 vs SM100 note

Update docs:

- SM120 = GeForce Blackwell, TMA+mbarrier, TN‑only, no TMEM/UMMA.  
- SM100 = Datacenter Blackwell, uses UMMA/TMEM/cluster (out of scope).

---

### Task 15 — Add extra head dims

Add instantiations for the following, including updating generate kernels:

- hdim 96  
- hdim 160

Use same 128×128×128 tile.

---

### Task 16 — Final cleanup

Modify:

- Top `README.md`: note SM120 support.  
- Add/modify `CHANGELOG.md`.

---

## Local build instructions (for human)

After agent completes:

```
pip install ninja
setx CMAKE_GENERATOR "Ninja"
setx TORCH_CUDA_ARCH_LIST "12.0"

cd hopper
python setup.py develop
```

Verify:

```
cuobjdump --list-elf *.pyd
# Ensure sm_120 cubin exists
```

---

## Appendix — Files to be created/modified

**Build**  
- `hopper/setup.py`  
- `CMakeUserPresets.json`  

**CUDA/C++**  
- `hopper/sm120/sm120_traits.hpp`  
- `hopper/sm120/flash_fwd_mainloop_sm120.hpp`  
- `hopper/sm120/flash_bwd_mainloop_sm120.hpp`  
- `hopper/instantiations/flash_fwd_hdim64_fp16_sm120.cu`  
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm120.cu`  
- `hopper/instantiations/flash_bwd_hdim64_fp16_sm120.cu`  
- `hopper/instantiations/flash_bwd_hdim128_bf16_sm120.cu`  

**Python**  
- Runtime dispatch changes

**Docs**  
- This file  
- `docs/blackwell_sm120_windows.md`  
- `README.md`, `CHANGELOG.md`

**Lint**  
- `tools/lint/*`  
- Optional `.github/workflows/lint.yml`

---

## Phase 2 (later)

- Add NVFP4 support using CUTLASS SM120 `mxf4nvf4.block_scale` collectives.  
- Add scale tensor generation & propagation.  
- Add quant/dequant helpers.

---

