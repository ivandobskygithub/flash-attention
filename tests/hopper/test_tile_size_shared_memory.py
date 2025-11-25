import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMEM_LIMIT_BYTES = 101_376


def _build_tile_size_bridge(tmpdir: Path) -> Path:
    src = tmpdir / "tile_size_bridge.cpp"
    lib = tmpdir / "libtile_size_bridge.so"
    src.write_text(
        r'''
#include <tuple>
#include "hopper/tile_size.h"

extern "C" void tile_size_fwd_sm90_bridge(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size,
        bool v_colmajor, bool paged_kv_non_TMA, bool softcap,
        int* block_m, int* block_n) {
    auto result = tile_size_fwd_sm90(headdim, headdim_v, is_causal, is_local,
                                     element_size, v_colmajor, paged_kv_non_TMA, softcap);
    *block_m = std::get<0>(result);
    *block_n = std::get<1>(result);
}
''',
        encoding="utf-8",
    )
    compile_cmd = [
        "g++",
        "-std=c++20",
        "-fPIC",
        "-shared",
        "-O2",
        f"-I{REPO_ROOT}",
        str(src),
        "-o",
        str(lib),
    ]
    subprocess.run(compile_cmd, check=True)
    return lib


def _load_bridge() -> ctypes.CDLL:
    with tempfile.TemporaryDirectory() as td:
        lib = _build_tile_size_bridge(Path(td))
        return ctypes.CDLL(str(lib))


def estimate_smem_bytes(block_m: int, block_n: int, headdim: int, headdim_v: int, element_size: int) -> int:
    # Mirror the buffering-aware estimate used in hopper/tile_size.h.
    buffering = 1 if (headdim_v >= 256 or headdim + headdim_v >= 512) else 2
    return buffering * (block_m + block_n) * (headdim + headdim_v) * element_size


def test_tile_sizes_stay_within_blackwell_smem_budget():
    bridge = _load_bridge()
    bridge.tile_size_fwd_sm90_bridge.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    head_dims = (64, 96, 128, 160, 192, 256, 320)
    value_dims = (64, 96, 128, 160, 192, 256, 512)
    bools = (False, True)

    for headdim in head_dims:
        for headdim_v in value_dims:
            for is_causal in bools:
                for is_local in bools:
                    if is_causal and is_local:
                        continue  # invalid combination
                    for paged_kv_non_tma in bools:
                        block_m = ctypes.c_int()
                        block_n = ctypes.c_int()
                        bridge.tile_size_fwd_sm90_bridge(
                            headdim,
                            headdim_v,
                            is_causal,
                            is_local,
                            2,  # fp16/bf16 element size
                            False,  # v_colmajor
                            paged_kv_non_tma,
                            False,  # softcap
                            ctypes.byref(block_m),
                            ctypes.byref(block_n),
                        )
                        smem_bytes = estimate_smem_bytes(block_m.value, block_n.value, headdim, headdim_v, 2)
                        assert smem_bytes <= SMEM_LIMIT_BYTES, (
                            f"SMEM overrun for d={headdim}, dv={headdim_v}, causal={is_causal}, "
                            f"local={is_local}, paged={paged_kv_non_tma}: {smem_bytes}B"
                        )
