"""FlashAttention-3 environment validator for SM120/FlashAttention-3 setups.

This script performs incremental validation steps, from simple package
availability checks to running a minimal FlashAttention-3 forward/backward pass.
Use it to confirm the environment matches the expected Torch and CUDA versions
and that FlashAttention-3 kernels are usable on Blackwell-class GPUs.
"""

from __future__ import annotations

import argparse
import importlib
import os
import traceback
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch

from flash_attn_interface import flash_attn_func

EXPECTED_TORCH_VERSION = "2.9.0+cu128"
MIN_COMPUTE_CAPABILITY = (12, 0)


@dataclass
class StageResult:
    name: str
    passed: bool
    details: str


@dataclass
class ValidationStage:
    name: str
    description: str
    runner: Callable[[], str]


@dataclass
class SmokeTestConfig:
    batch: int
    seqlen: int
    heads: int
    headdim: int
    dtype: torch.dtype
    causal: bool


SMOKE_TEST_CONFIG = SmokeTestConfig(
    batch=1,
    seqlen=16,
    heads=2,
    headdim=64,
    dtype=torch.float16,
    causal=False,
)


def ensure_torch_version() -> str:
    version = torch.__version__
    if version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"Expected torch {EXPECTED_TORCH_VERSION}, but found {version}."
        )
    return f"Torch version matches expectation ({version})."


def ensure_cuda_environment() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; FlashAttention-3 requires CUDA GPUs.")

    capability = torch.cuda.get_device_capability()
    runtime = torch.version.cuda or "unknown"

    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"Compute capability {capability[0]}.{capability[1]} is below the required "
            f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]} for cu120 kernels."
        )
    if not runtime.startswith("12"):
        raise RuntimeError(
            f"Expected a CUDA 12.x runtime, but torch reports CUDA {runtime}."
        )

    return (
        "CUDA runtime and device capability are compatible: "
        f"CUDA {runtime}, compute capability {capability[0]}.{capability[1]}."
    )


def ensure_flash_attn3_installed() -> str:
    if importlib.util.find_spec("flash_attn_3") is None:
        raise RuntimeError("flash_attn_3 package is not installed.")
    if importlib.util.find_spec("flash_attn_3._C") is None:
        raise RuntimeError("flash_attn_3 CUDA extension is missing.")
    if importlib.util.find_spec("flash_attn_2") is not None:
        raise RuntimeError(
            "flash_attn_2 is present; ensure FlashAttention-3 is used exclusively."
        )

    importlib.import_module("flash_attn_3")
    importlib.import_module("flash_attn_3._C")

    return "flash_attn_3 package and CUDA extension detected (FlashAttention-2 absent)."


def ensure_flash_attn3_ops() -> str:
    required_ops = ["fwd", "bwd", "fwd_combine", "get_scheduler_metadata"]
    missing_ops = [op for op in required_ops if not hasattr(torch.ops.flash_attn_3, op)]
    if missing_ops:
        raise RuntimeError(f"Missing FlashAttention-3 torch.ops: {', '.join(missing_ops)}.")

    return "FlashAttention-3 operators are registered with torch.ops."


def describe_flash_attn3_build() -> str:
    flash_attn3 = importlib.import_module("flash_attn_3")
    flash_attn3_ext = importlib.import_module("flash_attn_3._C")

    version = getattr(flash_attn3, "__version__", "unknown")
    module_path = getattr(flash_attn3, "__file__", "<not found>")
    ext_path = getattr(flash_attn3_ext, "__file__", "<not found>")
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>")

    return (
        f"flash_attn_3 version={version} (py={module_path}, ext={ext_path}); "
        f"TORCH_CUDA_ARCH_LIST={env_arch}; torch.cuda.get_arch_list()={torch.cuda.get_arch_list()}"
    )


def run_flash_attn3_smoke_test() -> str:
    device = torch.device("cuda")
    batch = SMOKE_TEST_CONFIG.batch
    seqlen = SMOKE_TEST_CONFIG.seqlen
    nheads = SMOKE_TEST_CONFIG.heads
    headdim = SMOKE_TEST_CONFIG.headdim
    causal = SMOKE_TEST_CONFIG.causal
    dtype = SMOKE_TEST_CONFIG.dtype

    q = torch.randn(
        batch, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def _format_tensor(tensor: torch.Tensor, name: str) -> str:
        return (
            f"- {name}: shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
            f"dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}"
        )

    try:
        out, softmax_lse = flash_attn_func(
            q, k, v, return_attn_probs=True, causal=causal
        )
        _ = softmax_lse  # Returned for visibility in case debugging is needed.
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(device)
        stream = torch.cuda.current_stream(device)
        raise RuntimeError(
            "FlashAttention-3 forward/backward failed. Diagnostics:\n"
            f"- Device: {props.name} (cc {props.major}.{props.minor})\n"
            f"- torch.__version__: {torch.__version__}\n"
            f"- torch.version.cuda: {torch.version.cuda}\n"
            f"- CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', '<unset>')}\n"
            f"- torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}\n"
            f"- flash_attn_3 module: {getattr(importlib.import_module('flash_attn_3'), '__file__', '<not found>')}\n"
            f"- flash_attn_3._C extension: {getattr(importlib.import_module('flash_attn_3._C'), '__file__', '<not found>')}\n"
            f"- Current stream: {stream}\n"
            f"- Current device: {torch.cuda.current_device()}\n"
            f"- smoke_test config: batch={batch}, seqlen={seqlen}, heads={nheads}, headdim={headdim}, "
            f"dtype={dtype}, causal={causal}\n"
            f"{_format_tensor(q, 'Q')}\n"
            f"{_format_tensor(k, 'K')}\n"
            f"{_format_tensor(v, 'V')}\n"
            f"- torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        ) from exc

    max_out = out.abs().max().item()
    max_grad = q.grad.abs().max().item()
    return (
        "FlashAttention-3 forward/backward succeeded on CUDA: "
        f"max |out|={max_out:.6f}, max |dq|={max_grad:.6f}."
    )


def build_stages() -> List[ValidationStage]:
    return [
        ValidationStage(
            name="torch-version",
            description="Validate Torch build matches expected CUDA toolkit version.",
            runner=ensure_torch_version,
        ),
        ValidationStage(
            name="cuda-environment",
            description="Check CUDA availability, runtime version, and SM120 capability.",
            runner=ensure_cuda_environment,
        ),
        ValidationStage(
            name="flash-attn3-package",
            description="Confirm FlashAttention-3 is installed and FlashAttention-2 is absent.",
            runner=ensure_flash_attn3_installed,
        ),
        ValidationStage(
            name="flash-attn3-ops",
            description="Verify core FlashAttention-3 operators are registered.",
            runner=ensure_flash_attn3_ops,
        ),
        ValidationStage(
            name="flash-attn3-build-info",
            description="Report FlashAttention-3 Python/CUDA artifact locations and arch flags.",
            runner=describe_flash_attn3_build,
        ),
        ValidationStage(
            name="flash-attn3-smoke-test",
            description="Run a minimal FlashAttention-3 forward/backward on CUDA.",
            runner=run_flash_attn3_smoke_test,
        ),
    ]


def run_stages(
    stages: Sequence[ValidationStage],
    *,
    max_stage: int | None,
    stop_on_failure: bool,
    print_traceback: bool,
) -> List[StageResult]:
    results: List[StageResult] = []
    selected = stages if max_stage is None else stages[:max_stage]

    for index, stage in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {stage.name}: {stage.description}")
        try:
            details = stage.runner()
            print(f"  PASS: {details}")
            results.append(StageResult(stage.name, True, details))
        except Exception as exc:  # noqa: BLE001
            details = str(exc)
            print(f"  FAIL: {details}")
            if print_traceback:
                traceback.print_exc()
            results.append(StageResult(stage.name, False, details))
            if stop_on_failure:
                break

    return results


def summarize_results(results: Sequence[StageResult]) -> None:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print("\nSummary:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"- {result.name}: {status} — {result.details}")
    print(f"\nCompleted {passed}/{total} stages successfully.")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged validation for FlashAttention-3 on SM120 (CUDA 12.x) systems. "
            "By default, all stages execute even if a previous check fails."
        )
    )
    parser.add_argument(
        "--max-stage",
        type=int,
        default=None,
        metavar="N",
        help="Limit validation to the first N stages (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing stage instead of continuing.",
    )
    parser.add_argument(
        "--traceback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print full tracebacks for failing stages to aid debugging (default: on).",
    )
    parser.add_argument(
        "--launch-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CUDA_LAUNCH_BLOCKING=1 to pin errors to the failing kernel (default: on).",
    )
    parser.add_argument(
        "--smoke-seqlen",
        type=int,
        default=16,
        help="Sequence length for the smoke test (default: 16).",
    )
    parser.add_argument(
        "--smoke-heads",
        type=int,
        default=2,
        help="Number of attention heads for the smoke test (default: 2).",
    )
    parser.add_argument(
        "--smoke-headdim",
        type=int,
        default=64,
        help="Head dimension for the smoke test (default: 64).",
    )
    parser.add_argument(
        "--smoke-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Data type for the smoke test tensors (default: fp16).",
    )
    parser.add_argument(
        "--smoke-causal",
        action="store_true",
        help="Run the smoke test with causal=True to exercise that path.",
    )
    parser.add_argument(
        "--sync-debug-mode",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help=(
            "Set torch.cuda.set_sync_debug_mode for additional CUDA diagnostics (0=off, 1=warn, 2=error). "
            "Defaults to 2 for maximal diagnostics."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args([] if argv is None else argv)
    global SMOKE_TEST_CONFIG
    SMOKE_TEST_CONFIG = SmokeTestConfig(
        batch=1,
        seqlen=args.smoke_seqlen,
        heads=args.smoke_heads,
        headdim=args.smoke_headdim,
        dtype=torch.bfloat16 if args.smoke_dtype == "bf16" else torch.float16,
        causal=args.smoke_causal,
    )
    if args.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    if args.sync_debug_mode is not None:
        torch.cuda.set_sync_debug_mode(args.sync_debug_mode)
    stages = build_stages()
    results = run_stages(
        stages,
        max_stage=args.max_stage,
        stop_on_failure=args.stop_on_failure,
        print_traceback=args.traceback,
    )
    summarize_results(results)


if __name__ == "__main__":
    main()
