"""FlashAttention-3 environment validator for SM120/FlashAttention-3 setups.

This script performs incremental validation steps, from simple package
availability checks to running a minimal FlashAttention-3 forward/backward pass.
Use it to confirm the environment matches the expected Torch and CUDA versions
and that FlashAttention-3 kernels are usable on Blackwell-class GPUs.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch

from hopper.flash_attn_interface import flash_attn_func

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


def run_flash_attn3_smoke_test() -> str:
    device = torch.device("cuda")
    batch, seqlen, nheads, headdim = 1, 16, 2, 64

    q = torch.randn(
        batch, seqlen, nheads, headdim, device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out, softmax_lse = flash_attn_func(q, k, v, return_attn_probs=True)
    _ = softmax_lse  # Returned for visibility in case debugging is needed.
    loss = out.sum()
    loss.backward()

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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args([] if argv is None else argv)
    stages = build_stages()
    results = run_stages(stages, max_stage=args.max_stage, stop_on_failure=args.stop_on_failure)
    summarize_results(results)


if __name__ == "__main__":
    main()
