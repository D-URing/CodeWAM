from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", ROOT / "checkpoints"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CheckResult:
    label: str
    ok: bool
    required: bool
    detail: str = ""


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_any(pattern: str) -> bool:
    return len(glob.glob(str(MODEL_ROOT / pattern))) > 0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def module_check(label: str, module: str, required: bool, detail: str = "") -> CheckResult:
    return CheckResult(label=label, ok=has_module(module), required=required, detail=detail)


def path_check(label: str, path: Path, required: bool) -> CheckResult:
    return CheckResult(label=label, ok=path.exists(), required=required, detail=rel(path))


def pattern_check(label: str, pattern: str, required: bool) -> CheckResult:
    return CheckResult(label=label, ok=has_any(pattern), required=required, detail=str(MODEL_ROOT / pattern))


def import_check(label: str, module: str, required: bool) -> CheckResult:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - user-facing environment diagnostic
        return CheckResult(label=label, ok=False, required=required, detail=f"{type(exc).__name__}: {exc}")
    return CheckResult(label=label, ok=True, required=required)


def collect_checks(mode: str) -> list[CheckResult]:
    cluster = mode == "cluster"
    checks = [
        module_check("torch module", "torch", required=True),
        module_check("fastwam module", "fastwam", required=True, detail="run scripts/bootstrap_fastwam.sh"),
        module_check("codewam module", "codewam", required=True),
        module_check("hydra module", "hydra", required=True),
        module_check("imageio module", "imageio", required=True),
        module_check("pyarrow module", "pyarrow", required=not cluster),
        module_check("av module", "av", required=not cluster),
        module_check("deepspeed module", "deepspeed", required=cluster),
        import_check("StateCodebook import", "codewam.codebook", required=True),
        import_check("CodeWAM import", "codewam.model", required=True),
        path_check("model root", MODEL_ROOT, required=cluster),
        pattern_check("Wan2.2 DiT", "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors", required=cluster),
        CheckResult(
            label="Wan2.2 VAE",
            ok=has_any("Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
            or has_any("DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"),
            required=cluster,
            detail=str(MODEL_ROOT / "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        ),
        path_check(
            "ActionDiT backbone",
            ROOT / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
            required=cluster,
        ),
        path_check(
            "FastWAM LIBERO release",
            ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
            required=False,
        ),
        path_check(
            "FastWAM RoboTwin release",
            ROOT / "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt",
            required=False,
        ),
        path_check("Package Scan v6 local data", ROOT / "package_scan_v6/meta/info.json", required=False),
    ]
    return checks


def print_result(result: CheckResult) -> None:
    if result.ok:
        marker = "OK"
    elif result.required:
        marker = "MISSING"
    else:
        marker = "OPTIONAL"
    suffix = f" - {result.detail}" if result.detail else ""
    print(f"[{marker}] {result.label}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CodeWAM local or cluster environment readiness.")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local")
    args = parser.parse_args()

    print(f"mode={args.mode}")
    print(f"root={ROOT}")
    print(f"model_root={MODEL_ROOT}")
    missing_required = False
    for result in collect_checks(args.mode):
        print_result(result)
        missing_required = missing_required or (result.required and not result.ok)
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
