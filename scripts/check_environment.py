from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", ROOT / "checkpoints"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def status(label: str, ok: bool, detail: str = "") -> None:
    marker = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker}] {label}{suffix}")


def has_any(pattern: str) -> bool:
    return len(glob.glob(str(MODEL_ROOT / pattern))) > 0


def main() -> None:
    status("torch", has_module("torch"))
    status("fastwam", has_module("fastwam"), "run scripts/bootstrap_fastwam.sh if missing")
    status("codewam", has_module("codewam"))
    status("hydra", has_module("hydra"))
    status("model root", MODEL_ROOT.exists(), str(MODEL_ROOT))
    status(
        "Wan2.2 DiT",
        has_any("Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors"),
    )
    status(
        "Wan2.2 VAE",
        has_any("Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
        or has_any("DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"),
    )
    status(
        "ActionDiT backbone",
        (ROOT / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt").exists(),
    )


if __name__ == "__main__":
    main()
