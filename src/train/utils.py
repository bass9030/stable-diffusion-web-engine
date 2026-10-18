"""학습/추론 공통 유틸리티. 장치(cuda/cpu) 의존 코드는 이 파일에서만 처리한다."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: torch.device, allow_tf32: bool) -> None:
    """Ampere 이상 GPU에서 fp32 행렬곱/합성곱을 TF32 텐서 코어로 계산하게 한다.

    정밀도가 약간 낮아지는 대신 fp32 학습이 크게 빨라진다. Turing(GTX 1660 등)에서는 효과가 없다.
    """
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def free_device_memory(device: torch.device) -> None:
    """캐시 할당자가 잡고 있는 미사용 GPU 메모리를 반환한다."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """CPU에서는 half 연산이 느리거나 지원되지 않아 fp32로 바꾼다."""
    dtype = DTYPES[name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def noise_schedule_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """학습/샘플링 스케줄러에 공통으로 넘길 노이즈 스케줄 설정.

    zero_terminal_snr + v_prediction: Lin et al. 2023 "Common Diffusion Noise Schedules and
    Sample Steps are Flawed". 마지막 timestep에 원본 신호가 남지 않게 beta를 재조정한다.
    """
    return {
        "prediction_type": model_cfg.get("prediction_type", "epsilon"),
        "rescale_betas_zero_snr": model_cfg.get("zero_terminal_snr", False),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    status = run("status", "--porcelain")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def save_run_info(out_dir: Path, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    """설정 파일과 git 정보를 함께 기록한다 (CLAUDE.md 학습 규칙)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    info = {**git_info(), **(extra or {})}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
