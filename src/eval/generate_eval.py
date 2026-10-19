"""Phase 4: 고정 검증셋에 대해 모델 버전별 생성 이미지를 만든다.

사용법:
    python -m src.eval.generate_eval --config configs/base.yaml configs/eval.yaml [--models phase3_html] [--seed 1234]

출력 (eval.out_dir 아래):
    eval_set.json                     세트별로 고른 샘플 id 목록
    {model}/{set}/{id}.png            생성 이미지 (이미 있으면 건너뛰므로 중단 후 다시 실행해도 된다)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.datagen.config import load_config
from src.infer.generate import encode, load_models
from src.infer.sampler import decode_latents, sample_latents
from src.train.cache_features import read_metadata
from src.train.utils import ROOT, free_device_memory, resolve_device

log = logging.getLogger("generate_eval")


def select_eval_set(data_dir: Path, num_samples: int, seed: int) -> list[str]:
    """val split에서 시드로 num_samples개를 고른다 (id 순으로 정렬해 저장)."""
    val_ids = [r["id"] for r in read_metadata(data_dir) if r["split"] == "val"]
    picked = random.Random(seed).sample(val_ids, min(num_samples, len(val_ids)))
    return sorted(picked)


def build_eval_sets(ec: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    return {name: {"data_dir": s["data_dir"], "ids": select_eval_set(ROOT / s["data_dir"], s["num_samples"], seed)}
            for name, s in ec["sets"].items()}


def load_model(spec: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = ROOT / spec["ckpt"]
    cfg = load_config([ckpt / "config.yaml"])
    if spec["kind"] == "lora":
        return cfg, load_models(cfg, ckpt, device)
    return cfg, load_models(cfg, None, device, full_dir=ckpt)


@torch.no_grad()
def generate_one(html: str, models: dict[str, Any], cfg: dict[str, Any], ec: dict[str, Any], seed: int,
                 uncond: tuple[torch.Tensor, torch.Tensor | None], device: torch.device):
    cond, cond_mask = encode(html, models, device)
    latents = sample_latents(models["unet"], models["scheduler"], cond, uncond[0], ec["num_inference_steps"],
                             ec["guidance_scale"], cfg["data"]["resolution"] // 8,
                             torch.Generator("cpu").manual_seed(seed), cfg["sample"].get("guidance_rescale", 0.0),
                             cond_mask=cond_mask, uncond_mask=uncond[1])
    return decode_latents(models["vae"], latents)[0]


def generate_model(name: str, spec: dict[str, Any], eval_sets: dict[str, dict[str, Any]], ec: dict[str, Any],
                   out_dir: Path, seed: int, device: torch.device) -> None:
    todo = [(s, i, k) for s, es in eval_sets.items() for k, i in enumerate(es["ids"])
            if not (out_dir / name / s / f"{i}.png").exists()]
    if not todo:
        log.info("%s: 모두 생성돼 있어 건너뛴다", name)
        return
    cfg, models = load_model(spec, device)
    uncond = encode("", models, device)
    for set_name, sample_id, k in tqdm(todo, desc=name):
        html = (ROOT / eval_sets[set_name]["data_dir"] / f"{sample_id}.html").read_text(encoding="utf-8")
        img = generate_one(html, models, cfg, ec, seed + k, uncond, device)
        path = out_dir / name / set_name / f"{sample_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    del models
    gc.collect()
    free_device_memory(device)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Phase 4 평가용 이미지 생성")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=None, help="생성할 모델 이름 (기본값: 설정의 전체)")
    p.add_argument("--seed", type=int, default=None, help="기본값: eval.seed")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    seed = args.seed if args.seed is not None else ec["seed"]
    out_dir = ROOT / ec["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    eval_sets = build_eval_sets(ec, seed)
    (out_dir / "eval_set.json").write_text(json.dumps({"seed": seed, "sets": eval_sets}, indent=1), encoding="utf-8")
    for name in args.models or list(ec["models"]):
        generate_model(name, ec["models"][name], eval_sets, ec, out_dir, seed, device)
    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
