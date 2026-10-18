"""HTML을 입력받아 학습된 LoRA로 렌더링 이미지를 생성한다.

사용법:
    python -m src.infer.generate --lora outputs/lora_short/checkpoints/step_003000 \
        --html data/short/000000.html --out outputs/infer/000000.png [--seed 42]
    python -m src.infer.generate --lora ... --html-text '<body style="background:#e33"></body>' --out x.png

--lora 를 생략하면 베이스 모델(SD 1.5) 그대로 생성한다.
모델/샘플링 설정은 체크포인트의 config.yaml(없으면 --config)을 따른다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.datagen.config import load_config
from src.infer.sampler import decode_latents, sample_latents
from src.train.utils import ROOT, noise_schedule_kwargs, resolve_device, resolve_dtype, set_seed

log = logging.getLogger("generate")


def load_models(cfg: dict[str, Any], lora_dir: Path | None, device: torch.device) -> dict[str, Any]:
    m = cfg["model"]
    unet = UNet2DConditionModel.from_pretrained(
        m["name"], subfolder="unet", variant=m["variant"], torch_dtype=resolve_dtype(m["unet_dtype"], device))
    if lora_dir is not None:
        unet.load_lora_adapter(str(lora_dir), weight_name="pytorch_lora_weights.safetensors", prefix="unet")
    return {
        "tokenizer": CLIPTokenizer.from_pretrained(m["name"], subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(
            m["name"], subfolder="text_encoder", variant=m["variant"],
            torch_dtype=resolve_dtype(m["text_encoder_dtype"], device)).to(device).eval(),
        "vae": AutoencoderKL.from_pretrained(
            m["name"], subfolder="vae", variant=m["variant"],
            torch_dtype=resolve_dtype(m["vae_dtype"], device)).to(device).eval(),
        "unet": unet.to(device).eval(),
        "scheduler": DPMSolverMultistepScheduler.from_pretrained(
            m["name"], subfolder="scheduler", timestep_spacing=cfg["sample"]["timestep_spacing"],
            **noise_schedule_kwargs(m)),
    }


@torch.no_grad()
def encode(text: str, models: dict[str, Any], device: torch.device) -> torch.Tensor:
    tok = models["tokenizer"]
    n_tokens = len(tok(text, truncation=False, verbose=False).input_ids)
    if n_tokens > tok.model_max_length:
        log.warning("입력이 %d토큰으로 %d토큰을 넘어 뒷부분이 잘린다", n_tokens, tok.model_max_length)
    ids = tok([text], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(device)
    return models["text_encoder"](ids)[0]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="HTML → 렌더링 이미지 생성")
    p.add_argument("--lora", type=Path, default=None, help="LoRA 체크포인트 폴더")
    p.add_argument("--config", type=Path, nargs="+",
                   default=[ROOT / "configs" / "base.yaml", ROOT / "configs" / "train_lora.yaml"])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--html", type=Path, help="HTML 파일")
    src.add_argument("--html-text", type=str, help="HTML 문자열")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=None, help="기본값: sample.num_inference_steps")
    p.add_argument("--guidance", type=float, default=None, help="기본값: sample.guidance_scale")
    args = p.parse_args()

    ckpt_cfg = args.lora / "config.yaml" if args.lora else None
    cfg = load_config([ckpt_cfg] if ckpt_cfg and ckpt_cfg.exists() else args.config)
    set_seed(args.seed)
    device = resolve_device()
    html = args.html.read_text(encoding="utf-8") if args.html else args.html_text

    models = load_models(cfg, args.lora, device)
    cond = encode(html, models, device)
    uncond = encode("", models, device)
    latents = sample_latents(
        models["unet"], models["scheduler"], cond, uncond,
        args.steps or cfg["sample"]["num_inference_steps"],
        args.guidance if args.guidance is not None else cfg["sample"]["guidance_scale"],
        cfg["data"]["resolution"] // 8, torch.Generator("cpu").manual_seed(args.seed),
        cfg["sample"].get("guidance_rescale", 0.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    decode_latents(models["vae"], latents)[0].save(args.out)
    log.info("저장 -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
