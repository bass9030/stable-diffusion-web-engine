"""VAE latent와 CLIP 텍스트 임베딩을 미리 계산해 저장한다.

텍스트 인코더와 VAE는 학습하지 않으므로 한 번만 계산하면 된다.
학습 중 GPU 메모리에 UNet만 올릴 수 있고 step당 시간도 줄어든다.

사용법:
    python -m src.train.cache_features --config configs/base.yaml configs/train_lora.yaml

출력 (cache.dir 아래):
    train.pt / val.pt   {"ids": [...], "latents": [N,4,H/8,W/8] fp16, "embeds": [N,77,768] fp16}
    empty.pt            빈 프롬프트 임베딩 [1,77,768] (caption dropout, CFG용)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.datagen.config import load_config
from src.train.utils import ROOT, resolve_device, resolve_dtype

log = logging.getLogger("cache_features")


def read_metadata(data_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (data_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]


def load_image_tensor(path: Path, resolution: int) -> torch.Tensor:
    """PNG를 [-1, 1] 범위의 [3,H,W] 텐서로 읽는다."""
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BICUBIC)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0


@torch.no_grad()
def encode_texts(texts: list[str], tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
                 device: torch.device) -> torch.Tensor:
    ids = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


@torch.no_grad()
def encode_images(images: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    # 결정적 캐시를 위해 latent 분포의 평균을 사용한다
    return vae.encode(images).latent_dist.mean * vae.config.scaling_factor


def cache_split(rows: list[dict[str, Any]], data_dir: Path, cfg: dict[str, Any], vae: AutoencoderKL,
                tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device) -> dict[str, Any]:
    bs = cfg["cache"]["batch_size"]
    res = cfg["data"]["resolution"]
    latents: list[torch.Tensor] = []
    embeds: list[torch.Tensor] = []
    for i in tqdm(range(0, len(rows), bs), desc="cache"):
        batch = rows[i:i + bs]
        imgs = torch.stack([load_image_tensor(data_dir / f"{r['id']}.png", res) for r in batch])
        latents.append(encode_images(imgs.to(device, vae.dtype), vae).to("cpu", torch.float16))
        texts = [(data_dir / f"{r['id']}.html").read_text(encoding="utf-8") for r in batch]
        embeds.append(encode_texts(texts, tokenizer, text_encoder, device).to("cpu", torch.float16))
    return {"ids": [r["id"] for r in rows], "latents": torch.cat(latents), "embeds": torch.cat(embeds)}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="VAE latent / 텍스트 임베딩 캐시 생성")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device()
    m = cfg["model"]
    data_dir = ROOT / cfg["data"]["dir"]
    out_dir = ROOT / cfg["cache"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(m["name"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        m["name"], subfolder="text_encoder", variant=m["variant"],
        torch_dtype=resolve_dtype(m["text_encoder_dtype"], device)).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        m["name"], subfolder="vae", variant=m["variant"],
        torch_dtype=resolve_dtype(m["vae_dtype"], device)).to(device).eval()

    rows = read_metadata(data_dir)
    for split in ("train", "val"):
        split_rows = [r for r in rows if r["split"] == split]
        log.info("%s: %d개", split, len(split_rows))
        data = cache_split(split_rows, data_dir, cfg, vae, tokenizer, text_encoder, device)
        if not torch.isfinite(data["latents"]).all() or not torch.isfinite(data["embeds"]).all():
            raise SystemExit(f"{split} 캐시에 NaN/Inf가 있다. vae_dtype/text_encoder_dtype을 float32로 바꿔 다시 실행한다.")
        torch.save(data, out_dir / f"{split}.pt")
    torch.save(encode_texts([""], tokenizer, text_encoder, device).to("cpu", torch.float16), out_dir / "empty.pt")
    (out_dir / "source.json").write_text(json.dumps(
        {"data_dir": cfg["data"]["dir"], "model": m, "resolution": cfg["data"]["resolution"]}, indent=2), encoding="utf-8")
    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
