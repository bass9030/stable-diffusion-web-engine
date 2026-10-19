"""Phase 3: VAE latent와 CodeT5+ 인코더 출력(가변 길이)을 미리 계산해 저장한다.

인코더는 학습하지 않으므로 출력을 한 번만 계산한다. projection은 학습 대상이라 캐시하지 않는다.

사용법:
    python -m src.train.cache_html --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml \
        configs/train_html.yaml

출력 (cache.dir 아래):
    train.pt / val.pt   {"ids", "latents": [N,4,H/8,W/8] fp16,
                         "hidden": [전체 토큰 수, 768] fp16 (샘플별로 이어 붙임), "offsets": [N+1] int64}
                        샘플 i의 인코더 출력은 hidden[offsets[i]:offsets[i+1]]
    empty.pt            빈 HTML의 인코더 출력 [L, 768] fp16 (caption dropout, CFG용)
    source.json         데이터 경로, 인코더/모델 설정, 해상도
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from src.datagen.config import load_config
from src.models.html_encoder import HtmlTextEncoder
from src.train.cache_features import encode_images, load_image_tensor, read_metadata
from src.train.utils import ROOT, resolve_device, resolve_dtype

log = logging.getLogger("cache_html")


def cache_split(rows: list[dict[str, Any]], data_dir: Path, cfg: dict[str, Any], vae: AutoencoderKL,
                encoder: HtmlTextEncoder) -> dict[str, Any]:
    bs = cfg["cache"]["batch_size"]
    res = cfg["data"]["resolution"]
    latents: list[torch.Tensor] = []
    hidden: list[torch.Tensor] = []
    lengths: list[int] = []
    for i in tqdm(range(0, len(rows), bs), desc="cache"):
        batch = rows[i:i + bs]
        imgs = torch.stack([load_image_tensor(data_dir / f"{r['id']}.png", res) for r in batch])
        latents.append(encode_images(imgs.to(encoder.device, vae.dtype), vae).to("cpu", torch.float16))
        texts = [(data_dir / f"{r['id']}.html").read_text(encoding="utf-8") for r in batch]
        h, mask = encoder.encode(texts)
        for j in range(len(batch)):
            n = int(mask[j].sum())
            hidden.append(h[j, :n].to("cpu", torch.float16))
            lengths.append(n)
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(lengths), 0)
    return {"ids": [r["id"] for r in rows], "latents": torch.cat(latents), "hidden": torch.cat(hidden),
            "offsets": offsets}


def check_finite(data: dict[str, Any], split: str) -> None:
    if not torch.isfinite(data["latents"]).all() or not torch.isfinite(data["hidden"]).all():
        raise SystemExit(f"{split} 캐시에 NaN/Inf가 있다. vae_dtype/text_encoder.dtype을 float32로 바꿔 다시 실행한다.")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Phase 3 VAE latent / CodeT5+ 인코더 출력 캐시 생성")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    p.add_argument("--max-samples", type=int, default=None, help="split마다 앞에서부터 이 개수만 (점검용)")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device()
    m, tc = cfg["model"], cfg["text_encoder"]
    data_dir = ROOT / cfg["data"]["dir"]
    out_dir = ROOT / cfg["cache"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = HtmlTextEncoder(tc["name"], tc["max_length"], device, resolve_dtype(tc["dtype"], device))
    vae = AutoencoderKL.from_pretrained(
        m["name"], subfolder="vae", variant=m["variant"],
        torch_dtype=resolve_dtype(m["vae_dtype"], device)).to(device).eval()

    rows = read_metadata(data_dir)
    n_truncated = sum(1 for r in rows if r.get("codet5p_tokens", 0) > tc["max_length"])
    if n_truncated:
        log.warning("인코더 최대 길이 %d를 넘어 잘리는 샘플 %d개", tc["max_length"], n_truncated)
    for split in ("train", "val"):
        split_rows = [r for r in rows if r["split"] == split][:args.max_samples]
        log.info("%s: %d개", split, len(split_rows))
        data = cache_split(split_rows, data_dir, cfg, vae, encoder)
        check_finite(data, split)
        torch.save(data, out_dir / f"{split}.pt")
        lengths = (data["offsets"][1:] - data["offsets"][:-1]).float()
        log.info("%s: 토큰 수 평균 %.0f / 최대 %d", split, lengths.mean(), int(lengths.max()))
    empty, mask = encoder.encode([""])
    torch.save(empty[0, :int(mask[0].sum())].to("cpu", torch.float16), out_dir / "empty.pt")
    (out_dir / "source.json").write_text(json.dumps(
        {"data_dir": cfg["data"]["dir"], "model": m, "text_encoder": tc, "resolution": cfg["data"]["resolution"],
         "max_samples": args.max_samples}, indent=2), encoding="utf-8")
    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
