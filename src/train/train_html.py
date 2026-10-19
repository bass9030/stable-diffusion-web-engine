"""Phase 3: 긴 HTML 조건 입력 학습 (고정 CodeT5+ 인코더 + 학습 projection + UNet full fine-tuning).

캐시(cache_html.py)의 인코더 출력을 projection으로 UNet cross-attention 공간에 옮기고, 패딩은 마스크로 가린다.
UNet은 Phase 2 full FT 체크포인트(model.unet_init)에서 시작한다. 처음 freeze_unet_steps 동안은 UNet을 고정하고
projection만 학습해 조건 임베딩을 먼저 맞춘다.

사용법:
    python -m src.train.train_html --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml \
        configs/train_html.yaml [configs/train_full_24gb.yaml] [--resume <checkpoints/step_XXXXXX>] [--seed 42]

출력 (outputs/{run_name}/): config.yaml, run_info.json, log.jsonl, samples/ (train_lora.py와 같음)
    checkpoints/step_XXXXXX/unet/                       UNet (checkpoint_dtype)
    checkpoints/step_XXXXXX/html_projection.safetensors projection (fp32)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from src.datagen.config import load_config
from src.infer.sampler import decode_latents, make_grid, sample_latents
from src.models.html_encoder import HtmlProjection, pad_sequences
from src.train.train_full import autocast_context, build_optimizer, save_full_checkpoint
from src.train.train_lora import training_loss
from src.train.utils import (ROOT, configure_backend, free_device_memory, noise_schedule_kwargs, resolve_device,
                             resolve_dtype, save_run_info, set_seed)

log = logging.getLogger("train_html")


class HiddenCache:
    """cache_html.py가 만든 가변 길이 인코더 출력. 인덱스로 샘플별 [L_i, D]를 꺼낸다."""

    def __init__(self, data: dict[str, Any], limit: int | None = None) -> None:
        n = len(data["ids"]) if limit is None else min(limit, len(data["ids"]))
        self.ids = data["ids"][:n]
        self.latents = data["latents"][:n]
        self.hidden = data["hidden"]
        self.offsets = data["offsets"][:n + 1]

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, i: int) -> torch.Tensor:
        return self.hidden[self.offsets[i]:self.offsets[i + 1]]


def load_cache(cache_dir: Path, max_train: int | None) -> tuple[HiddenCache, HiddenCache, torch.Tensor]:
    train = HiddenCache(torch.load(cache_dir / "train.pt"), max_train)
    val = HiddenCache(torch.load(cache_dir / "val.pt"))
    return train, val, torch.load(cache_dir / "empty.pt")


def load_unet(cfg: dict[str, Any], device: torch.device, resume: Path | None) -> UNet2DConditionModel:
    """재개 체크포인트 > model.unet_init(Phase 2 체크포인트) > 베이스 모델 순으로 UNet을 불러온다 (fp32 마스터 가중치)."""
    m = cfg["model"]
    if resume is not None:
        unet = UNet2DConditionModel.from_pretrained(resume / "unet", torch_dtype=torch.float32)
    elif m.get("unet_init"):
        unet = UNet2DConditionModel.from_pretrained(ROOT / m["unet_init"] / "unet", torch_dtype=torch.float32)
    else:
        unet = UNet2DConditionModel.from_pretrained(m["name"], subfolder="unet", variant=m["variant"],
                                                    torch_dtype=torch.float32)
    if cfg["train"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    return unet.to(device)


def load_projection(cfg: dict[str, Any], unet: UNet2DConditionModel, hidden_dim: int, device: torch.device,
                    resume: Path | None) -> HtmlProjection:
    if resume is not None:
        return HtmlProjection.load(resume).to(device)
    return HtmlProjection(hidden_dim, unet.config.cross_attention_dim).to(device)


def make_batch(cache: HiddenCache, idx: torch.Tensor, empty: torch.Tensor, drop: torch.Tensor,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """latent, 패딩한 인코더 출력, 마스크. drop인 샘플은 빈 HTML 출력으로 바꾼다 (caption dropout)."""
    seqs = [empty if d else cache.get(int(i)) for i, d in zip(idx, drop)]
    hidden, mask = pad_sequences(seqs)
    latents = cache.latents[idx].to(device, torch.float32)
    return latents, hidden.to(device, torch.float32), mask.to(device)


def sample_set(train: HiddenCache, val: HiddenCache, dc: dict[str, Any]) -> dict[str, Any]:
    """고정 샘플링 세트: 검증 num_sample_val개 + 학습 num_sample_train개."""
    pairs = [(val, i) for i in range(dc["num_sample_val"])] + [(train, i) for i in range(dc["num_sample_train"])]
    return {"latents": torch.stack([c.latents[i] for c, i in pairs]), "hidden": [c.get(i) for c, i in pairs],
            "ids": [c.ids[i] for c, i in pairs]}


@torch.no_grad()
def save_samples(step: int, out_dir: Path, unet: UNet2DConditionModel, proj: HtmlProjection, vae: AutoencoderKL,
                 ss: dict[str, Any], empty: torch.Tensor, cfg: dict[str, Any], device: torch.device, seed: int) -> None:
    sc = cfg["sample"]
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg["model"]["name"], subfolder="scheduler",
                                                            timestep_spacing=sc["timestep_spacing"],
                                                            **noise_schedule_kwargs(cfg["model"]))
    unet.eval()
    proj.eval()
    gen = torch.Generator("cpu").manual_seed(seed)
    uncond, uncond_mask = pad_sequences([empty])
    uncond = proj(uncond.to(device, torch.float32))
    uncond_mask = uncond_mask.to(device)
    bs = sc.get("batch_size", 2)
    generated, targets = [], []
    for i in range(0, len(ss["hidden"]), bs):
        hidden, mask = pad_sequences(ss["hidden"][i:i + bs])
        cond = proj(hidden.to(device, torch.float32))
        latents = sample_latents(unet, scheduler, cond, uncond, sc["num_inference_steps"], sc["guidance_scale"],
                                 ss["latents"].shape[-1], gen, sc.get("guidance_rescale", 0.0),
                                 cond_mask=mask.to(device), uncond_mask=uncond_mask)
        generated += decode_latents(vae, latents)
        targets += decode_latents(vae, ss["latents"][i:i + bs].to(device))
    unet.train()
    proj.train()
    free_device_memory(device)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    make_grid([[t, g] for t, g in zip(targets, generated)]).save(out_dir / "samples" / f"step_{step:06d}.png")
    # 정답(VAE 복원) 대비 픽셀 MSE [0,1]. 앞 num_sample_val개는 검증, 나머지는 학습 샘플
    mse = [float((((np.asarray(t, np.float32) - np.asarray(g, np.float32)) / 255.0) ** 2).mean())
           for t, g in zip(targets, generated)]
    n_val = cfg["data"]["num_sample_val"]
    rec = {"step": step, "val_mse": float(np.mean(mse[:n_val])), "train_mse": float(np.mean(mse[n_val:])),
           "per_sample": [round(x, 5) for x in mse], "ids": ss["ids"]}
    with open(out_dir / "samples" / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log.info("step %d 샘플 MSE: val %.4f / train %.4f", step, rec["val_mse"], rec["train_mse"])


def save_checkpoint(step: int, out_dir: Path, unet: UNet2DConditionModel, proj: HtmlProjection,
                    cfg: dict[str, Any]) -> None:
    # projection을 먼저 저장해야 save_full_checkpoint의 오래된 체크포인트 정리 후에도 최신 폴더에 함께 남는다
    proj.save(out_dir / "checkpoints" / f"step_{step:06d}")
    save_full_checkpoint(step, out_dir, unet, cfg)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Phase 3: 긴 HTML 조건 입력 학습")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=None, help="기본값: 설정 파일의 seed")
    p.add_argument("--resume", type=Path, default=None, help="이어서 학습할 체크포인트 폴더 (step_XXXXXX)")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    cfg["seed"] = seed
    device = resolve_device()
    configure_backend(device, cfg["train"].get("allow_tf32", False))
    tc, dc = cfg["train"], cfg["data"]
    start_step = 0
    if args.resume is not None:
        start_step = json.loads((args.resume / "run_info.json").read_text(encoding="utf-8"))["step"]
    set_seed(seed + start_step)
    out_dir = ROOT / cfg["paths"]["output_dir"] / tc["run_name"]
    save_run_info(out_dir, cfg, {"device": str(device), "resumed_from": str(args.resume) if args.resume else None,
                                 "start_step": start_step})
    log.info("장치: %s, 출력: %s, 시작 step: %d", device, out_dir, start_step)

    train, val, empty = load_cache(ROOT / cfg["cache"]["dir"], dc["max_train_samples"])
    ss = sample_set(train, val, dc)
    log.info("학습 샘플 %d개, 샘플링 세트 %s", len(train), ss["ids"])

    unet = load_unet(cfg, device, args.resume)
    proj = load_projection(cfg, unet, train.hidden.shape[1], device, args.resume)
    vae = AutoencoderKL.from_pretrained(cfg["model"]["name"], subfolder="vae", variant=cfg["model"]["variant"],
                                        torch_dtype=resolve_dtype(cfg["model"]["vae_dtype"], device)).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(cfg["model"]["name"], subfolder="scheduler",
                                                    **noise_schedule_kwargs(cfg["model"]))
    optimizer = build_optimizer([
        {"params": list(unet.parameters()), "lr": tc["learning_rate"]},
        {"params": list(proj.parameters()), "lr": tc["projection_learning_rate"]},
    ], tc)
    lr_scheduler = get_scheduler(tc["lr_scheduler"], optimizer=optimizer, num_warmup_steps=tc["lr_warmup_steps"],
                                 num_training_steps=tc["max_steps"] - start_step)
    scaler = torch.amp.GradScaler(device.type, enabled=tc["mixed_precision"] == "float16")
    params = [*unet.parameters(), *proj.parameters()]
    log.info("UNet %.1fM + projection %.2fM, mixed_precision=%s, 8bit_adam=%s, freeze_unet_steps=%d",
             sum(p.numel() for p in unet.parameters()) / 1e6, sum(p.numel() for p in proj.parameters()) / 1e6,
             tc["mixed_precision"], tc["use_8bit_adam"], tc["freeze_unet_steps"])

    gen = torch.Generator("cpu").manual_seed(seed + start_step)
    unet.train()
    proj.train()
    unet_frozen = start_step < tc["freeze_unet_steps"]
    unet.requires_grad_(not unet_frozen)
    if start_step == 0:
        save_samples(0, out_dir, unet, proj, vae, ss, empty, cfg, device, seed)

    with open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f, \
            tqdm(range(start_step + 1, tc["max_steps"] + 1), desc="train") as pbar:
        t0 = time.time()
        for step in pbar:
            if unet_frozen and step > tc["freeze_unet_steps"]:
                unet_frozen = False
                unet.requires_grad_(True)
                log.info("step %d: UNet 학습 시작", step)
            total = 0.0
            for _ in range(tc["grad_accum"]):
                idx = torch.randint(0, len(train), (tc["batch_size"],), generator=gen)
                drop = torch.rand(len(idx), generator=gen) < tc["caption_dropout"]
                latents, hidden, mask = make_batch(train, idx, empty, drop, device)
                with autocast_context(device, tc["mixed_precision"]):
                    embeds = proj(hidden)
                    loss = training_loss(unet, noise_scheduler, latents, embeds, gen, tc,
                                         encoder_attention_mask=mask) / tc["grad_accum"]
                scaler.scale(loss).backward()
                total += loss.item()
            if not math.isfinite(total):
                raise SystemExit(f"step {step}: loss가 NaN/Inf. mixed_precision을 bfloat16 또는 no로 바꿔 본다.")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % tc["log_every"] == 0:
                rec = {"step": step, "loss": total, "lr": lr_scheduler.get_last_lr()[0],
                       "unet_frozen": unet_frozen, "elapsed_s": round(time.time() - t0, 1)}
                if device.type == "cuda":
                    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                pbar.set_postfix(loss=f"{total:.4f}")
            if step % tc["sample_every"] == 0 or step == tc["max_steps"]:
                save_samples(step, out_dir, unet, proj, vae, ss, empty, cfg, device, seed)
            if step % tc["checkpoint_every"] == 0 or step == tc["max_steps"]:
                save_checkpoint(step, out_dir, unet, proj, cfg)
                free_device_memory(device)

    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
