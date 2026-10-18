"""Plan B: SD 1.5 UNet full fine-tuning (클라우드 GPU용, 캐시된 latent / 텍스트 임베딩 사용).

LoRA 학습(train_lora.py)과 같은 캐시·손실·샘플링 코드를 쓰고, UNet 전체 파라미터를 학습한다.
fp32 마스터 가중치 + AdamW 기준 약 14GB(가중치·기울기·옵티마이저 상태, 추정)에 activation이 더해지므로
로컬 6GB GPU에서는 돌릴 수 없다. A100/L4는 bf16, T4는 fp16 + 8-bit Adam 설정을 쓴다.

사용법:
    python -m src.train.train_full --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml \
        [configs/train_full_t4.yaml] [--resume outputs/full_short/checkpoints/step_001000] [--seed 42]

출력 (outputs/{run_name}/):
    config.yaml, run_info.json, log.jsonl, samples/   (train_lora.py와 같음)
    checkpoints/step_XXXXXX/unet/                     UNet 가중치 (checkpoint_dtype) + config.json
    checkpoints/step_XXXXXX/config.yaml, run_info.json

--resume 은 UNet 가중치와 step만 이어받는다. 옵티마이저 상태는 저장하지 않으므로(체크포인트 크기 절약)
재개 시 옵티마이저와 warmup이 새로 시작된다.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm import tqdm

from src.datagen.config import load_config
from src.train.train_lora import load_cache, save_samples, subset, training_loss
from src.train.utils import (DTYPES, ROOT, configure_backend, free_device_memory, noise_schedule_kwargs,
                             resolve_device, resolve_dtype, save_run_info, set_seed)

log = logging.getLogger("train_full")


def build_full_unet(cfg: dict[str, Any], device: torch.device, resume: Path | None) -> UNet2DConditionModel:
    """학습용 UNet. mixed precision을 쓰더라도 마스터 가중치는 fp32로 둔다."""
    if resume is not None:
        unet = UNet2DConditionModel.from_pretrained(resume / "unet", torch_dtype=torch.float32)
    else:
        m = cfg["model"]
        unet = UNet2DConditionModel.from_pretrained(m["name"], subfolder="unet", variant=m["variant"],
                                                    torch_dtype=torch.float32)
    unet.requires_grad_(True)
    if cfg["train"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    return unet.to(device)


def build_optimizer(params: list[torch.nn.Parameter], tc: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = {"lr": tc["learning_rate"], "weight_decay": tc["weight_decay"]}
    if tc["use_8bit_adam"]:
        import bitsandbytes as bnb  # CUDA 전용, 옵티마이저 상태를 8비트로 저장해 메모리 절약

        return bnb.optim.AdamW8bit(params, **kwargs)
    return torch.optim.AdamW(params, **kwargs)


def autocast_context(device: torch.device, mixed_precision: str) -> contextlib.AbstractContextManager:
    if mixed_precision == "no":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=DTYPES[mixed_precision])


def save_full_checkpoint(step: int, out_dir: Path, unet: UNet2DConditionModel, cfg: dict[str, Any]) -> None:
    """UNet을 checkpoint_dtype으로 저장한다 (학습 중인 fp32 가중치는 건드리지 않음)."""
    tc = cfg["train"]
    ckpt = out_dir / "checkpoints" / f"step_{step:06d}"
    unet_dir = ckpt / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[tc["checkpoint_dtype"]]
    state = {k: v.detach().to("cpu", dtype).contiguous() for k, v in unet.state_dict().items()}
    save_file(state, unet_dir / "diffusion_pytorch_model.safetensors")
    unet.save_config(unet_dir)
    save_run_info(ckpt, cfg, {"step": step})
    prune_checkpoints(out_dir / "checkpoints", tc["keep_last_checkpoints"])


def prune_checkpoints(ckpt_root: Path, keep: int | None) -> None:
    """최근 keep개만 남긴다 (Drive 용량 절약). None이면 모두 남긴다."""
    if not keep:
        return
    for old in sorted(ckpt_root.glob("step_*"))[:-keep]:
        shutil.rmtree(old)
        log.info("오래된 체크포인트 삭제: %s", old.name)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SD 1.5 UNet full fine-tuning")
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
    # 재개 시 같은 난수열을 반복하지 않도록 시작 step을 시드에 섞는다
    set_seed(seed + start_step)
    out_dir = ROOT / cfg["paths"]["output_dir"] / tc["run_name"]
    save_run_info(out_dir, cfg, {"device": str(device), "resumed_from": str(args.resume) if args.resume else None,
                                 "start_step": start_step})
    log.info("장치: %s, 출력: %s, 시작 step: %d", device, out_dir, start_step)

    train, val, empty = load_cache(ROOT / cfg["cache"]["dir"])
    train = subset(train, dc["max_train_samples"])
    n_train = len(train["ids"])
    sample_set = {
        k: torch.cat([val[k][:dc["num_sample_val"]], train[k][:dc["num_sample_train"]]])
        for k in ("latents", "embeds")
    }
    log.info("학습 샘플 %d개, 샘플링 세트 %d개", n_train, len(sample_set["latents"]))

    unet = build_full_unet(cfg, device, args.resume)
    vae = AutoencoderKL.from_pretrained(cfg["model"]["name"], subfolder="vae", variant=cfg["model"]["variant"],
                                        torch_dtype=resolve_dtype(cfg["model"]["vae_dtype"], device)).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(cfg["model"]["name"], subfolder="scheduler",
                                                    **noise_schedule_kwargs(cfg["model"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    log.info("학습 파라미터 %.1fM, mixed_precision=%s, 8bit_adam=%s",
             sum(p.numel() for p in params) / 1e6, tc["mixed_precision"], tc["use_8bit_adam"])
    optimizer = build_optimizer(params, tc)
    lr_scheduler = get_scheduler(tc["lr_scheduler"], optimizer=optimizer, num_warmup_steps=tc["lr_warmup_steps"],
                                 num_training_steps=tc["max_steps"] - start_step)
    # fp16은 기울기 underflow 방지를 위해 loss scaling이 필요하다 (bf16은 불필요)
    scaler = torch.amp.GradScaler(device.type, enabled=tc["mixed_precision"] == "float16")

    gen = torch.Generator("cpu").manual_seed(seed + start_step)
    latents_all, embeds_all = train["latents"], train["embeds"]
    empty_dev = empty.to(device)
    unet.train()
    if start_step == 0:
        save_samples(0, out_dir, unet, vae, sample_set, empty, cfg, device, seed)

    with open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f, \
            tqdm(range(start_step + 1, tc["max_steps"] + 1), desc="train") as pbar:
        t0 = time.time()
        for step in pbar:
            total = 0.0
            for _ in range(tc["grad_accum"]):
                idx = torch.randint(0, n_train, (tc["batch_size"],), generator=gen)
                latents = latents_all[idx].to(device, torch.float32)
                embeds = embeds_all[idx].to(device, torch.float32)
                drop = torch.rand(len(idx), generator=gen) < tc["caption_dropout"]
                if drop.any():
                    embeds[drop.to(device)] = empty_dev.float()
                with autocast_context(device, tc["mixed_precision"]):
                    loss = training_loss(unet, noise_scheduler, latents, embeds, gen, tc) / tc["grad_accum"]
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
                       "elapsed_s": round(time.time() - t0, 1)}
                if device.type == "cuda":
                    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                pbar.set_postfix(loss=f"{total:.4f}")
            if step % tc["sample_every"] == 0 or step == tc["max_steps"]:
                save_samples(step, out_dir, unet, vae, sample_set, empty, cfg, device, seed)
            if step % tc["checkpoint_every"] == 0 or step == tc["max_steps"]:
                save_full_checkpoint(step, out_dir, unet, cfg)
                free_device_memory(device)

    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
