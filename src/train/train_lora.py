"""Phase 2: SD 1.5 UNet LoRA 학습 (캐시된 latent / 텍스트 임베딩 사용).

사용법:
    python -m src.train.train_lora --config configs/base.yaml configs/train_lora.yaml [--seed 42]

출력 (outputs/{run_name}/):
    config.yaml, run_info.json   병합된 설정과 git commit
    log.jsonl                    step별 loss, lr
    samples/step_XXXXXX.png      행마다 [정답 | 생성] (검증 샘플 → 학습 샘플 순)
    checkpoints/step_XXXXXX/     LoRA 가중치 + config.yaml + run_info.json
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
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm

from src.datagen.config import load_config
from src.infer.sampler import decode_latents, make_grid, sample_latents
from src.train.utils import (ROOT, configure_backend, free_device_memory, noise_schedule_kwargs, resolve_device,
                             resolve_dtype, save_run_info, set_seed)

log = logging.getLogger("train_lora")


def load_cache(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    train = torch.load(cache_dir / "train.pt")
    val = torch.load(cache_dir / "val.pt")
    empty = torch.load(cache_dir / "empty.pt")
    return train, val, empty


def subset(data: dict[str, Any], n: int | None) -> dict[str, Any]:
    if n is None:
        return data
    return {"ids": data["ids"][:n], "latents": data["latents"][:n], "embeds": data["embeds"][:n]}


def build_unet(cfg: dict[str, Any], device: torch.device) -> UNet2DConditionModel:
    m, lc = cfg["model"], cfg["lora"]
    unet = UNet2DConditionModel.from_pretrained(
        m["name"], subfolder="unet", variant=m["variant"], torch_dtype=resolve_dtype(m["unet_dtype"], device))
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=lc["rank"], lora_alpha=lc["alpha"], init_lora_weights="gaussian",
                                target_modules=lc["target_modules"]))
    # 학습 파라미터(LoRA)는 항상 fp32로 둔다
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if cfg["train"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    return unet.to(device)


def sample_timesteps(n: int, num_train_timesteps: int, high_prob: float, high_min: int,
                     generator: torch.Generator) -> torch.Tensor:
    """균등 분포로 뽑되, high_prob 확률로 [high_min, T) 구간에서 뽑는다.

    단색 위주 이미지는 낮은 timestep의 노이즈 예측이 쉬워 전역 색/레이아웃을 정하는
    높은 timestep 구간의 학습 신호가 부족하다 (Phase 2 overfit 테스트 관찰).
    """
    t = torch.randint(0, num_train_timesteps, (n,), generator=generator)
    t_high = torch.randint(high_min, num_train_timesteps, (n,), generator=generator)
    use_high = torch.rand(n, generator=generator) < high_prob
    return torch.where(use_high, t_high, t)


def training_loss(unet: UNet2DConditionModel, noise_scheduler: DDPMScheduler, latents: torch.Tensor,
                  embeds: torch.Tensor, generator: torch.Generator, tc: dict[str, Any]) -> torch.Tensor:
    noise_offset = tc["noise_offset"]
    noise = torch.randn(latents.shape, generator=generator, device="cpu")
    if noise_offset > 0:
        # offset noise: 채널별 상수 성분을 더해 단색/평균 밝기 학습을 돕는다
        noise = noise + noise_offset * torch.randn((*latents.shape[:2], 1, 1), generator=generator, device="cpu")
    noise = noise.to(latents.device)
    timesteps = sample_timesteps(latents.shape[0], noise_scheduler.config.num_train_timesteps,
                                 tc["high_timestep_prob"], tc["high_timestep_min"], generator).to(latents.device)
    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
    pred = unet(noisy.to(unet.dtype), timesteps, encoder_hidden_states=embeds.to(unet.dtype)).sample
    if noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    return F.mse_loss(pred.float(), target.float())


def save_samples(step: int, out_dir: Path, unet: UNet2DConditionModel, vae: AutoencoderKL,
                 sample_set: dict[str, torch.Tensor], empty: torch.Tensor, cfg: dict[str, Any],
                 device: torch.device, seed: int) -> None:
    sc = cfg["sample"]
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg["model"]["name"], subfolder="scheduler",
                                                            timestep_spacing=sc["timestep_spacing"],
                                                            **noise_schedule_kwargs(cfg["model"]))
    unet.eval()
    gen = torch.Generator("cpu").manual_seed(seed)
    # 한 번에 샘플링하면 CFG로 배치가 두 배가 되어 VRAM을 넘길 수 있어 작은 묶음으로 나눈다
    bs = sc.get("batch_size", 2)
    generated, targets = [], []
    for i in range(0, len(sample_set["latents"]), bs):
        latents = sample_latents(unet, scheduler, sample_set["embeds"][i:i + bs].to(device), empty.to(device),
                                 sc["num_inference_steps"], sc["guidance_scale"],
                                 sample_set["latents"].shape[-1], gen, sc.get("guidance_rescale", 0.0))
        generated += decode_latents(vae, latents)
        targets += decode_latents(vae, sample_set["latents"][i:i + bs].to(device))
    unet.train()
    # 샘플링 중 캐시된 메모리가 학습 중 VRAM 초과(공유 메모리 사용)로 이어지지 않게 비운다
    free_device_memory(device)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    make_grid([[t, g] for t, g in zip(targets, generated)]).save(out_dir / "samples" / f"step_{step:06d}.png")
    # 정답(VAE 복원) 대비 픽셀 MSE [0,1] 범위. 앞 num_sample_val개는 검증, 나머지는 학습 샘플
    mse = [float(((np.asarray(t, np.float32) - np.asarray(g, np.float32)) / 255.0).__pow__(2).mean())
           for t, g in zip(targets, generated)]
    n_val = cfg["data"]["num_sample_val"]
    rec = {"step": step, "val_mse": float(np.mean(mse[:n_val])), "train_mse": float(np.mean(mse[n_val:])),
           "per_sample": [round(x, 5) for x in mse]}
    with open(out_dir / "samples" / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log.info("step %d 샘플 MSE: val %.4f / train %.4f", step, rec["val_mse"], rec["train_mse"])


def save_checkpoint(step: int, out_dir: Path, unet: UNet2DConditionModel, cfg: dict[str, Any]) -> None:
    ckpt = out_dir / "checkpoints" / f"step_{step:06d}"
    # PEFT 형식(lora_A/lora_B) 그대로 저장한다. convert_state_dict_to_diffusers는 어텐션 키만 옛 형식(lora.down/up)으로
    # 바꾸고 conv/proj 키는 남겨 형식이 섞이고, 불러올 때 어텐션 LoRA가 빠진다 (v8 체크포인트에서 확인)
    lora_state = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(ckpt, unet_lora_layers=lora_state, safe_serialization=True)
    save_run_info(ckpt, cfg, {"step": step})


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SD 1.5 UNet LoRA 학습")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=None, help="기본값: 설정 파일의 seed")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    cfg["seed"] = seed
    set_seed(seed)
    device = resolve_device()
    configure_backend(device, cfg["train"].get("allow_tf32", False))
    tc, dc = cfg["train"], cfg["data"]
    out_dir = ROOT / cfg["paths"]["output_dir"] / tc["run_name"]
    save_run_info(out_dir, cfg, {"device": str(device)})
    log.info("장치: %s, 출력: %s", device, out_dir)

    train, val, empty = load_cache(ROOT / cfg["cache"]["dir"])
    train = subset(train, dc["max_train_samples"])
    n_train = len(train["ids"])
    sample_set = {
        k: torch.cat([val[k][:dc["num_sample_val"]], train[k][:dc["num_sample_train"]]])
        for k in ("latents", "embeds")
    }
    log.info("학습 샘플 %d개, 샘플링 세트 %d개 (val %s / train %s)", n_train, len(sample_set["latents"]),
             val["ids"][:dc["num_sample_val"]], train["ids"][:dc["num_sample_train"]])

    unet = build_unet(cfg, device)
    vae = AutoencoderKL.from_pretrained(cfg["model"]["name"], subfolder="vae", variant=cfg["model"]["variant"],
                                        torch_dtype=resolve_dtype(cfg["model"]["vae_dtype"], device)).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(cfg["model"]["name"], subfolder="scheduler",
                                                    **noise_schedule_kwargs(cfg["model"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    log.info("학습 파라미터 %.2fM", sum(p.numel() for p in params) / 1e6)
    optimizer = torch.optim.AdamW(params, lr=tc["learning_rate"])
    lr_scheduler = get_scheduler(tc["lr_scheduler"], optimizer=optimizer,
                                 num_warmup_steps=tc["lr_warmup_steps"], num_training_steps=tc["max_steps"])

    gen = torch.Generator("cpu").manual_seed(seed)
    latents_all, embeds_all = train["latents"], train["embeds"]
    empty_dev = empty.to(device)
    unet.train()
    save_samples(0, out_dir, unet, vae, sample_set, empty, cfg, device, seed)

    with open(out_dir / "log.jsonl", "a", encoding="utf-8") as log_f, \
            tqdm(range(1, tc["max_steps"] + 1), desc="train") as pbar:
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
                loss = training_loss(unet, noise_scheduler, latents, embeds, gen, tc) / tc["grad_accum"]
                loss.backward()
                total += loss.item()
            if not math.isfinite(total):
                raise SystemExit(f"step {step}: loss가 NaN/Inf. unet_dtype을 float32로 두었는지 확인한다.")
            torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % tc["log_every"] == 0:
                rec = {"step": step, "loss": total, "lr": lr_scheduler.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0, 1)}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                pbar.set_postfix(loss=f"{total:.4f}")
            if step % tc["sample_every"] == 0 or step == tc["max_steps"]:
                save_samples(step, out_dir, unet, vae, sample_set, empty, cfg, device, seed)
            if step % tc["checkpoint_every"] == 0 or step == tc["max_steps"]:
                save_checkpoint(step, out_dir, unet, cfg)

    log.info("완료 -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
