"""텍스트 임베딩으로부터 이미지를 생성하는 최소 샘플링 루프.

학습 중 샘플 저장과 추론 스크립트가 같은 코드를 쓰도록 분리했다.
"""
from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from PIL import Image


@torch.no_grad()
def sample_latents(unet: UNet2DConditionModel, scheduler: DPMSolverMultistepScheduler, cond: torch.Tensor,
                   uncond: torch.Tensor, num_steps: int, guidance_scale: float, latent_size: int,
                   generator: torch.Generator, guidance_rescale: float = 0.0,
                   cond_mask: torch.Tensor | None = None, uncond_mask: torch.Tensor | None = None) -> torch.Tensor:
    """cond: [B,L,D], uncond: [1,L',D]. guidance_scale <= 1이면 CFG를 쓰지 않는다.

    guidance_rescale > 0이면 CFG로 커진 분산을 조건부 예측 수준으로 되돌린다 (zero terminal SNR 논문 권장).
    cond_mask / uncond_mask: 가변 길이 조건(Phase 3)의 [B,L] / [1,L'] 마스크 (1=유효 토큰). CLIP(고정 77토큰)이면 None.
    """
    device, dtype = cond.device, unet.dtype
    b = cond.shape[0]
    latents = torch.randn((b, unet.config.in_channels, latent_size, latent_size),
                          generator=generator, device="cpu").to(device, dtype)
    scheduler.set_timesteps(num_steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    use_cfg = guidance_scale > 1.0
    if use_cfg:
        embeds, mask = concat_for_cfg(uncond.expand(b, -1, -1), _expand_mask(uncond_mask, b), cond, cond_mask)
    else:
        embeds, mask = cond, cond_mask
    for t in scheduler.timesteps:
        inp = torch.cat([latents] * 2) if use_cfg else latents
        inp = scheduler.scale_model_input(inp, t)
        pred = unet(inp, t, encoder_hidden_states=embeds.to(dtype), encoder_attention_mask=mask).sample
        if use_cfg:
            pred_uncond, pred_cond = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            if guidance_rescale > 0:
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=guidance_rescale)
        latents = scheduler.step(pred, t, latents).prev_sample
    return latents


def _expand_mask(mask: torch.Tensor | None, b: int) -> torch.Tensor | None:
    return None if mask is None else mask.expand(b, -1)


def concat_for_cfg(uncond: torch.Tensor, uncond_mask: torch.Tensor | None, cond: torch.Tensor,
                   cond_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CFG용으로 [uncond; cond]를 이어 붙인다. 길이가 다르면 짧은 쪽을 0으로 패딩하고 마스크로 가린다."""
    if cond_mask is None and uncond_mask is None:
        return torch.cat([uncond, cond]), None
    width = max(uncond.shape[1], cond.shape[1])

    def pad(x: torch.Tensor, m: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        m = torch.ones(x.shape[:2], dtype=torch.long, device=x.device) if m is None else m.to(x.device)
        extra = width - x.shape[1]
        if extra:
            x = torch.cat([x, x.new_zeros(x.shape[0], extra, x.shape[2])], dim=1)
            m = torch.cat([m, m.new_zeros(m.shape[0], extra)], dim=1)
        return x, m

    u, um = pad(uncond, uncond_mask)
    c, cm = pad(cond, cond_mask)
    return torch.cat([u, c]), torch.cat([um, cm])


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> list[Image.Image]:
    imgs = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
    imgs = ((imgs.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(x) for x in imgs.permute(0, 2, 3, 1).cpu().numpy()]


def make_grid(rows: list[list[Image.Image]]) -> Image.Image:
    h, w = rows[0][0].height, rows[0][0].width
    grid = Image.new("RGB", (w * max(len(r) for r in rows), h * len(rows)), "black")
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            grid.paste(img, (j * w, i * h))
    return grid

