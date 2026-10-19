"""Phase 4 평가 지표. 모두 (생성 이미지, 정답 이미지) 쌍을 받는다.

- pixel_mse: [0,1] 픽셀 MSE (낮을수록 좋음). 큰 블록 위치 오차에 민감하다.
- ssim: 구조적 유사도 (높을수록 좋음, 최대 1).
- LpipsMetric: 학습된 지각 거리 (낮을수록 좋음).
- ClipImageSimilarity: CLIP 이미지 임베딩 코사인 유사도 (높을수록 좋음). 입력이 HTML이라 이미지-텍스트 CLIP score 대신 쓴다.
- OcrReader + text_similarity: OCR로 읽은 글자의 문자 단위 유사도 (1 - 정규화 편집거리, 높을수록 좋음).
  정답 렌더링에서 읽히는 글자를 기준으로 삼는다 (작거나 대비가 낮은 글자는 정답에서도 안 읽힘).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def pixel_mse(gen: Image.Image, target: Image.Image) -> float:
    return float((((to_array(gen) - to_array(target)) / 255.0) ** 2).mean())


def ssim(gen: Image.Image, target: Image.Image) -> float:
    return float(structural_similarity(to_array(gen), to_array(target), channel_axis=2, data_range=255.0))


class LpipsMetric:
    def __init__(self, net: str, device: torch.device) -> None:
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(to_array(img)).permute(2, 0, 1)[None] / 127.5 - 1.0

    @torch.no_grad()
    def __call__(self, gen: Image.Image, target: Image.Image) -> float:
        a, b = self._to_tensor(gen).to(self.device), self._to_tensor(target).to(self.device)
        return float(self.model(a, b).item())


class ClipImageSimilarity:
    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        pixels = self.processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device)
        emb = self.model(pixel_values=pixels).image_embeds[0]
        return emb / emb.norm()

    def __call__(self, gen: Image.Image, target: Image.Image) -> float:
        return float((self.embed(gen) @ self.embed(target)).item())


class OcrReader:
    def __init__(self, languages: list[str], device: torch.device) -> None:
        import easyocr

        self.reader = easyocr.Reader(languages, gpu=device.type == "cuda", verbose=False)

    def read(self, img: Image.Image) -> str:
        """읽은 글자 조각을 위→아래, 왼쪽→오른쪽 순으로 공백으로 이어 붙인다."""
        results = self.reader.readtext(to_array(img).astype(np.uint8))
        boxes = sorted(results, key=lambda r: (round(min(p[1] for p in r[0]) / 10), min(p[0] for p in r[0])))
        return " ".join(text for _, text, _ in boxes).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_similarity(gen_text: str, target_text: str) -> float | None:
    """대소문자·공백을 정규화한 1 - 편집거리/최대 길이. 정답에서 읽힌 글자가 없으면 None (평균에서 제외)."""
    a, b = " ".join(gen_text.lower().split()), " ".join(target_text.lower().split())
    if not b:
        return None
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))
