"""Phase 3: 긴 HTML 조건 입력용 인코더 (CodeT5+ 인코더 + projection).

- 토크나이저: CodeT5+의 바이트 단위 BPE. HTML을 손실 없이 토큰화한다 (T5/FLAN-T5 어휘에는 `<`가 없어 unk가 됨).
  transformers 5.20에서 이 저장소의 토크나이저 설정을 불러오지 못해 `tokenizers`로 vocab/merges를 직접 읽는다.
- 인코더: CodeT5+ 220M의 T5 인코더(d_model 768). 학습하지 않는다.
- projection: 인코더 출력을 UNet cross-attention 입력 공간으로 옮기는 LayerNorm + Linear. 학습한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from transformers import T5EncoderModel

PROJECTION_FILE = "html_projection.safetensors"


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    bos: int
    eos: int


class CodeT5pTokenizer:
    """CodeT5+ 바이트 단위 BPE 토크나이저. 앞뒤에 <s>, </s>를 붙인다."""

    def __init__(self, model_name: str) -> None:
        self._tok = ByteLevelBPETokenizer(hf_hub_download(model_name, "vocab.json"),
                                          hf_hub_download(model_name, "merges.txt"))
        vocab = self._tok.get_vocab()
        self.special = SpecialIds(pad=vocab["<pad>"], bos=vocab["<s>"], eos=vocab["</s>"])

    def encode(self, text: str) -> list[int]:
        return [self.special.bos, *self._tok.encode(text).ids, self.special.eos]

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def batch(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """오른쪽 패딩한 input_ids와 attention_mask. max_length를 넘으면 자르고 끝을 </s>로 맞춘다."""
        seqs = []
        for t in texts:
            ids = self.encode(t)
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [self.special.eos]
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.special.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return input_ids, mask


class HtmlTextEncoder:
    """고정된 CodeT5+ 인코더. 가변 길이 HTML을 [B, L, 768] 은닉 상태와 마스크로 바꾼다."""

    def __init__(self, model_name: str, max_length: int, device: torch.device, dtype: torch.dtype) -> None:
        self.tokenizer = CodeT5pTokenizer(model_name)
        self.max_length = max_length
        self.device = device
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.model.config.d_model

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, mask = self.tokenizer.batch(texts, self.max_length)
        input_ids, mask = input_ids.to(self.device), mask.to(self.device)
        hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
        return hidden, mask


class HtmlProjection(nn.Module):
    """인코더 출력 → UNet cross-attention 입력. LayerNorm 후 선형 변환."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(hidden))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in self.state_dict().items()}
        save_file(state, directory / PROJECTION_FILE)

    @classmethod
    def load(cls, directory: Path) -> "HtmlProjection":
        state = load_file(directory / PROJECTION_FILE)
        out_dim, in_dim = state["linear.weight"].shape
        proj = cls(in_dim, out_dim)
        proj.load_state_dict(state)
        return proj


def pad_sequences(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """가변 길이 [L_i, D] 텐서들을 [B, max L, D]로 오른쪽 패딩하고 마스크 [B, max L]를 만든다."""
    width = max(s.shape[0] for s in seqs)
    out = seqs[0].new_zeros((len(seqs), width, seqs[0].shape[1]))
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
        mask[i, :s.shape[0]] = 1
    return out, mask
