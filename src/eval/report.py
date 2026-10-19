"""Phase 4: 생성 이미지(generate_eval.py)와 정답 렌더링을 비교해 지표 표와 비교 그리드를 만든다.

사용법:
    python -m src.eval.report --config configs/base.yaml configs/eval.yaml [--models ...] [--no-ocr]

출력 (eval.out_dir 아래):
    metrics.csv       샘플별 지표 (model, set, id, mse, ssim, lpips, clip, ocr)
    summary.json      모델 × 세트별 평균과 샘플 수
    report.md         비교표 (마크다운)
    grid_{set}.png    행마다 [정답 | 모델들...] (세트별 앞 grid_samples개)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.datagen.config import load_config
from src.eval.metrics import ClipImageSimilarity, LpipsMetric, OcrReader, pixel_mse, ssim, text_similarity
from src.infer.sampler import make_grid
from src.train.utils import ROOT, resolve_device

log = logging.getLogger("report")

METRICS = ["mse", "ssim", "lpips", "clip", "ocr"]
# 표에 함께 적는 좋은 방향
DIRECTION = {"mse": "↓", "ssim": "↑", "lpips": "↓", "clip": "↑", "ocr": "↑"}


def evaluate(models: list[str], eval_sets: dict[str, Any], out_dir: Path, mc: dict[str, Any], use_ocr: bool,
             device) -> list[dict[str, Any]]:
    lpips_metric = LpipsMetric(mc["lpips_net"], device)
    clip_metric = ClipImageSimilarity(mc["clip_model"], device)
    ocr = OcrReader(mc["ocr_languages"], device) if use_ocr else None
    rows: list[dict[str, Any]] = []
    for set_name, es in eval_sets.items():
        for sample_id in tqdm(es["ids"], desc=set_name):
            target = Image.open(ROOT / es["data_dir"] / f"{sample_id}.png").convert("RGB")
            target_text = ocr.read(target) if ocr else ""
            target_emb = clip_metric.embed(target)
            for model in models:
                path = out_dir / model / set_name / f"{sample_id}.png"
                if not path.exists():
                    log.warning("생성 이미지 없음: %s", path)
                    continue
                gen = Image.open(path).convert("RGB")
                rows.append({
                    "model": model, "set": set_name, "id": sample_id,
                    "mse": pixel_mse(gen, target), "ssim": ssim(gen, target), "lpips": lpips_metric(gen, target),
                    "clip": float(clip_metric.embed(gen) @ target_emb),
                    "ocr": text_similarity(ocr.read(gen), target_text) if ocr else None,
                })
    return rows


def summarize(rows: list[dict[str, Any]], models: list[str], sets: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for model in models:
        for set_name in sets:
            sub = [r for r in rows if r["model"] == model and r["set"] == set_name]
            entry: dict[str, Any] = {"n": len(sub)}
            for m in METRICS:
                vals = [r[m] for r in sub if r[m] is not None]
                entry[m] = float(np.mean(vals)) if vals else None
                entry[f"{m}_n"] = len(vals)
            summary.setdefault(model, {})[set_name] = entry
    return summary


def markdown_table(summary: dict[str, dict[str, Any]], sets: list[str], seed: int) -> str:
    lines = [f"# Phase 4 평가 결과 (eval seed {seed})", ""]
    for set_name in sets:
        lines += [f"## {set_name}", "", "| 모델 | n | " + " | ".join(f"{m} {DIRECTION[m]}" for m in METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        best = {}
        for m in METRICS:
            vals = {mod: s[set_name][m] for mod, s in summary.items() if s[set_name][m] is not None}
            if vals:
                best[m] = (min if DIRECTION[m] == "↓" else max)(vals, key=vals.get)
        for model, s in summary.items():
            e = s[set_name]
            cells = []
            for m in METRICS:
                v = e[m]
                text = "-" if v is None else f"{v:.4f}"
                cells.append(f"**{text}**" if best.get(m) == model else text)
            lines.append(f"| {model} | {e['n']} | " + " | ".join(cells) + " |")
        ocr_n = {mod: s[set_name]["ocr_n"] for mod, s in summary.items()}
        lines += ["", f"OCR은 정답에서 글자가 읽힌 샘플만 평균한다 (모델별 샘플 수: {ocr_n}).", ""]
    return "\n".join(lines)


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 * len(text) + 8, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return img


def save_grids(models: list[str], eval_sets: dict[str, Any], out_dir: Path, n: int) -> None:
    for set_name, es in eval_sets.items():
        rows = []
        for sample_id in es["ids"][:n]:
            row = [label(Image.open(ROOT / es["data_dir"] / f"{sample_id}.png").convert("RGB"), f"target {sample_id}")]
            for model in models:
                path = out_dir / model / set_name / f"{sample_id}.png"
                img = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", row[0].size, "black")
                row.append(label(img, model))
            rows.append([im.resize((256, 256)) for im in row])
        if rows:
            make_grid(rows).save(out_dir / f"grid_{set_name}.png")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Phase 4 지표 계산과 비교표 생성")
    p.add_argument("--config", type=Path, nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=None, help="비교할 모델 이름 (기본값: 설정의 전체)")
    p.add_argument("--no-ocr", action="store_true", help="OCR 지표를 건너뛴다 (가장 느림)")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec, mc = cfg["eval"], cfg["metrics"]
    out_dir = ROOT / ec["out_dir"]
    eval_info = json.loads((out_dir / "eval_set.json").read_text(encoding="utf-8"))
    eval_sets, seed = eval_info["sets"], eval_info["seed"]
    models = args.models or list(ec["models"])

    rows = evaluate(models, eval_sets, out_dir, mc, not args.no_ocr, resolve_device())
    with open(out_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "set", "id", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, models, list(eval_sets))
    (out_dir / "summary.json").write_text(json.dumps({"seed": seed, "summary": summary}, indent=1), encoding="utf-8")
    report = markdown_table(summary, list(eval_sets), seed)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    save_grids(models, eval_sets, out_dir, mc["grid_samples"])
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
