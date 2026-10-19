"""Phase 1 합성 데이터셋 생성.

사용법:
    python -m src.datagen.build_dataset [--seed 42] [--num-samples N] [--out-dir data] [--overwrite]

출력 (out-dir 아래):
    {id}.html, {id}.png   샘플 쌍 (id = 6자리 인덱스)
    metadata.jsonl        샘플별 시드, 분할(train/val), 해시
    render_env.json       렌더링 환경(브라우저 버전, viewport, 폰트 등)과 생성 설정
    failures.jsonl        렌더링에 실패한 샘플 (실패가 있을 때만)
    summary.json          성공/실패/토큰 초과로 건너뛴 개수

dataset.max_tokens(이전 이름 max_clip_tokens)가 설정되면 토큰 수(dataset.tokenizer_type 기준)가 넘는 샘플은 렌더링 전에 건너뛰고,
통과한 샘플이 num_samples 개가 될 때까지 인덱스를 늘려간다 (인덱스·시드는 그대로 결정적).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import load_config
from .html_gen import GenConfig, generate_html
from .renderer import Renderer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [ROOT / "configs" / "base.yaml", ROOT / "configs" / "datagen.yaml"]

log = logging.getLogger("build_dataset")


def derive_seed(base_seed: int, index: int, purpose: str) -> int:
    """(기본 시드, 인덱스, 용도)로부터 결정적인 64비트 시드를 만든다."""
    digest = hashlib.sha256(f"{purpose}:{base_seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_split(sample_seed: int, val_ratio: float) -> str:
    """HTML 생성 시드만으로 train/val을 정한다 (생성 순서·개수와 무관)."""
    digest = hashlib.sha256(f"split:{sample_seed}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if u < val_ratio else "train"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_is_dirty() -> bool | None:
    """커밋되지 않은 변경이 있으면 True (git_commit 만으로는 코드 상태를 재현할 수 없음)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def build_render_env(cfg: dict[str, Any], base_seed: int, browser_version: str) -> dict[str, Any]:
    return {
        "base_seed": base_seed,
        "browser": "chromium",
        "browser_version": browser_version,
        "playwright_version": metadata.version("playwright"),
        "render": cfg["render"],
        "generator": cfg["generator"],
        "dataset": cfg["dataset"],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class TokenCounter:
    """토큰 수(BOS/EOS 포함)를 센다. 설정이 없으면 세지 않는다.

    tokenizer_type: "clip"(Phase 2, SD 1.5 CLIP) | "codet5p"(Phase 3, CodeT5+ 바이트 단위 BPE)
    """

    def __init__(self, tokenizer_name: str | None, tokenizer_type: str = "clip") -> None:
        self.tokenizer_type = tokenizer_type
        self._count = None
        if not tokenizer_name:
            return
        if tokenizer_type == "clip":
            from transformers import CLIPTokenizer

            tok = CLIPTokenizer.from_pretrained(tokenizer_name, subfolder="tokenizer")
            self._count = lambda text: len(tok(text, truncation=False, verbose=False).input_ids)
        elif tokenizer_type == "codet5p":
            from src.models.html_encoder import CodeT5pTokenizer

            self._count = CodeT5pTokenizer(tokenizer_name).count
        else:
            raise ValueError(f"알 수 없는 tokenizer_type: {tokenizer_type}")

    @property
    def metadata_key(self) -> str:
        return f"{self.tokenizer_type}_tokens"

    def count(self, text: str) -> int | None:
        return None if self._count is None else self._count(text)


def build_sample(index: int, base_seed: int, gen_cfg: GenConfig, val_ratio: float) -> dict[str, Any]:
    seed = derive_seed(base_seed, index, "html")
    return {
        "id": f"{index:06d}",
        "index": index,
        "seed": seed,
        "split": assign_split(seed, val_ratio),
        "html": generate_html(seed, gen_cfg),
    }


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "metadata.jsonl"
    if meta.exists() and not overwrite:
        raise SystemExit(f"{meta} 가 이미 있다. 덮어쓰려면 --overwrite 를 지정한다.")
    for name in ("metadata.jsonl", "failures.jsonl", "summary.json"):
        (out_dir / name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 합성 HTML-스크린샷 데이터셋 생성")
    p.add_argument("--config", type=Path, nargs="+", default=DEFAULT_CONFIGS, help="순서대로 병합할 yaml")
    p.add_argument("--seed", type=int, default=None, help="기본 시드 (기본값: 설정 파일의 seed)")
    p.add_argument("--num-samples", type=int, default=None, help="생성 개수 (기본값: dataset.num_samples)")
    p.add_argument("--out-dir", type=Path, default=None, help="출력 폴더 (기본값: paths.data_dir)")
    p.add_argument("--overwrite", action="store_true", help="기존 metadata.jsonl 이 있어도 덮어쓴다")
    return p.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    cfg = load_config(args.config)
    base_seed = args.seed if args.seed is not None else cfg["seed"]
    num_samples = args.num_samples if args.num_samples is not None else cfg["dataset"]["num_samples"]
    out_dir = args.out_dir if args.out_dir is not None else ROOT / cfg["paths"]["data_dir"]
    val_ratio = cfg["dataset"]["val_ratio"]
    gen_cfg = GenConfig.from_dict(cfg["generator"])
    cfg["dataset"] = {**cfg["dataset"], "num_samples": num_samples}

    prepare_out_dir(out_dir, args.overwrite)
    # max_clip_tokens는 Phase 2 설정의 이전 이름 (data/short의 render_env.json과 호환)
    max_tokens = cfg["dataset"].get("max_tokens", cfg["dataset"].get("max_clip_tokens"))
    counter = TokenCounter(cfg["dataset"].get("tokenizer") if max_tokens else None,
                           cfg["dataset"].get("tokenizer_type", "clip"))
    n_ok = n_fail = n_val = n_skipped = 0
    with Renderer.from_config(cfg["render"]) as renderer,             open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as meta_f,             tqdm(total=num_samples, desc="render") as pbar:
        env = build_render_env(cfg, base_seed, renderer.browser_version)
        (out_dir / "render_env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

        index = 0
        while n_ok + n_fail < num_samples:
            sample = build_sample(index, base_seed, gen_cfg, val_ratio)
            index += 1
            n_tokens = counter.count(sample["html"])
            if max_tokens and n_tokens > max_tokens:
                n_skipped += 1
                continue
            pbar.update(1)
            html_bytes = sample["html"].encode("utf-8")
            try:
                png = renderer.render(sample["html"])
            except Exception as e:  # noqa: BLE001
                # 실패 샘플은 버리지 않고 기록한다 (CLAUDE.md 데이터 규칙)
                n_fail += 1
                log.warning("렌더링 실패 id=%s: %s", sample["id"], e)
                with open(out_dir / "failures.jsonl", "a", encoding="utf-8") as fail_f:
                    fail_f.write(json.dumps({
                        "id": sample["id"], "seed": sample["seed"], "html": sample["html"],
                        "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                    }, ensure_ascii=False) + "\n")
                continue

            (out_dir / f"{sample['id']}.html").write_bytes(html_bytes)
            (out_dir / f"{sample['id']}.png").write_bytes(png)
            meta_f.write(json.dumps({
                "id": sample["id"], "index": sample["index"], "seed": sample["seed"], "split": sample["split"],
                "html_chars": len(sample["html"]), counter.metadata_key: n_tokens,
                "html_sha256": sha256_hex(html_bytes), "png_sha256": sha256_hex(png),
            }) + "\n")
            n_ok += 1
            n_val += sample["split"] == "val"

    summary = {"ok": n_ok, "val": n_val, "failed": n_fail, "skipped_over_tokens": n_skipped, "last_index": index - 1}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("완료: 성공 %d (val %d) / 실패 %d / 토큰 초과 건너뜀 %d -> %s", n_ok, n_val, n_fail, n_skipped, out_dir)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
