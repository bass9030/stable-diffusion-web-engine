"""데이터 생성 시드 재현성 검사.

생성된 데이터셋의 render_env.json(시드, 생성/렌더 설정)으로 샘플을 새 브라우저에서 다시 만들어
저장된 HTML/PNG 해시와 비교한다.

사용법:
    python scripts/check_datagen_repro.py [--data-dir data] [--num 50] [--seed 42]

--seed 는 검사할 샘플을 고르는 시드다. 불일치가 하나라도 있으면 종료 코드 1을 반환한다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datagen.build_dataset import build_sample, sha256_hex  # noqa: E402
from src.datagen.html_gen import GenConfig  # noqa: E402
from src.datagen.renderer import Renderer  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="데이터 생성 시드 재현성 검사")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--num", type=int, default=50, help="검사할 샘플 수")
    p.add_argument("--seed", type=int, default=42, help="검사 샘플 선택용 시드")
    args = p.parse_args()

    env = json.loads((args.data_dir / "render_env.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.data_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    picked = random.Random(args.seed).sample(rows, min(args.num, len(rows)))
    gen_cfg = GenConfig.from_dict(env["generator"])

    mismatches = 0
    with Renderer.from_config(env["render"]) as renderer:
        if renderer.browser_version != env["browser_version"]:
            print(f"[WARN] 브라우저 버전이 다름: 데이터 {env['browser_version']} / 현재 {renderer.browser_version}")
        for row in picked:
            sample = build_sample(row["index"], env["base_seed"], gen_cfg, env["dataset"]["val_ratio"])
            html_ok = sha256_hex(sample["html"].encode("utf-8")) == row["html_sha256"]
            png_ok = sha256_hex(renderer.render(sample["html"])) == row["png_sha256"]
            split_ok = sample["split"] == row["split"]
            if not (html_ok and png_ok and split_ok):
                mismatches += 1
                print(f"[FAIL] id={row['id']} html={html_ok} png={png_ok} split={split_ok}")

    print(f"결과: {len(picked)}개 중 불일치 {mismatches}개")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
