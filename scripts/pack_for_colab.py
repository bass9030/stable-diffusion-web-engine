"""Colab / 대여 GPU(Vast.ai, RunPod 등)에 올릴 번들 zip을 만든다.

git이 추적하는 파일(코드, 설정, 문서)과 latent/임베딩 캐시를 하나의 zip으로 묶는다.
데이터를 Colab에서 다시 렌더링하면 폰트(Arial) 등 렌더링 환경이 달라지므로 로컬에서 만든 캐시를 가져간다.

사용법:
    python scripts/pack_for_colab.py [--cache-dir outputs/cache/short] [--out outputs/colab_bundle.zip] [--include-data]
    # Phase 3: 캐시는 대여 GPU에서 만들므로 원본 데이터와 시작점 UNet(Phase 2 체크포인트)을 넣는다
    python scripts/pack_for_colab.py --no-cache --data-dir data/long         --extra outputs/full_short_rental/checkpoints/step_006000

--include-data 를 주면 캐시 원본 데이터(data/short 의 html/png/metadata)도 포함한다 (Colab에서 캐시를 다시 만들 때만 필요).
--data-dir 는 지정한 데이터 폴더를, --extra 는 지정한 파일/폴더(하위 포함)를 그대로 넣는다 (여러 번 지정 가능).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_tracked_files() -> list[Path]:
    """추적 중인 파일 + .gitignore에 걸리지 않는 새 파일 (작업 트리 기준)."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]


def git_status() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip())
    return commit, dirty


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Colab 업로드용 번들 생성")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "cache" / "short")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "colab_bundle.zip")
    p.add_argument("--include-data", action="store_true", help="캐시 원본 데이터 폴더도 포함")
    p.add_argument("--no-cache", action="store_true", help="캐시를 넣지 않는다 (대여 GPU에서 캐시를 만들 때)")
    p.add_argument("--data-dir", type=Path, default=None, help="넣을 데이터 폴더 (예: data/long)")
    p.add_argument("--extra", type=Path, action="append", default=[], help="추가로 넣을 파일/폴더")
    args = p.parse_args()

    cache_dir = args.cache_dir.resolve()
    if not args.no_cache:
        for name in ("train.pt", "val.pt", "empty.pt", "source.json"):
            if not (cache_dir / name).exists():
                raise SystemExit(f"{cache_dir / name} 가 없다. 먼저 python -m src.train.cache_features 를 실행한다.")

    commit, dirty = git_status()
    if dirty:
        print("[WARN] 커밋되지 않은 변경이 있다. 번들에는 작업 트리의 현재 파일이 들어간다.")
    files = git_tracked_files()
    if not args.no_cache:
        files += sorted(cache_dir.iterdir())
    if args.include_data:
        data_dir = ROOT / json.loads((cache_dir / "source.json").read_text(encoding="utf-8"))["data_dir"]
        files += sorted(f for f in data_dir.iterdir() if f.is_file())
    if args.data_dir is not None:
        files += sorted(f for f in (ROOT / args.data_dir).iterdir() if f.is_file())
    for extra in args.extra:
        path = (ROOT / extra).resolve()
        if not path.exists():
            raise SystemExit(f"{path} 가 없다.")
        files += [path] if path.is_file() else sorted(f for f in path.rglob("*") if f.is_file())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    # .pt 파일은 거의 압축되지 않으므로 그대로 저장해 시간을 줄인다
    with zipfile.ZipFile(args.out, "w") as zf:
        zf.writestr("BUNDLE_INFO.json", json.dumps({"git_commit": commit, "git_dirty": dirty}, indent=2))
        for f in files:
            compress = zipfile.ZIP_STORED if f.suffix in (".pt", ".png", ".safetensors") else zipfile.ZIP_DEFLATED
            zf.write(f, f.relative_to(ROOT).as_posix(), compress_type=compress)
            total += f.stat().st_size
    print(f"파일 {len(files)}개 ({total / 1024**2:.0f}MB) -> {args.out} ({args.out.stat().st_size / 1024**2:.0f}MB)")
    print(f"git commit {commit[:7]}{' (dirty)' if dirty else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
