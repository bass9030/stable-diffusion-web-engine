"""Phase 0 환경 점검 스크립트.

사용법:
    python scripts/check_env.py [--skip-browser]

FAIL 이 하나라도 있으면 종료 코드 1, 없으면 0 을 반환한다.
"""
from __future__ import annotations

import argparse
import importlib
import io
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (import 이름, pip 패키지 이름)
REQUIRED_PACKAGES = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("datasets", "datasets"),
    ("playwright", "playwright"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("lpips", "lpips"),
    ("skimage", "scikit-image"),
]

EXPECTED_DIRS = [
    "configs", "scripts", "src/datagen", "src/train", "src/models",
    "src/eval", "src/infer", "data", "outputs", "notebooks",
]

MIN_FREE_DISK_GB = 20
MIN_VRAM_GB = 8


class Report:
    def __init__(self) -> None:
        self.counts = {"OK": 0, "WARN": 0, "FAIL": 0}

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.counts[level] += 1
        print(f"[{level:<4}] {name}: {detail}")


def check_python(rep: Report) -> None:
    rep.add("OK", "Python", f"{platform.python_version()} ({platform.system()} {platform.machine()})")


def check_dirs(rep: Report) -> None:
    missing = [d for d in EXPECTED_DIRS if not (ROOT / d).is_dir()]
    if missing:
        rep.add("WARN", "디렉터리 구조", "없는 디렉터리: " + ", ".join(missing))
    else:
        rep.add("OK", "디렉터리 구조", "CLAUDE.md 4절 구조와 일치")


def check_config(rep: Report) -> None:
    path = ROOT / "configs" / "base.yaml"
    try:
        import yaml

        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        vp = cfg["render"]["viewport"]
        rep.add("OK", "configs/base.yaml", f"seed={cfg['seed']}, viewport={vp['width']}x{vp['height']}")
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", "configs/base.yaml", f"{type(e).__name__}: {e}")


def check_packages(rep: Report) -> None:
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            rep.add("OK", pip_name, metadata.version(pip_name))
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0] if str(e) else ""
            rep.add("FAIL", pip_name, f"import 실패 ({type(e).__name__}: {msg}) -> pip install -r requirements.txt")


def check_torch(rep: Report) -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", "torch", f"import 실패 ({type(e).__name__}) -> requirements-torch.txt 참고")
        return

    rep.add("OK", "torch", torch.__version__)
    try:
        import torchvision

        rep.add("OK", "torchvision", torchvision.__version__)
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", "torchvision", f"import 실패 ({type(e).__name__})")

    device = "cpu"
    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        device = "cuda"
        level = "OK" if vram >= MIN_VRAM_GB else "WARN"
        note = "" if vram >= MIN_VRAM_GB else f" (fine-tuning 권장 VRAM {MIN_VRAM_GB}GB 미만)"
        rep.add(level, "GPU", f"{backend}, {props.name}, VRAM {vram:.1f}GB{note}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        rep.add("OK", "GPU", "Apple MPS")
    else:
        try:
            import torch_directml  # type: ignore

            rep.add("WARN", "GPU", f"DirectML 사용 가능 ({torch_directml.device_name(0)}); 학습은 클라우드 GPU 권장")
        except Exception:  # noqa: BLE001
            rep.add("WARN", "GPU", "가속 장치 없음(CPU만 사용). 학습은 클라우드 GPU 사용 전제 (CLAUDE.md Phase 0)")

    try:
        x = torch.randn(64, 64, device=device)
        y = (x @ x).sum().item()
        rep.add("OK", f"torch 연산({device})", f"행렬곱 테스트 통과 (결과 finite={y == y})")
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", f"torch 연산({device})", f"{type(e).__name__}: {e}")


def check_browser(rep: Report) -> None:
    try:
        from PIL import Image
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        rep.add("FAIL", "Playwright 렌더링", f"import 실패 ({type(e).__name__})")
        return

    html = (
        "<html><body style='margin:0;background:#fff'>"
        "<div style='width:200px;height:100px;background:#f00'></div></body></html>"
    )
    shots: list[bytes] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            version = browser.version
            for _ in range(2):
                page = browser.new_page(viewport={"width": 512, "height": 512}, device_scale_factor=1)
                page.set_content(html)
                shots.append(page.screenshot())
                page.close()
            browser.close()
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else ""
        rep.add("FAIL", "Playwright 렌더링", f"{type(e).__name__}: {msg} -> `playwright install chromium` 실행 필요할 수 있음")
        return

    rep.add("OK", "Chromium", version)
    img = Image.open(io.BytesIO(shots[0])).convert("RGB")
    if img.size != (512, 512):
        rep.add("FAIL", "스크린샷 크기", f"{img.size} (기대: 512x512)")
    elif img.getpixel((10, 10)) != (255, 0, 0) or img.getpixel((400, 400)) != (255, 255, 255):
        rep.add("FAIL", "스크린샷 내용", "기대한 색상과 다름")
    else:
        rep.add("OK", "스크린샷", "512x512, 색상 검증 통과")

    if shots[0] == shots[1]:
        rep.add("OK", "렌더 재현성", "같은 HTML을 두 번 렌더링한 결과가 바이트 단위로 동일")
    else:
        rep.add("WARN", "렌더 재현성", "같은 HTML의 두 렌더링 결과가 다름 (Phase 1 시드 재현성에 영향)")


def check_disk(rep: Report) -> None:
    free = shutil.disk_usage(ROOT).free / 1024**3
    level = "OK" if free >= MIN_FREE_DISK_GB else "WARN"
    note = "" if free >= MIN_FREE_DISK_GB else f" (데이터/체크포인트용으로 {MIN_FREE_DISK_GB}GB 이상 권장)"
    rep.add(level, "디스크 여유", f"{free:.1f}GB{note}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phase 0 환경 점검")
    parser.add_argument("--skip-browser", action="store_true", help="Playwright 렌더링 검사를 건너뛴다")
    args = parser.parse_args()

    rep = Report()
    check_python(rep)
    check_dirs(rep)
    check_config(rep)
    check_packages(rep)
    check_torch(rep)
    if not args.skip_browser:
        check_browser(rep)
    check_disk(rep)

    c = rep.counts
    print(f"\n결과: OK {c['OK']} / WARN {c['WARN']} / FAIL {c['FAIL']}")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
