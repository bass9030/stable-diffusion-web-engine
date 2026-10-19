# stable-diffusion-engine

HTML 소스를 입력받아 렌더링 결과 이미지를 생성하는 Stable Diffusion 계열 모델을 만드는 취미 프로젝트.
계획과 작업 가이드라인은 `CLAUDE.md` 참고.

## 환경 구성 (Windows, PowerShell 기준)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1

# 1) torch 설치: requirements-torch.txt 상단 주석 참고 (GPU 종류에 따라 명령이 다름)
#    NVIDIA GPU(드라이버가 CUDA 13 지원) 예시. 개발 PC(GTX 1660 SUPER)에서 확인한 명령:
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu130

# 2) 나머지 패키지
pip install -r requirements.txt

# 3) 렌더링용 브라우저
playwright install chromium

# 4) 환경 점검
python scripts/check_env.py
```

## Phase 1: 합성 데이터셋 생성

```powershell
# configs/base.yaml + configs/datagen.yaml 설정으로 data/ 에 생성 (기본 1만 쌍, seed 42)
python -m src.datagen.build_dataset
# 생성된 샘플 일부를 새 브라우저로 다시 만들어 HTML/PNG 해시가 같은지 확인
python scripts/check_datagen_repro.py --num 100
```

출력: `data/{id}.html`, `data/{id}.png`, `data/metadata.jsonl`(시드, train/val 분할, 해시),
`data/render_env.json`(브라우저 버전, viewport, 폰트, 생성 설정, git commit), 실패 시 `data/failures.jsonl`.
폰트는 Arial 고정이며 렌더링 때만 `render.base_css`로 주입한다 (저장되는 HTML에는 없음).

## Phase 2: SD 1.5 학습

```powershell
# 짧은 HTML 데이터셋(CLIP 77토큰 이내) 생성 → data/short
python -m src.datagen.build_dataset --config configs/base.yaml configs/datagen.yaml configs/datagen_short.yaml
# VAE latent / 텍스트 임베딩 캐시 → outputs/cache/short
python -m src.train.cache_features --config configs/base.yaml configs/train_lora.yaml
# LoRA 본 학습 (로컬 GPU, v8 설정)
python -m src.train.train_lora --config configs/base.yaml configs/train_lora.yaml configs/train_lora_v8.yaml
# 추론
python -m src.infer.generate --lora outputs/lora_short_v8/checkpoints/step_003000 --html data/short/000011.html --out outputs/infer/000011.png
```

### Plan B: 클라우드 GPU에서 UNet full fine-tuning (Colab)
로컬 6GB GPU로는 full fine-tuning이 불가능해 Colab용 노트북을 둔다.
1. 변경 사항을 커밋하고 `python scripts/pack_for_colab.py` 로 `outputs/colab_bundle.zip`(코드 + 캐시)을 만든다.
2. Google Drive `MyDrive/sd-engine/` 에 올리고 `notebooks/colab_full_finetune.ipynb` 를 Colab에서 연다.
3. GPU 런타임에서 셀을 차례로 실행한다 (GPU에 따라 `configs/train_full_{24gb,t4}.yaml` 자동 선택, 출력은 Drive에 저장, 끊기면 이어서 학습).

추론: `python -m src.infer.generate --full <checkpoints/step_XXXXXX> --html ... --out ...`

### 대여 GPU (Vast.ai / RunPod 등 일반 Jupyter)
같은 번들을 인스턴스의 `/workspace/`에 올리고 `notebooks/rental_train.ipynb`를 실행한다.
`MODE = "lora"`(추천 GPU: RTX 4090)는 v8 LoRA 본 학습, `MODE = "full"`(추천 GPU: A100 80GB)은 UNet full fine-tuning.
결과는 `/workspace/results_<run>.zip`으로 묶이며, 인스턴스를 삭제하기 전에 내려받는다.

## Phase 3: 긴 HTML 조건 입력 (CodeT5+ 인코더 + projection)

CLIP(77토큰) 대신 CodeT5+ 220M 인코더(고정, 최대 512토큰)의 출력을 projection(LayerNorm + Linear)으로 UNet cross-attention에 넣는다.
UNet은 Phase 2 full FT 체크포인트에서 시작해 projection과 함께 학습한다. 학습은 대여 GPU를 전제로 한다.

```powershell
# 긴 HTML 데이터셋 3만 쌍 (CodeT5+ 512토큰 초과는 렌더링 전 제외) → data/long
python -m src.datagen.build_dataset --config configs/base.yaml configs/datagen.yaml configs/datagen_long.yaml
# 대여 GPU용 번들: 원본 데이터 + 시작점 UNet (캐시는 대여 GPU에서 생성)
python scripts/pack_for_colab.py --no-cache --data-dir data/long --extra outputs/full_short_rental/checkpoints/step_006000
```

대여 인스턴스에서 `notebooks/rental_train.ipynb`를 `MODE = "html"`로 실행한다 (캐시 생성 → overfit → 본 학습).
추론은 Phase 2와 같이 `python -m src.infer.generate --full <checkpoints/step_XXXXXX> --html ... --out ...`
(체크포인트의 config가 codet5p면 CodeT5+ 인코더와 projection을 자동으로 쓴다).

## Phase 4: 평가

고정 검증셋(짧은/긴 HTML val에서 시드로 50개씩, 학습에 쓰지 않음)으로 모델 버전을 나란히 비교한다.
지표: 픽셀 MSE, SSIM, LPIPS, CLIP 이미지 유사도(생성-정답), OCR 텍스트 일치율(정답에서 읽힌 글자 기준).

```powershell
# 모델별 생성 이미지 (이미 있으면 건너뜀) → outputs/eval/{model}/{set}/{id}.png
python -m src.eval.generate_eval --config configs/base.yaml configs/eval.yaml
# 지표 표(report.md, metrics.csv, summary.json)와 비교 그리드(grid_{set}.png)
python -m src.eval.report --config configs/base.yaml configs/eval.yaml
```

비교할 모델은 `configs/eval.yaml`의 `eval.models`에 추가한다.

`check_env.py`는 패키지 import, torch 장치(CUDA/ROCm/MPS/DirectML/CPU), Playwright 렌더링과 렌더 결과 재현성, 디스크 여유 공간을 검사한다.
`[FAIL]`이 하나라도 있으면 종료 코드 1을 반환한다. GPU 가속이 없는 것은 `[WARN]`이며, 학습은 클라우드 GPU를 쓰는 것을 전제로 한다.
