# stable-diffusion-web-engine

![정답 렌더링(왼쪽)과 Phase 3 생성 결과(오른쪽)](https://github.com/bass9030/stable-diffusion-web-engine/blob/main/docs/images/sample_001877.png)

HTML 소스를 입력하면 그 HTML을 브라우저로 렌더링한 것처럼 보이는 이미지를 생성하는 Stable Diffusion 모델을 만드는 취미/학습 프로젝트입니다.
정확한 렌더링은 브라우저의 몫이고, 이 프로젝트의 목표는 **단순한 HTML을 넣었을 때 레이아웃(블록 위치, 배경색, 대략적인 요소 배치)이 원본 렌더링과 눈에 띄게 닮은 이미지**를 만드는 것입니다.

- 베이스 모델: Stable Diffusion 1.5 (처음부터 학습하지 않고 fine-tuning)
- 학습 데이터: 템플릿으로 무작위 생성한 HTML과 Playwright(headless Chromium)로 찍은 512×512 스크린샷 쌍
- 최종 모델: HTML을 CodeT5+ 인코더로 읽어 UNet에 조건으로 넣는 방식 (최대 512토큰, 약 1.5KB HTML)

계획과 작업 규칙은 [CLAUDE.md](CLAUDE.md), 실험 기록과 의사결정 과정은 [handoff.md](handoff.md)에 있습니다.

## 왜 이딴걸 만들었나요?

X(구 트위터)를 보는데 [재미있는 트윗](https://x.com/shiftpsh/status/2111436630871724539)을 발견했고 전 참을 수 없었습니다.

## 결과 요약

고정 검증셋(학습에 쓰지 않은 HTML, 세트마다 50개)에서 모델 버전을 비교한 결과입니다 (`outputs/eval/report.md`).

**긴 HTML** (제한 없는 생성기, CodeT5+ 기준 512토큰 이하)

| 모델                                             | MSE ↓      | SSIM ↑     | LPIPS ↓    | CLIP ↑     | OCR ↑      |
| ------------------------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| LoRA v8 (CLIP)                                   | 0.1266     | 0.4909     | 0.8153     | 0.5638     | 0.1628     |
| Phase 2: UNet full FT (CLIP)                     | 0.0318     | 0.8996     | 0.2384     | 0.7199     | 0.1891     |
| **Phase 3: CodeT5+ + projection + UNet full FT** | **0.0189** | **0.9096** | **0.1634** | **0.7309** | **0.1916** |

**짧은 HTML** (CLIP 77토큰 이하)

| 모델                                             | MSE ↓      | SSIM ↑     | LPIPS ↓    | CLIP ↑     | OCR ↑      |
| ------------------------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| LoRA v8 (CLIP)                                   | 0.0945     | 0.4658     | 0.8261     | 0.6308     | 0.1495     |
| Phase 2: UNet full FT (CLIP)                     | 0.0041     | 0.9681     | 0.0466     | **0.8528** | **0.2305** |
| **Phase 3: CodeT5+ + projection + UNet full FT** | **0.0030** | **0.9810** | **0.0285** | 0.8276     | 0.1466     |

- Phase 3 모델은 배경색, 박스와 칼럼의 위치, 버튼 같은 **레이아웃을 가장 잘 재현**합니다. 긴 HTML에서는 CLIP(77토큰)을 쓰는 Phase 2가 입력이 잘려 첫 요소 정도만 그리는 반면, Phase 3는 페이지 전체 구조를 그립니다.
- **글자는 알아보기 어렵습니다.** VAE가 8배로 줄인 latent에서 동작해 작은 글자를 복원하기 어렵고, 짧은 HTML의 글자 지표(OCR, CLIP)는 Phase 2가 약간 낫습니다.
- LoRA(rank 64, 어텐션 + conv/proj)로는 성공 기준에 미치지 못했고, UNet 전체를 학습하고 나서야 레이아웃이 맞기 시작했습니다.
- CLIP은 생성 이미지와 정답 이미지의 CLIP 임베딩 유사도이고, OCR은 정답 이미지에서 읽힌 글자 기준의 문자 단위 유사도입니다. OCR 결과는 대략적인 참고용입니다.

## 동작 방식

```
HTML → CodeT5+ 토크나이저(바이트 단위 BPE, 최대 512토큰)
     → CodeT5+ 220M 인코더 (고정)  → projection (LayerNorm + Linear, 학습)
     → UNet cross-attention (패딩은 attention mask로 가림)
노이즈 latent → UNet (SD 1.5, 전체 학습) → VAE 디코더 → 512×512 이미지
```

- 원래 T5/FLAN-T5 어휘에는 `<`가 없어서 HTML 태그가 깨집니다. CodeT5+는 HTML을 손실 없이 토큰화하고, 은닉 차원(768)이 SD 1.5 cross-attention과 같습니다.
- 학습 설정: v-prediction + zero terminal SNR(마지막 timestep에 원본 신호가 남지 않도록 노이즈 스케줄 재조정). 단색 배경이 많은 데이터에서 SD 1.5 기본 스케줄은 전역 색과 레이아웃을 잘 배우지 못했습니다.
- Phase 3의 UNet은 Phase 2 체크포인트(CLIP 조건으로 HTML 렌더링을 이미 배운 UNet)에서 시작하고, 처음 500 step은 UNet을 고정한 채 projection만 학습합니다.

## 요구 사항

- Python 3.11 이상 (개발 환경: Windows 11, Python 3.14.7)
- NVIDIA GPU
    - 로컬: 데이터 생성, 캐시, 추론, 평가는 6GB(GTX 1660 SUPER)에서 확인했습니다. 추론은 fp32 UNet 기준 약 5GB를 씁니다.
    - 학습: UNet full fine-tuning은 24GB 이상이 필요해 클라우드 GPU(Vast.ai A100 40GB에서 확인)를 사용합니다.
- 디스크: 데이터와 캐시, 체크포인트로 수십 GB

## 설치 (Windows, PowerShell)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1

# 1) torch: GPU/드라이버에 맞는 빌드를 설치한다 (requirements-torch.txt 주석 참고)
#    드라이버가 CUDA 13을 지원하는 NVIDIA GPU 예시
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu130

# 2) 나머지 패키지
pip install -r requirements.txt

# 3) 렌더링용 브라우저
playwright install chromium

# 4) 환경 점검: [FAIL]이 없어야 한다 (GPU가 없거나 VRAM이 작으면 [WARN])
python scripts/check_env.py
```

SD 1.5(`stable-diffusion-v1-5/stable-diffusion-v1-5`)와 CodeT5+(`Salesforce/codet5p-220m`) 가중치는 처음 실행할 때 Hugging Face에서 자동으로 내려받습니다.

## 추론

학습된 체크포인트는 저장소에 들어 있지 않습니다(`outputs/`는 git에서 제외). 아래 "전체 파이프라인"으로 직접 학습하거나 Release 탭에서 체크포인트 폴더를 받아 `outputs/` 아래에 두고 실행합니다.

```powershell
# Phase 3 모델 (체크포인트의 config.yaml을 보고 CodeT5+ 인코더와 projection을 자동으로 사용)
python -m src.infer.generate --full outputs/html_long_rental/checkpoints/step_010000 `
    --html-text '<body style="margin:0;padding:16px;background:#def"><h1>Hello world</h1><div style="width:50%;height:120px;background:#e33"></div></body>' `
    --out outputs/infer/hello.png

# HTML 파일 입력, 시드/스텝/guidance 지정
python -m src.infer.generate --full <체크포인트 폴더> --html page.html --out out.png --seed 42 --steps 25 --guidance 3.0

# Phase 2 (CLIP) full FT 체크포인트도 --full, LoRA 체크포인트는 --lora
python -m src.infer.generate --lora outputs/lora_short_v8/checkpoints/step_003000 --html page.html --out out.png
```

입력 HTML이 인코더 최대 길이(Phase 3: 512토큰, Phase 2: 77토큰)를 넘으면 경고를 남기고 뒷부분을 자릅니다.

## 전체 파이프라인

### 1. 합성 데이터셋 생성

```powershell
# 1만 쌍 (제한 없는 생성기) → data/
python -m src.datagen.build_dataset
# 짧은 HTML 5천 쌍 (CLIP 77토큰 이하만) → data/short   (Phase 2용)
python -m src.datagen.build_dataset --config configs/base.yaml configs/datagen.yaml configs/datagen_short.yaml
# 긴 HTML 3만 쌍 (CodeT5+ 512토큰 이하만) → data/long   (Phase 3용)
python -m src.datagen.build_dataset --config configs/base.yaml configs/datagen.yaml configs/datagen_long.yaml

# 재현성 확인: 일부 샘플을 새 브라우저로 다시 만들어 HTML/PNG 해시 비교
python scripts/check_datagen_repro.py --data-dir data/long --num 100
```

- 생성기는 사각형 div, 제목, 문단, 버튼, 2~3칼럼 레이아웃을 무작위로 조합하고, 인라인 CSS만 씁니다. 외부 리소스와 JS는 쓰지 않습니다.
- 폰트는 Arial로 고정하고, 렌더링할 때만 `render.base_css`로 주입합니다(저장되는 HTML에는 없음).
- 출력: `{id}.html`, `{id}.png`, `metadata.jsonl`(시드, train/val 분할, 토큰 수, 해시), `render_env.json`(브라우저 버전, viewport, 폰트, 생성 설정, git commit), `summary.json`. 렌더링에 실패하면 `failures.jsonl`에 남깁니다.
- train/val 분할은 샘플 시드로 정해지고, 검증셋은 학습에 쓰지 않습니다.

### 2. Phase 2: 짧은 HTML, CLIP 조건

```powershell
# VAE latent / CLIP 임베딩 캐시 → outputs/cache/short
python -m src.train.cache_features --config configs/base.yaml configs/train_lora.yaml

# (로컬 GPU) LoRA 학습. 이 프로젝트에서는 성공 기준에 미달
python -m src.train.train_lora --config configs/base.yaml configs/train_lora.yaml configs/train_lora_v8.yaml

# (클라우드 GPU) UNet full fine-tuning. 24GB GPU는 뒤에 configs/train_full_24gb.yaml을 붙인다
python -m src.train.train_full --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml
```

### 3. Phase 3: 긴 HTML, CodeT5+ 조건

```powershell
# latent / CodeT5+ 인코더 출력(가변 길이) 캐시 → outputs/cache/long
python -m src.train.cache_html --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml configs/train_html.yaml

# 학습 (model.unet_init의 Phase 2 체크포인트에서 시작). 24GB GPU는 뒤에 configs/train_full_24gb.yaml
python -m src.train.train_html --config configs/base.yaml configs/train_lora.yaml configs/train_full.yaml configs/train_html.yaml
```

모든 학습 스크립트에 공통으로 적용되는 동작:

- 먼저 32장으로 overfit 테스트를 해서 파이프라인을 확인합니다(`configs/*_overfit.yaml`을 뒤에 붙임).
- 일정 step마다 고정 샘플 세트로 `samples/step_XXXXXX.png`(행마다 정답 | 생성)와 `samples/metrics.jsonl`(픽셀 MSE)을 저장합니다.
- 체크포인트에는 병합된 `config.yaml`과 `run_info.json`(git commit, dirty 여부)을 함께 저장합니다.
- `train_full` / `train_html`은 `--resume <checkpoints/step_XXXXXX>`로 이어서 학습할 수 있습니다(옵티마이저 상태는 새로 시작).

### 4. 클라우드 GPU에서 학습하기 (Vast.ai / RunPod / Colab)

로컬에서 번들(zip)을 만들어 올리고, 노트북으로 설치부터 학습, 결과 묶기까지 실행합니다.

```powershell
# Phase 2: 코드 + outputs/cache/short
python scripts/pack_for_colab.py
# Phase 3: 코드 + 원본 데이터 + 시작점 UNet (캐시는 클라우드에서 생성)
python scripts/pack_for_colab.py --no-cache --data-dir data/long --extra outputs/full_short_rental/checkpoints/step_006000
```

- **Vast.ai / RunPod 등 일반 Jupyter**: `outputs/colab_bundle.zip`과 [notebooks/rental_train.ipynb](notebooks/rental_train.ipynb)를 인스턴스의 `/workspace/`에 올리고, `MODE`를 `"lora"`, `"full"`, `"html"`(Phase 3) 중 하나로 정해 차례로 실행합니다. 결과는 `/workspace/results_<run>.zip`으로 묶입니다. 인스턴스를 삭제하면 디스크도 지워지니, 그 전에 내려받으세요.
- **Colab**: [notebooks/colab_full_finetune.ipynb](notebooks/colab_full_finetune.ipynb) (Phase 2 full FT, Google Drive에 번들을 올리고 출력도 Drive에 저장).
- GPU 프리셋은 노트북이 자동으로 고릅니다. 80GB는 기본 설정(배치 8, bf16), 24~40GB는 `train_full_24gb.yaml`(배치 4×2, 8-bit Adam), bf16을 지원하지 않는 T4는 `train_full_t4.yaml`입니다.
- 참고: A100 40GB에서 Phase 2(6000 step)는 약 1시간, Phase 3(10000 step)는 약 1시간 40분 걸렸습니다.

### 5. 평가

```powershell
# 모델별 평가 이미지 생성 (이미 있으면 건너뜀) → outputs/eval/{model}/{set}/{id}.png
python -m src.eval.generate_eval --config configs/base.yaml configs/eval.yaml
# 지표 계산과 비교표/그리드 → outputs/eval/report.md, metrics.csv, summary.json, grid_{set}.png
python -m src.eval.report --config configs/base.yaml configs/eval.yaml
```

비교할 모델과 검증셋 크기는 [configs/eval.yaml](configs/eval.yaml)에서 정합니다. 지표는 픽셀 MSE, SSIM, LPIPS, CLIP 이미지 유사도, OCR 텍스트 일치율(EasyOCR)입니다.

## 설정 파일

설정은 `--config`에 여러 yaml을 순서대로 넘기면 깊은 병합되고, 뒤 파일이 앞 파일을 덮어씁니다. 하이퍼파라미터는 코드에 하드코딩하지 않습니다.

| 파일                                                                                       | 내용                                              |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| `base.yaml`                                                                                | 공통: seed, 경로, 렌더링(512×512, Arial)          |
| `datagen.yaml`, `datagen_short.yaml`, `datagen_long.yaml`                                  | 데이터 생성기와 토큰 수 필터                      |
| `train_lora.yaml`                                                                          | 모델, 데이터, 캐시, LoRA, 학습, 샘플링의 기본값   |
| `train_lora_v8.yaml`, `train_lora_rental.yaml`, `train_lora_overfit*.yaml`                 | LoRA 실험 설정                                    |
| `train_full.yaml`, `train_full_24gb.yaml`, `train_full_t4.yaml`, `train_full_overfit.yaml` | UNet full fine-tuning과 GPU 프리셋                |
| `train_html.yaml`, `train_html_overfit.yaml`                                               | Phase 3 (CodeT5+ 인코더, projection, 시작점 UNet) |
| `eval.yaml`                                                                                | Phase 4 평가                                      |

## 디렉터리 구조

```
stable-diffusion-engine/
├── configs/          # 설정 (yaml)
├── scripts/          # check_env.py(환경 점검), check_datagen_repro.py(재현성), pack_for_colab.py(클라우드 번들)
├── src/
│   ├── datagen/      # HTML 생성기, Playwright 렌더러, 데이터셋 빌드
│   ├── models/       # CodeT5+ 토크나이저/인코더, projection
│   ├── train/        # 캐시 생성, LoRA / full FT / Phase 3 학습
│   ├── infer/        # 샘플링 루프, 추론 스크립트
│   └── eval/         # 평가 이미지 생성, 지표, 리포트
├── notebooks/        # 클라우드 학습 노트북 (Colab, 대여 GPU)
├── data/             # 생성 데이터 (git 제외)
└── outputs/          # 캐시, 체크포인트, 샘플, 평가 결과 (git 제외)
```

## 재현성

- 모든 스크립트는 시드를 인자로 받고, 기본값은 설정 파일에 고정돼 있습니다.
- 데이터 생성은 시드로 결정됩니다. 같은 시드와 같은 렌더링 환경이면 HTML과 PNG가 바이트 단위로 같습니다(`check_datagen_repro.py`로 확인). 렌더링 환경은 `render_env.json`에 기록됩니다.
- 학습 출력과 체크포인트에는 설정과 git commit이 기록됩니다. 클라우드 번들은 `.git`이 없어 `run_info.json`의 commit이 비어 있고, 대신 번들의 `BUNDLE_INFO.json`에 기록됩니다.

## 한계와 알려진 이슈

- **글자**: 대략적인 모양만 나오고 읽을 수 있는 수준은 아닙니다.
- **입력 범위**: 생성기가 만드는 요소(div, 제목, 문단, 버튼, 칼럼, 인라인 스타일)만 학습했습니다. `<table>`, `<img>`, `<style>` 블록, 외부 CSS 등은 제대로 그리지 못할 가능성이 높습니다.
- **입력 길이**: Phase 3는 최대 512토큰(이 프로젝트 스타일 HTML 기준 약 1.4~1.7KB)입니다. 출력은 512×512 고정이라, 화면 밖으로 넘치는 내용은 그리지 않습니다.
- **GTX 16xx 계열**: fp16 UNet 학습에서 loss가 NaN이 되어 fp32를 씁니다(`model.unet_dtype: float32`).
- **6GB GPU에서 학습할 때**: 브라우저 등 다른 앱과 VRAM을 나눠 쓰다 공유 메모리로 넘치면 학습이 몇 배 느려지고, 앱을 닫아도 회복되지 않았습니다. 이때는 학습을 다시 시작해야 합니다.
- **CodeT5+ 토크나이저**: transformers 5.20에서 저장소 설정을 불러오지 못해, `tokenizers`로 vocab/merges를 직접 읽습니다(`src/models/html_encoder.py`).
- **클라우드 노트북**: 셸에 `python` 명령이 없는 템플릿이 있어 노트북은 `python3`로 실행합니다.

## Special Thanks

- X(구 트위터) 시프트([@shiftpsh](https://x.com/shiftpsh))님: [아이디어 제공](https://x.com/shiftpsh/status/2111436630871724539)

## 참고

- Hugging Face `diffusers` 문서와 text-to-image 학습 예제
- Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed" (2023): zero terminal SNR, v-prediction
- CodeT5+ (`Salesforce/codet5p-220m`)
- WebSight, Design2Code (HTML-스크린샷 쌍 데이터, 주로 스크린샷→HTML 방향)
- ControlNet, GLIGEN (레이아웃 조건 입력, 이 프로젝트에서는 미사용)
