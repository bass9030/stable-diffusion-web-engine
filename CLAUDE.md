# CLAUDE.md — stable-diffusion-engine

HTML 소스를 입력받아 렌더링 결과 이미지를 생성하는 Stable Diffusion 계열 모델을 만드는 취미 프로젝트.

프로젝트 루트: `C:\Users\bass9030\codes\stable-diffusion-engine` (프로젝트 관련 파일은 모두 이 폴더에 작성한다.)

## 1. 프로젝트 목적과 범위

- 목적: 재미/학습. 정확한 렌더링이 목적이 아님 (정확한 렌더링은 브라우저가 수행).
- 성공 기준: 단순한 HTML을 넣었을 때 레이아웃(블록 위치, 배경색, 대략적인 요소 배치)이 원본 렌더링과 눈에 띄게 닮은 이미지를 생성.
- 비목표: 작은 글자의 정확한 복원, 실제 웹 페이지 재현, 반응형/JS 동작.

## 2. 기술적 제약 (설계 전제)

- SD 1.x/2.x/XL의 CLIP 텍스트 인코더는 입력이 77토큰으로 제한됨. HTML을 그대로 넣을 수 없음.
- VAE가 이미지를 8배 다운샘플한 latent에서 동작하므로 작은 글자/얇은 선 복원이 어려움.
- 확산 모델은 확률적 근사이므로 렌더러와 픽셀 단위로 일치하지 않음.
- 처음부터 학습하지 않고 기존 가중치를 fine-tuning한다.

## 3. 단계별 계획

### Phase 0 — 환경 구성
- Python 가상환경, PyTorch, `diffusers`, `transformers`, `accelerate`, `peft`, `playwright`.
- GPU 확인. 학습용 GPU가 없으면 Colab 등 클라우드 GPU 사용을 전제로 스크립트를 작성한다 (로컬은 데이터 생성/평가용).
- 산출물: `requirements.txt`, `requirements-torch.txt`(GPU 환경별로 설치 명령이 달라 분리), `scripts/check_env.py`(환경 점검), `README.md`(설치 순서), `configs/base.yaml`, `.gitignore`.

### Phase 1 — 합성 데이터셋 생성
- 템플릿 기반 HTML 생성기 작성: 사각형 div, 배경색, 제목/문단, 버튼, 2~3칼럼 레이아웃 등 소수의 컴포넌트를 무작위 조합.
- Playwright(headless Chromium)로 고정 viewport(512×512)에서 스크린샷 캡처.
- 외부 리소스(웹폰트, 외부 이미지, JS) 금지. 폰트는 시스템 폰트 1종으로 고정.
- 저장 형식: `data/{id}.html`, `data/{id}.png`, `metadata.jsonl`.
- 초기 규모: 5천~1만 쌍으로 시작, 파이프라인 검증 후 확장.

### Phase 2 — 베이스라인 (짧은 HTML 입력)
- 압축 HTML(인라인 CSS, 불필요한 속성 제거)이 77토큰 이내가 되도록 생성기 복잡도를 제한.
- SD 1.5 UNet을 LoRA 또는 full fine-tuning으로 학습 (`diffusers` text-to-image 학습 스크립트 기반).
- 이 단계는 "파이프라인이 끝까지 도는지" 확인하는 용도.

### Phase 3 — 긴 HTML 조건 입력
- 방법 A: 텍스트 인코더를 T5 계열로 교체하고 projection layer를 거쳐 UNet cross-attention에 연결, 해당 부분과 UNet을 학습.
- 방법 B: DOM에서 요소별 bounding box/색/텍스트를 추출해 레이아웃 조건으로 입력 (ControlNet/GLIGEN 방식). 렌더링 일부를 직접 계산하게 되므로 취지와 trade-off가 있음.
- A를 우선 시도하고, 결과가 부족하면 B를 비교 실험한다.

### Phase 4 — 평가와 개선
- 지표: LPIPS, SSIM, CLIP score, OCR 기반 텍스트 일치율.
- 고정된 검증셋(학습에 사용하지 않음)에서 모델 버전별 결과를 나란히 비교.
- 개선 후보: 해상도 상향, 글자 영역 손실 가중치, 베이스 모델 변경(SD3/FLUX 등), VAE fine-tuning.

## 4. 디렉터리 구조

```
stable-diffusion-engine/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── requirements-torch.txt
├── configs/            # 학습/생성 설정 (yaml)
├── scripts/            # 환경 점검 등 보조 스크립트
├── src/
│   ├── datagen/        # HTML 생성기, 렌더러(Playwright)
│   ├── train/          # 학습 스크립트
│   ├── models/         # 텍스트 인코더/projection 등 커스텀 모듈
│   ├── eval/           # 지표 계산, 비교 시각화
│   └── infer/          # 추론 스크립트
├── data/               # 생성 데이터 (git 제외)
├── outputs/            # 체크포인트, 샘플 이미지 (git 제외)
└── notebooks/          # 실험용
```

## 5. 작업 가이드라인 (Claude용)

### 일반
- 한국어로 답변하고, 코드 주석/문서는 한국어, 식별자는 영어를 사용한다.
- 질문에 대한 답변은 검증 가능한 사실에 기반한다. 확실하지 않은 내용(라이브러리 API, 하이퍼파라미터 효과 등)은 추측임을 명시하고 공식 문서/소스 확인을 제안한다.
- 변경 전에 해당 단계(Phase)의 목표와 산출물을 먼저 확인한다. 한 번에 한 Phase만 진행한다.
- 새 의존성을 추가하면 `requirements.txt`에 버전과 함께 반영한다.

### 재현성
- 모든 스크립트는 시드를 인자로 받고 기본값을 고정한다.
- 데이터 생성은 시드로 결정적이어야 한다. 같은 시드면 같은 HTML/스크린샷이 나와야 한다.
- 실험 설정은 `configs/*.yaml`에 두고 하드코딩하지 않는다.

### 데이터
- 학습/검증 분할은 HTML 생성 시드 기준으로 먼저 나누고, 검증셋은 학습 중 노출하지 않는다.
- 렌더링 환경(브라우저 버전, viewport, 폰트)을 `metadata.jsonl` 또는 별도 파일에 기록한다.
- 렌더링이 실패한 샘플은 조용히 버리지 말고 로그로 남긴다.

### 학습
- 먼저 소규모 샘플(수십 장)로 overfit 테스트를 해 파이프라인 정상 여부를 확인한 뒤 본 학습을 시작한다.
- mixed precision, gradient checkpointing, LoRA 등 메모리 절감 옵션을 사용 가능한 범위에서 기본 적용한다.
- 일정 step마다 고정 프롬프트 세트로 샘플 이미지를 저장해 학습 경과를 육안 확인할 수 있게 한다.
- 체크포인트에는 설정 파일과 git commit hash를 함께 기록한다.

### 코드 품질
- 함수는 단일 책임으로 작성하고 타입 힌트를 사용한다.
- 학습/평가 코드에서 장치(cuda/cpu) 의존 코드는 한 곳에서 처리한다.
- 대용량 파일(데이터, 체크포인트)은 git에 커밋하지 않는다 (`.gitignore` 등록).

## 6. 마일스톤 체크리스트

- [x] Phase 0: 환경 점검 스크립트(`python scripts/check_env.py`)를 사용자 PC에서 실행해 FAIL 없음 확인 (2026-10-18, OK 24 / WARN 1 / FAIL 0)
- [x] Phase 1: 1만 쌍 데이터셋 생성, 시드 재현성 확인 (2026-10-18, 실패 0, val 517, 재현성 100/100 일치)
- [x] Phase 2: 짧은 HTML 입력 모델이 학습·추론까지 동작 (2026-10-19, SD 1.5 UNet full fine-tuning, 검증 샘플 픽셀 MSE 0.0025)
- [x] Phase 3: 긴 HTML 조건 입력 구현 및 학습 (2026-10-19, 방법 A: CodeT5+ 인코더 + projection + UNet full FT, 긴 HTML 검증 MSE 0.083→0.041)
- [ ] Phase 4: 검증셋 지표 측정, 버전별 비교표 작성

## 7. 참고 자료

- Hugging Face `diffusers` 문서 및 text-to-image 학습 예제
- WebSight, Design2Code (HTML-스크린샷 쌍 데이터/벤치마크, 주로 스크린샷→HTML 방향)
- ControlNet, GLIGEN (레이아웃 조건 입력)
