# handoff.md — stable-diffusion-engine

작성 시점: 2026-10-18 (Phase 0 완료: 사용자 PC에서 check_env.py FAIL 0 확인)
프로젝트 루트: `C:\Users\bass9030\codes\stable-diffusion-engine`

## 1. 프로젝트 개요

- 목표: HTML을 입력하면 HTML 렌더 결과 이미지를 생성하는 Stable Diffusion 계열 모델 제작.
- 성격: 재미/학습 목적의 취미 프로젝트. 정확한 렌더링은 목적이 아님.
- 성공 기준: 단순한 HTML에서 레이아웃(블록 위치, 배경색, 대략적 배치)이 원본 렌더링과 눈에 띄게 닮은 이미지 생성.
- 비목표: 작은 글자의 정확한 복원, 실제 웹 페이지 재현, 반응형/JS 동작.
- 상세 계획과 작업 가이드라인은 `CLAUDE.md`에 있음 (Phase 0~4, 디렉터리 구조, 재현성/데이터/학습/코드 품질 규칙, 마일스톤 체크리스트).

## 2. 진행 이력

1. 사전 조사 답변: 기본 SD의 CLIP 텍스트 인코더는 77토큰 제한, VAE 8배 다운샘플로 작은 글자 복원이 어렵다는 제약, 필요한 구성요소(데이터셋, 조건 입력 설계, 베이스 모델, 해상도 처리, 학습 환경, 평가 지표)를 정리함.
2. `CLAUDE.md` 작성: Phase 0~4 계획과 작업 가이드라인.
3. 작업 위치를 로컬 폴더 `stable-diffusion-engine`으로 지정받아, 이후 프로젝트 데이터는 모두 이 폴더에 작성하기로 함.
4. Phase 0 수행: 디렉터리 구조, 의존성 파일, 설정 파일, 환경 점검 스크립트, README, .gitignore 작성. `CLAUDE.md`를 새 이름/경로에 맞게 수정.

## 3. 결정된 사항

- 기존 가중치를 fine-tuning한다 (처음부터 학습하지 않음). Phase 2 베이스는 SD 1.5 (LoRA 또는 full fine-tuning).
- Phase 2는 압축 HTML이 77토큰 이내가 되도록 생성기 복잡도를 제한한 뒤 파이프라인 end-to-end 동작 확인용으로 사용.
- Phase 3은 방법 A(T5 계열 인코더 + projection layer를 cross-attention에 연결)를 우선 시도하고, 부족하면 방법 B(DOM 기반 레이아웃 조건, ControlNet/GLIGEN 방식)와 비교.
- 로컬에 학습용 GPU가 없을 수 있다는 전제: 로컬은 데이터 생성/평가용, 학습은 클라우드 GPU(예: Colab). 단, 사용자 PC의 GPU는 아직 확인하지 않음.
- 렌더링: Playwright(headless Chromium), 고정 viewport 512x512, device_scale_factor 1. 외부 리소스(웹폰트, 외부 이미지, JS) 금지, 폰트는 시스템 폰트 1종으로 고정 (폰트 종류는 Phase 1에서 결정).
- 재현성 규칙: 모든 스크립트가 시드를 인자로 받고, 데이터 생성은 시드로 결정적이어야 함. 설정은 `configs/*.yaml`에 두고 하드코딩하지 않음.
- 언어 규칙: 답변·주석·문서는 한국어, 식별자는 영어.
- torch/torchvision은 GPU 환경별로 설치 명령이 달라 `requirements-torch.txt`로 분리.
- 대용량 파일(`data/`, `outputs/`, 모델 가중치)은 git에서 제외.

## 4. 구현된 내용 (Phase 0 산출물)

```
stable-diffusion-engine/
├── CLAUDE.md                 # 계획 및 가이드라인
├── README.md                 # 환경 구성 순서 (Windows PowerShell 기준)
├── handoff.md                # 이 문서
├── requirements.txt          # torch 제외 패키지 (버전 고정)
├── requirements-torch.txt    # torch / torchvision
├── .gitignore
├── configs/base.yaml         # seed: 42, paths(data/outputs), render.viewport 512x512, device_scale_factor 1
├── scripts/check_env.py      # 환경 점검 스크립트
├── src/{datagen,train,models,eval,infer}/   # 비어 있음 (.gitkeep)
├── data/ outputs/ notebooks/                # 비어 있음 (.gitkeep)
```

### 의존성 (requirements.txt)
diffusers 0.41.0, transformers 5.20.1, accelerate 1.15.0, peft 0.21.2, safetensors 0.8.0, sentencepiece 0.2.2, datasets 5.1.0, playwright 1.64.0, pillow 13.0.0, numpy 2.5.4, pyyaml 6.0.3, tqdm 4.70.1, lpips 0.1.4, scikit-image 0.26.0.
requirements-torch.txt: torch 2.14.1, torchvision 0.29.1.
- 버전은 `pip index versions`로 조회한 최신 버전이며, 작업 환경(Python 3.13, Linux)에서 `pip install --dry-run`으로 의존성 해석이 충돌 없이 되는 것까지만 확인함. 실제 설치와 사용자 PC(Windows, Python 버전 미확인)에서의 동작은 확인되지 않음.

### 환경 점검 스크립트 로직 (`python scripts/check_env.py [--skip-browser]`)
각 항목을 `[OK]`/`[WARN]`/`[FAIL]`로 출력하고, FAIL이 하나라도 있으면 종료 코드 1을 반환.
1. Python 버전/플랫폼 보고 (판정 없음).
2. 디렉터리 구조가 CLAUDE.md 4절과 일치하는지 (없으면 WARN).
3. `configs/base.yaml` 파싱 및 seed/viewport 출력 (실패 시 FAIL).
4. 필수 패키지 14종을 실제로 import하고 버전 출력 (실패 시 FAIL).
5. torch: 버전, torchvision, 장치 감지 순서 CUDA/ROCm → MPS → DirectML → CPU. VRAM 8GB 미만이면 WARN, 가속 장치가 없으면 WARN(클라우드 GPU 전제). 감지된 장치에서 64x64 행렬곱 테스트.
6. Playwright: Chromium 실행, 512x512 스크린샷, 색상 픽셀 검증(빨간 박스 (10,10), 흰 배경 (400,400)), 같은 HTML을 두 번 렌더링해 바이트 동일 여부(렌더 재현성, 다르면 WARN). `--skip-browser`로 생략 가능.
7. 디스크 여유 공간 (20GB 미만이면 WARN).
- stdout을 UTF-8로 재설정해 Windows 콘솔에서도 한국어 출력되도록 처리.

## 5. 검증 상태

- `check_env.py` 문법 검사 통과.
- 작업 환경(Linux, Python 3.10, 패키지 미설치)에서 `--skip-browser`로 실행: 설치된 항목은 OK, 미설치 패키지는 FAIL로 판정되고 종료 코드 1 반환 → 스크립트 자체의 분기 동작은 확인됨.
- 미확인: Playwright 렌더링 검사 경로, torch 장치 검사 경로(두 경로 모두 실제 실행되지 않음), 사용자 PC에서의 전체 실행 결과.

### 사용자 PC 실행 결과 (2026-10-18, Phase 0 완료)
- 환경: Windows 11, Python 3.14.7 (`py -3.14`로 `.venv` 생성; 3.10도 설치돼 있으나 numpy 2.5.x 미지원으로 사용 안 함).
- torch: `--index-url https://download.pytorch.org/whl/cu130`로 2.14.1+cu130 설치 (README에 반영).
- `python scripts/check_env.py` 전체(브라우저 포함) 실행: OK 24 / WARN 1 / FAIL 0, 종료 코드 0.
  - WARN: GPU NVIDIA GeForce GTX 1660 SUPER, VRAM 6.0GB (기준 8GB 미만).
  - Chromium 156.0.8078.4, 스크린샷 색상 검증 통과, 렌더 재현성(바이트 동일) 통과. 디스크 여유 114GB.

## 6. 미해결 사항 / 다음 단계

1. (완료) 사용자 PC에서 Phase 0 마무리.
2. 학습 장소 결정 미정: 로컬 GPU는 VRAM 6GB(GTX 1660 SUPER). SD 1.5 LoRA는 메모리 절감 옵션을 쓰면 가능할 수도 있으나 미검증(추측). GTX 16xx 계열은 fp16에서 NaN/검은 이미지 문제가 보고된 바 있어 확인 필요(추측, 공식 확인 안 됨). Phase 2 overfit 테스트 때 로컬에서 시도해보고 안 되면 클라우드 사용.
3. Phase 1 시작: 템플릿 기반 HTML 생성기(div, 배경색, 제목/문단, 버튼, 2~3칼럼 등을 무작위 조합), Playwright 렌더러, 저장 형식 `data/{id}.html`, `data/{id}.png`, `metadata.jsonl`, 초기 5천~1만 쌍, 시드 재현성 확인. 폰트 1종 고정 필요.
4. 알려진 잔여물: `scripts/__pycache__/`가 남아 있음 (삭제 권한이 없어 지우지 못함, .gitignore에 포함).
5. git 저장소 초기화 완료 (브랜치 `main`, Phase 0 산출물로 첫 커밋). 원격 저장소는 없음.
