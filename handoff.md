# handoff.md — stable-diffusion-engine

작성 시점: 2026-10-20 (Phase 4 완료: 고정 검증셋 지표·버전별 비교표)
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
5. Phase 0 마무리: 사용자 PC에서 설치 및 check_env.py FAIL 0 확인, `git init` 후 첫 커밋(a2788e9).
6. Phase 1 수행: HTML 생성기·렌더러·데이터셋 빌드·재현성 검사 스크립트 작성, 1만 쌍 생성. 커밋 8f7fefa.
7. Phase 2 진행 중: 짧은 데이터셋, 캐시, LoRA 학습/추론 코드 작성, overfit 테스트 v1~v8. 코드는 0e660c9로 커밋 (v8 설정 파일은 미커밋).

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

## 4-1. Phase 1 구현 (합성 데이터셋)

- `configs/base.yaml`: `render.font_family: Arial`, `render.base_css: "body{font-family:Arial}"` 추가.
- `configs/datagen.yaml`: 생성 개수 10000, val_ratio 0.05, 컴포넌트 가중치/개수 범위 등 생성기 파라미터.
- `src/datagen/config.py`: yaml 여러 개를 깊은 병합 (base → datagen).
- `src/datagen/html_gen.py`: `generate_html(seed, GenConfig)`. 컴포넌트 heading/paragraph/button/box/columns(2~3칼럼, flex). 3자리 hex 팔레트, 영문 단어 목록. `random.Random(seed)`만 사용해 결정적.
- `src/datagen/renderer.py`: `Renderer` (Playwright Chromium, JS 끔, 모든 네트워크 요청 abort, 페이지 1개 재사용). 렌더링 때 `<style>{base_css}</style>`를 HTML 앞에 붙임 → 저장되는 HTML(모델 입력)에는 폰트 지정이 없음.
- `src/datagen/build_dataset.py` (`python -m src.datagen.build_dataset`): 샘플 시드 = sha256("html:{base_seed}:{index}") 앞 8바이트, 분할 = 샘플 시드 해시로 결정(val_ratio). 출력 `data/{id}.html|png`, `metadata.jsonl`(id, index, seed, split, html_chars, 해시), `render_env.json`(브라우저/playwright 버전, render/generator 설정, platform, git commit, git_dirty), 실패 시 `failures.jsonl` + 종료 코드 1. 기존 metadata.jsonl이 있으면 `--overwrite` 필요.
- `scripts/check_datagen_repro.py`: render_env.json 기준으로 무작위 샘플을 새 브라우저에서 재생성해 HTML/PNG 해시·분할 비교.

### 생성 결과 (data/)
- 10000쌍, 렌더링 실패 0, val 517 / train 9483. Chromium 156.0.8078.4, 약 25장/초.
- 재현성: 무작위 100개 재생성 결과 HTML/PNG/분할 모두 일치. (40개 테스트셋에서도 일치)
- HTML 길이(40개 테스트 기준): 최소 120자, 중앙값 282자, 최대 1122자 → 다수가 CLIP 77토큰을 넘을 것으로 예상(토큰 수는 미측정).
- 주의: 이 데이터는 Phase 1 코드가 커밋되기 전에 생성되어 render_env.json의 git_commit이 a2788e9(Phase 0)로 기록됨. `git_dirty` 필드는 생성 이후 추가됨. 생성 로직은 이후 커밋과 동일(재생성 시 해시 일치 확인 가능).
- 알려진 한계: 글자 색을 배경과 무관하게 팔레트에서 골라 대비가 낮아 거의 안 보이는 글자가 일부 있음 (글자 복원은 비목표라 유지).

## 4-2. Phase 2 진행 상황

### 짧은 HTML 데이터셋
- 기존 1만 쌍의 CLIP 토큰 수(BOS/EOS 포함): 중앙값 128, p90 327, 77 이하 22.7%.
- `configs/datagen_short.yaml`(컴포넌트 1~3개, 문단 2~6단어, `wrap_html: false`) + build_dataset의 `dataset.max_clip_tokens: 77` 필터(렌더링 전 건너뜀, `summary.json`에 개수 기록).
- `data/short/`: 5000쌍(val 248), 토큰 초과로 건너뜀 5169개, 실패 0, 재현성 100/100 일치. 기존 data/ 재현성도 20/20 유지.

### GPU 점검 (GTX 1660 SUPER, 6GB)
- fp16 UNet 학습: loss NaN (실측). fp16이 fp32보다 느리기도 함(4.0s vs 1.1s/step). → `unet_dtype: float32`.
- fp32 + LoRA r8 + gradient checkpointing, 512px 배치 1: 최대 메모리 3.95GB, 약 1.1~1.2s/step.
- fp16 VAE/텍스트 인코더는 NaN 없음.

### 코드
- `configs/train_lora.yaml`, `configs/train_lora_overfit.yaml`(32장, run_name은 실험마다 v2/v3/v4로 변경).
- `src/train/cache_features.py`: VAE latent(분포 평균×scaling)와 CLIP 임베딩을 `outputs/cache/short/{train,val,empty}.pt`로 저장. 5000장에 약 1시간 45분 소요.
- `src/train/train_lora.py`: 캐시 기반 LoRA 학습 (fp32 UNet, offset noise, 높은 timestep 비중 옵션, caption dropout, 샘플 그리드 [정답|생성], 체크포인트에 config.yaml/run_info.json(git commit/dirty)).
- `src/infer/sampler.py`(DPMSolver + CFG 샘플링 루프, 학습/추론 공용), `src/infer/generate.py`(HTML → 이미지, LoRA 로드).
- `src/train/utils.py`: 장치/dtype/시드/git 정보 (장치 의존 코드는 여기서만).

### overfit 테스트 결과 (학습 32장, 샘플: val 2 + train 4)
| 버전 | 변경 | 결과 |
|---|---|---|
| v1 | lr 1e-4, 600 step | 회색 질감 화면. 정답과 무관 |
| v2 | + offset noise 0.1, 1000 step | 깔끔한 웹페이지 형태, 그러나 배경 항상 연회색, 입력별 차이 작음 |
| v3 | lr 5e-4, 1500 step | 웹 UI 요소 등장, 입력 단어 일부 반영. 배경 여전히 회색, 레이아웃이 입력과 무관 |
| v4 | + timestep 50%를 [700,1000)에서, trailing 샘플링 | 가장 나음. 배경색이 정답 쪽으로 일부 이동. 박스 위치/색은 아직 못 외움 |
| v5 | rank 64, 3000 step (v4 설정 유지) | step 1500까지 train MSE 0.077, 이후 발산(얼룩 텍스처, 0.247). 박스 없음 |
| v6 | + zero terminal SNR + v-prediction, lr 2e-4, offset noise 끔 (높은 t 편향 유지) | **전역 색/큰 블록 위치가 처음으로 정답을 따라감** (보라/초록 블록 위치 일치). 그러나 세부가 노이즈 질감. train MSE 0.052~0.095 |
| v7 | v6에서 timestep 균등 분포로 | 질감 노이즈는 줄었으나 전역 구조 일치가 v6보다 약해짐. train MSE 0.070~0.124 |
| v8 | v6 + LoRA 대상에 conv1/conv2(각 22), proj_in/proj_out(각 16) 추가. 학습 파라미터 47.3M, 4.33GB, 1.24s/step | 일부 학습 샘플(흰 배경+초록 박스)은 위치·크기·색까지 일치. 다른 샘플엔 자주 본 요소(초록 박스)가 섞여 나옴, 배경 얼룩 질감 남음. train MSE 0.064~0.184 (step 3000: 0.069) |
- 샘플 MSE: `samples/metrics.jsonl`에 정답(VAE 복원) 대비 픽셀 MSE 기록 (v5부터). 샘플 6개라 checkpoint 간 변동이 큼.
- v6 체크포인트로 샘플러 비교(DPMSolver 25, DDIM 25/50, Euler 25): 결과 거의 동일 → 노이즈 질감은 샘플러가 아니라 모델 문제.
- SD1.5 기본 스케줄의 마지막 timestep 신호 √ᾱ_T = 0.068 (실측), zero SNR 적용 시 0.
- 판단(추측): LoRA(어텐션만)로 eps→v 전환과 레이아웃 암기를 동시에 하기엔 용량/학습량이 부족할 수 있음.
- loss는 처음부터 ~0.005로 낮아 지표로 쓸모가 적음 (단색 이미지라 낮은 t의 노이즈 예측이 쉬움).
- 확인한 것: LoRA B 가중치가 모든 블록(self/cross attention)에서 학습됨 → gradient checkpointing 버그 아님. 샘플 간 CLIP 임베딩 평균 cos 0.61 → 입력 구별 정보는 있음.
- 가설(미검증): SD1.5 노이즈 스케줄의 종단 SNR이 0이 아니어서 전역 색/레이아웃 학습이 느림. 근본 대책은 zero terminal SNR + v-prediction (Lin et al. 2023), 다만 eps 모델을 LoRA만으로 v-pred로 바꾸는 것은 부담이 큼.

### 본 학습 결과 (`outputs/lora_short_v8`, 커밋 f7c9d4c, 3000 step, 약 4.2시간)
- 재시작 이력: VRAM 넘침(공유 메모리 사용)으로 두 번 중단 → `lora_short_v8_aborted_slow`(샘플링 메모리, 수정 623723b), `lora_short_v8_aborted_vram`/`_vram2`(Firefox 등 데스크톱 앱과 VRAM 경합, 한 번 공유 메모리로 넘친 프로세스는 앱을 닫아도 속도가 돌아오지 않음). 정상 속도 약 5초/step.
- 학습 중 샘플 MSE(val 4 / train 2, CFG 3.0 + rescale 0.7): step 0 val 0.188 → 1000: 0.113 → 2000: 0.098 → 3000: 0.102. step 2000 이후 정체.
- 처음 보는 HTML 추론(검증 4개 + 직접 만든 3개, guidance 1/3/5): 검증 MSE 약 0.10, guidance 차이 거의 없음. 밝기(밝은/어두운 배경)는 대체로 따라가고 흰 배경 위쪽 파란 박스는 위치가 대략 맞지만, 색상은 자주 틀리고(주황→초록, 빨강→회색/갈색) 전체에 얼룩 질감, 블록 경계 불분명. → 성공 기준 미달.
- 버그 수정(미커밋): 체크포인트 저장 시 `convert_state_dict_to_diffusers`가 어텐션 키만 옛 형식(lora.down/up)으로 바꿔 conv/proj 키(lora_A/B)와 섞였고, `generate.py`에서 불러오면 어텐션 LoRA 128개가 빠져 회색 화면만 나왔음. 저장은 PEFT 형식 그대로, 로드는 키를 PEFT 형식으로 통일(`load_lora_state_dict`)하도록 수정. 기존 v8 체크포인트 로드(204개 모듈) 및 저장→로드 왕복(차이 0) 확인. 학습 중 샘플/MSE는 메모리상 모델로 만들어 영향 없음. 어텐션만 학습한 v1~v7 체크포인트도 영향 없음.
- 판단: 같은 설정의 LoRA를 대여 GPU에서 다시 돌리면 속도만 빨라질 뿐 결과는 비슷할 것으로 예상. 다음은 full fine-tuning(대여 A100 80GB) 또는 LoRA 설정 변경(rank/데이터/step)이 후보.

### full fine-tuning 결과 (Vast.ai A100-SXM4-40GB, `outputs/full_short_rental`, 번들 커밋 766ff90)
- 설정: `train_full.yaml` + `train_full_24gb.yaml`(VRAM 39.5GB라 자동 선택: 배치 4×2, bf16, 8-bit Adam), v-pred + zero SNR, lr 1e-5, 6000 step. 약 59분, 최대 메모리 10.8GB. overfit 테스트 후 본 학습.
- run_info.json의 git_commit은 null (번들에 .git이 없음). 코드 기준은 번들의 BUNDLE_INFO.json(766ff90). 노트북 셸 명령은 인스턴스에서 python3로 고쳐 실행 (이후 5bd8a38로 반영).
- 학습 중 샘플 MSE(val 4 / train 2): step 0 val 0.188 → 1500: 0.016 → 3500: 0.0036 → 6000: 0.0034. 검증 샘플에서 배경색·버튼·제목 위치·박스까지 정답과 거의 같음. 글자는 대략적인 모양만.
- 처음 보는 HTML 추론(로컬 GTX 1660, fp32, LoRA v8과 같은 7개 입력, guidance 1/3/5): 검증 4개 MSE 약 0.0025(LoRA v8 0.10), 직접 만든 빨간 페이지 0.003, 흰 배경+파란 박스 0.016으로 위치·크기·색 일치. 어두운 배경 + 2칼럼(min-height 200px) 입력만 칼럼 대신 작은 노란 박스(0.088) → 짧은 데이터셋에서 칼럼 컴포넌트가 드물어서로 추정. guidance 영향 거의 없음.
- 결론: Phase 2 성공 기준 충족. LoRA(어텐션+conv/proj, rank 64)로는 부족했고 UNet 전체 학습이 필요했음.

## 4-3. Plan B: 클라우드 GPU 학습 (Colab / 대여 GPU, 클라우드 실행 전)

LoRA 본 학습(`outputs/lora_short_v8`, 커밋 623723b 기준, 로컬 백그라운드 실행 중) 결과가 부족할 때를 대비.
- `src/train/train_full.py`: train_lora.py의 캐시·손실·샘플링 함수를 재사용하고 UNet 전체를 학습. fp32 마스터 가중치 + autocast(bf16/fp16, fp16이면 GradScaler), 8-bit Adam 옵션(bitsandbytes), 체크포인트는 UNet만 fp16 safetensors로 저장(약 1.7GB), 최근 N개만 유지, `--resume`(가중치·step만 이어받음, 옵티마이저/warmup은 새로 시작).
- 설정: `configs/train_full.yaml`(A100 기준: 배치 8, bf16, lr 1e-5, warmup 500, 6000 step, v-pred + zero SNR, 높은 t 편향 0.5), `train_full_24gb.yaml`(24GB bf16 GPU: RTX 4090/3090/L4 등, 4×2, 8-bit Adam; 이전 이름 train_full_l4.yaml), `train_full_t4.yaml`(2×4, fp16, 8-bit Adam), `train_full_overfit.yaml`(32장, 1500 step, lr 2e-5).
- `scripts/pack_for_colab.py`: git 추적 파일 + 무시되지 않은 새 파일 + `outputs/cache/short` → `outputs/colab_bundle.zip`(약 720MB). Colab에서 다시 렌더링하면 폰트가 달라지므로 로컬 캐시를 가져감.
- `notebooks/colab_full_finetune.ipynb`: Drive 마운트 → 번들 풀기 → requirements 설치(Colab torch 사용) → check_env → GPU별 프리셋 자동 선택 → overfit → 본 학습(자동 재개) → 결과 보기 → 추론. 출력은 Drive `MyDrive/sd-engine/outputs/`.
- `src/infer/generate.py`: `--full <ckpt>` 옵션 추가.
- requirements.txt에 `bitsandbytes==0.50.2` 추가 (로컬 venv에도 설치, import 확인).
- 검증: 로컬 CPU에서 작은 무작위 UNet으로 바꿔치기한 스모크 테스트로 bf16/fp16 학습, 샘플·체크포인트 저장, 재개, 체크포인트 정리, `generate --full` 로드 확인. 노트북 코드 셀 문법 검사 통과.
- 대여 GPU(Vast.ai / RunPod)용 `notebooks/rental_train.ipynb`: Drive 없이 `/workspace`에 번들 업로드, `MODE`("lora" | "full")로 LoRA v8 본 학습과 full FT를 같은 노트북에서 실행. torch가 고정 버전과 다르면 드라이버 CUDA 버전에 따라 cu130/cu126으로 재설치, 결과를 `/workspace/results_<run>.zip`으로 묶음.
- `configs/train_lora_rental.yaml`: 대여 GPU에서 v8 LoRA를 돌릴 때 실행 환경 값만 변경 (TF32 켬, 샘플 6장 한 번에, 250 step 간격). 학습 하이퍼파라미터는 로컬 v8과 동일.
- `train.allow_tf32` 옵션 추가 (`src/train/utils.py: configure_backend`). LoRA 기본 false(로컬 재현), full FT true.
- GPU 대여 조사(2026-10-18): LoRA는 RTX 4090(현재 LoRA는 fp32 UNet이라 fp32 성능이 높은 4090이 A100보다 유리할 것으로 추정), full FT는 A100 80GB 추천. 최저가는 Vast.ai(4090 중앙값 약 $0.51/h, A100 80GB 약 $0.60~0.67/h), 편의성은 RunPod Community Cloud(4090 $0.34/h, A100 80GB $1.19~1.39/h).
- 미검증: 실제 SD UNet full FT의 메모리/속도(표의 GPU별 설정은 추정), Colab Python/패키지 버전 호환(requirements 고정 버전이 Colab 기본 환경과 충돌할 수 있음), T4 fp16 안정성, 8-bit Adam 실제 동작(CUDA 필요).

## 4-4. Phase 3: 긴 HTML 조건 입력 (방법 A, 커밋 0598cc0, 학습 완료)

### 결정 사항 (사용자 확인)
- UNet은 Phase 2 full FT 체크포인트(`outputs/full_short_rental/checkpoints/step_006000`)에서 시작, 데이터는 3만 쌍으로 확장, 인코더는 고정.

### 인코더 선택 (측정)
- 기존 1만 쌍 HTML 기준: T5/FLAN-T5는 `<`가 어휘에 없어 토큰의 7.7%가 unk(디코딩 시 `<html>`→`html>`). ByT5는 무손실이지만 토큰 중앙값 338, 512 이내 68%.
- **CodeT5+ 220M**(`Salesforce/codet5p-220m`): 바이트 단위 BPE로 무손실, 토큰 중앙값 107 / p90 270 / 최대 693, 512 이내 99.8%, d_model 768(SD 1.5 cross-attention과 동일). 인코더 109.6M.
- transformers 5.20에서 CodeT5 계열 토크나이저 설정을 불러오지 못해(TypeError, RobertaTokenizerFast 생성자는 빈 토큰 출력) `tokenizers.ByteLevelBPETokenizer`로 vocab.json/merges.txt를 직접 읽음. requirements에 `tokenizers==0.23.3` 명시, check_env에 추가.

### 데이터 (`data/long`, `configs/datagen_long.yaml`)
- 생성기 설정은 Phase 1과 동일(제한 없음), 3만 쌍, CodeT5+ 토큰 512 초과는 렌더링 전 제외(`dataset.max_tokens`, `tokenizer_type: codet5p`). build_dataset의 토큰 필터를 일반화(이전 키 `max_clip_tokens` 호환, 메타데이터 키 `{type}_tokens`).
- 결과: 성공 30000(val 1503), 실패 0, 토큰 초과 92, 재현성 100/100. 같은 인덱스는 data/와 해시 일치(50/50 확인).

### 코드
- `src/models/html_encoder.py`: `CodeT5pTokenizer`, `HtmlTextEncoder`(고정), `HtmlProjection`(LayerNorm+Linear, `html_projection.safetensors`로 저장), `pad_sequences`.
- `src/train/cache_html.py`: VAE latent + 인코더 출력(가변 길이를 이어 붙인 `hidden` + `offsets`)과 빈 HTML 출력 저장.
- `src/train/train_html.py`: projection(lr 1e-4) + UNet(lr 1e-5) 학습, 패딩은 `encoder_attention_mask`로 가림, 처음 `freeze_unet_steps`(500) 동안 UNet 고정, 체크포인트에 UNet + projection, `--resume`.
- `train_lora.training_loss`와 `sampler.sample_latents`에 마스크 인자 추가(없으면 기존 동작). CFG에서 uncond/cond 길이가 다르면 패딩+마스크(`concat_for_cfg`).
- `generate.py`: config의 `text_encoder.type == codet5p`면 CodeT5+ 인코더 + 체크포인트 projection 사용, 512토큰 초과 시 경고 후 자름.
- 설정: `configs/train_html.yaml`(10000 step, 배치 8, sample 1000/checkpoint 2000 간격), `configs/train_html_overfit.yaml`(32장, 1500 step). 24GB GPU는 `train_full_24gb.yaml`을 뒤에 붙임.
- `scripts/pack_for_colab.py`: `--no-cache`, `--data-dir`, `--extra` 옵션 추가(.safetensors 무압축). `notebooks/rental_train.ipynb`: `MODE = "html"`(캐시 생성 셀 5-1 → overfit → 본 학습 → 추론).

### 검증 (로컬)
- 캐시: 50쌍으로 실제 VAE + CodeT5+ 실행, 샘플별 길이 = 메타데이터 토큰 수, 빈 HTML 2토큰.
- 학습: 작은 무작위 UNet으로 바꿔치기한 CPU 스모크(bf16) — UNet 고정→해제 전환, 샘플/체크포인트(UNet+projection), 재개, 오래된 체크포인트 정리 확인.
- 단위 확인: UNet 고정 시 projection에만 grad, 해제 시 UNet 208개 파라미터 모두 grad, 패딩 값을 바꿔도 마스크 적용 출력 동일(마스크 없으면 달라짐).
- 추론: Phase 3 스모크 체크포인트로 `generate --full` 동작(512 초과 입력 경고), Phase 2 체크포인트 추론 회귀 없음(빨간 페이지 평균색 (240,63,60) ≈ #e33).
- 미검증: 실제 SD UNet으로의 Phase 3 학습 메모리/속도, 3만 쌍 캐시 크기와 시간(인코더 출력 약 5GB 추정), projection 학습률과 freeze_unet_steps의 효과.

### 학습 결과 (Vast.ai A100 40GB, `outputs/html_long_rental`, 번들 커밋 c52989d)
- 설정: train_full + train_html + train_full_24gb(배치 4×2, bf16, 8-bit Adam), 10000 step, UNet은 Phase 2 체크포인트에서 시작, 처음 500 step UNet 고정. 약 100분, 최대 메모리 10.8GB(UNet 고정 구간 4.9GB). git_commit은 null(번들에 .git 없음).
- 학습 중 샘플 MSE(긴 HTML, val 4 / train 2): step 0 val 0.099 → 1000: 0.169(UNet 적응 중 일시 악화) → 3000: 0.048 → 8000: 0.022 → 10000: 0.024. 검증 샘플에서 칼럼 배치·버튼·박스·배경색 대부분 일치.
- 처음 보는 HTML 비교(로컬 GTX 1660 fp32, guidance 3.0, `eval_phase3` 스크래치 스크립트):
  - 짧은 HTML 7개(Phase 2 평가와 동일): Phase 2 평균 0.0166 → Phase 3 0.0054. Phase 2가 틀린 2칼럼 입력을 Phase 3는 맞힘(0.088 → 0.017). 일부 샘플은 약간 나빠짐(0.0026→0.0039 등).
  - 긴 HTML(data/long 검증 중 CodeT5+ 150토큰 초과 6개): Phase 2 평균 약 0.083 → Phase 3 약 0.041. Phase 2는 77토큰에서 잘려 배경과 첫 요소만, Phase 3는 페이지 전체 구조를 그림.
  - 직접 만든 긴 페이지 2개: 구조(내비 바·3칼럼·버튼·푸터)는 Phase 3만 재현하지만, 칼럼 높이 차이로 MSE가 오히려 높게 나오는 경우가 있음(0.119 → 0.197) → 픽셀 MSE는 큰 블록 위치 오차에 민감. Phase 4에서 LPIPS/SSIM 등 병행 필요.
- 결론: Phase 3 마일스톤 달성. 방법 A로 충분해 방법 B(DOM 레이아웃 조건) 비교는 보류.

### 대여 GPU 실행 순서
1. 커밋 후 번들: `python scripts/pack_for_colab.py --no-cache --data-dir data/long --extra outputs/full_short_rental/checkpoints/step_006000` (약 2GB 예상).
2. 인스턴스(A100 권장, 디스크 60GB 이상 권장: 캐시 수 GB + 체크포인트)에 번들과 노트북 업로드 → `MODE = "html"`, `RUN_OVERFIT = True`.

## 4-5. Phase 4: 평가 (미커밋)

### 구현
- `configs/eval.yaml`: 고정 검증셋(data/short, data/long의 val에서 eval seed 1234로 50개씩, 학습 미사용), 비교 모델(lora_v8, phase2_full, phase3_html), 생성 설정 고정(25 step, guidance 3.0, 샘플 i의 시드 = 1234 + i).
- `src/eval/generate_eval.py`: 모델별 생성 이미지 `outputs/eval/{model}/{set}/{id}.png` (이미 있으면 건너뜀), `eval_set.json`에 선택된 id 기록.
- `src/eval/metrics.py`: 픽셀 MSE, SSIM(scikit-image), LPIPS(alex), CLIP 이미지 유사도(openai/clip-vit-base-patch32, 생성-정답 코사인; 입력이 HTML이라 이미지-텍스트 CLIP score 대신), OCR 텍스트 일치율(EasyOCR, 정답 이미지 OCR 기준 1-정규화 편집거리, 정답에서 글자가 안 읽히면 제외).
- `src/eval/report.py`: metrics.csv, summary.json, report.md, grid_{set}.png.
- requirements: `easyocr==1.7.2` 추가 (opencv-python-headless 등 의존성 설치, torch 변경 없음).
- 검증: 동일 이미지 → MSE 0 / SSIM 1 / LPIPS 0 / CLIP 1 / OCR 1, 다른 이미지에서 모두 악화 확인.
- 생성 시간: 로컬 GTX 1660 fp32 약 17~24초/장, 300장 약 2시간.

### 결과 (`outputs/eval/report.md`)
## short

| 모델 | n | mse ↓ | ssim ↑ | lpips ↓ | clip ↑ | ocr ↑ |
|---|---|---|---|---|---|---|
| lora_v8 | 50 | 0.0945 | 0.4658 | 0.8261 | 0.6308 | 0.1495 |
| phase2_full | 50 | 0.0041 | 0.9681 | 0.0466 | **0.8528** | **0.2305** |
| phase3_html | 50 | **0.0030** | **0.9810** | **0.0285** | 0.8276 | 0.1466 |

OCR은 정답에서 글자가 읽힌 샘플만 평균한다 (모델별 샘플 수: {'lora_v8': 36, 'phase2_full': 36, 'phase3_html': 36}).

## long

| 모델 | n | mse ↓ | ssim ↑ | lpips ↓ | clip ↑ | ocr ↑ |
|---|---|---|---|---|---|---|
| lora_v8 | 50 | 0.1266 | 0.4909 | 0.8153 | 0.5638 | 0.1628 |
| phase2_full | 50 | 0.0318 | 0.8996 | 0.2384 | 0.7199 | 0.1891 |
| phase3_html | 50 | **0.0189** | **0.9096** | **0.1634** | **0.7309** | **0.1916** |

OCR은 정답에서 글자가 읽힌 샘플만 평균한다 (모델별 샘플 수: {'lora_v8': 43, 'phase2_full': 43, 'phase3_html': 43}).


### 짝 비교 (phase3_html - phase2_full, 같은 샘플, 부트스트랩 95% CI)
- 긴 HTML: LPIPS phase3 우세 88% (평균 차이 +0.075, CI [+0.056, +0.095]), MSE 우세 80% (+0.013, CI [+0.006, +0.021]). SSIM/CLIP/OCR은 CI가 0을 포함(유의하지 않음).
- 짧은 HTML: MSE/SSIM/LPIPS는 phase3 우세(62~64%, CI가 0 미포함, 차이는 작음). **CLIP(-0.025, CI [-0.039, -0.012])와 OCR(-0.084, CI [-0.134, -0.036])은 phase2 우세**.
- 그리드 관찰: 두 full FT 모두 글자는 거의 알아볼 수 없음(가끔 phase2가 단어를 더 비슷하게, 예: "Home" 정확). phase3는 일부 색이 약간 더 진함(주황 배경). LoRA v8은 모든 지표에서 가장 나쁨(얼룩 질감).
- 한계: OCR은 정답 이미지에서도 작은 글자를 잘못 읽어(예: "Home ['tyoyd' P: Tegjm") 대략적인 참고용. 표본 50개라 작은 차이는 신중히 해석.

### 개선 후보 (CLAUDE.md Phase 4)
- 글자 품질: 글자 영역 손실 가중치, 해상도 상향(현재 512px, VAE 8배 다운샘플로 작은 글자 손실), VAE fine-tuning.
- 짧은 HTML 글자에서 phase2가 나은 원인 확인(CLIP 인코더의 자연어 단어 표현 vs CodeT5+ 추정, 미검증).
- 베이스 모델 변경(SD3/FLUX 등)은 비용이 크므로 후순위.

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
3. (완료) Phase 1: 템플릿 기반 HTML 생성기(div, 배경색, 제목/문단, 버튼, 2~3칼럼 등을 무작위 조합), Playwright 렌더러, 저장 형식 `data/{id}.html`, `data/{id}.png`, `metadata.jsonl`, 초기 5천~1만 쌍, 시드 재현성 확인. 폰트 1종 고정 필요.
4. 알려진 잔여물: `scripts/__pycache__/`가 남아 있음 (삭제 권한이 없어 지우지 못함, .gitignore에 포함).
5. git 저장소 초기화 완료 (브랜치 `main`, Phase 0 산출물로 첫 커밋). 원격 저장소는 없음.
6. (완료) 토큰 수 측정 및 짧은 데이터셋 생성.
7. (완료) Phase 2: full fine-tuning으로 성공 기준 충족 (4-2절). 이전 LoRA 기록: overfit 테스트 미통과 (v1~v8). 시도 완료: step 증가, rank 64, zero terminal SNR + v-prediction, conv/proj LoRA. 남은 후보:
   - UNet full fine-tuning: fp32 + AdamW 기준 약 14GB 필요(추정) → 로컬 6GB 불가. Colab 노트북 준비됨 (4-3절)
   - 256px로 낮춰 실험 반복 속도 4배 (SD1.5는 저해상도 품질이 떨어지는 점 감안)
   - (완료) v8 설정 본 학습: 성공 기준 미달 (4-2절 '본 학습 결과')
8. (완료) Phase 3 방법 A — 4-4절. (완료) Phase 4 평가 — 4-5절. 다음은 개선 후보 중 선택(사용자 결정).
9. 결과 zip은 `outputs/results_full_short.zip`(Phase 2), `outputs/results_html_long.zip`(Phase 3), 압축은 각각 `outputs/full_short_rental/`, `outputs/html_long_rental/`에 풀어 둠(git 제외, .ipynb_checkpoints 제외).
