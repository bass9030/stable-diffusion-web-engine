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

`check_env.py`는 패키지 import, torch 장치(CUDA/ROCm/MPS/DirectML/CPU), Playwright 렌더링과 렌더 결과 재현성, 디스크 여유 공간을 검사한다.
`[FAIL]`이 하나라도 있으면 종료 코드 1을 반환한다. GPU 가속이 없는 것은 `[WARN]`이며, 학습은 클라우드 GPU를 쓰는 것을 전제로 한다.
