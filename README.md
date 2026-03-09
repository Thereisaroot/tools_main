# ISAC Labelr

Win/Mac/Ubuntu에서 동작하는 영상 ROI 진입/출입 메타데이터 생성 GUI 앱.

## 주요 기능
- 영상 프리뷰 + ROI 다중 드래그 지정
- 버튼 기반 빠른 조작: `Open File`, `Rotate -90/+90`, `Start Full/Partial Analysis`, `Stop Analysis`
- 회전(0/90/180/270) 후 동일 좌표계로 분석
- 모드 A(진입만), 모드 B(진입+출입 방향)
- 사람 검출(ONNX Runtime) + 추적(ByteTrack-like)
- 이벤트 프레임 중심 OCR(Unix ms 13자리)
- 타임스탬프 보정(+/- ms)
- 이벤트 리스트 클릭 시 프레임 이동
- 결과 저장: `<video>.json`, `<video>_debug.json`, `<video>_run_log.txt`

## 재생/분석 동작 구분 (중요)
- `Play Preview` 버튼: **프리뷰 재생만 수행** (분석 시작 아님)
- 분석 시작: `Start Full Analysis` 또는 `Start Partial Analysis` 버튼
- 분석 중지: `Stop Analysis` 버튼

## 실행
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m isac_labelr
```

## 설치+실행 스크립트
- macOS: `scripts/setup_run_macos.sh`
- Ubuntu: `scripts/setup_run_ubuntu.sh`
- Windows: `scripts/setup_run_windows.ps1`
- Windows(권장): `scripts/setup_run_windows.cmd`  (ExecutionPolicy 우회 실행)

위 스크립트는 가상환경 생성, 의존성 설치, 모델 자동 다운로드 시도 후 앱 실행까지 수행한다.
또한 파이썬 버전 fallback을 포함한다.
- `python3.12`가 있으면 우선 사용(권장).
- `py` 런처가 없거나 실패하면 `python/python3` 명령을 자동 탐색한다.
- 3.13 이상이면 `rapidocr` 설치를 건너뛰고 `pytesseract` fallback OCR 사용.
- `av` 설치 실패 시 OpenCV 디코더 fallback 사용.

## 플랫폼별 빌드
- macOS: `scripts/build_macos.sh`
- Ubuntu: `scripts/build_ubuntu.sh`
- Windows: `scripts/build_windows.ps1`
- Windows(권장): `scripts/build_windows.cmd`  (ExecutionPolicy 우회 실행)

각 스크립트는 PyInstaller one-folder 방식으로 `dist/isac_labelr_portable`를 생성한다.
빌드 스크립트도 실행 스크립트와 동일하게 Python/version fallback을 적용한다.

## 모델
기본 검출 모델 경로: `models/person_detector.onnx`

### 자동 다운로드
```bash
python scripts/download_default_model.py --output models/person_detector.onnx
```

인증서 오류(`CERTIFICATE_VERIFY_FAILED`)가 나면:
```bash
MODEL_DOWNLOAD_INSECURE=1 ./scripts/setup_run_macos.sh
```
또는 직접:
```bash
python scripts/download_default_model.py --output models/person_detector.onnx --insecure
```

기본값은 TLS 검증을 유지하며, `--insecure`는 마지막 수단으로만 사용한다.

### 수동 배치
자동 다운로드가 실패하면 YOLO ONNX 파일을 직접 준비해서 아래 경로로 복사한다.

- `models/person_detector.onnx`

주의: 현재 코드는 person class index를 `0`으로 가정한다.

## OCR fallback
- 1순위: `rapidocr-onnxruntime` (Python < 3.13에서 시도)
- 2순위: `pytesseract` + 시스템 `tesseract` 바이너리
- macOS 기본값: 메모리 안정성을 위해 `pytesseract` 우선 사용(설치되어 있을 때)

강제 선택:
```bash
ISAC_OCR_BACKEND=pytesseract python -m isac_labelr
ISAC_OCR_BACKEND=rapidocr python -m isac_labelr
```

`tesseract`가 없으면 OCR 인식률이 급격히 떨어지거나 실패할 수 있다.
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install -y tesseract-ocr`
- Windows: `winget install UB-Mannheim.TesseractOCR` 후 새 터미널에서 실행

## macOS AV 경고 대응
macOS에서 `cv2`와 `av`가 FFmpeg dylib를 중복 로드하는 경고가 날 수 있다.  
기본값은 `PyAV` 비활성(CV2 디코딩 사용)으로 설정되어 있으며, 필요할 때만 아래로 강제 활성 가능:

```bash
ISAC_USE_PYAV=1 python -m isac_labelr
```

## 메모리 튜닝(장시간 분석)
- 기본값으로 `cv2` 캡처 재오픈은 비활성이다. (macOS에서 재오픈 시 footprint 누적 가능)
- 필요 시에만 명시적으로 재오픈:

```bash
ISAC_CV2_REOPEN_EVERY_CHUNKS=1 python -m isac_labelr
```

- 프리뷰 재생 중 캡처 재오픈도 기본 비활성이며, 필요 시만 설정:

```bash
ISAC_PREVIEW_REOPEN_INTERVAL=900 python -m isac_labelr
```
