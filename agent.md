# agent.md: 유튜브 자동 편집기

AI 코딩 에이전트(Claude Code, Codex 등)가 이 저장소에서 일할 때 읽는 안내서입니다.

## 무엇을 하는 프로그램인가
유튜브 원본 영상을 넣으면 다음을 한 번에 처리합니다.
1. **1차 컷**: 무음 구간과 "어/음" 필러 제거. 이해를 해치지 않는 것이 최우선이라, 위치가 불확실한 필러는 남깁니다.
2. **설명 일러스트**: 대본에서 설명이 필요한 구간에 플랫 벡터 + 손그림 카툰 그림을 넣습니다(Krea 2 Turbo, ComfyUI). 글자는 손글씨 자막 2~8자만 씁니다.
3. **개인정보 모자이크**: 화면 속 전화번호, 이메일, 주민번호, 카드·계좌번호, 주소, 차량번호를 EasyOCR로 찾아 가립니다.

## 구조
```
autoedit/
  run.py         명령 진입점: prepare / images / sheet / render / cutonly
  config.py      경로, 컷 기준값, 그림체 문구
  transcribe.py  Whisper large-v3(GPU) 받아쓰기, 단어별 시각 → words.json
  cutplan.py     무음·필러 찾기 → cuts.json, 원본↔컷 후 시각 변환(TimeMap), transcript_cut.txt
  privacy.py     EasyOCR + 정규식 → privacy.json  (Python313으로 따로 실행)
  illustrate.py  plan.json의 묘사로 Krea 2 그림 생성 (ComfyUI API, 꺼져 있으면 자동 실행)
  compose.py     원본 프레임을 읽어 컷 + 모자이크 + 일러스트를 입혀 NVENC로 인코딩
  assets/fonts   나눔펜 (OFL 라이선스)
```
작업 결과는 `D:\AutoEdit\jobs\<이름>\`에 쌓입니다(영상이 커서 저장소와 OneDrive 밖).

## 실행 환경 (이 PC 전용, 경로는 config.py)
| 용도 | 위치 |
|---|---|
| 메인 파이썬 | `D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe` (faster-whisper, cv2, soundfile, PIL) |
| OCR 파이썬 | `D:\ProgramFiles\Python313\python.exe` (easyocr, torch cu126) |
| Whisper GPU용 cublas/cudnn | Python313의 `torch\lib`에서 빌려 씀 (`transcribe._cuda_dlls`) |
| ComfyUI | `D:\ProgramFiles\ComfyUI_windows_portable`, `127.0.0.1:8188` |
| ffmpeg | `D:\ProgramFiles\ffmpeg\bin` (h264_nvenc 사용) |

PowerShell에서 실행할 때는 먼저 `$env:PYTHONIOENCODING="utf-8"`.

## 시간 기준 (헷갈리기 쉬움)
- `words.json`, `cuts.json`의 keep/removed, `privacy.json`은 **원본 영상 시각**.
- `transcript_cut.txt`, `plan.json`(illustrations, masks)은 **컷 후 시각**.
- 변환은 `cutplan.TimeMap`(to_new / to_orig). cuts.json을 다시 만들면 plan.json 시각이 밀리므로 함께 고칩니다.

## 지켜야 할 것
- **단어를 자르지 않는다.** Whisper의 필러 시각은 0.2~1초씩 틀립니다. 필러는 앞뒤 실제 단어의 소리 덩어리 *사이*의 짧은 덩어리만 자르고, 못 찾으면 남깁니다(`cutplan.snap_filler`). 이 규칙을 느슨하게 바꾸면 "그리고" 같은 단어가 잘린 적이 있습니다.
- 컷 로직을 바꾸면 결과 영상을 다시 받아써서 원본 대본과 대조해 확인합니다.
- 일러스트 프롬프트에 글자·숫자를 그리라고 쓰지 않습니다(AI가 엉터리 글자를 그림). 화살표는 방향과 끝을 명시합니다.
- 영상, 음성, 작업 폴더, 모델 파일은 커밋하지 않습니다(.gitignore).
- 개인정보 탐지 결과를 로그에 출력할 때는 가린 형태(`mask_text`)로만 씁니다.

## 테스트
테스트 영상 만들기(Windows 한국어 음성 Heami + ffmpeg 화면): `D:\AutoEdit\test\sample.mp4`.
```
python autoedit/run.py prepare D:\AutoEdit\test\sample.mp4 --name test1
python autoedit/run.py images test1
python autoedit/run.py render test1
```
기대 결과: 43.5초 → 약 22.7초, 필러 4개 제거, 전화번호·이메일 모자이크, 다시 받아쓴 대본에 빠진 단어 없음.

## 알려진 한계
- 말로 읽은 개인정보는 가리지 않음(대본에서 발견하면 사람에게 알림).
- 얼굴 가리기(`--faces`)는 `autoedit/assets/face_detection_yunet_2023mar.onnx`가 있어야 동작.
- 렌더는 파이썬 프레임 루프라 1080p에서 원본 길이의 1~2배 시간이 걸림.
