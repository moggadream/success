# 유튜브 자동 편집기

영상을 넣으면 다음을 해 줍니다.
1. **1차 컷**: 무음 구간과 "어/음" 필러를 자름 (단어가 잘릴 위험이 있으면 남겨 둠)
2. **설명 일러스트**: 대본을 읽고 필요한 곳에 플랫 벡터 + 손그림 카툰 그림을 넣음 (Krea 2, 손글씨 자막 2~8자)
3. **개인정보 모자이크**: 화면 속 전화번호, 이메일, 주민번호, 카드·계좌번호, 주소, 차량번호

## 쓰는 법 (Claude Code에게)
> "D:\영상\원본.mp4 편집해줘"

Claude가 `yt-autoedit` 스킬로 아래를 순서대로 실행합니다. 2단계(어디에 어떤 그림을 넣을지)는 Claude가 대본을 읽고 정합니다.

## 직접 실행
```bash
D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe run.py prepare D:\영상\원본.mp4 --name 내영상
```
```bash
D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe run.py images 내영상
```
```bash
D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe run.py render 내영상
```
일러스트 없이 컷만: `run.py cutonly <영상>`

결과: `D:\AutoEdit\jobs\<이름>\final.mp4`

## 파일
| 파일 | 하는 일 |
|---|---|
| `transcribe.py` | Whisper large-v3(GPU) 받아쓰기, 단어별 시각 |
| `cutplan.py` | 무음·필러 찾기, 원본↔컷 후 시각 변환 |
| `privacy.py` | EasyOCR(GPU)로 화면 글자 읽고 개인정보 패턴 찾기 (Python313으로 실행) |
| `illustrate.py` | ComfyUI의 Krea 2 Turbo로 일러스트 생성 |
| `compose.py` | 컷 + 모자이크 + 일러스트를 한 번에 렌더 (NVENC) |
| `config.py` | 경로, 컷 기준값, 그림체 문구 |
