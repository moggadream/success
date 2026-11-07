"""경로와 기본값. 이 PC에 맞춰져 있다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONT = ROOT / "assets" / "fonts" / "NanumPen.otf"

# 작업 폴더(영상이 커서 OneDrive 밖에 둔다)
JOBS = Path(r"D:\AutoEdit\jobs")

# 파이썬: Whisper(GPU)는 ClaudeMV venv, OCR은 easyocr가 깔린 Python313
PY_MAIN = r"D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe"
PY_OCR = r"D:\ProgramFiles\Python313\python.exe"

WHISPER_MODEL = r"D:\ProgramFiles\asr\models\whisper\models--Systran--faster-whisper-large-v3\snapshots\edaa852ec7e145841d8ffdb056a99866b5f0a478"
CUDA_DLL_DIR = r"D:\ProgramFiles\Python313\Lib\site-packages\torch\lib"
EASYOCR_MODELS =r"D:\ProgramFiles\EasyOCR\model"

COMFY = "http://127.0.0.1:8188"
COMFY_DIR = Path(r"D:\ProgramFiles\ComfyUI_windows_portable")
COMFY_OUT = COMFY_DIR / "ComfyUI" / "output"

# 컷 편집 기본값 (초)
SILENCE_MIN = 0.45      # 이보다 긴 무음만 줄인다
SILENCE_KEEP = 0.12     # 무음을 자를 때 앞뒤로 남기는 여유
FILLER_PAD = 0.03       # 필러 단어 앞뒤 여유
MIN_KEEP = 0.12         # 이보다 짧게 남는 조각은 버린다
EDGE_FADE = 0.008       # 이음새 딸깍 소리 방지 페이드

# 무조건 지우는 필러 (단어 전체가 이것일 때만)
FILLERS = {"어", "음", "으", "엄", "흠", "어어", "음음", "으음", "어음", "음어", "에", "에에", "으으", "어으"}
# 앞뒤에 쉼이 있을 때만 지우는 감탄사 ("아, 그렇구나"의 '아'는 살린다)
SOFT_FILLERS = {"아", "그", "저", "뭐", "막"}
SOFT_GAP = 0.25

# 일러스트 그림체
ILLUST_STYLE = (
    "flat vector illustration mixed with hand-drawn cartoon doodle style, "
    "thick slightly wobbly black ink outlines, simple rounded shapes, flat pastel colors, "
    "clean off-white paper background, lots of empty space, cute and friendly, "
    "no text, no letters, no words, no numbers, no captions, no watermark"
)
