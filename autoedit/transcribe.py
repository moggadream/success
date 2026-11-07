"""받아쓰기: 단어마다 시각이 붙은 words.json을 만든다.

필러("어", "음")는 Whisper가 보통 빼먹는다. 필러가 섞인 예시 문장을
initial_prompt로 주면 들리는 그대로 적는 쪽으로 기운다.
"""
import json
import subprocess
import sys
from pathlib import Path

import config

PROMPT = "음, 어, 그러니까 오늘은요, 어어 이거를 한번 해 볼 건데요. 음... 아, 네. 으음, 그래서 어 이렇게 됩니다."


def extract_audio(video, job):
    """16kHz 모노(받아쓰기용)와 48kHz 스테레오(최종 음성용)를 뽑는다."""
    a16 = job / "audio16k.wav"
    a48 = job / "audio48k.wav"
    for out, args in ((a16, ["-ac", "1", "-ar", "16000"]), (a48, ["-ac", "2", "-ar", "48000"])):
        if not out.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", *args,
                            "-c:a", "pcm_s16le", str(out)], check=True)
    return a16, a48


def _cuda_dlls():
    """faster-whisper(GPU)가 쓰는 cublas/cudnn을 Python313의 torch 폴더에서 빌려 온다."""
    import os
    d = Path(config.CUDA_DLL_DIR)
    if d.exists():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def transcribe(job, language="ko"):
    _cuda_dlls()
    from faster_whisper import WhisperModel

    a16 = job / "audio16k.wav"
    model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="float16")
    segs, info = model.transcribe(
        str(a16), language=language, word_timestamps=True, initial_prompt=PROMPT,
        vad_filter=False, condition_on_previous_text=False, beam_size=5,
    )
    words, segments = [], []
    for s in segs:
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                          "word": w.word.strip(), "p": round(w.probability, 3)})
        print(f"  [{s.start:7.2f}] {s.text.strip()}", flush=True)
    (job / "words.json").write_text(json.dumps({"segments": segments, "words": words},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    return words


if __name__ == "__main__":
    job = Path(sys.argv[1])
    transcribe(job)
