"""1차 컷: 무음 구간과 필러 단어를 지울 구간으로 정하고 cuts.json을 쓴다.

남길 구간(keep)은 프레임 경계에 맞춰 둔다. 그래야 음성과 영상 길이가 정확히 같다.
"""
import json
import re
import subprocess
from bisect import bisect_right
from pathlib import Path

import numpy as np
import soundfile as sf

import config

PUNCT = re.compile(r"[\s.,!?~…·\-\"'“”‘’]+")


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    # 29.97 같은 값은 그대로 두되, 59.94 이상은 30으로 내려 처리량을 줄인다
    if fps > 31:
        fps = fps / 2
    return {"w": st["width"], "h": st["height"], "fps": round(fps, 3), "duration": float(j["format"]["duration"])}


def silences(wav, hop=0.02):
    x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    n = int(sr * hop)
    frames = len(x) // n
    rms = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    speech = np.percentile(db, 90)
    # 잡음 바닥보다 충분히 크고, 말소리보다 충분히 작은 지점을 문턱으로
    thr = max(floor + 10, speech - 28)
    quiet = db < thr
    out, gaps, i = [], [], 0
    while i < frames:
        if quiet[i]:
            j = i
            while j < frames and quiet[j]:
                j += 1
            if (j - i) * hop >= config.SILENCE_MIN:
                out.append([i * hop, j * hop])
            if (j - i) * hop >= 0.1:
                gaps.append([i * hop, j * hop])
            i = j
        else:
            i += 1
    # 소리 덩어리: 0.1초 이상 쉼으로 나뉜 구간
    blobs, t = [], 0.0
    for s, e in gaps:
        if s - t > 0.04:
            blobs.append([t, s])
        t = e
    if frames * hop - t > 0.04:
        blobs.append([t, frames * hop])
    return out, float(thr), len(x) / sr, blobs


def norm(w):
    return PUNCT.sub("", w)


def snap_filler(s, e, k, words, blobs):
    """Whisper의 필러 시각은 0.2~1초씩 틀린다. 그래서 순서로 찾는다.

    앞 일반 단어의 소리 덩어리와 뒤 일반 단어의 소리 덩어리를 먼저 찾고,
    그 '사이'에 있는 짧은 덩어리를 필러로 본다. 확실하지 않으면 None(자르지 않음).
    """
    def is_real(j):
        return norm(words[j]["word"]) not in config.FILLERS

    def blob_of(w):
        """단어와 가장 많이 겹치는 덩어리, 없으면 가장 가까운 덩어리의 번호"""
        def key(i):
            a, b = blobs[i]
            o = max(0.0, min(b, w["end"]) - max(a, w["start"]))
            d = max(a - w["end"], w["start"] - b, 0.0)
            return (o, -d)
        return max(range(len(blobs)), key=key) if blobs else None

    p = next((j for j in range(k - 1, -1, -1) if is_real(j)), None)
    n = next((j for j in range(k + 1, len(words)) if is_real(j)), None)
    ip = blob_of(words[p]) if p is not None else -1
    inn = blob_of(words[n]) if n is not None else len(blobs)
    if ip is None or inn is None:
        return None
    cands = [blobs[i] for i in range(ip + 1, inn) if blobs[i][1] - blobs[i][0] <= 0.9]
    if not cands:
        return None
    mid = (s + e) / 2
    a, b = min(cands, key=lambda ab: abs((ab[0] + ab[1]) / 2 - mid))
    return a - config.FILLER_PAD, b + config.FILLER_PAD


def filler_spans(words, blobs=()):
    spans = []
    for k, w in enumerate(words):
        t = norm(w["word"])
        if not t:
            continue
        hard = t in config.FILLERS
        soft = False
        if t in config.SOFT_FILLERS:
            gap_before = w["start"] - words[k - 1]["end"] if k > 0 else 9
            gap_after = words[k + 1]["start"] - w["end"] if k + 1 < len(words) else 9
            soft = gap_before >= config.SOFT_GAP and gap_after >= config.SOFT_GAP
        if hard or soft:
            ab = snap_filler(w["start"], w["end"], k, words, blobs)
            if ab is None:
                print(f"  필러 위치가 불확실해 남겨 둠: '{t}' ({w['start']:.1f}s)")
                continue
            spans.append([ab[0], ab[1], t, k])
    return spans


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def build(job, video):
    info = probe(video)
    words = json.loads((job / "words.json").read_text(encoding="utf-8"))["words"]
    sil, thr, dur, blobs = silences(job / "audio16k.wav")
    dur = min(dur, info["duration"])
    fps = info["fps"]

    removed = []
    for s, e in sil:
        a, b = s + config.SILENCE_KEEP, e - config.SILENCE_KEEP
        # 영상 맨 앞뒤 무음은 통째로 지운다
        if s <= 0.05:
            a = 0.0
        if e >= dur - 0.05:
            b = dur
        if b - a > 0.05:
            removed.append({"start": a, "end": b, "reason": "silence"})
    for s, e, t, k in filler_spans(words, blobs):
        removed.append({"start": max(0, s), "end": min(dur, e), "reason": "filler", "word": t, "idx": k})

    cut = merge([[r["start"], r["end"]] for r in removed])
    keep, t = [], 0.0
    for s, e in cut:
        if s - t >= config.MIN_KEEP:
            keep.append([t, s])
        t = max(t, e)
    if dur - t >= config.MIN_KEEP:
        keep.append([t, dur])

    # 프레임 경계로 맞춤
    snapped = []
    for s, e in keep:
        a, b = round(s * fps) / fps, round(e * fps) / fps
        if b - a >= 1 / fps:
            snapped.append([round(a, 4), round(b, 4)])
    keep = snapped

    new_dur = sum(e - s for s, e in keep)
    data = {"video": str(video), **info, "threshold_db": round(thr, 1), "keep": keep,
            "removed": removed, "orig_duration": round(dur, 3), "new_duration": round(new_dur, 3)}
    (job / "cuts.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    write_cut_transcript(job, data, words)
    nf = sum(1 for r in removed if r["reason"] == "filler")
    print(f"  원본 {dur:.1f}s → 컷 후 {new_dur:.1f}s ({dur - new_dur:.1f}s 줄임, 필러 {nf}개)")
    return data


class TimeMap:
    """원본 시각 ↔ 컷 후 시각 변환"""

    def __init__(self, keep):
        self.keep = keep
        self.starts = [s for s, _ in keep]
        self.offset = []
        acc = 0.0
        for s, e in keep:
            self.offset.append(acc)
            acc += e - s
        self.total = acc

    def to_new(self, t):
        i = bisect_right(self.starts, t) - 1
        if i < 0:
            return 0.0
        s, e = self.keep[i]
        return self.offset[i] + min(t, e) - s

    def to_orig(self, t):
        i = bisect_right(self.offset, t) - 1
        i = max(0, min(i, len(self.keep) - 1))
        return self.keep[i][0] + (t - self.offset[i])


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def write_cut_transcript(job, data, words):
    """컷 후 시각 기준 대본. 일러스트 계획을 세울 때 읽는다."""
    tm = TimeMap(data["keep"])
    gone = {r["idx"] for r in data["removed"] if r["reason"] == "filler"}
    segs = json.loads((job / "words.json").read_text(encoding="utf-8"))["segments"]
    lines = []
    wi = 0
    for sg in segs:
        ws = []
        while wi < len(words) and words[wi]["start"] < sg["end"] + 0.01:
            if wi not in gone:
                ws.append(words[wi])
            wi += 1
        if not ws:
            continue
        a, b = tm.to_new(ws[0]["start"]), tm.to_new(ws[-1]["end"])
        lines.append(f"[{fmt(a)} - {fmt(b)}] " + " ".join(w["word"] for w in ws))
    (job / "transcript_cut.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
