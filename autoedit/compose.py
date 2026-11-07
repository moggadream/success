"""최종 렌더: 컷 편집 + 개인정보 모자이크 + 설명 일러스트를 한 번에 입힌다.

원본을 ffmpeg로 프레임 단위로 읽어, 남길 프레임만 골라 처리한 뒤 NVENC로 인코딩한다.
"""
import json
import subprocess
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

import config
from cutplan import TimeMap

FADE = 0.25  # 일러스트 들어오고 나가는 시간


def cut_audio(job, keep_frames, fps):
    x, sr = sf.read(str(job / "audio48k.wav"), dtype="float32", always_2d=True)
    nf = int(sr * config.EDGE_FADE)
    ramp = np.linspace(0, 1, nf, dtype=np.float32)[:, None]
    parts = []
    for a, b in keep_frames:
        s, e = round(a / fps * sr), round(b / fps * sr)
        seg = x[s:e].copy()
        if len(seg) < e - s:
            seg = np.vstack([seg, np.zeros((e - s - len(seg), x.shape[1]), np.float32)])
        if len(seg) > 2 * nf:
            seg[:nf] *= ramp
            seg[-nf:] *= ramp[::-1]
        parts.append(seg)
    out = job / "audio_cut.wav"
    sf.write(str(out), np.vstack(parts), sr, subtype="PCM_16")
    return out


def pixelate(img, box, block):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad = 0.12
    x0 = int(max(0, (x0 - bw * pad) * W - 4)); x1 = int(min(W, (x1 + bw * pad) * W + 4))
    y0 = int(max(0, (y0 - bh * pad) * H - 4)); y1 = int(min(H, (y1 + bh * pad) * H + 4))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = img[y0:y1, x0:x1]
    sw, sh = max(1, (x1 - x0) // block), max(1, (y1 - y0) // block)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def caption_layer(text, W, H):
    """손글씨 글자를 둥근 흰 종이 위에 얹은 RGBA 조각과 위치"""
    size = int(H * 0.105)
    font = ImageFont.truetype(str(config.FONT), size)
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    px, py = int(size * 0.45), int(size * 0.28)
    w, h = tw + 2 * px, th + 2 * py
    im = Image.new("RGBA", (w + 12, h + 12), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 10, w + 8, h + 10], radius=int(h * 0.35), fill=(40, 40, 40, 70))  # 그림자
    d.rounded_rectangle([2, 2, w + 2, h + 2], radius=int(h * 0.35), fill=(255, 253, 246, 245),
                        outline=(30, 30, 30, 255), width=max(3, size // 22))
    d.text((2 + px - l, 2 + py - t), text, font=font, fill=(25, 25, 25, 255))
    im = im.rotate(-1.5, resample=Image.BICUBIC, expand=True)
    arr = np.array(im)  # RGBA
    bgra = arr[:, :, [2, 1, 0, 3]].astype(np.float32)
    x = (W - arr.shape[1]) // 2
    y = int(H * 0.93) - arr.shape[0]
    return bgra, x, y


def blend(dst, bgra, x, y, alpha=1.0):
    h, w = bgra.shape[:2]
    H, W = dst.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return
    a = (bgra[:, :, 3:4] / 255.0) * alpha
    roi = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (roi * (1 - a) + bgra[:, :, :3] * a).astype(np.uint8)


class Illust:
    def __init__(self, il, path, W, H):
        self.start, self.end = float(il["start"]), float(il["end"])
        self.mode = il.get("mode", "cutaway")
        img = cv2.imread(str(path))
        # 화면을 꽉 채우도록 자르기 (줌 여유 6%)
        s = max(W / img.shape[1], H / img.shape[0]) * 1.06
        img = cv2.resize(img, (int(img.shape[1] * s + 0.5), int(img.shape[0] * s + 0.5)), interpolation=cv2.INTER_AREA)
        self.img, self.W, self.H = img, W, H
        cap = il.get("caption", "").strip()
        self.cap = caption_layer(cap, W, H) if cap else None

    def frame(self, T):
        p = (T - self.start) / max(0.01, self.end - self.start)
        z = 1.06 - 0.06 * (1 - p)  # 천천히 다가가는 줌
        ih, iw = self.img.shape[:2]
        # 그림은 출력보다 6% 크게 줄여 두었다. z=1.0이면 전체, z=1.06이면 가운데 1:1
        vw, vh = self.W * 1.06 / z, self.H * 1.06 / z
        x0 = (iw - vw) / 2
        y0 = (ih - vh) / 2
        crop = self.img[int(y0):int(y0 + vh), int(x0):int(x0 + vw)]
        out = cv2.resize(crop, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.cap:
            blend(out, *self.cap)
        return out

    def alpha(self, T):
        return float(np.clip(min((T - self.start) / FADE, (self.end - T) / FADE), 0, 1))


def pip(base, small_src, W, H):
    """일러스트 위 오른쪽 아래에 원본 화면을 작게"""
    w = int(W * 0.27)
    h = int(w * small_src.shape[0] / small_src.shape[1])
    m, bd = int(W * 0.025), max(4, W // 320)
    sm = cv2.resize(small_src, (w, h), interpolation=cv2.INTER_AREA)
    x, y = W - w - m, H - h - m
    cv2.rectangle(base, (x - bd, y - bd), (x + w + bd, y + h + bd), (25, 25, 25), -1)
    base[y:y + h, x:x + w] = sm
    return base


def render(job, out_path=None, max_h=1080, preview=False):
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    fps = cuts["fps"]
    tm = TimeMap(cuts["keep"])
    keep_frames = [(round(s * fps), round(e * fps)) for s, e in cuts["keep"]]
    keep_frames = [(a, b) for a, b in keep_frames if b > a]
    total = sum(b - a for a, b in keep_frames)

    W0, H0 = cuts["w"], cuts["h"]
    mh = 360 if preview else max_h
    H = min(H0, mh) // 2 * 2
    W = int(W0 * H / H0) // 2 * 2

    plan = {}
    if (job / "plan.json").exists():
        plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
    ills = []
    for il in plan.get("illustrations", []):
        p = job / "illust" / f"{il['id']}.png"
        if p.exists():
            ills.append(Illust(il, p, W, H))
        else:
            print("  그림 없음, 건너뜀:", il["id"])
    ills.sort(key=lambda i: i.start)

    dets = []
    hold = 0.35
    if (job / "privacy.json").exists():
        pj = json.loads((job / "privacy.json").read_text(encoding="utf-8"))
        hold = pj.get("hold", hold)
        ignore = set(plan.get("privacy_ignore", []))
        dets = sorted((d for i, d in enumerate(pj["detections"]) if i not in ignore), key=lambda d: d["t"])
    dts = [d["t"] for d in dets]
    masks = plan.get("masks", [])
    block = max(8, H // 40)  # 모자이크 칸 크기 (1080p에서 27px)

    audio = cut_audio(job, keep_frames, fps)
    out_path = Path(out_path or job / ("preview.mp4" if preview else "final.mp4"))

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", cuts["video"], "-an", "-vf",
                            f"fps={fps},scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)
    venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"] if preview else \
        ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-i", str(audio), *venc, "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_path)],
                           stdin=subprocess.PIPE)

    fsize = W * H * 3
    k, n_out, seg = 0, 0, 0
    last_pct = -1
    while seg < len(keep_frames):
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        a, b = keep_frames[seg]
        if k < a:
            k += 1
            continue
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = k / fps
        T = n_out / fps

        # 1) 개인정보 모자이크 (원본 시각 기준)
        i0, i1 = bisect_left(dts, t - hold), bisect_right(dts, t + hold)
        for d in dets[i0:i1]:
            pixelate(frame, d["box"], block)
        for m in masks:
            if m["start"] <= T <= m["end"]:
                pixelate(frame, m["box"], block)

        # 2) 설명 일러스트 (컷 후 시각 기준)
        for il in ills:
            if il.start - 0.01 <= T <= il.end + 0.01:
                al = il.alpha(T)
                art = il.frame(T)
                if il.mode == "pip":
                    art = pip(art, frame, W, H)
                frame = cv2.addWeighted(art, al, frame, 1 - al, 0) if al < 1 else art
                break

        enc.stdin.write(frame.tobytes())
        n_out += 1
        k += 1
        if k >= b:
            seg += 1
        pct = int(100 * n_out / total)
        if pct % 5 == 0 and pct != last_pct:
            print(f"  렌더 {pct}%", flush=True)
            last_pct = pct

    dec.kill()  # 남은 구간은 읽을 필요 없음
    dec.wait()
    enc.stdin.close()
    enc.wait()
    print(f"  완성: {out_path}  ({n_out / fps:.1f}s)")
    return out_path
