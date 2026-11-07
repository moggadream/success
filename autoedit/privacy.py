"""화면 속 개인정보 찾기 (Python313 + EasyOCR GPU로 실행).

사용: python privacy.py <작업폴더> [--step 0.33] [--faces]
결과: <작업폴더>/privacy.json  (원본 시각 기준 박스, 좌표는 0~1 비율)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

# 숫자 사이 구분자: 하이픈, 점, 공백, 물결 등
SEP = r"[\s\-\.~·]?"
PATTERNS = [
    ("주민번호", re.compile(r"\d{6}\s?[-~]\s?[1-4]\d{6}|\d{6}\s?[-~]\s?[1-4]\*{6}")),
    ("카드번호", re.compile(r"\d{4}" + SEP + r"\d{4}" + SEP + r"\d{4}" + SEP + r"\d{3,4}")),
    ("휴대폰", re.compile(r"01[016789]" + SEP + r"\d{3,4}" + SEP + r"\d{4}")),
    ("전화번호", re.compile(r"0[2-6]\d?" + SEP + r"\d{3,4}" + SEP + r"\d{4}")),
    ("이메일", re.compile(r"[\w.+-]+\s?@\s?[\w-]+(\.[\w-]+)+")),
    ("계좌번호", re.compile(r"\d{2,6}-\d{2,6}-\d{2,8}(-\d{1,4})?")),
    ("차량번호", re.compile(r"\d{2,3}\s?[가-힣]\s?\d{4}")),
    ("주소", re.compile(r"[가-힣]+(시|구|군|읍|면)\s+[가-힣\d]+(로|길)\s?\d+|\d+동\s?\d+호|[가-힣]+(아파트|빌라|오피스텔)\s?\d+동")),
    ("여권번호", re.compile(r"\b[MSRODmsrod]\d{8}\b")),
]
# 라벨 뒤에 오는 값은 패턴과 무관하게 가린다 ("이름: 홍길동")
LABELS = re.compile(r"(성명|이름|주소|생년월일|연락처|전화|휴대폰|핸드폰|계좌|주민|e-?mail|이메일|비밀번호|password|사번|학번)\s*[:：]?",
                    re.I)


def classify(text):
    t = text.strip()
    if len(t) < 3:
        return None
    for kind, pat in PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        if kind == "계좌번호":
            s = m.group(0)
            if sum(ch.isdigit() for ch in s) < 10 or re.fullmatch(r"(19|20)\d\d-\d\d?-\d\d?", s):
                continue  # 날짜(2026-11-07) 같은 것은 제외
        return kind
    return None


def line_groups(res):
    """같은 줄에 붙어 있는 OCR 조각을 합친다 (010 / 1234 / 5678로 쪼개지는 경우)."""
    items = []
    for box, text, conf in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({"b": [min(xs), min(ys), max(xs), max(ys)], "t": text, "c": conf})
    items.sort(key=lambda d: (d["b"][1], d["b"][0]))
    groups = []
    for it in items:
        h = it["b"][3] - it["b"][1]
        cy = (it["b"][1] + it["b"][3]) / 2
        for g in groups:
            gh = g["b"][3] - g["b"][1]
            gcy = (g["b"][1] + g["b"][3]) / 2
            if abs(cy - gcy) < 0.5 * max(h, gh) and 0 <= it["b"][0] - g["b"][2] < 1.5 * max(h, gh):
                g["b"] = [g["b"][0], min(g["b"][1], it["b"][1]), it["b"][2], max(g["b"][3], it["b"][3])]
                g["t"] += " " + it["t"]
                g["parts"].append(it)
                break
        else:
            groups.append({"b": list(it["b"]), "t": it["t"], "parts": [it]})
    return items, groups


def find_pii(res):
    items, groups = line_groups(res)
    hits = []
    for g in groups:
        kind = classify(g["t"]) or classify(g["t"].replace(" ", ""))
        if kind:
            hits.append((kind, g["b"], g["t"]))
            continue
        # 라벨이 있으면 라벨 뒤 값 조각을 가린다
        m = LABELS.search(g["t"])
        if m and len(g["parts"]) >= 1:
            after = [p for p in g["parts"] if not LABELS.fullmatch(p["t"].strip())]
            if after:
                b = [min(p["b"][0] for p in after), min(p["b"][1] for p in after),
                     max(p["b"][2] for p in after), max(p["b"][3] for p in after)]
                hits.append(("라벨값", b, g["t"]))
    return hits


def mask_text(t):
    return re.sub(r"[0-9A-Za-z가-힣]", "*", t)[:4] + f"…({len(t)}자)"


def face_detector(w, h):
    model = HERE / "assets" / "face_detection_yunet_2023mar.onnx"
    if not model.exists():
        print("  얼굴 모델이 없어 얼굴 가리기를 건너뜀:", model)
        return None
    return cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.8, 0.3, 5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--step", type=float, default=0.33)
    ap.add_argument("--faces", action="store_true", help="화면 속 얼굴도 가림(가장 큰 얼굴=본인은 제외)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    import easyocr

    job = Path(a.job)
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    reader = easyocr.Reader(["ko", "en"], gpu=not a.cpu, model_storage_directory=config.EASYOCR_MODELS,
                            download_enabled=False, verbose=False)

    cap = cv2.VideoCapture(cuts["video"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1600 / max(W, H))
    fdet = face_detector(int(W * scale), int(H * scale)) if a.faces else None

    # 남길 구간 안에서만 step 간격으로 샘플
    times = []
    for s, e in cuts["keep"]:
        t = s
        while t < e:
            times.append(t)
            t += a.step
    targets = sorted({int(round(t * src_fps)) for t in times})

    dets, faces = [], []
    prev_small, prev_hits, prev_faces = None, [], []
    fi, ti, ocr_runs = 0, 0, 0
    while ti < len(targets):
        ok = cap.grab()
        if not ok:
            break
        if fi != targets[ti]:
            fi += 1
            continue
        ok, frame = cap.retrieve()
        fi += 1
        ti += 1
        if not ok:
            continue
        t = (fi - 1) / src_fps
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        small = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        same = prev_small is not None and np.abs(small - prev_small).mean() < 1.5
        if same:
            hits, fbs = prev_hits, prev_faces
        else:
            res = reader.readtext(img, paragraph=False)
            ocr_runs += 1
            hits = [(k, [b[0] / img.shape[1], b[1] / img.shape[0], b[2] / img.shape[1], b[3] / img.shape[0]], txt)
                    for k, b, txt in find_pii(res)]
            fbs = []
            if fdet is not None:
                _, fs = fdet.detect(img)
                if fs is not None and len(fs) > 1:
                    fs = sorted(fs, key=lambda f: -f[2] * f[3])[1:]  # 가장 큰 얼굴(진행자)은 남김
                    for f in fs:
                        x, y, w, h = f[:4]
                        fbs.append([x / img.shape[1], y / img.shape[0], (x + w) / img.shape[1], (y + h) / img.shape[0]])
            prev_small, prev_hits, prev_faces = small, hits, fbs
            for k, b, txt in hits:
                print(f"  {t:7.2f}s  {k}  {mask_text(txt)}", flush=True)
        for k, b, txt in hits:
            dets.append({"t": round(t, 3), "kind": k, "box": [round(v, 4) for v in b]})
        for b in fbs:
            faces.append({"t": round(t, 3), "kind": "얼굴", "box": [round(float(v), 4) for v in b]})

    out = {"step": a.step, "hold": round(a.step * 0.6 + 0.15, 3), "detections": dets + faces}
    (job / "privacy.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {}
    for d in out["detections"]:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"  샘플 {len(targets)}장 중 OCR {ocr_runs}번 실행, 발견: {kinds or '없음'}")


if __name__ == "__main__":
    main()
