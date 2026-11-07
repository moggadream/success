r"""유튜브 자동 편집기

  python run.py prepare <영상> [--name 이름] [--faces]
      받아쓰기 → 무음·필러 컷 계획 → 화면 개인정보 탐지
      결과: D:\AutoEdit\jobs\<이름>\  (transcript_cut.txt, cuts.json, privacy.json)
  (plan.json 작성: 컷 후 대본을 읽고 설명 일러스트 구간·묘사·한두 단어 자막을 정한다)
  python run.py images <작업폴더> [--redo il01,il03] [--seed 1]
  python run.py sheet <작업폴더>          일러스트 모아보기 이미지
  python run.py render <작업폴더> [--preview]
  python run.py cutonly <영상>            1차 컷만 바로 렌더 (일러스트·모자이크 없이)

실행 파이썬: D:\ProgramFiles\ClaudeMV\.venv\Scripts\python.exe
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402


def job_dir(arg):
    p = Path(arg)
    return p if p.is_dir() else config.JOBS / arg


def prepare(video, name=None, faces=False, skip_privacy=False):
    import cutplan
    import transcribe

    video = Path(video).resolve()
    name = name or re.sub(r"[^\w가-힣\-]+", "_", video.stem)[:40]
    job = config.JOBS / name
    job.mkdir(parents=True, exist_ok=True)
    (job / "source.txt").write_text(str(video), encoding="utf-8")
    print(f"[1/4] 음성 추출 → {job}")
    transcribe.extract_audio(video, job)
    if not (job / "words.json").exists():
        print("[2/4] 받아쓰기 (Whisper large-v3, GPU)")
        transcribe.transcribe(job)
    else:
        print("[2/4] 받아쓰기: 이전 결과 사용")
    print("[3/4] 무음·필러 컷 계획")
    cutplan.build(job, video)
    if skip_privacy:
        print("[4/4] 개인정보 탐지 건너뜀")
    else:
        print("[4/4] 화면 개인정보 탐지 (EasyOCR, GPU)")
        cmd = [config.PY_OCR, str(HERE / "privacy.py"), str(job)] + (["--faces"] if faces else [])
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"\n완료. 대본: {job / 'transcript_cut.txt'}")
    print("다음: plan.json을 쓰고  run.py images → run.py render")
    return job


def sheet(job):
    import cv2
    import numpy as np

    plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
    tiles = []
    for il in plan["illustrations"]:
        p = job / "illust" / f"{il['id']}.png"
        if not p.exists():
            continue
        im = cv2.resize(cv2.imread(str(p)), (480, 270))
        cv2.putText(im, il["id"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(im)
    while len(tiles) % 3:
        tiles.append(np.full((270, 480, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    out = job / "illust_sheet.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print("  ", out)


def main():
    ap = argparse.ArgumentParser(description="유튜브 자동 편집기")
    ap.add_argument("cmd", choices=["prepare", "images", "sheet", "render", "cutonly"])
    ap.add_argument("target")
    ap.add_argument("--name")
    ap.add_argument("--faces", action="store_true")
    ap.add_argument("--no-privacy", action="store_true")
    ap.add_argument("--redo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.cmd == "prepare":
        prepare(a.target, a.name, a.faces, a.no_privacy)
    elif a.cmd == "cutonly":
        import compose
        job = prepare(a.target, a.name, skip_privacy=True)
        compose.render(job, job / "cut_only.mp4")
    elif a.cmd == "images":
        import illustrate
        illustrate.generate(job_dir(a.target), a.redo.split(",") if a.redo else None, a.seed)
    elif a.cmd == "sheet":
        sheet(job_dir(a.target))
    elif a.cmd == "render":
        import compose
        compose.render(job_dir(a.target), a.out, preview=a.preview)


if __name__ == "__main__":
    main()
