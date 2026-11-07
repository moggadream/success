"""설명 일러스트 생성: plan.json의 그림 묘사를 Krea 2 Turbo(ComfyUI)로 그린다.

plan.json 형식 (시각은 컷 후 영상 기준, 초):
{
  "illustrations": [
    {"id": "il01", "start": 12.4, "end": 17.0, "caption": "복리", "prompt": "a small coin growing into ..."}
  ],
  "masks": [ {"start": 30, "end": 34, "box": [0.1, 0.2, 0.4, 0.3]} ]   # 선택: 직접 가릴 곳
}
"""
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import config


def _get(path, timeout=5):
    return json.loads(urllib.request.urlopen(config.COMFY + path, timeout=timeout).read())


def _post(path, data):
    req = urllib.request.Request(config.COMFY + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def ensure_comfy():
    try:
        _get("/system_stats")
        return
    except Exception:
        pass
    print("  ComfyUI를 켭니다…")
    d = config.COMFY_DIR
    subprocess.Popen([str(d / "python_embeded" / "python.exe"), "-s", "ComfyUI\\main.py",
                      "--windows-standalone-build", "--disable-auto-launch"], cwd=str(d),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(180):
        time.sleep(2)
        try:
            _get("/system_stats")
            return
        except Exception:
            pass
    raise RuntimeError("ComfyUI가 켜지지 않았습니다")


def graph(name, prompt, seed, w=1536, h=864):
    return {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_int8_convrot.safetensors", "type": "krea2", "device": "default"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}},
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["6", 0], "negative": ["13", 0],
                                                   "latent_image": ["5", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "autoedit/" + name}},
    }


def generate(job, redo=None, seed_offset=0):
    """redo: 다시 그릴 id 목록(None이면 없는 것만 그림)"""
    plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
    out = job / "illust"
    out.mkdir(exist_ok=True)
    todo = [il for il in plan.get("illustrations", [])
            if (redo and il["id"] in redo) or (not redo and not (out / f"{il['id']}.png").exists())]
    if not todo:
        print("  그릴 일러스트 없음")
        return
    ensure_comfy()
    for k, il in enumerate(todo):
        t0 = time.time()
        seed = il.get("seed", 7 + k) + seed_offset
        prompt = il["prompt"].rstrip(". ") + ". " + config.ILLUST_STYLE
        pid = _post("/prompt", {"prompt": graph(il["id"], prompt, seed)})["prompt_id"]
        while True:
            h = _get(f"/history/{pid}")
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    print("  오류", il["id"], json.dumps(st.get("messages", ""), ensure_ascii=False)[:600])
                    break
                imgs = [im for o in h[pid]["outputs"].values() for im in o.get("images", [])]
                if imgs:
                    im = imgs[0]
                    dst = out / f"{il['id']}.png"
                    if dst.exists():  # 되돌리기용으로 이전 그림 보관
                        dst.rename(out / f"{il['id']}_old{int(time.time())}.png")
                    shutil.copy(config.COMFY_OUT / im.get("subfolder", "") / im["filename"], dst)
                    print(f"  {il['id']} ({il.get('caption', '')}) {time.time() - t0:.1f}s", flush=True)
                break
            time.sleep(1)
