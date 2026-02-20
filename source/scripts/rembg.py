#!/usr/bin/env python3.11
import os
import redis
import json
import time
import traceback
from PIL import Image
from dotenv import load_dotenv
# ─── Specialist import ───
from carvekit.api.high import HiInterface

# Force CPU-only (keep your existing choice)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─── Globals & lazy interface ───
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_INTERFACE = None

def get_interface():
    global _INTERFACE
    if _INTERFACE is None:
        _INTERFACE = HiInterface(
            object_type="hairs-like",
            batch_size_seg=2,
            batch_size_matting=1,
            device='cpu',
            seg_mask_size=320,
            matting_mask_size=1024,
            trimap_prob_threshold=231,
            trimap_dilation=15,
            trimap_erosion_iters=5,
            fp16=False
        )
    return _INTERFACE

# ─── Redis ───
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

print("🚀 Simple BG Removal Worker (just CarveKit → RGBA WebP) waiting...")

while True:
    try:
        job = r.blpop("queue:remove_bg", timeout=60)
        if not job:
            continue

        volatile_id = job[1]
        redis_key = f"forge:volatile:{volatile_id}"
        master_path = r.hget(redis_key, "master_path")

        if not master_path or not os.path.exists(master_path):
            r.hset(redis_key, mapping={"status": "failed", "error": "missing master path"})
            continue

        try:
            # Open image
            with Image.open(master_path) as img:
                input_rgb = img.convert("RGB")

                # Run CarveKit (returns RGBA with alpha)
                interface = get_interface()
                result_rgba = interface([input_rgb])[0]

                # Save as lossless WebP (preserves transparency)
                output_path = os.path.join(os.path.dirname(master_path), f"{volatile_id}.webp")
                result_rgba.save(output_path, "WEBP", quality=100, lossless=True)
                alpha_channel = result_rgba.getchannel('A')
                mask_path = os.path.join(os.path.dirname(master_path), f"mask_{volatile_id}.jpg")
                alpha_channel.save(mask_path, "JPEG", quality=92)   # 90–95 is usually excellent for masks

            r.hset(redis_key, mapping={"status": "bg_completed"})
            print(f"✅ {volatile_id}  →  bg removed & saved as {output_path}")

        except Exception as e:
            print(f"❌ Processing failed for {volatile_id}: {e}")
            r.hset(redis_key, mapping={"status": "failed_bg", "error": str(e)})

    except Exception as outer:
        print(f"Worker crashed: {outer}")
        time.sleep(2)  # prevent fast crash loop