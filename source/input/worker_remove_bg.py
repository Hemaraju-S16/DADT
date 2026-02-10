#!/usr/bin/env python3.11
import os
import redis
import json
import time
import traceback
from PIL import Image
from dotenv import load_dotenv

# --- SPECIALIST IMPORTS ---
from carvekit.api.high import HiInterface

# --- FORCE CPU-ONLY MODE ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1. Environment & Global Interface Cache
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_INTERFACE = None

# 🎯 Canonical Orthographic Constants
TARGET_CANVAS = 768
VIEW_SPAN = 1.1        # Total units from (-0.55 to 0.55)
UNIT_CUBE_SIZE = 1.0   # Target normalization within the view frustum

def get_interface():
    """Lazy-load the CarveKit HiInterface with stabilized CPU settings"""
    global _INTERFACE
    if _INTERFACE is None:
        _INTERFACE = HiInterface(
            object_type="hairs-like", 
            batch_size_seg=2,          # Optimized for single-task CPU stability
            batch_size_matting=1,
            device='cpu',
            seg_mask_size=320,
            matting_mask_size=1024,    # Balanced for high detail vs memory pressure
            trimap_prob_threshold=231,
            trimap_dilation=15,        # Faster CPU inference without losing edge quality
            trimap_erosion_iters=5,
            fp16=False
        )
    return _INTERFACE

# 2. Redis Connection
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

print(f"🚀 BG Removal Specialist (Canonical {TARGET_CANVAS}px | RGBA + JPEG Mask) awaiting jobs...")

# -------------------------------------------------------------------------
# MAIN WORKER LOOP
# -------------------------------------------------------------------------
while True:
    try:
        # Blocking pop for efficient single-task processing
        job = r.blpop("queue:remove_bg", timeout=60)
        if not job: continue
            
        volatile_id = job[1]
        redis_key = f"forge:volatile:{volatile_id}"
        master_path = r.hget(redis_key, "master_path")

        if not master_path or not os.path.exists(master_path):
            r.hset(redis_key, mapping={"status": "failed", "error": "MISSING_MASTER_PATH"})
            continue

        try:
            # 🛡️ Resource Safety: Using context managers for image handles
            with Image.open(master_path) as img:
                subject_raw = img.convert("RGB")
                
                interface = get_interface()
                # HiInterface returns RGBA with high-fidelity matting
                isolated_subject = interface([subject_raw])[0] 

                # 🛡️ Step 1: Tight Crop to Content (Removes user's original padding)
                bbox = isolated_subject.getbbox() 
                if not bbox: raise ValueError("NO_SUBJECT_DETECTED")
                
                with isolated_subject.crop(bbox) as subject_cropped:
                    # 🛡️ Step 2: Deterministic Canonical Math
                    # Derive limit from Unit Cube vs View Span ratio (approx 698.18px)
                    pixel_limit = TARGET_CANVAS * (UNIT_CUBE_SIZE / VIEW_SPAN) 
                    
                    orig_w, orig_h = subject_cropped.size
                    # Scale based on the most constrained side to maintain aspect ratio
                    scale = min(pixel_limit / orig_w, pixel_limit / orig_h)
                    
                    # Round-off to integer occurs only at the final step
                    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                    
                    with subject_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS) as subject_resized:
                        # 🛡️ Step 3: Master RGBA Composition (Transparent Background)
                        canvas = Image.new("RGBA", (TARGET_CANVAS, TARGET_CANVAS), (0, 0, 0, 0))
                        
                        # Symmetric Origin Centering for Azimuth alignment
                        offset_x = (TARGET_CANVAS - new_w) // 2
                        offset_y = (TARGET_CANVAS - new_h) // 2
                        
                        # Paste using the subject as its own mask to preserve transparency
                        canvas.paste(subject_resized, (offset_x, offset_y), subject_resized)

                        # 🛡️ Step 4: Safety Alpha Mask Extraction (Grayscale)
                        # Extract the 'A' channel and place it on a black canvas
                        alpha_channel = subject_resized.getchannel('A')
                        mask_canvas = Image.new("L", (TARGET_CANVAS, TARGET_CANVAS), 0) # 8-bit black
                        mask_canvas.paste(alpha_channel, (offset_x, offset_y))

                        # 🏁 Step 5: Dual Asset Preservation
                        job_dir = os.path.dirname(master_path)
                        
                        # A. The RGBA Master WebP (Lossless for Stage 2)
                        output_path = os.path.join(job_dir, f"{volatile_id}.webp")
                        canvas.save(output_path, "WEBP", quality=100, lossless=True)

                        # B. The Safety Alpha Mask (Grayscale JPEG @ Q100)
                        mask_path = os.path.join(job_dir, f"mask_{volatile_id}.jpg")
                        mask_canvas.save(mask_path, "JPEG", quality=100)

            r.hset(redis_key, mapping={"status": "bg_completed"})
            print(f"✅ Job {volatile_id} | Canonical Master + JPEG Mask Saved")

        except Exception as ml_err:
            print(f"❌ Worker Logic Error: {ml_err}")
            r.hset(redis_key, mapping={"status": "failed_bg", "error": str(ml_err)})

    except Exception as global_err:
        traceback.print_exc()
        time.sleep(1)