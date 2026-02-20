import cv2
import numpy as np
import os
import cv2
import numpy as np
import os
import imageio


def preview_combined_falloff(fg_mask_path, delta_path,save_dir="previews", fade_width_pct=0.25, bottom_fade_pct=0.30):
    os.makedirs(save_dir, exist_ok=True)
    
    
    #loadn the delta
    if isinstance(delta_path, str):
        delta = np.load(delta_path)
        print(f"Loaded delta from: {delta_path}")
    else:
        print("Using provided delta array directly.")
    
    
    # 1. Load and Clean Mask
    mask_img = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError("Check mask path!")
    h, w = mask_img.shape
    fg_mask = (mask_img > 150).astype(np.uint8)

    # 2. CREATE PLATEAU FALLOFF (Inner Uniformity)
    # This removes the "hard white line" ridge from your previous image
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    max_d = dist.max()
    
    if max_d > 0:
        # We scale the distance so it hits 1.0 quickly
        # Lower multiplier = wider uniform center
        limit = max_d * fade_width_pct 
        mask_falloff = np.clip(dist / limit, 0, 1)
        
        # Smoothstep for organic transition at the boundary
        mask_falloff = 3 * mask_falloff**2 - 2 * mask_falloff**3
    else:
        mask_falloff = np.zeros_like(dist)

    # 3. CREATE S-CURVE BOTTOM FADE
    # This handles the bottom 25-30% of the image height
    y_coords = np.linspace(0, 1, h)
    threshold = 1.0 - bottom_fade_pct
    
    # Linear ramp for the bottom section
    v = np.clip((y_coords - threshold) / bottom_fade_pct, 0, 1)
    # Invert and S-Curve (so 1 is top/middle, 0 is very bottom)
    s_curve = 1.0 - (3 * v**2 - 2 * v**3)
    y_gradient = s_curve.reshape(h, 1).astype(np.float32)

    # 4. COMBINE
    # Multiplication ensures both conditions must be met for high intensity
    final_falloff = mask_falloff * y_gradient

    # 5. SAVE PREVIEW
    preview_path = os.path.join(save_dir, "final_sculpt_brush_preview.png")
    cv2.imwrite(preview_path, (final_falloff * 255).astype(np.uint8))
    print(f"Success! Preview saved to: {preview_path}")
    
    return delta * final_falloff 

# Usage
# preview_combined_falloff("your_mask.png", fade_width_pct=0.25, bottom_fade_pct=0.30)



def save_delta_as_exr(mask_path, delta_npy_path, output_exr_path,fade_width_pct=0.63, bottom_fade_pct=0.55):
    delta_with_fall_off = preview_combined_falloff(mask_path,delta_npy_path,
                         fade_width_pct=fade_width_pct, bottom_fade_pct=bottom_fade_pct)

    print(delta_with_fall_off.min())
    print(delta_with_fall_off.max())
    os.makedirs(output_exr_path, exist_ok=True)
    imageio.imwrite(f'{output_exr_path}/face_deta_brush.exr', delta_with_fall_off.astype(np.float32))
    print("Exported 32-bit EXR for Blender.")
            