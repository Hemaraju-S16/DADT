import cv2
import numpy as np
import os

# --- THE FUNCTION (STAYS THE SAME) ---
def fix_and_stabilize(raw_npy, guide_image):
    # 1. FIX SHAPE: If it's (3, 768, 3), it's likely transposed or corrupted.
    # We need to ensure we are working with (H, W, 3)
    if raw_npy.shape[0] == 3 and raw_npy.shape[1] != 3:
        raw_npy = np.transpose(raw_npy, (1, 2, 0)) 

    # 2. FIX RANGE: If values are near zero, we must re-scale 
    # Normal vectors MUST have a length of 1.
    # If your NPY is 0-255, scale to -1, 1. If it's already float, normalize it.
    norm = np.linalg.norm(raw_npy, axis=2, keepdims=True)
    raw_npy = raw_npy / (norm + 1e-6)

    # 3. STABILIZE (Channel by Channel)
    guide_float = guide_image.astype(np.float32)
    stabilized_channels = []
    
    # Map to [0, 1] for the filter
    n_for_filter = (raw_npy + 1.0) / 2.0
    
    for i in range(3):
        chan = cv2.ximgproc.jointBilateralFilter(
            guide_float, n_for_filter[:,:,i].astype(np.float32), 
            d=5, sigmaColor=25, sigmaSpace=5
        )
        stabilized_channels.append(chan)
        
    stabilized = cv2.merge(stabilized_channels)
    
    # 4. FINAL RE-NORMALIZATION (The "Curvature" Restorer)
    n_centered = stabilized * 2.0 - 1.0
    final_norm = np.linalg.norm(n_centered, axis=2, keepdims=True)
    n_normalized = n_centered / (final_norm + 1e-6)
    
    return n_normalized

# --- EXECUTION WITH DUAL SAVE ---

npy_input_path = '/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/masked_normals/normals_masked.npy'
guide_img_path = '/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/original_rgb.webp'
save_dir = '/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/stabilized_normals'

os.makedirs(save_dir, exist_ok=True)

# 1. Load
raw_npy = np.load(npy_input_path)
guide = cv2.imread(guide_img_path)

# 2. Process
clean_normals_float = fix_and_stabilize(raw_npy, guide)

# 3. Save NPY (For the Pipeline/Step 3)
np.save(f'{save_dir}/stabilized_normals.npy', clean_normals_float)

# 4. Save Image (For Visualization/Blender)
# Convert [-1, 1] back to [0, 255] uint8
vis_map = ((clean_normals_float + 1.0) / 2.0 * 255.0).astype(np.uint8)
# OpenCV uses BGR, so if your NPY is RGB, swap channels for correct visual colors
vis_map_bgr = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR) 
cv2.imwrite(f'{save_dir}/stabilized_normals_vis.png', vis_map_bgr)

print(f"Done! \nData: stabilized_normals.npy \nVisual: stabilized_normals_vis.png")