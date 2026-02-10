import cv2
import numpy as np
import os

def generate_final_hole_mask(full_body_mask, uncertainty_npy, raw_depth, threshold=0.6):
    """
    full_body_mask: 1 for person, 0 for background
    uncertainty_npy: Marigold uncertainty map [0, 1]
    raw_depth: The depth map we want to fix
    """
    # 1. Standardize full_body_mask to binary (0 or 1)
    # If it's a 3-channel image, convert to grayscale first
    if len(full_body_mask.shape) == 3:
        full_body_mask = cv2.cvtColor(full_body_mask, cv2.COLOR_BGR2GRAY)
    body_binary = (full_body_mask > 127).astype(np.uint8)

    # 2. Find 'unreliable' spots based on your rules
    # High uncertainty OR missing depth
    unreliable = (uncertainty_npy > threshold) | (raw_depth <= 0)
    
    # 3. Create Hole Mask: Only pixels INSIDE the body that are UNRELIABLE
    final_mask = np.zeros_like(body_binary)
    final_mask[(body_binary == 1) & (unreliable)] = 1
    
    return final_mask

# --- PATHS ---
hoel_mask_save_path = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/holes_mask"
os.makedirs(hoel_mask_save_path, exist_ok=True)

# Use cv2.imread for the .jpg mask
full_body_mask_img = cv2.imread("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/mask_8141d04d-6885-421b-830d-031ad3b0bd3f.jpg")

uncertainty_npy = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_uncertainity_files/masked_uncertainity.npy")

raw_depth = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/raw_depth_marigold/masked_depth_marigold/depth_masked_no_halo.npy")

# --- EXECUTION ---
final_hole_mask = generate_final_hole_mask(full_body_mask_img, uncertainty_npy, raw_depth)

# --- SAVE OUTPUTS ---
# 1. Save NPY for the math pipeline (Step 3)
np.save(f"{hoel_mask_save_path}/hole_mask.npy", final_hole_mask)

# 2. Save Image for visualization (White = Hole to be fixed, Black = OK)
cv2.imwrite(f"{hoel_mask_save_path}/hole_mask_vis.png", (final_hole_mask * 255).astype(np.uint8))

print(f"Hole masks saved successfully in: {hoel_mask_save_path}")