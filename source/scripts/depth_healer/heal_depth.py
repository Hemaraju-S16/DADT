import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import cv2
import os

def heal_depth_with_poisson_and_fg(depth_npy, normals_npy, hole_mask_npy, fg_mask_img):
    H, W = depth_npy.shape
    num_pixels = H * W
    
    # 1. Clean Foreground Mask
    if len(fg_mask_img.shape) == 3:
        fg_mask_img = cv2.cvtColor(fg_mask_img, cv2.COLOR_BGR2GRAY)
    fg_binary = (fg_mask_img > 127).astype(np.uint8)

    # 2. Target gradients from normals
    nx, ny, nz = normals_npy[:,:,0], normals_npy[:,:,1], normals_npy[:,:,2]
    
    # Increase the safety margin for nz
    nz_safe = np.where(np.abs(nz) < 0.1, 0.1 * np.sign(nz + 1e-9), nz)
    
    target_gx = -nx / nz_safe
    target_gy = -ny / nz_safe

    # --- CRITICAL SAFETY STEP ---
    # Replace any NaN or Inf with 0.0 before they hit the solver
    target_gx = np.nan_to_num(target_gx, nan=0.0, posinf=0.0, neginf=0.0)
    target_gy = np.nan_to_num(target_gy, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Optional: Clamp extreme gradients (stops 'depth spikes')
    target_gx = np.clip(target_gx, -5, 5)
    target_gy = np.clip(target_gy, -5, 5)
    # ----------------------------

    # 3. Build Linear System
    A = sp.lil_matrix((num_pixels, num_pixels))
    b = np.zeros(num_pixels)
    idx = np.arange(num_pixels).reshape(H, W)

    for y in range(H):
        for x in range(W):
            curr_idx = idx[y, x]
            
            # IF BACKGROUND: Keep depth at a fixed background value (e.g., 0)
            if fg_binary[y, x] == 0:
                A[curr_idx, curr_idx] = 1
                b[curr_idx] = 0 
                continue

            # IF FOREGROUND & RELIABLE (hole_mask == 0): Anchor to Marigold depth
            if hole_mask_npy[y, x] == 0:
                A[curr_idx, curr_idx] = 1
                b[curr_idx] = depth_npy[y, x]
            
            # IF FOREGROUND & HOLE (hole_mask == 1): Solve via Poisson
            else:
                # Standard 4-neighbor Laplacian for internal pixels
                if 0 < y < H-1 and 0 < x < W-1:
                    A[curr_idx, idx[y, x+1]] = 1
                    A[curr_idx, idx[y, x-1]] = 1
                    A[curr_idx, idx[y+1, x]] = 1
                    A[curr_idx, idx[y-1, x]] = 1
                    A[curr_idx, curr_idx] = -4
                    
                    div = (target_gx[y, x+1] - target_gx[y, x-1]) / 2.0 + \
                          (target_gy[y+1, x] - target_gy[y-1, x]) / 2.0
                    b[curr_idx] = div
                else:
                    # Fallback for edge-of-frame pixels
                    A[curr_idx, curr_idx] = 1
                    b[curr_idx] = depth_npy[y, x]

    # 4. Solve
    A = A.tocsr()
    healed_depth = spsolve(A, b).reshape(H, W)
    return healed_depth

# --- EXECUTION ---
# Load your 4th input: the full_body_mask.jpg
fg_mask = cv2.imread("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/mask_8141d04d-6885-421b-830d-031ad3b0bd3f.jpg")


# Load your 3 finalized inputs
depth = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/raw_depth_marigold/masked_depth_marigold/depth_masked_no_halo.npy")

#traspose normals to (H, W, 3) if needed
normals = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/stabilized_normals/stabilized_normals.npy")
#normals = np.transpose(normals, (1, 2, 0))

print("Loaded depth and normals. Shapes:", depth.shape, normals.shape)


hole_mask = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/holes_mask/hole_mask.npy")

# Run the body-aware healing
healed_depth = heal_depth_with_poisson_and_fg(depth, normals, hole_mask, fg_mask)



healed_depth_save_path = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/healed_depth/stabilized_normals_healed_depth.npy"
np.save(healed_depth_save_path, healed_depth)

print(f"Healed Depth Min: {healed_depth.min()}")
print(f"Healed Depth Max: {healed_depth.max()}")

