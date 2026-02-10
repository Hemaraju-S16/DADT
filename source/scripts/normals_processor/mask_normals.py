import numpy as np
import cv2

# ---------------- CONFIG ----------------
normals_npy = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/original_rgb_normals.npy"   # shape (3,768,768)
mask_path = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/mask_8141d04d-6885-421b-830d-031ad3b0bd3f.jpg"

out_normals = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/masked_normals/normals_masked.npy"
out_vis = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/masked_normals/normals_masked_vis_original.png"
# --------------------------------------

# Load normals (C,H,W)
normals = np.load(normals_npy).astype(np.float32)

print("loaded normals min and max:", normals.min(), normals.max())

# Load mask (H,W)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask_bin = (mask > 10).astype(np.float32)

# Sanity check
assert normals.shape[1:] == mask_bin.shape

# Expand mask to (1,H,W)
mask3 = mask_bin[None, :, :]

# -----------------------------
# 1. Apply mask
# -----------------------------
normals_masked = normals * mask3

# -----------------------------
# 2. Renormalize normals inside mask
# -----------------------------
eps = 1e-6
norm = np.linalg.norm(normals_masked, axis=0, keepdims=True)
normals_masked = normals_masked / (norm + eps)

# Save masked normals
np.save(out_normals, normals_masked)
print(f"masked normals min and max: {normals_masked.min()}, {normals_masked.max()}")

# -----------------------------
# 3. Visualization (convert to RGB)
# -----------------------------
# Normals assumed in [-1,1] → map to [0,255]
vis = normals_masked.transpose(1,2,0)   # HWC
vis = (vis + 1.0) * 0.5                 # -> [0,1]
vis = np.clip(vis, 0, 1)
vis = (vis * 255).astype(np.uint8)

cv2.imwrite(out_vis, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

print("Saved masked normals:", out_normals)
print("Saved visualization:", out_vis)