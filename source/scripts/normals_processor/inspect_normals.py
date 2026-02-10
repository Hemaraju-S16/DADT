import numpy as np
import cv2

normals = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/masked_normals/normals_masked.npy")   # C,H,W
nz = normals[2]                           # Z channel

vis = (nz + 1) * 0.5
cv2.imwrite("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_normals_files/masked_normals/nz.png", (vis*255).astype(np.uint8))