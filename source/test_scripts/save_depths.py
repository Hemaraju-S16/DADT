import numpy as np
import cv2

import os


def save_depth_png(npy_path, png_path):
    depth = np.load(npy_path).astype(np.float32)
    print(depth.min())
    print(depth.max())

    valid = np.isfinite(depth) & (depth > 0)
    dmin, dmax = depth[valid].min(), depth[valid].max()

    depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)
    depth_norm[~valid] = 0.0

    depth_16 = (depth_norm * 65535).astype(np.uint16)

    cv2.imwrite(png_path, depth_16)
    print(f"✅ Saved PNG depth: {png_path}")
    


def save_depth_png_p3d(npy_path, png_path):
    depth = np.load(npy_path).astype(np.float32)
    print(depth.min())
    print(depth.max())
    # valid = rendered pixels only
    valid = np.isfinite(depth) & (depth > 0)

    dmin, dmax = depth[valid].min(), depth[valid].max()

    # normalize to [0,1]
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)

    # invert for Blender (near = white)
    depth_disp = 1.0 - depth_norm

    depth_disp[~valid] = 0.0
    
    #inverse depth to 16 bit
    
    depth_16 = (depth_disp * 65535).astype(np.uint16)
    #invert depth for saving
   
    cv2.imwrite(png_path, depth_16)

    print(f"✅ Saved P3D displacement PNG: {png_path}")
    
save_path = "source/test_scripts/depth_visualization_saves"
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_name = "test2.png"

save_depth_png("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/intermediate_outputs/course_mesh_metric_depth_cleaned.npy",
               f"{save_path}/{save_name}")
    
