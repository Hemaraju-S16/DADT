import numpy as np


depth = np.load("/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/healed_depth/healed_depth.npy")

print(depth.shape)
print(depth.min(), depth.max())