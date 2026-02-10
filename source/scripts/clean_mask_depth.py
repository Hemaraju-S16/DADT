import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter

# ────────────────────────────────────────────────
#               CONFIG - EDIT PATHS ONLY
# ────────────────────────────────────────────────
ORIGINAL_DEPTH_NPY   = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/raw_depth_marigold/original_rgb_depth.npy"
TRANSPARENT_PNG      = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/mask_8141d04d-6885-421b-830d-031ad3b0bd3f.jpg"
OUTPUT_CLEAN_NPY     = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/raw_depth_marigold/masked_depth_marigold/depth_masked_no_halo.npy"
OUTPUT_VIS_PNG       = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_depth_files/raw_depth_marigold/masked_depth_marigold/depth_masked_no_halo_vis.png"

# Edge Cleaning Settings
EROSION_ITER         = 2      
ALPHA_THRESHOLD      = 150    
GAUSSIAN_SIGMA       = 0.3  

# ────────────────────────────────────────────────
#                  Main processing
# ────────────────────────────────────────────────

print("Loading files...")
depth = np.load(ORIGINAL_DEPTH_NPY).astype(np.float32)

# 1. Load alpha and create mask
img = Image.open(TRANSPARENT_PNG).convert("RGBA")
# Extract alpha channel to ensure we have a clean mask
alpha = np.array(img.split()[3]) 
mask_bin = alpha > ALPHA_THRESHOLD

# 2. Erosion (Removes background bleed/halos)
if EROSION_ITER > 0:
    mask_clean = binary_erosion(mask_bin, iterations=EROSION_ITER)
else:
    mask_clean = mask_bin

# 3. Mask Smoothing (Softens the edge of the cutout)
mask_final = mask_clean.astype(np.float32)
if GAUSSIAN_SIGMA > 0:
    mask_final = gaussian_filter(mask_final, sigma=GAUSSIAN_SIGMA)

# 4. NORMALIZATION (Foreground ONLY)
# We isolate the object pixels to find their specific depth range
person_pixels = depth[mask_clean] 

if len(person_pixels) > 0:
    d_min = person_pixels.min()
    d_max = person_pixels.max()
    # Apply Min-Max normalization based ONLY on foreground values
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
else:
    depth_norm = depth
    print("Warning: No foreground detected!")

# 5. Apply Mask (Pure Black Background)
# Foreground * Mask = Foreground. Background * 0 = 0.

BG_EPSILON = -0.003
clean_depth = np.where(mask_final > 0, depth_norm * mask_final, BG_EPSILON)
#clean_depth = depth_norm * mask_final

# 6. Save .npy Data (Masked and Normalized)
np.save(OUTPUT_CLEAN_NPY, clean_depth)
print(f"✅ Data Saved (Normalized 0-1, BG 0): {OUTPUT_CLEAN_NPY}")

# 7. Visualization logic (16-bit PNG)
# Mapping 0.0-1.0 to 0-65535
vis_16bit = (clean_depth * 65535).astype(np.uint16)
Image.fromarray(vis_16bit).save(OUTPUT_VIS_PNG)

print(f"✅ Image Saved (Pure Black BG): {OUTPUT_VIS_PNG}")
print(f"Stats: FG Min={clean_depth[mask_clean].min()}, FG Max={clean_depth[mask_clean].max()}, BG={clean_depth[~mask_clean].min()}")