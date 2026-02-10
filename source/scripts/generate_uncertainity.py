from diffusers import MarigoldDepthPipeline
import torch
import cv2
import numpy as np
import os
from scipy.ndimage import binary_erosion

# ---------------- CONFIG ----------------
image_path = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/original_rgb.webp"
mask_path = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/input/mask_8141d04d-6885-421b-830d-031ad3b0bd3f.jpg"
uncertainty_npy_out = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_uncertainity_files/masked_uncertainity.npy"
uncertainty_png_out = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/source/marigold_uncertainity_files/masked_uncertainity.png"

# Mask Refinement Settings
ALPHA_THRESHOLD = 150  # Ignore "soft" edges
EROSION_ITER    = 3  # Cut 3 pixels into the silhouette to kill the halo
# --------------------------------------

# Ensure output directory exists
os.makedirs(os.path.dirname(uncertainty_npy_out), exist_ok=True)

pipe = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Load image
# input_image = cv2.imread(image_path)
# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
dark_gray = (64, 64, 64)

from PIL import Image
input_image = Image.open(image_path)
dark_gray = (64, 64, 64)
black_bg = Image.new("RGBA", input_image.size, (*dark_gray, 255))
input_image = Image.alpha_composite(black_bg, input_image).convert("RGB")



# Run Marigold
output = pipe(
    input_image,
    num_inference_steps=25,
    ensemble_size=8,
    output_uncertainty=True,
    generator=torch.Generator().manual_seed(42)
)

# Extract uncertainty
if not hasattr(output, "uncertainty") or output.uncertainty is None:
    raise RuntimeError("Uncertainty not returned. Ensure ensemble_size >= 3.")

uncertainty = output.uncertainty.squeeze(0).squeeze(-1).astype(np.float32)

# -----------------------------
# REFINED MASKING LOGIC
# -----------------------------
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 1. Strict threshold to remove fuzzy boundaries
mask_bin = mask_img > ALPHA_THRESHOLD

# 2. Erode the mask to move the edge inward
if EROSION_ITER > 0:
    mask_clean = binary_erosion(mask_bin, iterations=EROSION_ITER)
else:
    mask_clean = mask_bin

# Match resolution if needed
if mask_clean.shape != uncertainty.shape:
    mask_clean = cv2.resize(mask_clean.astype(np.uint8), (uncertainty.shape[1], uncertainty.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

# 3. Apply Clean Mask
uncertainty_masked = uncertainty * mask_clean.astype(np.float32)

# Save RAW masked uncertainty
np.save(uncertainty_npy_out, uncertainty_masked)

# -----------------------------
# Visualization
# -----------------------------
# Normalizing only based on non-zero pixels for better contrast
valid_pixels = uncertainty_masked[uncertainty_masked > 0]
if len(valid_pixels) > 0:
    vmin, vmax = valid_pixels.min(), valid_pixels.max()
    vis = np.where(uncertainty_masked > 0, (uncertainty_masked - vmin) / (vmax - vmin + 1e-8) * 255, 0)
    cv2.imwrite(uncertainty_png_out, vis.astype(np.uint8))
else:
    cv2.imwrite(uncertainty_png_out, np.zeros_like(uncertainty_masked).astype(np.uint8))

print("Masked uncertainty saved to:", uncertainty_npy_out)
print("Visualization saved to:", uncertainty_png_out)