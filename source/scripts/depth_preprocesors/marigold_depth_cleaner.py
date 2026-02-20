import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter
import os


class MarigoldMaskedDepthCleaner:
    """
    Cleans depth maps using an external mask image and removes halo artifacts.

    Produces:
        - Cleaned depth (.npy)
        - Foreground-normalized depth (.npy)
        - 16-bit visualization PNG
    """

    def __init__(
        self,
        depth_npy,
        mask_image,
        save_clean_npy,
        save_norm_npy,
        save_vis_png,
        mask_threshold=150,
        erosion_iter=2,
        gaussian_sigma=0.3,
    ):
        self.depth_npy = depth_npy
        self.mask_image = mask_image
        self.save_clean_npy = save_clean_npy
        self.save_norm_npy = save_norm_npy
        self.save_vis_png = save_vis_png

        self.mask_threshold = mask_threshold
        self.erosion_iter = erosion_iter
        self.gaussian_sigma = gaussian_sigma

        self.depth = None
        self.mask = None

    # ---------------------------------------------------
    # Public API
    # ---------------------------------------------------

    def run(self):
        print("Running Marigold Masked Depth Cleaner...")

        self._load_depth()
        self._load_mask()

        clean_depth = self._apply_mask()
        norm_depth = self._normalize_foreground(clean_depth)

        self._save_outputs(clean_depth, norm_depth)
        self._save_visualization(norm_depth)

        print("Pipeline complete.")

    # ---------------------------------------------------
    # Core Steps
    # ---------------------------------------------------

    def _load_depth(self):
        self.depth = np.load(self.depth_npy).astype(np.float32)
        print(f"Depth loaded: {self.depth.shape}")

    def _load_mask(self):
        """
        Loads mask directly from file (grayscale or binary).
        """
        mask_img = Image.open(self.mask_image).convert("L")
        mask_np = np.array(mask_img)

        mask_bin = mask_np > self.mask_threshold

        if self.erosion_iter > 0:
            mask_bin = binary_erosion(mask_bin, iterations=self.erosion_iter)

        mask = mask_bin.astype(np.float32)

        if self.gaussian_sigma > 0:
            mask = gaussian_filter(mask, sigma=self.gaussian_sigma)

        # Safety: ensure same resolution
        if mask.shape != self.depth.shape:
            raise ValueError("Mask and depth resolution mismatch.")

        self.mask = mask

        print("Mask loaded and processed.")

    def _apply_mask(self):
        clean_depth = self.depth * self.mask
        self._log_stats(clean_depth, "Cleaned")
        return clean_depth

    def _normalize_foreground(self, clean_depth):
        norm = np.zeros_like(clean_depth)

        fg = clean_depth[clean_depth > 0]

        if len(fg) == 0:
            print("Warning: No foreground detected.")
            return norm

        fg_min = fg.min()
        fg_max = fg.max()

        if fg_max > fg_min:
            norm[clean_depth > 0] = (fg - fg_min) / (fg_max - fg_min)
        else:
            norm[clean_depth > 0] = 1.0

        print(f"Normalized FG range: {fg_min:.4f} → {fg_max:.4f}")

        return norm

    # ---------------------------------------------------
    # Output
    # ---------------------------------------------------

    def _save_outputs(self, clean, norm):
        os.makedirs(os.path.dirname(self.save_clean_npy) or ".", exist_ok=True)

        np.save(self.save_clean_npy, clean)
        np.save(self.save_norm_npy, norm)

        print(f"Saved cleaned depth: {self.save_clean_npy}")
        print(f"Saved normalized depth: {self.save_norm_npy}")

    def _save_visualization(self, norm_depth):
        vis = np.clip(norm_depth, 0.0, 1.0)
        vis_16 = (vis * 65535).astype(np.uint16)
        Image.fromarray(vis_16).save(self.save_vis_png)

        print(f"Saved visualization: {self.save_vis_png}")

    # ---------------------------------------------------
    # Utils
    # ---------------------------------------------------

    def _log_stats(self, depth, name):
        fg = depth[depth > 0]

        if len(fg) == 0:
            print(f"{name}: all masked.")
            return

        print(f"{name} min: {fg.min():.4f}")
        print(f"{name} max: {fg.max():.4f}")
        print(f"{name} median: {np.median(fg):.4f}")
