import OpenEXR
import Imath
import numpy as np
import cv2
import os
from scipy.ndimage import binary_erosion


class MetricDepthCleaner:
    """
    Cleans EXR metric depth using a binary mask and distance filtering.
    Outputs:
        - Cleaned metric depth (.npy)
        - Foreground-normalized depth (.npy)
    """

    def __init__(
        self,
        exr_path: str,
        mask_path: str,
        save_clean_path: str,
        save_norm_path: str,
        alpha_threshold: int = 150,
        erosion_iter: int = 2,
        max_valid_depth: float = 50.0,
        metic_provider=None,
    ):
        self.exr_path = exr_path
        self.mask_path = mask_path
        self.save_clean_path = save_clean_path
        self.save_norm_path = save_norm_path
        self.alpha_threshold = alpha_threshold
        self.erosion_iter = erosion_iter
        self.max_valid_depth = max_valid_depth
        self.metric_provider = metic_provider

        self.depth = None
        self.mask = None
        self.final_mask = None

    # ---------------------------------------------------
    # Public API
    # ---------------------------------------------------

    def run(self):
        print(f"{self.metric_provider} Running Metric Depth Cleaning Pipeline... ")

        self._load_exr()
        self._load_mask()
        self._build_final_mask()

        depth_cleaned = self._apply_mask()
        depth_normalized = self._normalize_foreground(depth_cleaned)

        self._save_outputs(depth_cleaned, depth_normalized)

        print("Pipeline completed successfully.")

    # ---------------------------------------------------
    # Core Steps
    # ---------------------------------------------------

    def _load_exr(self):
        """Load EXR depth channel."""
        exr = OpenEXR.InputFile(self.exr_path)
        dw = exr.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr.channel("V", FLOAT)

        self.depth = np.frombuffer(depth_str, dtype=np.float32).reshape(h, w)

        print(f"Loaded EXR depth: {self.depth.shape}")

    def _load_mask(self):
        """Load, threshold, and optionally erode mask."""
        mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Mask not found at {self.mask_path}")

        mask_bin = mask_img > self.alpha_threshold

        if self.erosion_iter > 0:
            mask_bin = binary_erosion(mask_bin, iterations=self.erosion_iter)

        # Resize to match depth resolution if needed
        if mask_bin.shape != self.depth.shape:
            h, w = self.depth.shape
            mask_bin = cv2.resize(
                mask_bin.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        self.mask = mask_bin
        print("Mask loaded and processed.")

    def _build_final_mask(self):
        """Combine person mask with valid depth filtering."""
        depth_filter = (self.depth > 0) & (self.depth < self.max_valid_depth)
        self.final_mask = self.mask & depth_filter

        kept = np.sum(self.final_mask)
        print(f"Final mask built. Kept pixels: {kept}")

    def _apply_mask(self):
        """Apply final mask to depth."""
        depth_cleaned = np.where(self.final_mask, self.depth, 0.0)

        self._log_depth_stats(depth_cleaned, name="Cleaned")

        return depth_cleaned

    def _normalize_foreground(self, depth_cleaned):
        """Min-max normalize foreground only."""
        depth_normalized = np.zeros_like(depth_cleaned)

        fg_values = depth_cleaned[self.final_mask]

        if len(fg_values) == 0:
            print("Warning: No foreground pixels found.")
            return depth_normalized

        fg_min = fg_values.min()
        fg_max = fg_values.max()

        if fg_max > fg_min:
            depth_normalized[self.final_mask] = (
                (fg_values - fg_min) / (fg_max - fg_min)
            )
        else:
            depth_normalized[self.final_mask] = 1.0

        print(
            f"Normalization complete using FG Min: {fg_min:.4f}, "
            f"Max: {fg_max:.4f}"
        )

        return depth_normalized

    # ---------------------------------------------------
    # Utilities
    # ---------------------------------------------------

    def _save_outputs(self, depth_cleaned, depth_normalized):
        os.makedirs(os.path.dirname(self.save_clean_path), exist_ok=True)

        np.save(self.save_clean_path, depth_cleaned)
        np.save(self.save_norm_path, depth_normalized)

        print(f"Saved cleaned depth to: {self.save_clean_path}")
        print(f"Saved normalized depth to: {self.save_norm_path}")

    def _log_depth_stats(self, depth, name="Depth"):
        non_zero = depth[depth > 0]

        if len(non_zero) == 0:
            print(f"{name} depth: All values masked out.")
            return

        print(f"{name} depth range: {non_zero.min():.4f} → {non_zero.max():.4f}")
        print(f"{name} median depth: {np.median(non_zero):.4f}")
