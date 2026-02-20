import numpy as np
import open3d as o3d
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import os
import matplotlib.pyplot as plt


class MaskedHealer:

    def __init__(self, output_dir="stage4_out"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, msg):
        print(msg)

    def run(
        self,
        filled_depth_path,
        filled_normal_path,
        mask_path,
        fg_min=1.65,
        fg_max=1.77,
        lambda_fidelity=0.9
    ):

        self.log(f"Running Metric Poisson Healing (Lambda={lambda_fidelity})...")

        # -------------------------------------------------
        # Load depth + mask
        # -------------------------------------------------

        D_norm = np.load(filled_depth_path).astype(np.float64)

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask_img > 127

        H, W = D_norm.shape

        # -------------------------------------------------
        # Re-inject metric
        # -------------------------------------------------

        D_init = np.zeros_like(D_norm)
        D_init[mask] = D_norm[mask] * (fg_max - fg_min) + fg_min

        # -------------------------------------------------
        # Load normals RGB
        # -------------------------------------------------

        normal_img = cv2.imread(filled_normal_path)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB).astype(np.float64)

        N = (normal_img / 255.0) * 2.0 - 1.0
        N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-8
        N[~mask] = 0

        nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]

        # -------------------------------------------------
        # Slopes
        # -------------------------------------------------

        pixel_pitch = 1.1 / max(H, W)

        nz_safe = np.where(np.abs(nz) < 0.01, 0.01 * np.sign(nz + 1e-6), nz)

        Sx = -nx / nz_safe
        Sy = -ny / nz_safe

        Sx[~np.isfinite(Sx)] = 0
        Sy[~np.isfinite(Sy)] = 0

        Sx *= pixel_pitch
        Sy *= pixel_pitch

        div_S = np.gradient(Sx, axis=1) + np.gradient(Sy, axis=0)
        div_S[~np.isfinite(div_S)] = 0

        # -------------------------------------------------
        # Build Poisson system
        # -------------------------------------------------

        mask_ids = np.full((H, W), -1, np.int32)
        mask_ids[mask] = np.arange(np.sum(mask))

        rows, cols = np.where(mask)
        ids = mask_ids[rows, cols]

        b = div_S[mask] + lambda_fidelity * D_init[mask]

        A_r, A_c, A_v = [], [], []
        diag = np.full(len(ids), 4 + lambda_fidelity)

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rn = rows + dr
            cn = cols + dc

            valid = (rn>=0)&(rn<H)&(cn>=0)&(cn<W)
            rn, cn, cid = rn[valid], cn[valid], ids[valid]

            nid = mask_ids[rn, cn]
            var = nid != -1

            A_r.append(cid[var])
            A_c.append(nid[var])
            A_v.append(np.full(np.sum(var), -1))

            fixed = ~var
            b[cid[fixed]] += D_init[rn[fixed], cn[fixed]]

        A_r.append(ids)
        A_c.append(ids)
        A_v.append(diag)

        A = sp.csr_matrix(
            (np.concatenate(A_v),
            (np.concatenate(A_r), np.concatenate(A_c))),
            shape=(len(ids), len(ids))
        )

        self.log("Solving sparse system...")

        Z = spsolve(A, b)

        # -------------------------------------------------
        # Reconstruct
        # -------------------------------------------------

        D_healed = D_init.copy()
        D_healed[mask] = Z

        out = os.path.join(self.output_dir, "stage4_healed_depth.npy")
        np.save(out, D_healed.astype(np.float32))

        self.log(f"Saved: {out}")

        return out


MaskedHealer().run(
    "source/outputs/marigold/marigold_face_depth_normalized.npy",
    "libs/Marigold/output/in-the-wild_example/normals_vis/face_cam_original_normals.png",
    "source/inputs/original_image/mask_face_cam_original.jpg",
    fg_min=1.65,
    fg_max=1.77
)
