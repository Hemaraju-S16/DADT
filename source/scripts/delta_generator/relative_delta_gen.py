import numpy as np
import cv2
import os



# ---------------- paths ----------------

## working with face camera
def generatre_delta():
    depth_unique = np.load("source/intermediate_outputs/unique_metric_depth_normalized.npy")
    mask_unique = cv2.imread("source/intermediate_inputs/mask_unique_face.png", 0) > 150
    print("unique raw shape is", depth_unique.shape)

    depth_mesh = np.load("source/intermediate_outputs/course_mesh_metric_depth_cleaned.npy")
    mask_mesh = cv2.imread("source/intermediate_inputs/mask_course_mesh.png", 0) > 150
    print("mesh depth shape is", depth_mesh.shape)






    save_dir = "source/intermediate_outputs/delta_files"

    os.makedirs(save_dir, exist_ok=True)


    # ---------------- Resize logic ----------------
    # We target the smaller resolution (usually the mesh render)
    target_h, target_w = depth_mesh.shape

    if depth_unique.shape != depth_mesh.shape:
        print(f"Resizing Marigold ({depth_unique.shape}) to match Mesh ({depth_mesh.shape})")
        # INTER_NEAREST or INTER_AREA is best for depth to avoid halo artifacts
        depth_unique = cv2.resize(depth_unique, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # -------------------------------------

    # Intersection foreground
    fg = mask_mesh & mask_unique

    # Interior mask for stats
    kernel = np.ones((2,2), np.uint8)
    fg_inner = cv2.erode(fg.astype(np.uint8), kernel).astype(bool)

    # Also require valid depths
    fg_inner = fg_inner & (depth_unique > 0) & (depth_mesh > 0)

    # ---------------- normalize Marigold ----------------
    # if True: # If Marigold is not already normalized to 0-1, do it here. Otherwise skip.
    #     m_vals = depth_mari[fg_inner]
    #     m_min, m_max = m_vals.min(), m_vals.max()

    #     depth_mari_rel = np.zeros_like(depth_mari, dtype=np.float32)
    #     depth_mari_rel[fg] = (depth_mari[fg] - m_min) / (m_max - m_min + 1e-8)
    #     depth_mari_rel = np.clip(depth_mari_rel, 0.0, 1.0)
    # If depth_mari is already 0-1 and masked:




    # ---------------- normalize unique ----------------
    # We assume it's already 0-1, so we skip re-stretching the values,
    # but we MUST re-apply the mask to ensure a clean background.
    depth_unique_rel = depth_unique.copy().astype(np.float32)

    # Ensure background is PURE black (0.0) based on the combined mask
    depth_unique_rel[~fg] = 0.0

    # Optional: Ensure no stray values above 1.0 from the resize/interpolation
    depth_unique_rel = np.clip(depth_unique_rel, 0.0, 1.0)


    #flip polarity so that 1.0 is Near and 0.0 is Far, to match the Mesh convention
    depth_unique_rel[fg] = 1.0 - depth_unique_rel[fg]

    np.save(os.path.join(save_dir, "unique_relative_normalised.npy"), depth_unique_rel)





    # ---------------- normalize Mesh ----------------
    g_vals = depth_mesh[fg_inner]
    print("Mesh raw depth min/max in foreground:", g_vals.min(), g_vals.max())
    g_min, g_max = g_vals.min(), g_vals.max()

    depth_mesh_rel = np.zeros_like(depth_mesh, dtype=np.float32)
    depth_mesh_rel[fg] = (depth_mesh[fg] - g_min) / (g_max - g_min + 1e-8)



    # --- CRITICAL FIX: Flip Mesh Polarity to match unique ---
    # Now 1.0 will be Near and 0.0 will be Far for the Mesh too
    depth_mesh_rel[fg] = 1.0 - depth_mesh_rel[fg]



    depth_mesh_rel = np.clip(depth_mesh_rel, 0.0, 1.0)
    #save relative normalised mesh depth for debugging
    np.save(os.path.join(save_dir, "mesh_relative_normalised.npy"), depth_mesh_rel)



    # ---------------- delta depth ----------------
    delta =  depth_unique_rel - depth_mesh_rel



    ############### main delta is saved as face_cam_delta_depth.npy  ###################    
    delta[~fg] = 0.0
    print(delta.shape, delta.dtype)
    np.save(os.path.join(save_dir, "face_cam_delta_depth.npy"), delta)


    #save clipedd delta for debugging
    delta_clipped = np.clip(delta, -0.3, 1.0)
    #np.save(os.path.join(save_dir, "delta_depth_clipped.npy"), delta_clipped)
    #print(f"cliped delta min/max: {delta_clipped[fg].min()}, {delta_clipped[fg].max()}")
    # visualization
    vis = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(save_dir, "face_cam_delta_depth.png"), vis.astype(np.uint8))

    print("Marigold rel min/max:", depth_unique_rel[fg].min(), depth_unique_rel[fg].max())
    print("Mesh rel min/max:", depth_mesh_rel[fg].min(), depth_mesh_rel[fg].max())
    print("Delta min/max:", delta[fg].min(), delta[fg].max())


    ###############  story board   ###################
    import matplotlib.pyplot as plt

    def norm01(x):
        m, M = np.nanmin(x), np.nanmax(x)
        return (x - m) / (M - m + 1e-8)

    # ---------- Absolute Delta ----------
    abs_delta = np.abs(delta)
    abs_delta[~fg] = 0

    # ---------- Signed Delta (bipolar) ----------
    v = np.percentile(np.abs(delta[fg]), 98)
    signed = np.clip(delta, -v, v)

    # ---------- High-pass Delta (detail noise) ----------
    delta_lp = cv2.GaussianBlur(delta, (41,41), 0)
    delta_hp = delta - delta_lp
    delta_hp[~fg] = 0

    # ---------- Plot Grid ----------
    plt.figure(figsize=(16,12))

    # Row 1
    plt.subplot(3,3,1); plt.title("Raw Marigold")
    plt.imshow(norm01(depth_unique), cmap='gray'); plt.axis('off')

    plt.subplot(3,3,2); plt.title("Raw Mesh Depth")
    plt.imshow(norm01(depth_mesh), cmap='gray'); plt.axis('off')

    plt.subplot(3,3,3); plt.title("Foreground Mask")
    plt.imshow(fg, cmap='gray'); plt.axis('off')

    # Row 2
    plt.subplot(3,3,4); plt.title("Marigold Relative")
    plt.imshow(depth_unique_rel, cmap='gray'); plt.axis('off')

    plt.subplot(3,3,5); plt.title("Mesh Relative")
    plt.imshow(depth_mesh_rel, cmap='gray'); plt.axis('off')

    plt.subplot(3,3,6); plt.title("Absolute Delta")
    plt.imshow(norm01(abs_delta), cmap='hot'); plt.axis('off')

    # Row 3
    plt.subplot(3,3,7); plt.title("Signed Delta (Blue↔Red)")
    plt.imshow(signed, cmap='bwr', vmin=-v, vmax=v); plt.axis('off')

    plt.subplot(3,3,8); plt.title("High-Pass Delta (Noise Layer)")
    plt.imshow(norm01(delta_hp), cmap='hot'); plt.axis('off')

    plt.subplot(3,3,9); plt.title("Delta Histogram")
    plt.hist(delta_hp[fg].flatten(), bins=200)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "DELTA_STORYBOARD.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved delta storyboard:", out_path)

