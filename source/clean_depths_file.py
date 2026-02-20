from scripts.depth_preprocesors.metric_depth_cleaner import MetricDepthCleaner  
from scripts.depth_preprocesors.marigold_depth_cleaner import MarigoldMaskedDepthCleaner



mesh_depth_cleaner = MetricDepthCleaner(
    exr_path="source/inputs/face_metric_depth/face_cam.exr",
    mask_path="source/inputs/face_metric_depth/mask_course_mesh_face_cam.jpg",
    save_clean_path="source/outputs/metric/course_mesh_face_depth_cleaned.npy",
    save_norm_path="source/outputs/metric/course_mesh_face_depth_normalized.npy",
    alpha_threshold=150,
    erosion_iter=2,
    max_valid_depth=50.0,
)


marigold_depth_cleaner = MarigoldMaskedDepthCleaner(
    depth_npy="source/depth_from_normal/marigold_face_depth_norm.npy", # using integrated depth for better results
    mask_image="source/inputs/original_image/mask_face_cam_original.jpg",
    save_clean_npy="source/outputs/marigold/marigold_face_depth_cleaned.npy",
    save_norm_npy="source/outputs/marigold/marigold_face_depth_normalized.npy",
    save_vis_png="source/outputs/marigold/marigold_face_depth_vis.png",
    mask_threshold=190,
    erosion_iter=6,
    gaussian_sigma=0.3,
)





marigold_depth_cleaner.run()
mesh_depth_cleaner.run()
