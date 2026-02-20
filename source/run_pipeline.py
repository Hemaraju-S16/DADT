from scripts.depth_preprocesors.metric_depth_cleaner import MetricDepthCleaner


unique_depth_cleaner = MetricDepthCleaner(
    exr_path ="source/intermediate_inputs/unique_metric_depth.exr",
    mask_path="source/intermediate_inputs/mask_unique_face.png",
    save_clean_path="source/intermediate_outputs/unique_metric_depth_cleaned.npy",
    save_norm_path="source/intermediate_outputs/unique_metric_depth_normalized.npy",
    alpha_threshold=150,
    erosion_iter=2,
    max_valid_depth=50.0,
    metic_provider="Unique Depth",
)

course_mesh_cleaner = MetricDepthCleaner(
    exr_path = "source/intermediate_inputs/course_mesh_metric_depth.exr",
    mask_path="source/intermediate_inputs/mask_course_mesh.png",
    save_clean_path="source/intermediate_outputs/course_mesh_metric_depth_cleaned.npy",
    save_norm_path="source/intermediate_outputs/course_mesh_metric_depth_normalized.npy",
    alpha_threshold=150,
    erosion_iter=2,     
    max_valid_depth=50.0,
    metic_provider="Course Mesh Depth",
)

unique_depth_cleaner.run()
course_mesh_cleaner.run()


# Generate the delta
from scripts.delta_generator.relative_delta_gen import generatre_delta
## using hardcoded paths inside the function for now, can refactor to take args later
generatre_delta()


#convert delta to exr with falloff
from scripts.delta_post_processer.post_process_delta import save_delta_as_exr
mask_path = "source/intermediate_inputs/mask_course_mesh.png"
delta_npy_path = "source/intermediate_outputs/delta_files/face_cam_delta_depth.npy"
delta_brsh_save_path = "source/intermediate_outputs/sculpt_delta_brush"

save_delta_as_exr(mask_path, delta_npy_path, output_exr_path=delta_brsh_save_path ,fade_width_pct=0.63, bottom_fade_pct=0.55)


