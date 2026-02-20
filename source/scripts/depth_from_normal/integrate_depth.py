# code modified from https://github.com/YertleTurtleGit/depth-from-normals
import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
from typing import Tuple, List, Union
import numba


def calculate_gradients(
    normals: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    horizontal_angle_map = np.arccos(np.clip(normals[:, :, 0], -1, 1))
    left_gradients = np.zeros(normals.shape[:2])
    left_gradients[mask != 0] = (1 - np.sin(horizontal_angle_map[mask != 0])) * np.sign(
        horizontal_angle_map[mask != 0] - np.pi / 2
    )

    vertical_angle_map = np.arccos(np.clip(normals[:, :, 1], -1, 1))
    top_gradients = np.zeros(normals.shape[:2])
    top_gradients[mask != 0] = -(1 - np.sin(vertical_angle_map[mask != 0])) * np.sign(
        vertical_angle_map[mask != 0] - np.pi / 2
    )

    return left_gradients, top_gradients


@numba.jit(nopython=True)
def integrate_gradient_field(
    gradient_field: np.ndarray, axis: int, mask: np.ndarray
) -> np.ndarray:
    heights = np.zeros(gradient_field.shape)

    for d1 in numba.prange(heights.shape[1 - axis]):
        sum_value = 0
        for d2 in range(heights.shape[axis]):
            coordinates = (d1, d2) if axis == 1 else (d2, d1)

            if mask[coordinates] != 0:
                sum_value = sum_value + gradient_field[coordinates]
                heights[coordinates] = sum_value
            else:
                sum_value = 0

    return heights


def calculate_heights(
    left_gradients: np.ndarray, top_gradients, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_heights = integrate_gradient_field(left_gradients, 1, mask)
    right_heights = np.fliplr(
        integrate_gradient_field(np.fliplr(-left_gradients), 1, np.fliplr(mask))
    )
    top_heights = integrate_gradient_field(top_gradients, 0, mask)
    bottom_heights = np.flipud(
        integrate_gradient_field(np.flipud(-top_gradients), 0, np.flipud(mask))
    )
    return left_heights, right_heights, top_heights, bottom_heights


def combine_heights(*heights: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(heights, axis=0), axis=0)


def rotate(matrix: np.ndarray, angle: float) -> np.ndarray:
    h, w = matrix.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    corners = cv.transform(
        np.array([[[0, 0], [w, 0], [w, h], [0, h]]]), rotation_matrix
    )[0]

    _, _, w, h = cv.boundingRect(corners)

    rotation_matrix[0, 2] += w / 2 - center[0]
    rotation_matrix[1, 2] += h / 2 - center[1]
    result = cv.warpAffine(matrix, rotation_matrix, (w, h), flags=cv.INTER_LINEAR)

    return result


def rotate_vector_field_normals(normals: np.ndarray, angle: float) -> np.ndarray:
    angle = np.radians(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotated_normals = np.empty_like(normals)
    rotated_normals[:, :, 0] = (
        normals[:, :, 0] * cos_angle - normals[:, :, 1] * sin_angle
    )
    rotated_normals[:, :, 1] = (
        normals[:, :, 0] * sin_angle + normals[:, :, 1] * cos_angle
    )

    return rotated_normals


def centered_crop(image: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
    return image[
        (image.shape[0] - target_resolution[0])
        // 2 : (image.shape[0] - target_resolution[0])
        // 2
        + target_resolution[0],
        (image.shape[1] - target_resolution[1])
        // 2 : (image.shape[1] - target_resolution[1])
        // 2
        + target_resolution[1],
    ]


def integrate_vector_field(
    vector_field: np.ndarray,
    mask: np.ndarray,
    target_iteration_count: int,
    thread_count: int,
) -> np.ndarray:
    shape = vector_field.shape[:2]
    angles = np.linspace(0, 90, target_iteration_count, endpoint=False)

    def integrate_vector_field_angles(angles: List[float]) -> np.ndarray:
        all_combined_heights = np.zeros(shape)

        for angle in angles:
            rotated_vector_field = rotate_vector_field_normals(
                rotate(vector_field, angle), angle
            )
            rotated_mask = rotate(mask, angle)

            left_gradients, top_gradients = calculate_gradients(
                rotated_vector_field, rotated_mask
            )
            (
                left_heights,
                right_heights,
                top_heights,
                bottom_heights,
            ) = calculate_heights(left_gradients, top_gradients, rotated_mask)

            combined_heights = combine_heights(
                left_heights, right_heights, top_heights, bottom_heights
            )
            combined_heights = centered_crop(rotate(combined_heights, -angle), shape)
            all_combined_heights += combined_heights / len(angles)

        return all_combined_heights

    with Pool(processes=thread_count) as pool:
        heights = pool.map(
            integrate_vector_field_angles,
            np.array(
                np.array_split(angles, thread_count),
                dtype=object,
            ),
        )
        pool.close()
        pool.join()

    isotropic_height = np.zeros(shape)
    for height in heights:
        isotropic_height += height / thread_count

    return isotropic_height


def estimate_height_map(
    normal_map: np.ndarray,
    mask: Union[np.ndarray, None] = None,
    height_divisor: float = 1,
    target_iteration_count: int = 250,
    thread_count: int = cpu_count(),
    raw_values: bool = False,
) -> np.ndarray:
    if mask is None:
        if normal_map.shape[-1] == 4:
            mask = normal_map[:, :, 3] / 255
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            print("Extracted mask from alpha channel.")
        else:
            mask = np.ones(normal_map.shape[:2], dtype=np.uint8)
            print("No mask provided, using default mask of all ones.")

    normals = ((normal_map[:, :, :3].astype(np.float64) / 255) - 0.5) * 2
    heights = integrate_vector_field(
        normals, mask, target_iteration_count, thread_count
    )

    if raw_values:
        print("called with Raw values true")
        valid = mask > 0
        print("true height scale is", heights.max(), heights.min())

        h_min = heights[valid].min()
        h_max = heights[valid].max()

        print("Raw height range:", h_min, h_max)

        #save depth map as numpy array
        np.save(os.path.join("source/depth_from_normal", "marigold_face_raw_depth.npy"), heights)
        print("Saved raw height map as numpy array.")
        # Avoid divide-by-zero
        if abs(h_max - h_min) < 1e-8:
            raise ValueError("Height map has near-zero variation")

        
        
        # Normalize to [0, 1]
        heights_norm = (heights - h_min) / (h_max - h_min)
        
        

        # Outside mask → mid-gray
        heights_norm[~valid] = 0.0 #0.5
        
        #save normlaized height map as numpy array
        np.save(os.path.join("source/depth_from_normal", "marigold_face_depth_norm.npy"), heights_norm)
        print("Saved normalized height map as numpy array.")

        # Convert to uint16
        heights_u16 = np.clip(
            heights_norm * 65535,
            0, 65535
        ).astype(np.uint16)
        
        from PIL import Image

        Image.fromarray(heights_u16, mode="I;16").save("integrated_marigold_face_depth.png")
        print("Saved normalized height map:")

        return heights

    heights /= height_divisor
    heights[mask > 0] += 1 / 2
    heights[mask == 0] = 1 / 2

    heights *= 2**16 - 1

    if np.min(heights) < 0 or np.max(heights) > 2**16 - 1:
        raise OverflowError("Height values are clipping.")

    heights = np.clip(heights, 0, 2**16 - 1)
    heights = heights.astype(np.uint16)

    return heights

import cv2

import os

def main():
    NORMAL_MAP_PATH = "/home/hemraj/vs_code_files/Deterministic_Analytical_Detail_Transfer/libs/Marigold/output/in-the-wild_example/normals_vis/face_cam_original_normals.png"
    SAVE_DIR = "source/depth_from_normal"
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Read normal map (preserve channels)
    normal_map = cv2.imread(NORMAL_MAP_PATH, cv2.IMREAD_UNCHANGED)
    if normal_map is None:
        raise FileNotFoundError(f"Failed to load normal map: {NORMAL_MAP_PATH}")

    # Convert BGR → RGB (important!)
    if normal_map.shape[-1] >= 3:
        normal_map[..., :3] = normal_map[..., :3][:, :, ::-1]

    # Estimate height map (raw floating values)
    height_map = estimate_height_map(
        normal_map,
        raw_values=True,
        thread_count=8,
        target_iteration_count=96
    ).astype(np.float32)
    
if __name__ == "__main__":
    main()
    # Save height map as 16-bit PNG
    
   

    
    



