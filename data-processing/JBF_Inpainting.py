import torch
import numpy as np
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
import time
from DepthMetrics import DepthMetrics

def rgbd_inpaint_parallel(color_image, depth_image, mask, max_window_size=5, n_jobs=-1):
    """
    Inpaint missing depth values using joint bilateral filtering with parallelization.

    Parameters:
    - color_image: 3D RGB color image (HxWx3, uint8).
    - depth_image: 2D depth map with missing values (HxW, float32).
    - mask: Binary mask of valid depth values (1 for valid, 0 for missing).
    - max_window_size: Maximum window size for the bilateral filter.
    - n_jobs: Number of parallel jobs for computation.

    Returns:
    - inpainted_depth: 2D inpainted depth map (HxW, float32).
    """
    color_image = color_image.astype(np.float32) / 255.0  # Normalize color image

    h, w = depth_image.shape

    def process_pixel(x, y):
        if mask[y, x] == 1:  # Skip valid depth pixels
            return depth_image[y, x]

        weight_sum = 0.0
        depth_sum = 0.0

        for dy in range(-max_window_size, max_window_size + 1):
            for dx in range(-max_window_size, max_window_size + 1):
                ny, nx = y + dy, x + dx

                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1:
                    spatial_weight = np.exp(-(dx**2 + dy**2) / (2 * (max_window_size / 3)**2))

                    color_diff = color_image[y, x] - color_image[ny, nx]
                    range_weight = np.exp(-np.sum(color_diff**2) / (2 * 0.1**2))

                    weight = spatial_weight * range_weight
                    weight_sum += weight
                    depth_sum += weight * depth_image[ny, nx]

        return depth_sum / weight_sum if weight_sum > 0 else 0.0

    # Use joblib to parallelize pixel-wise processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel)(x, y) for y in range(h) for x in range(w)
    )

    return np.array(results).reshape(h, w)



def rgbd_inpaint(color_image, depth_image, mask, max_window_size=5):
    """
    Inpaint missing depth values using joint bilateral filtering.

    Parameters:
    - color_image: 3D RGB color image (HxWx3, uint8).
    - depth_image: 2D depth map with missing values (HxW, float32).
    - mask: Binary mask of valid depth values (1 for valid, 0 for missing).
    - max_window_size: Maximum window size for the bilateral filter.

    Returns:
    - inpainted_depth: 2D inpainted depth map (HxW, float32).
    """
    # Ensure inputs are in the correct format
    color_image = color_image.astype(np.float32) / 255.0  # Normalize color image
    inpainted_depth = depth_image.copy()

    h, w = depth_image.shape

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1:  # Skip valid depth pixels
                continue

            # Initialize weights and depth sums
            weight_sum = 0.0
            depth_sum = 0.0

            for dy in range(-max_window_size, max_window_size + 1):
                for dx in range(-max_window_size, max_window_size + 1):
                    ny, nx = y + dy, x + dx

                    # Check bounds
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1:
                        # Compute spatial weight
                        spatial_weight = np.exp(-(dx**2 + dy**2) / (2 * (max_window_size / 3)**2))

                        # Compute range weight
                        color_diff = color_image[y, x] - color_image[ny, nx]
                        range_weight = np.exp(-np.sum(color_diff**2) / (2 * 0.1**2))

                        # Total weight
                        weight = spatial_weight * range_weight
                        weight_sum += weight
                        depth_sum += weight * depth_image[ny, nx]

            # Assign inpainted value
            if weight_sum > 0:
                inpainted_depth[y, x] = depth_sum / weight_sum
            else:
                inpainted_depth[y, x] = 0.0  # Default fallback for isolated pixels

    return inpainted_depth


def display_depth_in_color(depth_image):
    """
    Visualize depth map as a colorized image using OpenCV.
    """
    depth_min = np.min(depth_image[depth_image > 0])
    depth_max = np.max(depth_image)
    depth_image_normalized = (depth_image - depth_min) / (depth_max - depth_min)
    depth_colorized = cv2.applyColorMap((depth_image_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return depth_colorized

if __name__ == "__main__":
    VIZ = 1
    # Load the image
    image_path1 = '/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/00000-cam1_rgb.png'
    depth_path1 = '/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/depth_img0_cam1_5000.tiff'
   
    image_path2 = '/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/00000-cam3.png'
    depth_path2 = '/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/depth_img0_cam3_2500.tiff'
    depth_path3 = '/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/depth_img0_cam3_5000.tiff'
    ## Depth Downsampling Start
    # Load the image
    image = cv2.imread(image_path2, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path3, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    start = time.time()
    print("Input Shape", image.shape, depth.shape)
    # Target dimensions
    target_width, target_height = 768, 1024
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    print("After Reshaping", image.shape, depth.shape)

    # resize image and depth using inter nearest
    # image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    # original_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Create a mask for missing depth values (assume 0 indicates missing pixels)
    mask = (depth > 0).astype(np.uint8)

    # Perform joint bilateral filtering for inpainting
    inpainted_depth = rgbd_inpaint_parallel(image, depth, mask)
    inpainted_depth = inpainted_depth.astype(np.float32)
    # Refine with Tealea
    new_mask = (inpainted_depth == 0).astype(np.uint8)
    print("Mask/Inpainted Shape", new_mask.shape, inpainted_depth.shape, inpainted_depth.dtype)
    telea = cv2.inpaint(inpainted_depth, new_mask, 3, cv2.INPAINT_NS) 
    print("Time taken", time.time() - start)

    if VIZ:
        #plot
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        axs[1].imshow(depth, cmap='inferno')
        axs[1].set_title('Original Depth')
        axs[1].axis('off')
        axs[2].imshow(inpainted_depth, cmap='inferno')
        axs[2].set_title('Inpainted Depth')
        axs[2].axis('off')
        axs[3].imshow(telea, cmap='inferno')
        axs[3].set_title('Inpainting Refinement')
        axs[3].axis('off')

        plt.show()

        DepthMetrics_inpainted = DepthMetrics(depth, inpainted_depth)
        DepthMetrics_refined = DepthMetrics(depth, telea)
        are_inpainted = DepthMetrics_inpainted.absolute_relative_error()
        are_refined = DepthMetrics_refined.absolute_relative_error()

        print("ARE Inpainted", are_inpainted, "ARE Refined", are_refined)

        # save the figure
        #fig.savefig('/home/rog-nuc/Desktop/depth-upsampling/radius_test/guided-depth-completion/inpainted_depth5.png')
    # SIlogloss_inpainted = SiLogLoss(torch.tensor(inpainted_depth), torch.tensor(depth), torch.tensor(mask))
    # SIlogloss_refined = SiLogLoss(torch.tensor(telea), torch.tensor(depth), torch.tensor(mask))

    # print("SiLogLoss Inpainted", SIlogloss_inpainted)
    # print("SiLogLoss Refined", SIlogloss_refined)