import numpy as np
from joblib import Parallel, delayed
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def process_pixel(i, j, color, depth, factor, sigma_w, sigma_c, w):
    """
    Process a single pixel for the Joint Bilateral Upsampling.

    Parameters:
    - i, j: Indices of the pixel in the high-resolution image
    - color: 3D array of the high-resolution color image
    - depth: 2D array of the low-resolution depth map
    - factor: Upsampling factor
    - sigma_w: Spatial standard deviation
    - sigma_c: Range standard deviation
    - w: Half window size for the filter

    Returns:
    - Computed value for the result[i, j]
    """
    id = i / factor
    jd = j / factor
    
    lowHeight, lowWidth = depth.shape
    iMin = int(np.ceil(max(id - w, 0)))
    iMax = int(np.floor(min(id + w, lowHeight - 1)))
    jMin = int(np.ceil(max(jd - w, 0)))
    jMax = int(np.floor(min(jd + w, lowWidth - 1)))
    
    depth_sec = depth[iMin:iMax+1, jMin:jMax+1]
    color_sec = color[int(iMin * factor):int((iMax+1) * factor):factor, 
                      int(jMin * factor):int((jMax+1) * factor):factor, :]
    
    # Compute Gaussian range weights
    dR = color_sec[:, :, 0] - color[i, j, 0]
    dG = color_sec[:, :, 1] - color[i, j, 1]
    dB = color_sec[:, :, 2] - color[i, j, 2]
    range_weights = np.exp(-(dR**2 + dG**2 + dB**2) / (2 * sigma_c**2))
    
    # Compute Gaussian spatial weights
    iw = np.arange(iMin, iMax+1) - id
    jw = np.arange(jMin, jMax+1) - jd
    mx, my = np.meshgrid(jw, iw)
    spatial_weights = np.exp(-(mx**2 + my**2) / (2 * sigma_w**2))
    
    # Compute bilateral weights and depth sum
    depth_weights = (depth_sec > 0) * range_weights * spatial_weights
    depth_sum = depth_sec * depth_weights
    
    if np.sum(depth_weights) > 0:
        return np.sum(depth_sum) / np.sum(depth_weights)
    return 0

def joint_bilateral_upsample_parallel(color, depth, factor, sigma_w, sigma_c, w):
    """
    Parallelized Joint Bilateral Upsampling using joblib.

    Parameters:
    - color: 3D NumPy array of shape (highHeight, highWidth, 3)
    - depth: 2D NumPy array of shape (lowHeight, lowWidth)
    - factor: Upsampling factor
    - sigma_w: Spatial standard deviation
    - sigma_c: Range standard deviation
    - w: Half window size for the filter

    Returns:
    - result: 2D NumPy array of shape (highHeight, highWidth)
    """
    if color.shape[2] != 3:
        raise ValueError("Color data must have 3 channels")
    
    depth = depth.astype(np.float64)
    color = color.astype(np.float64)
    
    highHeight, highWidth, _ = color.shape
    
    # Parallel processing for each pixel
    result = Parallel(n_jobs=-1)(
        delayed(process_pixel)(i, j, color, depth, factor, sigma_w, sigma_c, w)
        for i in range(highHeight)
        for j in range(highWidth)
    )
    
    result = np.array(result).reshape((highHeight, highWidth))
    return result

def nearest_neighbor_interpolation(depth_map, view=True):
    # Set all nan values to 0
    depth_map = np.nan_to_num(depth_map)
    #depth_map[np.isnan(depth_map)] = 0

    if len(depth_map.shape) > 2:
        print(f"Converting depth map from {depth_map.shape} to single-channel.")
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Convert the depth map to 32-bit float if it's not already
    if depth_map.dtype != np.float32:
        print(f"Converting depth map from {depth_map.dtype} to 32-bit float.")
        depth_map = depth_map.astype(np.float32)
    print(depth_map)
    mask = depth_map == 0 
    mask = mask.astype(np.uint8)
    dense_depth_map = cv2.inpaint(depth_map, mask.astype(np.uint8), 3, cv2.INPAINT_NS)

    if view:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(depth_map, cmap='inferno')
        ax[0].set_title('Sparse Depth')
        ax[1].imshow(dense_depth_map, cmap='inferno')
        ax[1].set_title('Inpainted Depth')
        plt.show()
    return dense_depth_map

def preprocess_jbu(image, depth, factor=4):
    ''' Resize the high resolution image to valid size for JBU'''

    # Resize the image
    depth_height, depth_width = depth.shape[:2]
    new_height = depth_height * factor
    new_width = depth_width * factor
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return depth, image

if __name__ == "__main__":

    # Load the image
    img = cv2.imread('/home/admin-anedunga/Downloads/depth-upsampling/frame_00000.jpg', cv2.IMREAD_COLOR)
    original_depth = cv2.imread('/home/admin-anedunga/Downloads/depth-upsampling/depth_00000.png', cv2.IMREAD_UNCHANGED).astype(np.float64)

    # Upsample the depth map using Joint Bilateral Upsampling
    start = time.time()
    # JBU Parameters
    scale_factor = 4
    sigma_w = 1.0
    sigma_c = 0.1
    window_size = 3
    
    pre_processed_depth, pre_processed_image = preprocess_jbu(img, original_depth, scale_factor)
    upsampled_depth = joint_bilateral_upsample_parallel(pre_processed_image, pre_processed_depth, scale_factor, sigma_w, sigma_c, window_size)
    
    end = time.time() - start
    print(f"Upsampling took {end:.2f} seconds.")
    # Inpaint the depth map using nearest neighbor interpolation
    inpainted_depth = nearest_neighbor_interpolation(upsampled_depth, view=True)

    print("Original Depth Map:", original_depth.shape, original_depth.dtype, original_depth.min(), original_depth.max())
    print("Upsampled Depth Map:", inpainted_depth.shape, inpainted_depth.dtype, inpainted_depth.min(), inpainted_depth.max())

    # Plot with matpliotlib side by side
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')  
    axs[1].imshow(original_depth, cmap='inferno')
    axs[1].set_title('Original Depth')
    axs[1].axis('off')
    axs[2].imshow(upsampled_depth, cmap='inferno')
    axs[2].set_title('Upsampled Depth')
    axs[2].axis('off')
    axs[3].imshow(inpainted_depth, cmap='inferno')
    axs[3].set_title('Inpainted Depth')
    axs[3].axis('off')
    plt.show()