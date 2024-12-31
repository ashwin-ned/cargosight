import os
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import math
from joblib import Parallel, delayed

def nearest_neighbor_interpolation(depth_map, view=False):
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
    mask = depth_map == 0 
    mask = mask.astype(np.uint8)
    if view:
        plt.imshow(mask, cmap='gray')
    dense_depth_map = cv2.inpaint(depth_map, mask.astype(np.uint8), 2, cv2.INPAINT_NS)

    if view:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(depth_map, cmap='inferno')
        ax[0].set_title('Sparse Depth')
        ax[1].imshow(dense_depth_map, cmap='inferno')
        ax[1].set_title('Inpainted Depth')
        plt.show()
    return dense_depth_map

def downsample_nearest_neighbor(depth, target_width, target_height):
    """
    Downsample depth map using nearest-neighbor interpolation.
    """
    downsampled_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return downsampled_depth


def downsample_median_pooling(depth, target_width, target_height):
    """
    Downsample depth map using median pooling followed by nearest-neighbor interpolation.
    """
    # Compute the downsampling ratio
    h, w = depth.shape
    scale_x = w / target_width
    scale_y = h / target_height

    # Perform median pooling
    pooled_depth = np.zeros((target_height, target_width), dtype=np.float32)
    for i in range(target_height):
        for j in range(target_width):
            # Define the region in the original depth map
            x1 = int(j * scale_x)
            x2 = int((j + 1) * scale_x)
            y1 = int(i * scale_y)
            y2 = int((i + 1) * scale_y)
            
            # Compute the median value in this region
            region = depth[y1:y2, x1:x2]
            if region.size > 0:
                pooled_depth[i, j] = np.median(region)
    
    # plt.imshow(pooled_depth, cmap='inferno')
    # plt.show()
    # Upscale back to target resolution using nearest-neighbor interpolation
    #downsampled_depth = cv2.resize(pooled_depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    downsampled_depth = pooled_depth

    return downsampled_depth

def gauss(spatialKern, rangeKern):
    """
    Compute the Gaussian spatial and range kernels.
    """
    gaussianSpatial = 1 / math.sqrt(2 * math.pi * (spatialKern ** 2))
    gaussianRange = 1 / math.sqrt(2 * math.pi * (rangeKern ** 2))

    # Precompute range kernel matrix
    matrix = np.exp(-np.arange(256) ** 2 * gaussianRange)

    # Compute spatial kernel
    xx = -spatialKern + np.arange(2 * spatialKern + 1)
    yy = -spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx, yy)
    spatialGS = gaussianSpatial * np.exp(-(x ** 2 + y ** 2) / (2 * spatialKern ** 2))

    return matrix, spatialGS

def padImage(img, spatialKern):
    """
    Pad an image with symmetric reflections of itself.
    Works for both single-channel and multi-channel images.
    """
    if len(img.shape) == 2:  # Single-channel image
        img = np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern)), 'symmetric')
    else:  # Multi-channel image
        img = np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    return img

def process_pixel(x, y, paddedImg, paddedGuidance, matrix, spatialGS, spatialKern, ch):
    """
    Process a single pixel or channel in the joint bilateral filtering process.
    """
    if ch == 1:  # Single-channel processing
        neighbourhood = paddedGuidance[x-spatialKern:x+spatialKern+1, y-spatialKern:y+spatialKern+1]
        central = paddedGuidance[x, y]
        rangeKernel = matrix[np.abs(neighbourhood - central).astype(np.uint8)]
        combinedKernel = rangeKernel * spatialGS
        norm = np.sum(combinedKernel)
        result = np.sum(combinedKernel * paddedImg[x-spatialKern:x+spatialKern+1, y-spatialKern:y+spatialKern+1]) / norm
    else:  # Multi-channel processing
        result = []
        for i in range(ch):  # Iterate through each channel
            neighbourhood = paddedGuidance[x-spatialKern:x+spatialKern+1, y-spatialKern:y+spatialKern+1, i]
            central = paddedGuidance[x, y, i]
            rangeKernel = matrix[np.abs(neighbourhood - central).astype(np.uint8)]
            combinedKernel = rangeKernel * spatialGS
            norm = np.sum(combinedKernel)
            result.append(
                np.sum(combinedKernel * paddedImg[x-spatialKern:x+spatialKern+1, y-spatialKern:y+spatialKern+1, i]) / norm
            )
        result = np.array(result)
    return result

def jointBilateralFilter(img, guidance, spatialKern, rangeKern):
    """
    Apply joint bilateral filtering to an image using a guidance image.
    Supports single-channel (depth) and multi-channel (RGB) guidance images.
    """
    h, w = img.shape[:2]
    ch = 1 if len(img.shape) == 2 else img.shape[2]
    paddedImg = padImage(img, spatialKern)
    paddedGuidance = padImage(guidance, spatialKern)
    matrix, spatialGS = gauss(spatialKern, rangeKern)

    # Parallel processing with joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_pixel)(x, y, paddedImg, paddedGuidance, matrix, spatialGS, spatialKern, ch)
        for x in range(spatialKern, spatialKern + h)
        for y in range(spatialKern, spatialKern + w)
    )

    # Reshape results into the output image
    outputImg = np.zeros((h, w), dtype=img.dtype) if ch == 1 else np.zeros((h, w, ch), dtype=img.dtype)
    for idx, result in enumerate(results):
        x = idx // w + spatialKern
        y = idx % w + spatialKern
        if ch == 1:
            outputImg[x - spatialKern, y - spatialKern] = result
        else:
            outputImg[x - spatialKern, y - spatialKern, :] = result

    return outputImg

def joint_bilateral_downsampling(depth, guidance, target_width, target_height, spatialKern, rangeKern):
    """
    Downsample depth map using joint bilateral filtering.
    Supports single-channel or 2-channel depth maps.
    """
    # Ensure depth and guidance images have matching spatial dimensions
    if depth.shape[:2] != guidance.shape[:2]:
        raise ValueError("Depth and guidance images must have the same spatial dimensions.")

    # Normalize depth for better processing
    if len(depth.shape) == 2:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        for i in range(depth.shape[2]):
            depth_norm[:, :, i] = cv2.normalize(depth[:, :, i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply joint bilateral filter
    filtered_depth = jointBilateralFilter(depth_norm, guidance, spatialKern, rangeKern)

    # Downsample the filtered depth map
    downsampled_depth = cv2.resize(
        filtered_depth, 
        (target_width, target_height), 
        interpolation=cv2.INTER_NEAREST
    )

    return downsampled_depth
if __name__ == "__main__":
    NAVVIS_FLAG = 1
    # Load the image
    navvis_depth_folder = f'/media/admin-anedunga/Extreme Pro/NAVVIS/Schenker_00-00-01_HS_SF2/depth/'
    navvis_image_folder = f'/media/admin-anedunga/Extreme Pro/NAVVIS/Schenker_00-00-01_HS_SF2/undistorted/'
    output_folder = f'/media/admin-anedunga/Extreme Pro/NAVVIS_DATA_DOWNSAMPLED/Schenker_00-00-01_HS_SF2/'
    
    
    if NAVVIS_FLAG == 1:
        for file in os.listdir(navvis_depth_folder):
            if file.endswith('.tiff'):
                original_depth  = cv2.imread(os.path.join(navvis_depth_folder, file), cv2.IMREAD_UNCHANGED).astype(np.float32)

                depth_file = file.split('_')
                print(f'Reading depth {depth_file} ...')
                img_number = depth_file[1].replace('img', '')
                camera_number = depth_file[2]
                # format as 5 digit number
                img_number = f"{int(img_number):05d}"
                image_filename = img_number + '-' + camera_number 
                image_filename = image_filename.replace('tiff', 'png')
                print(f'Reading image {image_filename} ...')
                image = cv2.imread(os.path.join(navvis_image_folder, image_filename), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(camera_number)
                if camera_number == 'cam0.tiff':
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    original_depth = cv2.rotate(original_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    original_depth = cv2.rotate(original_depth, cv2.ROTATE_90_CLOCKWISE)

                ## Depth Downsampling Start

                # Target dimensions
                target_width, target_height = 768, 1024

                # Inpainting Before
                original_depth = nearest_neighbor_interpolation(original_depth)

                # Downsample using nearest-neighbor interpolation
                depth_nn = downsample_nearest_neighbor(original_depth, target_width, target_height)

                
                


                # save 
                #plt.imsave(f'{output_folder}navvis_{image_filename}original_depth.png', original_depth, cmap='inferno')
                image_filename = image_filename.split('.')[0]
                print("Depth Shape: ", depth_nn.shape, depth_nn.dtype)
                print("Original Depth", original_depth.max(), original_depth.min())
                print("Depth NN", depth_nn.max(), depth_nn.min())
                
                
                # Resize image
                image = cv2.resize(image, (target_width, target_height))
                
                cv2.imwrite(f'{output_folder}{image_filename}_depth.tiff', depth_nn)
                cv2.imwrite(f'{output_folder}{image_filename}_rgb.png', image)
                plt.imsave(f'{output_folder}{image_filename}_depth.png', depth_nn, cmap='inferno')

                

                # # plot using matplotlib
                # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                # ax[0].imshow(original_depth, cmap='inferno')
                # ax[0].set_title('Original Depth')
                # ax[0].axis('off')
                # ax[1].imshow(depth_nn, cmap='inferno')
                # ax[1].set_title('Nearest Neighbor')
                # ax[1].axis('off')
                # ax[2].imshow(image)
                # ax[2].set_title('Image')
                # ax[2].axis('off')
                # plt.show()


    if NAVVIS_FLAG == 0:

        for file in os.listdir(navvis_folder):
            if file.endswith('.tiff'):
                original_depth  = cv2.imread(os.path.join(navvis_folder, file), cv2.IMREAD_UNCHANGED).astype(np.float32)

                depth_file = file.split('_')
                print(f'Reading depth {depth_file} ...')
                img_number = depth_file[1].replace('img', '')
                camera_number = depth_file[2]
                # format as 5 digit number
                img_number = f"{int(img_number):05d}"
                image_filename = img_number + '-' + camera_number 
                image_filename = image_filename.replace('tiff', 'png')
                print(f'Reading image {image_filename} ...')
                image = cv2.imread(os.path.join(navvis_folder, image_filename), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(camera_number)
                if camera_number == 'cam0.tiff':
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    original_depth = cv2.rotate(original_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    original_depth = cv2.rotate(original_depth, cv2.ROTATE_90_CLOCKWISE)

                ## Depth Downsampling Start

                # Target dimensions
                target_width, target_height = 768, 1024

                # Inpainting Before
                original_depth = nearest_neighbor_interpolation(original_depth)

                # Downsample using nearest-neighbor interpolation
                depth_nn = downsample_nearest_neighbor(original_depth, target_width, target_height)

                # Downsample using bilateral filtering + average pooling
                depth_median_pool = downsample_median_pooling(original_depth, target_width, target_height)

                # Downsample using joint bilateral filtering
                SpatialKernal = 30
                RangeKernal = 20
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                depth_joint_bilateral = joint_bilateral_downsampling(original_depth, gray_image, target_width, target_height, SpatialKernal, RangeKernal)


                # # Save results
                #cv2.imwrite(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling/navvis_{image_filename}depth_nn.png', depth_nn.astype(np.uint16))
                #cv2.imwrite(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling/navvis_{image_filename}depth_median_pool.png', depth_median_pool.astype(np.uint16))
                image_filename = image_filename.split('.')[0]
                plt.imsave(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling_sf2/navvis_{image_filename}original_depth.png', original_depth, cmap='inferno')
                plt.imsave(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling_sf2/navvis_{image_filename}depth_nn.png', depth_nn, cmap='inferno')
                plt.imsave(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling_sf2/navvis_{image_filename}depth_median_pool.png', depth_median_pool, cmap='inferno')
                plt.imsave(f'/home/admin-anedunga/Downloads/depth-upsampling/navvis_downsampling_sf2/navvis_{image_filename}depth_joint_bilateral.png', depth_joint_bilateral, cmap='inferno')

                print("Downsampling completed! Outputs saved.")

                # Inpainting Before
                # original_depth = nearest_neighbor_interpolation(original_depth)
                # depth_nn = nearest_neighbor_interpolation(depth_nn)
                # depth_median_pool = nearest_neighbor_interpolation(depth_median_pool)
                
                # # Display results using matplotlib
                # fig, ax = plt.subplots(1, 4, figsize=(18, 6))
                # ax[0].imshow(original_depth, cmap='inferno')
                # ax[0].set_title('Original Depth')
                # ax[0].axis('off')
                # ax[1].imshow(depth_nn, cmap='inferno')
                # ax[1].set_title('Nearest Neighbor')
                # ax[1].axis('off')
                # ax[2].imshow(depth_median_pool, cmap='inferno')
                # ax[2].set_title('Median Pooling')
                # ax[2].axis('off')
                # ax[3].imshow(depth_joint_bilateral, cmap='inferno')
                # ax[3].set_title('Joint Bilateral Filter + Nearest Neighbor')
                # ax[3].axis('off')

                # plt.show()