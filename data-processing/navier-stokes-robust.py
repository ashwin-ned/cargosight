import numpy as np
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
import time
import os 




def preprocess_and_inpaint_depth(depth, inpaint_radius=3, max_depth=25):
    """
    Preprocess the depth image, create masks, and perform inpainting.
    """
    # Step 1: Create a mask for missing values (depth == 0)
    missing_values_mask = (depth == 0).astype(np.uint8)

    # Step 2: Create a mask for values outside the valid depth range (depth > max_depth)
    invalid_depth_range_mask = (depth > max_depth).astype(np.uint8)

    # Step 3: Inpaint the missing values using Navier-Stokes (cv2.INPAINT_NS)
    binary_mask = missing_values_mask * 255
    inpainted_depth = cv2.inpaint(depth.astype(np.float32), binary_mask, inpaint_radius, flags=cv2.INPAINT_NS)

    # Step 4: Remove invalid depth values from the inpainted depth map
    inpainted_depth[invalid_depth_range_mask == 1] = 0

    return inpainted_depth

def main():
    DEBUG = 1

    navvis_depth_dir = f'/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/NavVis/Schenker_00-00-01_HS_SF2/depth/'
    navvis_image_dir = f'/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/NavVis/Schenker_00-00-01_HS_SF2/undistorted/'
    output_dir = f'/home/rog-nuc/Desktop/depth-upsampling/navvis-advanced-inpainting/'

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(navvis_depth_dir):
        # Load depth image
        depth = cv2.imread(navvis_depth_dir + file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image_name = file.replace('depth_img', '')
        img_nr = int(image_name.split('_')[0])
        cam_nr = (image_name.split('_')[1])  
        # Format image number with 5 digits
        image_name = str(img_nr).zfill(5) + "-" + cam_nr 
        image_name = image_name.replace('.tiff', '.png')

        print(image_name)
        # Load the corresponding undistorted image
        image = cv2.imread(os.path.join(navvis_image_dir, image_name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if cam_nr == 'cam0':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Preprocess depth: Rotate and resize
        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        target_width, target_height = 768, 1024
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        print("After Reshaping", image.shape, depth.shape)

        # Inpaint the depth image
        start = time.time()
        inpainted_depth = preprocess_and_inpaint_depth(depth)
        print(f"Inpainting completed in {time.time() - start:.2f} seconds.")

        # Optionally, smooth the inpainted depth image using a bilateral filter
        #smoothed_depth = cv2.bilateralFilter(inpainted_depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

        # Save the results
        output_path = os.path.join(output_dir, file)
        #cv2.imwrite(output_path, inpainted_depth.astype(np.float32))
        print(f"Saved inpainted depth image to {output_path}")

        if DEBUG:
            # Display the results
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(image)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            axs[1].imshow(depth, cmap='inferno')
            axs[1].set_title('Original Depth')
            axs[1].axis('off')
            axs[2].imshow(inpainted_depth, cmap='inferno')
            axs[2].set_title('Inpainted Depth')
            axs[2].axis('off')

            plt.show()

if __name__ == "__main__":
    main()