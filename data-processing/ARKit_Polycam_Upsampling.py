import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
from JBU_parallel_upsampling import preprocess_jbu, joint_bilateral_upsample_parallel, nearest_neighbor_interpolation








if __name__ == "__main__":
    
    dir = f'/media/admin-anedunga/ASHWIN-128/Duisberg_Polycam/2023-03-02/'
    output_dir = f'/media/admin-anedunga/ASHWIN-128/Duisburg_Polycam_Upsampled/'
    rgb_dirs = list()
    depth_dirs = list()

    for folder in os.listdir(dir):
        for subdir in os.listdir(dir+folder):
          
            if subdir == 'keyframes':
                
                depth_dir = dir+folder+'/'+subdir+'/'+'depth'
                rgb_dir = dir+folder+'/'+subdir+'/'+'images'
                rgb_dirs.append(rgb_dir)
                depth_dirs.append(depth_dir)



    for d in rgb_dirs:
        print("DIR:", d)
        for file in os.listdir(d):
            if file in os.listdir(output_dir):
                pass
            else:
                try:
                    print("File:", file)
                    rgb_image = cv2.imread(d+'/'+file, cv2.IMREAD_COLOR)
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    depth_path = (d+'/'+file).split('/')
                    depth_path[-2] = 'depth'
                    depth_path[-1] = depth_path[-1].replace('.jpg', '.png')
                    depth_path = '/'.join(depth_path)
                    print(depth_path)
                
                    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                    depth_image = depth_image.astype(np.float32)

                    # # plot rgb and depth
                    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    # ax[0].imshow(rgb_image)
                    # ax[0].set_title('RGB Image')
                    # ax[0].axis('off')
                    # ax[1].imshow(depth_image, cmap='inferno')
                    # ax[1].set_title('Depth Image')
                    # ax[1].axis('off')   
                    # plt.show()
                    
                    
                    #Upsample the depth map using Joint Bilateral Upsampling
                    # JBU Parameters
                    scale_factor = 4 # Upsampling factor
                    sigma_w = 1.0
                    sigma_c = 0.1
                    window_size = 3

                    pre_processed_depth, pre_processed_image = preprocess_jbu(rgb_image, depth_image, scale_factor)
                    upsampled_depth = joint_bilateral_upsample_parallel(pre_processed_image, pre_processed_depth, scale_factor, sigma_w, sigma_c, window_size)
                    inpainted_depth = nearest_neighbor_interpolation(upsampled_depth, view=False) 
                    print(f"Upsampled Depth Map: {inpainted_depth.shape}, {inpainted_depth.dtype}, {inpainted_depth.min()}, {inpainted_depth.max()}")

                    # # Plot using MatplotLib
                    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    # axs[0].imshow(rgb_image)
                    # axs[0].set_title('RGB Image')
                    # axs[0].axis('off')
                    # axs[1].imshow(upsampled_depth, cmap='inferno')
                    # axs[1].set_title('Upsampled Depth')
                    # axs[1].axis('off')
                    # axs[2].imshow(inpainted_depth, cmap='inferno')
                    # axs[2].set_title('Inpainted Depth')
                    # axs[2].axis('off')
                    # plt.show()

                    # Write the upsampled depth map to disk
        
                    print("f:", file)
                    cv2.imwrite(output_dir+file, cv2.cvtColor(pre_processed_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(output_dir+file.replace('.jpg', '.tiff'), inpainted_depth)
                    # Viz image
                    plt.imsave(output_dir+file.replace('.jpg', '.png'), inpainted_depth, cmap='inferno')
            
                except Exception as e:
                    print(f"Error: {e} Encountered for File: {file}")
                    pass