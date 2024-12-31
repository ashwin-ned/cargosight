import cv2
import os
import re
import json 
import matplotlib.pyplot as plt
import numpy as np
from JBU_parallel_upsampling import preprocess_jbu, joint_bilateral_upsample_parallel, nearest_neighbor_interpolation
from sympy import *

class DataLoaderARKit:
    """ Dataloader for Ipad/Iphone LiDAR data (ten-container-dataset) scanned with 3D scanner app on  IOS."""
    def __init__(self, folder_path, output_path=None, preprocess=True):
        # initialize the directory containing the images, the annotations file
        # Implement functions to read all camera/image paths at once
        self.folder_path = folder_path
        self.output_path = output_path
        self.preprocess = preprocess # Only keeps values for which there is a depth and image pair
   

        self.data = self.load_data()

        
    def load_data(self):
        # TODO: Error Handling; Separating the logic for reading specific data types
        data = dict()
        read_data = {"Depth Confidences": 0, "Depth": 0, "Image": 0, "Jsons": 0}
       
        for file in os.listdir(self.folder_path):
            filename = file.split("_")
            # Don't read points.ply
            if len(filename) >= 2:
                file_key = filename[1].split(".")[0]
            #print(f"Processing file {file_key}.")
            if file_key not in data:
                data[file_key] = {}
        
            if filename[0] == "conf":
                depth_conf = self.read_depth(os.path.join(self.folder_path, file))
                data[file_key]["depth_conf"] = depth_conf
                read_data["Depth Confidences"] += 1
            
            elif filename[0] == "depth":
                depth = self.read_depth(os.path.join(self.folder_path, file))
                data[file_key]["depth"] = depth
                read_data["Depth"] += 1

            elif filename[0] == "frame" and file.endswith(".jpg"):
                image = self.read_image(os.path.join(self.folder_path, file))
                data[file_key]["image"] = image
                read_data["Image"] += 1

            elif filename[0] == "frame" and file.endswith(".json"):
                json_frame = json.load(open(os.path.join(self.folder_path, file)))
                data[file_key]["json"] = json_frame
                read_data["Jsons"] += 1

        if self.preprocess:
            data = self.remove_unpaired_data(data)

        print("Data read: ", read_data)
        print("Data after removing unpaired data: ", len(data))

        return data

    def read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return image
    
    def read_depth(self, depth_path):
        # Read 16 bit int depth in meters float 32
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        return depth

    def bgr_to_rgb(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def remove_unpaired_data(self, data):
        # Remove data that does not have image and depth pairs
        data = {key: value for key, value in data.items() if "depth" in value and "image" in value}
        return data
    
if __name__ == "__main__":

    folder_path = "/home/admin-anedunga/Desktop/benchmarking_data/DBSchenker_Dortmund_Dataset/"
    output_path = "/media/admin-anedunga/ASHWIN-128/ARKit_Dortmund_Upsampled_Data/"

    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Reading files in directory:{directory}")
            data_loader = DataLoaderARKit(os.path.join(folder_path, root, directory), os.path.join(output_path, directory), preprocess=True)

    # for key, value in data_loader.data.items():
    #     print(key, value)
    #     cv2.imshow("Image", value["image"])
    #     cv2.imshow("Depth", value["depth"])
    #     cv2.waitKey(0)
            print("Processing data from directory:", directory)
            count = 0
            for key, value in data_loader.data.items():
                image = value["image"]
                depth = value["depth"]
                print(f"Original Depth Map: {depth.shape}, {depth.dtype}, {depth.min()}, {depth.max()}")
                # Upsample the depth map using Joint Bilateral Upsampling
                # JBU Parameters
                scale_factor = 4
                sigma_w = 1.0
                sigma_c = 0.1
                window_size = 3

                pre_processed_depth, pre_processed_image = preprocess_jbu(image, depth, scale_factor)
                upsampled_depth = joint_bilateral_upsample_parallel(pre_processed_image, pre_processed_depth, scale_factor, sigma_w, sigma_c, window_size)
                inpainted_depth = nearest_neighbor_interpolation(upsampled_depth, view=False) 
                print(f"Upsampled Depth Map: {inpainted_depth.shape}, {inpainted_depth.dtype}, {inpainted_depth.min()}, {inpainted_depth.max()}")
                # # view orginal and upsampled depth map with matplotlib
                # fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                # ax[0].imshow(depth, cmap='inferno')
                # ax[0].set_title("Original Depth")
                
                # ax[1].imshow(upsampled_depth, cmap='inferno')
                # ax[1].set_title("Upsampled Depth")

                # ax[2].imshow(inpainted_depth, cmap='inferno')
                # ax[2].set_title("Inpainted Depth")

                # plt.show()

                # Save the image
                plt.imsave(os.path.join(output_path, f"{directory.lower()}_{key}_rgb.png"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Only save the depth map if it is a prime number (reduce viz images)
                count+=1
                if isprime(count):
                    plt.imsave(os.path.join(output_path, f"{directory.lower()}_{key}_depth_upsampled.png"), inpainted_depth, cmap='inferno')
                
                # Save the depth as .tiff
                cv2.imwrite(os.path.join(output_path, f"{directory.lower()}_{key}_depth.tiff"), inpainted_depth * 1000)
