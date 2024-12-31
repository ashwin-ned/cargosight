import cv2
import os
import re
import json 
import matplotlib.pyplot as plt
import numpy as np
from JBU_parallel_upsampling import preprocess_jbu, joint_bilateral_upsample_parallel, nearest_neighbor_interpolation
from DepthMetrics import DepthMetrics

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
    
if __name__ == "__main__":

    folder_path = "/home/admin-anedunga/Desktop/benchmarking_data/DBSchenker_Dortmund_Dataset/"
    output_path = "/media/admin-anedunga/ASHWIN-128/upsampling-quality-check/"

    TOTAL_IMAGES = 0
    MEAN_AVERAGE = 0
    ROOT_MEAN_SQUARE = 0
    ABSOLUTE_RELATIVE = 0
    THRESHOLD_ACCURACY = 0


    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Reading files in directory:{directory}")
            data_loader = DataLoaderARKit(os.path.join(folder_path, root, directory), os.path.join(output_path, directory), preprocess=True)

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


                # # plot using matplotlib
                # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                # axs[0].imshow(depth, cmap='inferno')
                # axs[0].set_title(f'Original Depth Map')
                # axs[0].axis('off')
                # axs[1].imshow(inpainted_depth, cmap='inferno')
                # axs[1].set_title(f'Inpainted Depth Map')
                # axs[1].axis('off')
                # plt.tight_layout()
                # plt.show()

                # Compute depth metrics after resizing to original size

                original_shape = (depth.shape[1], depth.shape[0])
                inpainted_depth_qc = cv2.resize(inpainted_depth, original_shape, interpolation=cv2.INTER_NEAREST)
                depth_metrics = DepthMetrics(depth, inpainted_depth_qc, view_result=False)
                mae = depth_metrics.mean_absolute_error()
                rmse = depth_metrics.root_mean_squared_error()
                are = depth_metrics.absolute_relative_error()
                accuracy = depth_metrics.threshold_accuracy()
                print(f"*"*50, 'METRICS', f"*"*50)
                print(f"Mean Absolute Error: {mae}", f"Root Mean Squared Error: {rmse}", f"Absolute Relative Error: {are}", f"Threshold Accuracy: {accuracy}")
                print(f"*"*50, 'METRICS', f"*"*50)

                TOTAL_IMAGES += 1
                MEAN_AVERAGE += mae
                ROOT_MEAN_SQUARE += rmse
                ABSOLUTE_RELATIVE += are
                THRESHOLD_ACCURACY += accuracy
                
                # write metric for each image to file 
                with open(f"{output_path}/{directory}_metrics.txt", "a") as file:
                    file.write(f"{key};, {mae};, {rmse};, {are};, {accuracy}\n")
    
    print(f"-"*50, 'TOTAL METRICS', f"-"*50)
    print(f"Total Images: {TOTAL_IMAGES}")
    print(f"Mean Average Error: {MEAN_AVERAGE/TOTAL_IMAGES}")
    print(f"Root Mean Square Error: {ROOT_MEAN_SQUARE/TOTAL_IMAGES}")
    print(f"Absolute Relative Error: {ABSOLUTE_RELATIVE/TOTAL_IMAGES}")
    print(f"Threshold Accuracy: {THRESHOLD_ACCURACY/TOTAL_IMAGES}")
