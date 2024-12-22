import cv2
import os
import re
import json 
import numpy as np
from navvis_depth_masking import remove_invalid_depth_edges, resize_navvis, image_resize_in_ratio

""" Authored by Ashwin Nedungadi on 2024-05-12 """ 

class DataLoaderARKit:
    """ Dataloader for Ipad/Iphone LiDAR data (ten-container-dataset) scanned with 3D scanner app on  IOS."""
    def __init__(self, folder_path, output_path=None, preprocess=True, resize=True):
        # initialize the directory containing the images, the annotations file
        # Implement functions to read all camera/image paths at once
        self.folder_path = folder_path
        self.output_path = output_path
        self.preprocess = preprocess # Only keeps values for which there is a depth and image pair
        self.resize = resize

        self.data = self.load_data()
        # if set True resizes the image to dim
        self.resize_data()
        

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
        #print(data["00000"]["image"])
        #cv2.imshow("Image", data["00000"]["image"])
        #cv2.imshow("Depth", data["00000"]["depth"])
        #cv2.waitKey(0)
        return data

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        # Comment this line to not rotate image
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    def read_depth(self, depth_path):
        # Read 16 bit int depth in meters float 32
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        return depth

    def bgr_to_rgb(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def resize_image(self, image, width, height):
        resized_image = cv2.resize(image, (height, width))
        # Rotate image 
        resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return resized_image
    # TODO: Rename this to resize_image_to_depth
    def resize_data(self):
        if self.resize:
            for key, value in self.data.items():
                if "depth" in value and "image" in value:
                    value["image"] = self.resize_image(value["image"], value["depth"].shape[1], value["depth"].shape[0])
        return self.data
    
    def remove_unpaired_data(self, data):
        # Remove data that does not have image and depth pairs
        data = {key: value for key, value in data.items() if "depth" in value and "image" in value}
        return data
    
# TODO: Navvis depthmaps and images are same size, need to resize them uniformally
class DataLoaderNavvis:
    """ Dataloader for Navvis data scanned with Navvis VLX2 """
    def __init__(self, folder_path, output_path=None, preprocess=False):
        self.folder_path = folder_path
        self.output_path = output_path
        self.preprocess = preprocess
        self.data = self.load_data()

        if self.preprocess:
            self.preprocess_data()
    
    def load_data(self):
        data = dict()
        read_data = {"Depth": 0, "Image": 0}

        for file in os.listdir(self.folder_path):
            # Deal with Image
            if file.endswith(".png"):
                filename = file.split(".")
                file_key = filename[0]
                if file_key not in data:
                    data[file_key] = {}
                image = self.read_image(os.path.join(self.folder_path, file))
                image = self.bgr_to_rgb(image)
                data[file_key]["image"] = image
                read_data["Image"] += 1

            # Deal with Depth
            elif file.endswith(".tiff"):
                filename = file.split(".")
                temp_file = filename[0]
                file_key = self.convert_to_key(temp_file)
                if file_key not in data:
                    data[file_key] = {}
                depth = self.read_depth(os.path.join(self.folder_path, file))
                data[file_key]["depth"] = depth
                read_data["Depth"] += 1
        
        print("Data read: ", read_data)

        return data
    
    def convert_to_key(self, filename):
        """ Depth File format sucks, convert it to image format to match with image file """
        key = filename.split("_")
        img_name = key[1]
        cam = key[2]
        # Extract the numeric part from img_name using regex
        match = re.match(r"img(\d+)", img_name)
        if match:
            number = match.group(1)
            padded_number = number.zfill(5) # Pad the number to 5 digits
            img_name = f"img{padded_number}"

        result = img_name [3:] + "-" + cam    
        return result

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        # Comment this line to not rotate image
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    def read_depth(self, depth_path):
        # Read navvis depth in meters float 32
        # Navvis also needs depth rotation
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        rotated = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        return rotated

    def bgr_to_rgb(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def resize_image(self, image, width, height):
        resized_image = cv2.resize(image, (height, width))
        # Rotate image 
        resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return resized_image
    
    def preprocess_data(self):
        for key, value in self.data.items():
            if "depth" in value and "image" in value:
                depth = value["depth"]
                image = value["image"]
                depth, image, removed_area_str = remove_invalid_depth_edges(depth, image)
                # Resize the image and depth by downscaling 2x
                aspect_ratio_before = depth.shape[1] / depth.shape[0]
                aspect_ratio_before = depth.shape[1] / depth.shape[0]
                print("current shape", depth.shape, image.shape, "Aspect Ratio", aspect_ratio_before)
                ########## DEBUG ##########
                #image = image_resize_in_ratio(image, width=None, height=512)
                #depth = image_resize_in_ratio(depth, width=None, height=512)
                aspect_ratio_after = depth.shape[1] / depth.shape[0] 
                print("Resized shape", depth.shape, image.shape)
                #image, depth = resize_navvis(depth, image)
                cv2.imshow("Depth", depth)
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                # Filter weird aspect ratios
                if aspect_ratio_after >= 0.5:
                    value["depth"] = depth
                    value["image"] = image
                    #value["removed_area_str"] = removed_area_str
        return self.data

if __name__ == "__main__":

    # Example usage Navvis Data
    folder_path = "/home/admin-anedunga/Desktop/benchmarking_data/navvis_data"
    output_path = "/home/admin-anedunga/Desktop/benchmarking_data/"

    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Reading files in directory:{directory}")
            data_loader = DataLoaderNavvis(os.path.join(folder_path, root, directory), output_path, preprocess=True)

    """# Example usage Iphone LiDAR Data
    folder_path = "/home/admin-anedunga/Desktop/ten-container-dataset/"
    output_path = "/home/admin-anedunga/Desktop/ten-container-debug-output/"
    data_loader = DataLoaderARKit(folder_path, output_path, preprocess=True)

    for key, value in data_loader.data.items():
        print(key, value)
        cv2.imshow("Image", value["image"])
        cv2.imshow("Depth", value["depth"])
        cv2.waitKey(0)

    # Iterate through all 10 container scans
    # Will iterate through all 10 container scans 
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Reading files in directory:{directory}")
            data_loader = DataLoaderARKit(os.path.join(folder_path, root, directory), os.path.join(output_path, directory), preprocess=True)"""
            
