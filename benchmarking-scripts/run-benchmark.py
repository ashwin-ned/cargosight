import cv2
import numpy as np
import os, time, random, json
import gc
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from matplotlib import pyplot as plt
from BenchmarkLogger import BenchmarkLogger
from DepthMetrics import DepthMetrics
from sympy import *

# from DepthAnythingv2Metric import DepthAnythingInference
#from UniDepth import UniDepthInference
#from ZoeDepth import ZoeDepthInference
#from Metric3D import Metric3DInference
from models.DepthPro.src import depth_pro

torch.cuda.empty_cache()
gc.collect()

class DepthDataset(Dataset):
    def __init__(self, data_dir, max_depth=None, sample_size=None):
        self.data_dir = data_dir
        self.depth_files = [f for f in os.listdir(data_dir) if f.endswith(".tiff")]
        self.rgb_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        self.max_depth = max_depth
        self.sample_size = sample_size

        print(f"Dataset initialized with {len(self.depth_files)} depth files and {len(self.rgb_files)} RGB files.")

        if sample_size:
            self.depth_files = random.sample(self.depth_files, min(sample_size, len(self.depth_files)))
            self.rgb_files = random.sample(self.rgb_files, min(sample_size, len(self.rgb_files)))

    def __len__(self):
        return min(len(self.depth_files), len(self.rgb_files))

    def __getitem__(self, idx):
        depth_path = os.path.join(self.data_dir, self.depth_files[idx])
        rgb_path = depth_path.replace(".tiff", ".png")  # Assuming RGB file shares same name as depth

        depth = self._load_depth(depth_path)
        image = self._load_image(rgb_path)

        # if self.max_depth:
        #     depth[depth > self.max_depth] = 0

        return image, depth

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"RGB file not found at {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)/1000.0
        if depth is None:
            raise FileNotFoundError(f"Depth file not found at {path}")
        return depth


class FLWDataset(Dataset):
    def __init__(self, data_dir, max_depth=None, sample_size=None):
        self.data_dir = data_dir
        self.depth_files = [f for f in os.listdir(data_dir) if f.endswith(".tiff")]
        self.rgb_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        self.max_depth = max_depth
        self.sample_size = sample_size

        print(f"Dataset initialized with {len(self.depth_files)} depth files and {len(self.rgb_files)} RGB files.")

        if sample_size:
            self.depth_files = random.sample(self.depth_files, min(sample_size, len(self.depth_files)))
            self.rgb_files = random.sample(self.rgb_files, min(sample_size, len(self.rgb_files)))
        else:
            self.depth_files = sorted(self.depth_files)
            self.rgb_files = sorted(self.rgb_files)

    def __len__(self):
        return min(len(self.depth_files), len(self.rgb_files))

    def __getitem__(self, idx):
        depth_path = os.path.join(self.data_dir, self.depth_files[idx])
        rgb_path = depth_path.replace(".tiff", ".jpg")  # Assuming RGB file shares same name as depth

        depth = self._load_depth(depth_path)
        image = self._load_image(rgb_path)


        return image, depth

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"RGB file not found at {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found at {path}")
        return depth

    

# Read Data - Sample n random or all
# Filter out max depth and nan values
# Resize image to depth shape 
# Load intrinsics from json or env file depending on dataset
# Compute Metrics (only compute for  non-zero values)
# Save Metrics and average inference time (Per Image)




def infer_depthpro(image, model, transform, device, depth_pro_f=None):

    with torch.no_grad():
            # Prepare input
            rgb_image = Image.fromarray(image)
            image_tensor = transform(rgb_image).to(device)
            
            # Model inference
            prediction = model.infer(image_tensor, f_px=depth_pro_f)
            
            # Retrieve predicted depth
            predicted_depth = prediction["depth"].cpu().numpy()
            
            # Clean up
            del image_tensor, prediction  # Explicitly free variables
            torch.cuda.empty_cache()  # Release GPU memory
            gc.collect()  # Trigger garbage collection
            
    return predicted_depth




def main():
    #encoders = ['vits', 'vitb', 'vitl']
    #for encoder in encoders:

    DEBUG = 0
    SAVE = 1
    MODEL_ID = f"arkit-DepthPro"
    # ARKit Dataset
    arkit_data_dir = '/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/ARKit-Data/'
    # FLW Dataset
    #flw_data_dir = '/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/FLW-Data/val/'
    output_dir = '/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-benchmarking-logs/'
    model_csv_path = os.path.join(output_dir, f"{MODEL_ID}.csv")


    arkit_dataset = DepthDataset(arkit_data_dir, max_depth=5.0, sample_size=1500)
    print(f"Dataset loaded: {len(arkit_dataset)} samples.")

    #flw_dataset = FLWDataset(flw_data_dir)
    #print(f"Dataset loaded: {len(flw_dataset)} samples.")

    #######################################MODEL SETUP############################################
    # 1. DEPTH ANYTHINGv2
    #depth_anything = DepthAnythingInference(encoder)

    # 2. UniDepth
    # camera_intrinsics = np.array([[1593.36, 0.0, 949.35],  # f_x, 0, c_x
    #                               [0.0, 1593.36, 722.76],  # 0, f_y, c_y
    #                               [0.0, 0.0, 1.0]       # 0, 0, 1
    #                               ]).astype(np.float32)
    camera_intrinsics = [1598.7, 1598.7, 949.2, 722.7]
    #uni_depth = UniDepthInference(model='unidepthv2s', view_result=False, intrinsics=camera_intrinsics)

    # 3. ZoeDepth
    #zoe_depth = ZoeDepthInference(model='zoe_n', view_result=False)

    # 4. Metric 3D
    #metric3d = Metric3DInference(model='metric3d_convnext_large', view_result=False)

    # 5. DepthPro
    model, transform = depth_pro.create_model_and_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU once
    model.eval()  # Set the model to evaluation mode
    

    print("Starting inference...")
    metrics_list = {'mae': 0.0, 'are': 0.0, 'rmse': 0.0, 'delta1': 0.0}
    total_samples = 0

    for i, (image, depth) in enumerate(arkit_dataset):
        filename = arkit_dataset.depth_files[i]
        file_path = os.path.join(arkit_dataset.data_dir, filename)
        print(f"Processing sample {i + 1}: Image Shape: {image.shape}, Depth Shape: {depth.shape}")

        if image.shape[:2] != depth.shape:
            print("Resizing image to match depth dimensions.")
            image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
        # Time
        start_time = time.time()
        print(image.shape, image.dtype)

        ############################INFERENCE################################
        # 1. DepthAnythingv2
        #predicted_depth = depth_anything.inference_depth_anything(image)

        # 2. UniDepth
        #predicted_depth, _, pcd, pred_intrinsics = uni_depth.infer_depth(image)

        # 3. ZoeDepth
    
        #predicted_depth = zoe_depth.infer_depth(image)

        # 4. Metric3D
        #predicted_depth, _, confidence = metric3d.infer_depth(image, camera_intrinsics)

        # 5. DepthPro (No Intrinsics)
        print(file_path.replace(".tiff", ".png"))
        #depth_pro_rgb, _, depth_pro_f = depth_pro.load_rgb(file_path.replace(".tiff", ".png"))
        depth_pro_f = None
        #print(f"Predicted Focal Length for {filename}", depth_pro_f)

        print(f"Inferencing Image {filename}, Sample {i + 1}")
        predicted_depth = infer_depthpro(image, model, transform, device, depth_pro_f)
        # Time
        inference_time = time.time() - start_time

        print(f"Inference time: {inference_time:.2f} seconds")
        # torch.cuda.empty_cache()
        gc.collect()
        # plot the image, gt and predicted
        if DEBUG:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image)
            ax[0].set_title("Image")
            plt.axis('off')
            ax[1].imshow(depth, cmap="inferno")
            ax[1].set_title("Ground Truth")
            plt.axis('off')
            ax[2].imshow(predicted_depth, cmap="inferno")
            ax[2].set_title("Predicted Depth")
            plt.axis('off')
            plt.show()

        # TODO: Change logic to save specific files for comparision
        if SAVE and isprime(i):
            figs, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image)
            ax[0].set_title("Image")
            plt.axis('off')
            ax[1].imshow(depth, cmap="inferno")
            ax[1].set_title("Ground Truth")
            plt.axis('off')
            ax[2].imshow(predicted_depth, cmap="inferno")
            ax[2].set_title("Predicted Depth")
            plt.axis('off')
            plt.savefig(f'/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-benchmarking-logs/{MODEL_ID}_{filename}.png')



    # Use DepthMetrics class for metric calculation
        metrics_calculator = DepthMetrics(ground_truth=depth, predicted=predicted_depth)
        mean_absolute_error = metrics_calculator.mean_absolute_error()
        abs_rel = metrics_calculator.absolute_relative_error()
        rmse = metrics_calculator.root_mean_squared_error()
        delta1 = metrics_calculator.threshold_accuracy()

        print(f"Metrics for sample {i + 1}: MAE={mean_absolute_error}, AbsRel={abs_rel}, RMSE={rmse}, Delta1={delta1}")
        metrics_list['mae'] += mean_absolute_error
        metrics_list['are'] += abs_rel
        metrics_list['rmse'] += rmse
        metrics_list['delta1'] += delta1
        total_samples += 1

        if DEBUG:
            metrics_calculator.plot_depth_density()
            # metrics_calculator.plot_depth_histogram()
            #metrics_calculator.plot_hist_density_difference()

        # Log metrics to CSV for each image
        # Write metrics to CSV
        with open(model_csv_path, "a") as f:
            f.write(f"{MODEL_ID};{filename};{inference_time};{mean_absolute_error};{rmse};{abs_rel};{delta1}\n")
        

    # Compute and display average metrics
    avg_metrics = {key: value / total_samples for key, value in metrics_list.items()}
    
    print("Average Metrics:", avg_metrics)

if __name__ == "__main__":
    main()