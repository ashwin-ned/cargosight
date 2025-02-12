import numpy as np
import os, random
import matplotlib.pyplot as plt
import cv2
import time
import open3d as o3d
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging
from pcd_depth_metrics.boundary_metrics import SI_boundary_F1, SI_boundary_Recall
from pcd_depth_metrics.model_utils import fscore, calc_cd, calc_dcd
import point_cloud_utils as pcu
from skimage.metrics import structural_similarity

#from DepthAnythingv2Metricpcd import DepthAnythingInference
#from models.depthanythingv2.depth_anything_v2.dpt import DepthAnythingV2
from models.depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import gc


class EvaluationDataloader(Dataset):
    def __init__(self, data_dir, max_depth=None, sample_size=None):
        self.data_dir = data_dir
        self.max_depth = max_depth

        self.depth_files = [f for f in os.listdir(data_dir) if f.endswith("_inpainted.tiff")]
        self.rgb_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

        if sample_size:
            self.depth_files = random.sample(self.depth_files, min(sample_size, len(self.depth_files)))
            self.rgb_files = random.sample(self.rgb_files, min(sample_size, len(self.rgb_files)))

        else:
            self.depth_files = self.depth_files
            self.rgb_files = self.rgb_files
        
        # Sort the files
        self.depth_files.sort()
        self.rgb_files.sort()
        logging.info(f"Dataset initialized with {len(self.depth_files)} depth files and {len(self.rgb_files)} RGB files.")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        depth_path = os.path.join(self.data_dir, self.depth_files[idx])
        rgb_filename = self.depth_files[idx].replace("_inpainted.tiff", "_rgb.png")
        print(rgb_filename)
        rgb_path = os.path.join(self.data_dir, rgb_filename)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

        intrinsics = self.get_camera_intrinsics(rgb_filename)

        if self.max_depth:
            logging.warning(f"Max Depth Mode is Enabled. Clipping depth values greater than {self.max_depth} to 0")
            depth[depth > self.max_depth] = 0

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        return {
            'depth': depth,
            'rgb': rgb,
            'filename': rgb_filename,
            'intrinsics': intrinsics}
    
    def get_camera_intrinsics(self, filename):
        
        if filename.startswith("iphone"):
            return np.array([[2454.30, 0.0, 1214.21], 
                             [0.0, 2455.00, 1610.50], 
                             [0.0, 0.0, 1.0]]).astype(np.float32)
        elif filename.startswith("pixel"):
            return np.array([[5702.00, 0.0, 1504.80], 
                             [0.0, 5700.30, 2037.42], 
                             [0.0, 0.0, 1.0]]).astype(np.float32)
        # FLIR
        else:
            None
                                   

                                   

def compute_depth_metrics(predicted, ground_truth):

    predicted_copy = predicted.copy()
    ground_truth_copy = ground_truth.copy()

    # Create a valid mask to exclude zero or NaN values in the ground truth
    valid_mask = (ground_truth != 0) 

    # Apply the mask to filter out invalid pixels
    predicted = predicted[valid_mask]
    ground_truth = ground_truth[valid_mask]



    mae = np.mean(np.abs(predicted - ground_truth))
    abs_rel = np.mean(np.abs(predicted - ground_truth) / ground_truth)
    rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))
    ssim = structural_similarity(predicted, ground_truth, full=True, data_range=ground_truth.max() - ground_truth.min())
    delta_1 = np.mean((np.maximum(predicted / ground_truth, ground_truth / predicted) < 1.25).astype(np.float32))
    delta_2 = np.mean((np.maximum(predicted / ground_truth, ground_truth / predicted) < 1.25 ** 2).astype(np.float32))
    delta_3 = np.mean((np.maximum(predicted / ground_truth, ground_truth / predicted) < 1.25 ** 3).astype(np.float32))


    print(f"Predicted depth shape: {predicted.shape}, Type: {predicted.dtype}")
    print(f"Target depth shape: {ground_truth.shape} Type: {ground_truth.dtype}")

    si_boundary_f1 = SI_boundary_F1(predicted_copy, ground_truth_copy)
    si_boundary_recall = SI_boundary_Recall(predicted_copy, ground_truth_copy)

    return {
        "MAE": mae,
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "SSIM": ssim,
        "Delta1": delta_1,
        "Delta2": delta_2,
        "Delta3": delta_3,
        "SI-Boundary-F1": si_boundary_f1,
        "SI-Boundary-Recall": si_boundary_recall
    }

  

def compute_pointcloud_metrics(predicted_pcd, ground_truth_pcd):
    # Convert to PyTorch tensors
    predicted_tensor = torch.tensor(predicted_pcd, dtype=torch.float32).cuda()  # Add .cuda() if using GPU
    ground_truth_tensor = torch.tensor(ground_truth_pcd, dtype=torch.float32).cuda()  # Add .cuda() if using GPU

    # Add batch dimension
    predicted_tensor = predicted_tensor.unsqueeze(0)  # Shape: [1, num_points, 3]
    ground_truth_tensor = ground_truth_tensor.unsqueeze(0)  # Shape: [1, num_points, 3]

    # F-Score
    f_score = fscore(predicted_tensor, ground_truth_tensor)


    # Chamfer Distance
    chamfer_distances = calc_cd(predicted_tensor, ground_truth_tensor)
    density_aware_cd = calc_dcd(predicted_tensor, ground_truth_tensor)

    # Hausdorf
    # Compute one-sided squared Hausdorff distances (Hausdorf between Predicted-Groundtruth tells how far off (Worst Case) our prediction is from the LiDAR GT)
    hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(predicted_pcd, ground_truth_pcd)

    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance (Since LiDAR is always algined, shoudl be consistent across all samples)
    hausdorff_dist = pcu.hausdorff_distance(predicted_pcd, ground_truth_pcd)

    return {    "F-Score": f_score,
                "Chamfer Distance": chamfer_distances,
                "Density-Aware Chamfer Distance": density_aware_cd,
                "Haussdorf Distance": hausdorff_dist,
                "One-Sided Hausdorff": hausdorff_a_to_b,
            }

def generate_pointcloud(image, depth_map, intrinsic_matrix, output_directory, save=False):

    if intrinsic_matrix is None:
        raise ValueError("Intrinsic matrix is required to generate point cloud.")

    focal_length_x = intrinsic_matrix[0, 0]
    focal_length_y = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / focal_length_x
    y = (y - cy) / focal_length_y
    z = np.array(depth_map)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save:
        print(f"Saving point cloud to {output_directory}")
        o3d.io.write_point_cloud(output_directory, pcd)
        print(f"Point cloud saved to {output_directory}")

    return pcd



if __name__ == "__main__":
    DEBUG = 0
    SAVE = 1
    # Path to pretrained checkpoitns
    #checkpoint_path = "/home/rog-nuc/Downloads/DepthAnything-Pretrained/depth_anything_v2_metric_vkitti_vits.pth"
    # Path to trained checkpoint
    #checkpoint_path = "/home/rog-nuc/Downloads/ViTs-Final-Training/custom-trainingv4-ViTs_vits_5e-06_0.01_epochs40.pth"
    checkpoint_path = "/home/rog-nuc/Downloads/DepthAnything-Pretrained/depth_anything_v2_metric_vkitti_vitb.pth"
    files_to_save = ['iphone06_rgb.png', 'iphone10_rgb.png', 'iphone07_rgb.png', 'pixel06_rgb.png', 'pixel10_rgb.png', 'pixel07_rgb.png']
    
    model_encoder = 'vitb'
    # Path to Evaluation Data
    data_dir = f"/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/phone-rgb-depth/processed data/"
    output_metrics_path = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/pipeline_metrics/DA-Pretrained-Metrics-Navvis-Phone/"
    save_image_path = f"/home/rog-nuc/Pictures/high-quality-inference-pictures/"
    max_depth = 80
    batch_size = 1

    # Initialize the evaluation dataloader
    navvis_dataset = EvaluationDataloader(data_dir, max_depth=max_depth)
    navvis_dataloader = DataLoader(navvis_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Parameters
    model = 'xxx'
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}
    

    # Empty pytorch cache
    torch.cuda.empty_cache()
    gc.collect() 
    depth_anything_model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
    depth_anything_model.load_state_dict(torch.load(checkpoint_path))

    depth_anything_model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Pointcloud, Metrics Dictionary
    predicted_pointclouds = {}
    ground_truth_pointclouds = {}
    depth_metrics_dict = {}
    with open(f"{output_metrics_path}{model}-{model_encoder}_metrics.csv", "w") as f:
        f.write("Filename; MAE; AbsRel; RMSE; SSIM; Delta1; Delta2; Delta3; Chamfer Distance; Density-Aware Chamfer Distance; Hausdorf Distance; One Sided Hausdorf; F-Score \n")
    # phone_dataloader or flir_dataloader
    for batch in navvis_dataloader:
        filename = batch['filename']
        intrinsics = np.squeeze(batch['intrinsics'].numpy())
        rgb_batch = batch['rgb']  # A batch of RGB images
        depth_batch = batch['depth']  # A batch of depth maps
        print(f"Processing file: {filename}")
       
        for filename, rgb, depth in zip(filename, rgb_batch, depth_batch):  # Iterate over individual RGB and depth pairs
            start_time = time.time()

            h = 1120
            w = 980
            depth = cv2.resize(depth.cpu().numpy(), (w, h), cv2.INTER_NEAREST)

            # Check size
            if rgb.shape != depth.shape:
                print(f"Resizing rgb image to match depth shape: {depth.shape}")
                rgb = cv2.resize(np.asarray(rgb), (depth.shape[1], depth.shape[0]))

            rgb_torch = torch.tensor(rgb).float().permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
            rgb_torch = (rgb_torch - mean) / std
            rgb_torch = rgb_torch.unsqueeze(0)
            # Infer Depth with Trained model
            with torch.inference_mode():
                predicted = depth_anything_model(rgb_torch.to('cuda'))
                predicted = F.interpolate(predicted[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            predicted = predicted.cpu().numpy()
 
            # Visuzalize the predicted depth
            print(f"Predicted Depth Shape: {predicted.shape}, Type: {predicted.dtype}, RGB Shape: {rgb.shape}, Type: {rgb.dtype}")
            print(f"Predicted Depth Min: {predicted.min()}, Max: {predicted.max()}, Ground Truth Min: {depth.min()}, Max: {depth.max()}")
            if DEBUG:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(rgb)
                ax[0].set_title("RGB Image")
                ax[0].axis('off')
                ax[1].imshow(depth, cmap='inferno')
                ax[1].set_title("Ground Truth Depth")
                ax[1].axis('off')
                ax[2].imshow(predicted, cmap='inferno')
                ax[2].set_title("Predicted Depth")
                ax[2].axis('off')
                plt.show()
            if SAVE:
                if filename in files_to_save:
                    filename = filename.replace("_rgb.png", "")
                    # Crop 20 pixels from all sides
                    cropped_predicted = predicted[20:-20, 20:-20]
                    cropped_depth = depth[20:-20, 20:-20]
                    cropped_rgb = rgb[20:-20, 20:-20, :]  # Assuming RGB image has shape (H, W, 3)
                    
                    # Save the cropped images
                    plt.imsave(f"{save_image_path}{filename}_{model_encoder}_predicted.jpg", cropped_predicted, cmap='inferno', dpi=300)
                    plt.imsave(f"{save_image_path}{filename}_{model_encoder}_ground_truth.jpg", cropped_depth, cmap='inferno', dpi=300)
                    plt.imsave(f"{save_image_path}{filename}_{model_encoder}_rgb.jpg", cropped_rgb, dpi=300)
                 
            
            # Compute Depth Metrics
            depth_metrics = compute_depth_metrics(predicted, depth)
            depth_metrics_dict[filename] = depth_metrics

            # Generate Pointclouds
            predicted_pcd = generate_pointcloud(rgb, predicted, intrinsics, f"{output_metrics_path}{filename}_predicted.ply", save=False)
            ground_truth_pcd = generate_pointcloud(rgb, depth, intrinsics, f"{output_metrics_path}{filename}_ground_truth.ply", save=False)

            # Color the ground truth point cloud green (RGB: [0, 1, 0])
            ground_truth_colors = np.zeros_like(ground_truth_pcd.points)  # Initialize with zeros
            ground_truth_colors[:, 1] = 1.0                             # Set G channel to 1 (Green)
            ground_truth_pcd.colors = o3d.utility.Vector3dVector(ground_truth_colors)

            # Color the prediction point cloud orange (RGB: [1, 0.5, 0])
            predicted_colors = np.zeros_like(predicted_pcd.points)       # Initialize with zeros
            predicted_colors[:, 0] = 1.0                                # Set R channel to 1 (Red)
            predicted_colors[:, 1] = 0.5                                # Set G channel to 0.5 (Orange)
            predicted_pcd.colors = o3d.utility.Vector3dVector(predicted_colors)

            # Compute Pointcloud Metrics
            #pointcloud_metrics = compute_pointcloud_metrics(np.asarray(predicted_pcd.points), np.asarray(ground_truth_pcd.points))

            if DEBUG:
                # Use the original Open3D point clouds for visualization
                o3d.visualization.draw_geometries([predicted_pcd, ground_truth_pcd])


    #         print("*" * 50)
    #         print(f"Depth Metrics: {depth_metrics}")

    #         # Extract Chamfer Distance (ensure it's a number or list of numbers)
    #         chamfer_distance = pointcloud_metrics['Chamfer Distance']
    #         if isinstance(chamfer_distance, torch.Tensor):
    #             chamfer_distance = chamfer_distance.cpu().numpy().tolist()  # Convert tensor to list
    #         elif isinstance(chamfer_distance, list):
    #             chamfer_distance = [float(d) for d in chamfer_distance]  # Ensure all elements are floats

    #         # Extract Density-Aware Chamfer Distance
    #         density_aware_cd = pointcloud_metrics['Density-Aware Chamfer Distance']
    #         if isinstance(density_aware_cd, torch.Tensor):
    #             density_aware_cd = density_aware_cd.cpu().numpy().tolist()  # Convert tensor to list
    #         elif isinstance(density_aware_cd, list):
    #             density_aware_cd = [float(d) for d in density_aware_cd]  # Ensure all elements are floats

    #         # Extract F-Score (convert tensor values to lists or floats)
    #         pcd_fscore = pointcloud_metrics['F-Score']
    #         if isinstance(pcd_fscore, tuple):  # Handle tuple of tensors
    #             pcd_fscore = tuple(
    #                 item.cpu().numpy().tolist() if isinstance(item, torch.Tensor) else item 
    #                 for item in pcd_fscore
    #             )

    #         # Extract Hausdorff Distance (already a float but included here for consistency)
    #         hausdorff_distance = pointcloud_metrics['Haussdorf Distance']

    #         # Extract One-Sided Hausdorff Distance (handle tuple with potential tensors)
    #         one_sided_hausdorff = pointcloud_metrics['One-Sided Hausdorff']
    #         if isinstance(one_sided_hausdorff, tuple):  # Handle tuple with possible tensors
    #             one_sided_hausdorff = tuple(
    #                 item.cpu().numpy() if isinstance(item, torch.Tensor) else item 
    #                 for item in one_sided_hausdorff
    #             )

    #         print(depth_metrics_dict[filename])

    #         print(f"f-Score: {pcd_fscore}")
    #         print(f"Hausdorff Distance: {hausdorff_distance}")
    #         print(f"One-Sided Hausdorff Distance: {one_sided_hausdorff}")
    #         print("*" * 50)

    #         # Optional: Prepare data for saving into .csv format.
    #         metrics_to_save = {
    #             "Chamfer Distance": chamfer_distance,
    #             "Density-Aware Chamfer Distance": density_aware_cd,
    #             "F Score": pcd_fscore,
    #             "Hausdorff Distance": hausdorff_distance,
    #             "One Sided Hausdorff": one_sided_hausdorff,
    #         }
    #         print("-" * 50)
    #         print(f"Metrics to Save: {metrics_to_save}")
    #         print("-" * 50)

    #         with open(f"{output_metrics_path}{model}-{model_encoder}_metrics.csv", "a") as f:
    #             data = (f"{filename}; {depth_metrics_dict[filename]['MAE']}; {depth_metrics_dict[filename]['AbsRel']}; {depth_metrics_dict[filename]['RMSE']}; {depth_metrics_dict[filename]['SSIM'][0]}; "f"{depth_metrics_dict[filename]['Delta1']}; {depth_metrics_dict[filename]['Delta2']}; {depth_metrics_dict[filename]['Delta3']}; {metrics_to_save['Chamfer Distance']}; {metrics_to_save['Density-Aware Chamfer Distance']}; {metrics_to_save['Hausdorff Distance']}; {metrics_to_save['One Sided Hausdorff']}; {metrics_to_save['F Score']} \n")
    #             f.write(data)

    # print(f"Depth Metrics: {depth_metrics}")




   
