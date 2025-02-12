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
            return np.array([[853.673, 0.0, 622.400],               # f_x, 0, c_x
                            [0.0, 854.153, 507.496],                # 0, f_y, c_y
                            [0.0, 0.0, 1.0]]).astype(np.float32)    # 0, 0, 1
                                   

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
    SAVE = 0
    # Path to pretrained checkpoitns
    #checkpoint_path = "/home/rog-nuc/Downloads/DepthAnything-Pretrained/depth_anything_v2_metric_vkitti_vitl.pth"
    # Path to trained checkpoint
    checkpoint_path = "/media/rog-nuc/Extreme SSD/MDE-TRAINING-CODE/mde-training/Depth-Anything-V2/Trained_Checkpoints/TrainingV5-Loss-Functions-Test/custom-trainingv5-silog-pixelformer-ViTs_vits_5e-06_0.01_epochs25_epoch20.pth"
    # Path to Evaluation Data
    flir_data_dir = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/flir-rgb-depth/"
    phone_rgb_dir = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/phone-rgb-depth/"
    pointcloud_dir = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/pointclouds/"
    output_metrics_path = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/pipeline_metrics/Loss-Functions/"
    output_pcd_path = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/output_pointclouds/"
    sample_size = 19 # 36 for phone, 19 for flir
    max_depth = 80
    batch_size = 1

    # Load Datasets
    flir_dataset = EvaluationDataloader(flir_data_dir, max_depth=max_depth, sample_size=sample_size)
    # Phone_Dataset not being inferenced right now, need to load intrinsics for this dataset and run it
    phone_dataset = EvaluationDataloader(phone_rgb_dir, max_depth=max_depth, sample_size=sample_size)

    flir_dataloader = DataLoader(flir_dataset, batch_size=batch_size, shuffle=False)
    phone_dataloader = DataLoader(phone_dataset, batch_size=batch_size, shuffle=False)
    
    # Empty pytorch cache
    torch.cuda.empty_cache()
    gc.collect() 
    
    # Model Parameters
    model = 'DepthAnythingv2-Loss-Function-ScaledSilog2'
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}
    model_encoder = 'vits'


    depth_anything_model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
    depth_anything_model.load_state_dict(torch.load(checkpoint_path))

    depth_anything_model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Pointcloud, Metrics Dictionary
    predicted_pointclouds = {}
    ground_truth_pointclouds = {}
    depth_metrics_dict = {}

    # phone_dataloader or flir_dataloader
    for batch in flir_dataloader:
        filename = batch['filename']
        intrinsics = np.squeeze(batch['intrinsics'].numpy())
        rgb_batch = batch['rgb']  # A batch of RGB images
        depth_batch = batch['depth']  # A batch of depth maps
       
        for filename, rgb, depth in zip(filename, rgb_batch, depth_batch):  # Iterate over individual RGB and depth pairs
            start_time = time.time()

            # Convert shape to multiple of 14
            # Phone
            # h = 1260
            # w = 980
            # FLIR
            h = 980
            w = 1120
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
            
            # Compute Depth Metrics
            depth_metrics = compute_depth_metrics(predicted, depth)
            depth_metrics_dict[filename] = depth_metrics

            # Generate Point Clouds
            pcd = generate_pointcloud(rgb, predicted, intrinsics, f"{output_pcd_path}{filename}-{model_encoder}.ply", save=SAVE)

            predicted_pointclouds[filename] = pcd

            mde_inference_time = time.time()
            print(f" Model inferenced on image in {mde_inference_time - start_time} seconds")
            #print(f"Depth Metrics: {depth_metrics}")

            if isinstance(pcd, np.ndarray):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))

            if DEBUG:
                o3d.visualization.draw_geometries([pcd])

    # Compare predicted depth to ground truth depth
    for file in os.listdir(pointcloud_dir):
        if file.endswith("_lidar.pcd") and "projected" not in file.split("_"):
            lidar_pcd = o3d.io.read_point_cloud(os.path.join(pointcloud_dir, file))
            file = file.replace("_lidar.pcd", "_rgb.png")
            ground_truth_pointclouds[file] = lidar_pcd

    #print("PRED", predicted_pointclouds.keys())
    #print("GT", ground_truth_pointclouds.keys())

    with open(f"{output_metrics_path}{model}-{model_encoder}-flir-livox_metrics.csv", "w") as f:
        f.write("Filename; MAE; AbsRel; RMSE; SSIM; Delta1; Delta2; Delta3; SI-Boundary-F1; SI-Boundary-Recall; Chamfer Distance; Density-Aware Chamfer Distance; Hausdorf Distance; One Sided Hausdorf; fscore \n")
        # print("*" * 50)
        # print("GT PCDS:", ground_truth_pointclouds)
        # print("Pred PCDS:", predicted_pointclouds)
        # print("Computed Depth Metrics:", depth_metrics_dict)

        for filename in predicted_pointclouds.keys():
            if filename in ground_truth_pointclouds:
                print(f"Caculating PCD metrics for {filename}")
                predicted_pcd = predicted_pointclouds[filename]
                ground_truth_pcd = ground_truth_pointclouds[filename]

                # Compute metrics for point clouds
                ground_truth_pcd = np.asarray(ground_truth_pcd.points)
                predicted_pcd = np.asarray(predicted_pcd.points)
                pointcloud_metrics = compute_pointcloud_metrics(predicted_pcd, ground_truth_pcd)

                if DEBUG:
                    o3d.visualization.draw_geometries([ground_truth_pcd, predicted_pcd])

                # Handle Chamfer Distance (ensure it's a number, not a tensor)
                chamfer_distance = pointcloud_metrics['Chamfer Distance']
                if isinstance(chamfer_distance, torch.Tensor):
                    chamfer_distance = chamfer_distance.cpu().numpy().tolist()  # Convert tensor to list
                elif isinstance(chamfer_distance, list):
                    chamfer_distance = [float(d) for d in chamfer_distance]  # Convert list elements to float

                # Handle Density-Aware Chamfer Distance (ensure it's a number, not a tensor)
                density_aware_cd = pointcloud_metrics['Density-Aware Chamfer Distance']
                if isinstance(density_aware_cd, torch.Tensor):
                    density_aware_cd = density_aware_cd.cpu().numpy().tolist()  # Convert tensor to list
                elif isinstance(density_aware_cd, list):
                    density_aware_cd = [float(d) for d in density_aware_cd]  # Convert list elements to float
                print(depth_metrics_dict[filename])

                pcd_fscore = pointcloud_metrics['F-Score']
                hausdorff_distance = pointcloud_metrics['Haussdorf Distance']
                one_sided_hausdorff = pointcloud_metrics['One-Sided Hausdorff']

                print(f"f-Score: {pointcloud_metrics['F-Score']}")
                print(f"Hausdorff Distance: {hausdorff_distance}")
                print(f"One-Sided Hausdorff Distance: {one_sided_hausdorff}")
                print("*" * 50)
                # Write to CSV
                '''data = (f"{filename}; {depth_metrics_dict[filename]['MAE']}; {depth_metrics_dict[filename]['AbsRel']}; {depth_metrics_dict[filename]['RMSE']}; {depth_metrics_dict[filename]['SSIM'][0]}; "
                        f"{depth_metrics_dict[filename]['Delta1']}; {depth_metrics_dict[filename]['Delta2']}; {depth_metrics_dict[filename]['Delta3']}; "
                        f"{depth_metrics_dict[filename]['SI-Boundary-F1']}; {depth_metrics_dict[filename]['SI-Boundary-Recall']}; "
                        f"{chamfer_distance}; {density_aware_cd}; {hausdorff_distance}; {one_sided_hausdorff}; {pcd_fscore} \n")'''
                # Without extra pcd metrics
                data = (f"{filename}; {depth_metrics_dict[filename]['MAE']}; {depth_metrics_dict[filename]['AbsRel']}; {depth_metrics_dict[filename]['RMSE']}; {depth_metrics_dict[filename]['SSIM'][0]}; "
                        f"{depth_metrics_dict[filename]['Delta1']}; {depth_metrics_dict[filename]['Delta2']}; {depth_metrics_dict[filename]['Delta3']}; "
                        f"{depth_metrics_dict[filename]['SI-Boundary-F1']}; {depth_metrics_dict[filename]['SI-Boundary-Recall']}; "
                        f"{chamfer_distance}; {density_aware_cd}; {hausdorff_distance} \n")
                f.write(data)