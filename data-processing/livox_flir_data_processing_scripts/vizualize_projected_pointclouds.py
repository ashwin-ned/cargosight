import open3d as o3d
import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt

#import nksr
import torch

DEBUG = True  # Set to False to disable visualization and debug outputs

CAMERA = "pixel"  
if CAMERA == "pixel":
    width = 3072
    height = 4080
    fx = 5702.00
    fy = 5700.30
    cy = 2037.42
    cx = 1504.80
elif CAMERA == "iphone":
    width = 2448
    height = 3264
    fx = 2454.30
    fy = 2455.00
    cy = 1610.50
    cx = 1214.21
elif CAMERA == "flir":
    width = 1280
    height = 1024
    fx = 853.67375 
    cx = 622.40052
    fy = 854.15395
    cy = 507.49693

def paint_pcd_gradient(pcd, color_map=True):
    points = np.asarray(pcd.points)

    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)

    if not color_map:
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = normalized_z
        colors[:, 1] = 1 - normalized_z
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        colormap = plt.cm.viridis
        colors = colormap(normalized_z)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def reconstruct_with_nksr(points, normals, colors):
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    
    input_xyz = torch.tensor(points, dtype=torch.float32, device=device)
    input_normal = torch.tensor(normals, dtype=torch.float32, device=device)
    input_color = torch.tensor(colors, dtype=torch.float32, device=device)

    field = reconstructor.reconstruct(input_xyz, input_normal)
    field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))

    mesh = field.extract_dual_mesh(mise_iter=2)

    # Visualize using Open3D
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.v.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.f.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.c.cpu().numpy())

    o3d.visualization.draw_geometries([mesh_o3d], mesh_show_back_face=True)

if __name__ == "__main__":
    container = f'{3:02d}'

    # Load file paths
    flir_rgb_path = f"/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/flir-rgb-depth/C{container}_rgb.png"
    flir_sparse_path = f"/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/flir-rgb-depth/C{container}_depth.tiff"
    flir_depth_path = f"/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/flir-rgb-depth/C{container}_inpainted.tiff"
    ground_truth_pcd_path = f"/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/pointclouds/C{container}_lidar.pcd"

    # Load RGB and depth images
    flir_rgb = cv2.imread(flir_rgb_path, cv2.IMREAD_COLOR)
    flir_rgb = cv2.cvtColor(flir_rgb, cv2.COLOR_BGR2RGB)
    flir_depth = cv2.imread(flir_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Load point cloud
    ground_truth_pcd = o3d.io.read_point_cloud(ground_truth_pcd_path)

    # Estimate normals
    ground_truth_pcd = estimate_normals(ground_truth_pcd)

    # Visualize if DEBUG is True
    if DEBUG:
        colored_ground_truth_pcd = paint_pcd_gradient(ground_truth_pcd)
        o3d.visualization.draw_geometries([colored_ground_truth_pcd])
        print("Number of points in ground truth point cloud:", len(ground_truth_pcd.points))

    # Convert Open3D point cloud to numpy arrays
    points = np.asarray(ground_truth_pcd.points)
    normals = np.asarray(ground_truth_pcd.normals)
    colors = np.asarray(ground_truth_pcd.colors)
