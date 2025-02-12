import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

class DepthMapProcessor:
    def __init__(self, depth_path, rgb_path, extrinsics, intrinsics):
        self.depth_path = depth_path
        self.rgb_path = rgb_path
        self.depth_map = None
        self.rgb_image = None
        self.rgb_rectified = None
        self.dense_depth_map = None
        self.extrinsics = extrinsics  # 4x4 extrinsic matrix
        self.intrinsics = intrinsics  # o3d.camera.PinholeCameraIntrinsic

    def add_camera_and_lidar_boxes(self, camera_extrinsics, lidar_pcd):
        """
        Add small blue and orange boxes to represent the LiDAR and camera locations.

        Args:
            camera_extrinsics: 4x4 extrinsic matrix of the camera (T_{L}^{C}, LiDAR to Camera).
            lidar_pcd: The point cloud representing the LiDAR points.

        Returns:
            geometries: List of Open3D geometries including the LiDAR and camera boxes.
        """
        # Step 1: Create small boxes
        lidar_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.1)
        camera_box = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)

        # Step 2: Paint the boxes (LiDAR -> Blue, Camera -> Orange)
        lidar_box.paint_uniform_color([0, 0, 1])   # Blue
        camera_box.paint_uniform_color([1, 0.5, 0])  # Orange

        # Step 3: Assume the camera is at the origin (0, 0, 0) of the point cloud
        camera_position = np.array([0, 0, 0])

        # Step 4: Transform the camera position using the extrinsic matrix to get the LiDAR's position
        lidar_position_homogeneous = np.dot(camera_extrinsics, np.append(camera_position, 1))  # Homogeneous coordinates
        lidar_position = lidar_position_homogeneous[:3]  # Extract the x, y, z coordinates

        # Step 5: Translate the camera box to the camera position (origin)
        camera_box.translate(camera_position)

        # Step 6: Translate the LiDAR box to the LiDAR position
        lidar_box.translate(lidar_position)
        # Rotate 180 degrees about the X-axis to flip the Z-axis
        #z_flip_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi, np.pi])  # 180 degrees rotation along X
        #lidar_box.rotate(z_flip_rotation, center=lidar_box.get_center())

        # Step 7: Collect and return geometries (boxes and point cloud)
        geometries = [lidar_box, camera_box, lidar_pcd]
        return geometries
    
    def read_data(self):
        # Read depth map
        self.depth_map = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        #self.depth_map = cv2.rotate(self.depth_map, cv2.ROTATE_180)
        if self.depth_map is None:
            raise ValueError("Failed to read the depth map.")
        # Read RGB image
        self.rgb_image = cv2.imread(self.rgb_path, cv2.IMREAD_COLOR)
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        # # Padding to see if it fixes the projection
        # pad_width = 20
        # padding = np.zeros((self.rgb_image.shape[0], pad_width,  self.rgb_image.shape[2]), dtype= self.rgb_image.dtype)
        # # Concatenate the padding to the left of the image
        # self.rgb_image = np.concatenate((padding, self.rgb_image), axis=1)

        print(f"Depth map shape: {self.depth_map.shape}, RGB image shape: {self.rgb_image.shape}")
       
        if self.rgb_image is None:
            raise ValueError("Failed to read the RGB image.")
        # Resize RGB to depth dimensions
        #self.rgb_image = cv2.resize(self.rgb_image, (1285, 1044))
        
    def nearest_neighbor_interpolation(self, view=True):

        if len(self.depth_map.shape) > 2:
            print(f"Converting depth map from {self.depth_map.shape} to single-channel.")
            self.depth_map = cv2.cvtColor(self.depth_map, cv2.COLOR_BGR2GRAY)
    
        # Convert the depth map to 32-bit float if it's not already
        if self.depth_map.dtype != np.float32:
            print(f"Converting depth map from {self.depth_map.dtype} to 32-bit float.")
            self.depth_map = self.depth_map.astype(np.float32)
        
        mask = self.depth_map == 0
        mask = mask.astype(np.uint8)
        self.dense_depth_map = cv2.inpaint(self.depth_map, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

        if view:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(self.depth_map, cmap='inferno')
            ax[0].set_title('Sparse Depth')
            ax[1].imshow(self.dense_depth_map, cmap='inferno')
            ax[1].set_title('Inpainted Depth')
            plt.show()

    def project_to_3d(self):

        depth_raw = o3d.geometry.Image(self.depth_map)
        rgb_raw = o3d.geometry.Image(self.rgb_rectified)
        
        
        # Create RGBD image - Depending on how the Depth was scaled with reprojecting from LidAR need to adjust depth_scale
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_raw, depth_raw, depth_scale=1.0, depth_trunc=10000.0, convert_rgb_to_intensity=False)

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)
        
        # Transform the point cloud using the extrinsic matrix
        pcd.transform(self.extrinsics)
        return pcd

    def compare_pointclouds(self, pcd1, pcd2, voxel_size=0.05):
        # Voxel downsample for efficiency
        pcd1_down = pcd1.voxel_down_sample(voxel_size)
        pcd2_down = pcd2.voxel_down_sample(voxel_size)

        # Compute IoU based on overlap of voxels
        pcd1_voxels = np.asarray(pcd1_down.points)
        pcd2_voxels = np.asarray(pcd2_down.points)

        union = len(np.union1d(pcd1_voxels.view([('', pcd1_voxels.dtype)] * pcd1_voxels.shape[1]), 
                               pcd2_voxels.view([('', pcd2_voxels.dtype)] * pcd2_voxels.shape[1])))
        intersection = len(np.intersect1d(pcd1_voxels.view([('', pcd1_voxels.dtype)] * pcd1_voxels.shape[1]), 
                                          pcd2_voxels.view([('', pcd2_voxels.dtype)] * pcd2_voxels.shape[1])))

        iou = intersection / union
        return iou
    
    def rectify_image(self, rgb_image, camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, view=True):
        """
        Rectify the RGB image using the given camera matrix, distortion coefficients,
        rectification, and projection matrices.
        
        Args:
            rgb_image: Raw distorted image from the camera
            camera_matrix: 3x3 intrinsic camera matrix
            dist_coeffs: Distortion coefficients
            rectification_matrix: Rectification matrix (3x3)
            projection_matrix: 3x4 projection matrix

        Returns:
            rectified_image: Rectified RGB image
        """
        h, w = rgb_image.shape[:2]
        
        # Compute the rectification transformation maps
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, (w, h), cv2.CV_32FC1
        )
        
        # Apply the rectification to the image
        rectified_image = cv2.remap(rgb_image, map1, map2, cv2.INTER_LINEAR)

        # plot original and rectified images side by side
        if view:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(rgb_image)
            ax[0].set_title('Original Image')
            ax[1].imshow(rectified_image)
            ax[1].set_title('Rectified Image')
            plt.show()
        
        self.rgb_rectified = rectified_image
        return rectified_image
    
    def align_pointclouds(self, source_pcd, target_pcd, scale_factor=1.0):
        """
        Align the source point cloud to the target point cloud by scaling, rotating, and centering.

        Args:
            source_pcd: Source point cloud (projected point cloud)
            target_pcd: Target point cloud (original LiDAR point cloud)
            scale_factor: Factor by which to scale the source point cloud

        Returns:
            aligned_source_pcd: The aligned source point cloud
            transformation: The transformation matrix used for alignment
        """
        # Step 1: Scale the source point cloud
        source_pcd.scale(scale_factor, center=(0, 0, 0))  # Scale around origin

        # Step 2: Apply 90-degree rotation around x-axis (out of screen)
        R_x = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        source_pcd.rotate(R_x, center=(0, 0, 0))

        # Step 3: Apply 90-degree rotation around y-axis (away from viewer)
        R_y = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0]])
        source_pcd.rotate(R_y, center=(0, 0, 0))
        R_x_180 = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        source_pcd.rotate(R_x_180, center=(0, 0, 0))
        R_z_180 = np.array([[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])
        source_pcd.rotate(R_z_180, center=(0, 0, 0))

        # Step 4: Get the centers of both point clouds
        source_center = source_pcd.get_center()
        target_center = target_pcd.get_center()

        # Step 5: Compute the translation needed to align the centers
        translation = target_center - source_center

        # Step 6: Apply the translation to the source point cloud
        source_pcd.translate(translation)
        # Step 7: Apply ICP for fine alignment (optional, can refine after scaling and translation)
        threshold = 0.0005  # Distance threshold for ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            np.eye(4),  # Initial guess (identity matrix)
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Apply the ICP transformation to further refine the alignment
        aligned_source_pcd = source_pcd.transform(reg_p2p.transformation)
        
        return aligned_source_pcd, reg_p2p.transformation


# Example usage
# extrinsics = np.array([  
#     [-0.0390928,-0.999055,-0.0190137,0.231959],
#     [-0.0422467,0.0206638,-0.998894,0.604373],
#     [0.998342,-0.0382463,-0.0430146,-0.168554],
#     [0, 0, 0, 1]
# ])
# After Rectification

# extrinsics = np.array([
#         [0.33822,-0.938604,-0.0680429,-1.15113],
#         [-0.138813,0.0217542,-0.99008,1.09087],
#         [0.930773,0.34431,-0.122933,0.374216],
#         [0,  0,  0,  1]])

extrinsics = np.array([
        [0.090,-0.995,0.002, -0.187],
        [-0.094,-0.011,-0.995,0.502],
        [0.991,0.090,-0.095,-0.130],
        [0,0,0,1]])

# Camera intrinsic parameters (replace with actual values)
camera_intrinsics_np = np.array([[853.673, 0.0, 622.400],
                                    [0.0, 854.153, 507.496],
                                    [0.0, 0.0, 1.0]])

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=1280, height=1024, fx=853.673, fy=854.153, cx=622.400, cy=507.496
)

# Camera rectification parameters
dist_coeffs = np.array([-0.248604, 0.094251, -0.000100, -0.000317, 0.000000])
rectification_matrix = np.eye(3)  # Identity since rectification is already done
projection_matrix = np.array([[739.853, 0.000000, 617.655, 0.000000],
                              [0.000000, 777.037, 506.513, 0.000000],
                              [0.000000, 0.000000, 1.000000, 0.000000]])

# Initialize processor with paths, extrinsic, and intrinsic parameters
container = 11
processor = DepthMapProcessor(f'./c_{container}/c{container}_depth.tiff', f'./c_{container}/c{container}_flir_rgb.png', extrinsics, camera_intrinsics)

# Read and process data
processor.read_data()
rectified_rgb = processor.rectify_image(processor.rgb_image, camera_intrinsics_np, dist_coeffs, rectification_matrix, projection_matrix)

processor.nearest_neighbor_interpolation()
projected_pcd = processor.project_to_3d()

# Load original LiDAR point cloud
lidar_pcd = o3d.io.read_point_cloud(f"./c_{container}/c{container}_lidar.pcd")  
o3d.visualization.draw_geometries([lidar_pcd])
#o3d.io.write_point_cloud("c6_lidar.pcd", lidar_pcd) 
# Align the point clouds using ICP for scale matching
aligned_projected_pcd, transformation = processor.align_pointclouds(projected_pcd, lidar_pcd)
print("Transformation Matrix", transformation)

# # Compare the point clouds using IoU
# iou = processor.compare_pointclouds(aligned_projected_pcd, lidar_pcd)
# print(f"IoU between the projected and LiDAR point clouds: {iou:.4f}")

# Visualize both point clouds
o3d.visualization.draw_geometries([projected_pcd])
o3d.io.write_point_cloud(f"{container:02d}_projected.pcd", projected_pcd)
# Color the LiDAR point cloud green
lidar_pcd.paint_uniform_color([0, 1, 0])  # RGB: [0, 1, 0] -> Green
# Color the projected point cloud red
aligned_projected_pcd.paint_uniform_color([1, 0, 0])  # RGB: [1, 0, 0] -> Red
# Visualize the point clouds with the assigned colors
o3d.visualization.draw_geometries([aligned_projected_pcd, lidar_pcd])

# Call the function to add the LiDAR and camera boxes, with the LiDAR 6 meters away from the center
geometries = processor.add_camera_and_lidar_boxes(extrinsics, lidar_pcd)

# Visualize the point cloud with the LiDAR and camera boxes
o3d.visualization.draw_geometries(geometries)