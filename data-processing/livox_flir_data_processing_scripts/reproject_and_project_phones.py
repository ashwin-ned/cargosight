import open3d as o3d
import open3d.core as o3c
import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt

PHONE = "pixel"  
if PHONE == "pixel":
    width = 3072
    height = 4080
    fx = 5702.00
    fy = 5700.30
    cy = 2037.42
    cx = 1504.80
elif PHONE == "iphone":
    width = 2448
    height = 3264
    fx = 2454.30
    fy = 2455.00
    cy = 1610.50
    cx = 1214.21


CAMERA_INTRINSICS_O3D = o3d.core.Tensor([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

CAMERA_INTRINSICS_NP = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
CAMERA_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# EXTRINSICS = np.array([
#         [0.09082,-0.995863,0.00278281,-0.187652],
#         [-0.0949438,-0.0114402,-0.995417,0.502423],
#         [0.991331,0.0901396,-0.0955901,-0.130249],
#         [0,0,0,1]])
EXTRINSICS = np.eye(4)

######### CAMERA RECTIFICATION PARAMS ##########
if PHONE == "pixel":


    #DISTORTION_COEFFS = np.array([ 0.3063, -2.5996, 8.8741, 0.000000, 0.000000])
    DISTORTION_COEFFS = np.array([0., 0., 0., 0., 0.])
    RECTIFICATION_MAT = np.eye(3)  # Identity since rectification will be done
    PROJECTION_MAT = np.array([[fx, 0.000000, cx, 0.000000],
                              [0.000000, fy, cy, 0.000000],
                              [0.000000, 0.000000, 1.000000, 0.000000]])
elif PHONE == "iphone":
    #DISTORTION_COEFFS = np.array([0.1367, -0.2606, 0.1645, 0.000000, 0.000000])
    DISTORTION_COEFFS = np.array([0., 0., 0., 0., 0.])
    RECTIFICATION_MAT = np.eye(3)  # Identity since rectification will be done
    PROJECTION_MAT = np.array([[fx, 0.000000, cx, 0.000000],
                              [0.000000, fy, cy, 0.000000],
                              [0.000000, 0.000000, 1.000000, 0.000000]])

VIEW = 0
SAVE = 1

RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 1024

def detect_sparse_depth_contours(sparse_depth):

    # Create a binary mask where 255 represents areas to inpaint (missing data)
    mask = np.uint8(sparse_depth == 0)
    # invert the mask to get the projected depth region (non-zero where depth is valid)
    projected_area = cv2.bitwise_not(mask)

    # morphological closing
    kernel = np.ones((7, 7), np.uint8)
    projected_area = cv2.morphologyEx(projected_area, cv2.MORPH_CLOSE, kernel, iterations=5)

    # find the contou
    contours, _ = cv2.findContours(projected_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(projected_area, contours, -1, (255, 255, 255), 3)

    projected_area = cv2.normalize(projected_area, None, 0, 255, cv2.NORM_MINMAX)
    
    return projected_area

def project_pcd_depth(combined_ply, intrinsic, width, height):
    points = np.asarray(combined_ply.points)
    points[:, [0, 1, 2]] = points[:, [1, 2, 0]]
    points = o3c.Tensor(np.asarray(points), dtype=o3c.Dtype.Float32)

    if combined_ply.has_colors():
        colors = o3c.Tensor(np.asarray(combined_ply.colors), dtype=o3c.Dtype.Float32)
    else:
        colors = None

    if combined_ply.has_normals():
        normals = o3c.Tensor(np.asarray(combined_ply.normals), dtype=o3c.Dtype.Float32)
    else:
        normals = None

    t_pcd = o3d.t.geometry.PointCloud(o3c.Device("CPU:0"))   
    t_pcd.point["positions"] = points

    if colors is not None:
        t_pcd.point["colors"] = colors
    
    if normals is not None:
        t_pcd.point["normals"] = normals

    # Perform this transformation to align the point cloud with the camera
    rotate_180_z = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    t_pcd.transform(rotate_180_z)

    combined_ply.paint_uniform_color([0, 1, 0])
    
    depth_reproj = t_pcd.project_to_depth_image(
        width,
        height,
        intrinsic,
        depth_scale=1.0,
        depth_max=40.0
    )
    depth_reproj = np.asarray(depth_reproj)
    
    depth_map = Image.fromarray(depth_reproj[:, :, 0], 'F')
    
    print("Image mode: ", depth_map.mode)
    print("Image size: ", depth_map.size)

    return depth_map

def project_to_3d(projected_depth, rgb_image, intrinsics, extrinsics):

    depth_raw = o3d.geometry.Image(projected_depth)
    rgb_raw = o3d.geometry.Image(rgb_image)
    
    
    # Create RGBD image - Depending on how the Depth was scaled with reprojecting from LidAR need to adjust depth_scale
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_raw, depth_raw, depth_scale=1.0, depth_trunc=10000.0, convert_rgb_to_intensity=False)

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    # Transform the point cloud using the extrinsic matrix
    #pcd.transform(extrinsics)
    return pcd

def rectify_image(rgb_image, camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, view=True):

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
    

    return rectified_image

def nearest_neighbor_interpolation(depth_map, view=True):

    if len(depth_map.shape) > 2:
        print(f"Converting depth map from {depth_map.shape} to single-channel.")
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Convert the depth map to 32-bit float if it's not already
    if depth_map.dtype != np.float32:
        print(f"Converting depth map from {depth_map.dtype} to 32-bit float.")
        depth_map = depth_map.astype(np.float32)
    
    mask = depth_map == 0
    mask = mask.astype(np.uint8)
    dense_depth_map = cv2.inpaint(depth_map, mask.astype(np.uint8), 3, cv2.INPAINT_NS)

    if view:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(depth_map, cmap='inferno')
        ax[0].set_title('Sparse Depth')
        ax[1].imshow(dense_depth_map, cmap='inferno')
        ax[1].set_title('Inpainted Depth')
        plt.show()
    return dense_depth_map.astype(np.float32)

def align_pointclouds(source_pcd, target_pcd, scale_factor=1.0, p2p=True):
  
    source_pcd.scale(scale_factor, center=(0, 0, 0))  # Scale around origin

    rotate_180_z = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    source_pcd.transform(rotate_180_z)
    
    # Step 4: Get the centers of both point clouds
    source_center = source_pcd.get_center()
    target_center = target_pcd.get_center()

    # Step 5: Compute the translation needed to align the centers
    translation = target_center - source_center

    # Step 6: Apply the translation to the source point cloud
    source_pcd.translate(translation)
    # Step 7: Apply ICP for fine alignment (optional, can refine after scaling and translation)
    threshold = 0.0020  # Distance threshold for ICP
    if p2p:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            np.eye(4),  # Initial guess (identity matrix)
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        

    else:
        # estimate normals
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            np.eye(4),  # Initial guess (identity matrix)
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    
    # Apply the ICP transformation to further refine the alignment
    aligned_source_pcd = source_pcd.transform(reg_p2p.transformation)

    print(reg_p2p)
    return aligned_source_pcd, reg_p2p.transformation

def generate_depthmap_from_pointcloud(pointcloud, image_width, image_height, intrinsics_matrix):
    """
    Generates a 32-bit depth map from a projected colored point cloud.
    
    Parameters:
    - pointcloud (open3d.geometry.PointCloud): The input point cloud.
    - image_width (int): The width of the desired depth map.
    - image_height (int): The height of the desired depth map.
    - intrinsics_matrix (np.ndarray): 3x3 camera intrinsics matrix for projection.
    
    Returns:
    - depth_map (np.ndarray): The resulting 32-bit depth map.
    """

    # Initialize depth map with infinite values (max depth initially)
    depth_map = np.full((image_height, image_width), np.inf, dtype=np.float32)

    # Get points and colors from point cloud
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)  # in case you want to use color data

    # Project 3D points to 2D image plane
    for point in points:
        # Decompose point coordinates
        x, y, z = point

        # Project 3D points to 2D using intrinsic matrix
        if z > 0:  # Ignore points behind the camera
            pixel = intrinsics_matrix @ np.array([x, y, z])
            u, v = int(pixel[0] / pixel[2]), int(pixel[1] / pixel[2])

            # Check if projected points are within image boundaries
            if 0 <= u < image_width and 0 <= v < image_height:
                # Update depth map with closer points
                depth_map[v, u] = min(depth_map[v, u], z)
    
    # Convert infinite values to 0 for visualization
    depth_map[depth_map == np.inf] = 0

    return depth_map
    

if __name__ == "__main__":

    for i in range(1, 19):
        container = i
        container_folder_name = f"/home/admin-anedunga/Desktop/livox_flir_raw_data/data/Container{container:02d}/livox_horizon/"
        save_path = f"/home/admin-anedunga/Desktop/livox_flir_raw_data/phone_images/processed/"
        content = os.listdir(container_folder_name)
        plys = []
        for file_name in content:
            if file_name.endswith(".ply"):
                pcd = o3d.io.read_point_cloud(os.path.join(container_folder_name, file_name))
                plys.append(pcd)
        
        combined_ply = o3d.geometry.PointCloud()
        for ply in plys:
            combined_ply += ply

        if VIEW:
            o3d.visualization.draw_geometries([combined_ply], lookat = [0, 0, 0], up = [0, 0, 1], front = [-1, 0, 0], zoom = 0.25)

        
        projected_depth = project_pcd_depth(combined_ply, CAMERA_INTRINSICS_O3D, width, height)
        
        cv_depth = np.array(projected_depth)
        normalized_dm = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_view = cv2.applyColorMap(normalized_dm.astype(np.uint8), cv2.COLORMAP_INFERNO)

        if VIEW:
            cv2.imshow("Reprojected Depth Map", depth_map_view)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if SAVE:
            projected_depth.save(f"{save_path}/{PHONE}{container:02d}_projected_depth.tiff")
            cv2.imwrite(f"{save_path}/{PHONE}{container:02d}_projected_depth_view.png", depth_map_view)
            
        # Load images
        if PHONE == "pixel":
            image_folder_name = f"/home/admin-anedunga/Desktop/livox_flir_raw_data/phone_images/PXL7_Images/PXL_Container{container:02d}.jpg"
        else:
            image_folder_name = f"/home/admin-anedunga/Desktop/livox_flir_raw_data/phone_images/iPhone_Images/iPhone_Container{container:02d}.jpg"

        rgb_image = cv2.imread(image_folder_name, cv2.IMREAD_COLOR)
       
        

        rectified_rgb_image = rectify_image(rgb_image, CAMERA_INTRINSICS_NP, DISTORTION_COEFFS, RECTIFICATION_MAT, PROJECTION_MAT, view=VIEW)

        enhanced_rgb_image = cv2.cvtColor(rectified_rgb_image, cv2.COLOR_BGR2RGB)
        # Use either the enhanced_rgb_image or the rgb_image when projecting to pointcloud
        enhanced_rgb_image = cv2.resize(enhanced_rgb_image, (cv_depth.shape[1], cv_depth.shape[0]))
        if VIEW:
            cv2.imshow("Enhanced RGB Image", enhanced_rgb_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if SAVE:
            #cv2.imwrite(f"{save_path}rectified_rgb.png", rectified_rgb_image)
            enhanced_rgb_image = cv2.cvtColor(enhanced_rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{save_path}/{PHONE}{container:02d}_enhanced_rgb.png", enhanced_rgb_image)
        

    
        reprojected_pcd = project_to_3d(cv_depth, enhanced_rgb_image, CAMERA_INTRINSICS, EXTRINSICS)

    
        aligned_projected_pcd, transformation = align_pointclouds(reprojected_pcd, combined_ply)
        print("Transformation Matrix", transformation)
        if VIEW:
            o3d.visualization.draw_geometries([aligned_projected_pcd], lookat = [0, 0, 0], up = [0, 0, 1], front = [-1, 0, 0], zoom = 0.75)
            o3d.visualization.draw_geometries([aligned_projected_pcd, combined_ply], lookat = [0, 0, 0], up = [0, 0, 1], front = [-1, 0, 0], zoom = 0.75)

        if SAVE:
            # o3d.io.write_point_cloud(f"container{container:02d}_aligned_projected.pcd", aligned_projected_pcd)
            # o3d.io.write_point_cloud(f"container{container:02d}_combined_lidar.pcd", combined_ply)
            o3d.io.write_point_cloud(f"{save_path}/{PHONE}{container:02d}_projected_pointcloud.pcd", aligned_projected_pcd)
            #o3d.io.write_point_cloud(f"{save_path}lidar.pcd", combined_ply)

        # Inpaint Depth Map
        dense_depth_map = nearest_neighbor_interpolation(cv_depth, view=VIEW)
        depth_filter_mask = detect_sparse_depth_contours(cv_depth)
        interpolated_depth = np.where(depth_filter_mask==255, dense_depth_map, depth_filter_mask)
        dense_depth_map = interpolated_depth

        # project inpainted depth
        # inpainted_pcd = project_to_3d(dense_depth_map, rgb_image, CAMERA_INTRINSICS, EXTRINSICS)
        # o3d.visualization.draw_geometries([inpainted_pcd])

        # Save inpainted depthmap as .tiff
        if SAVE:
            dense_depth_map = Image.fromarray(dense_depth_map)
            dense_depth_map.save(f"{save_path}/{PHONE}{container:02d}_dense_depth_map.tiff")
            # dense_depth_map = np.array(dense_depth_map).astype(np.uint8)
            # dense_depth_map = cv2.normalize(dense_depth_map, None, 0, 255, cv2.NORM_MINMAX)
            # colored_depth = cv2.applyColorMap(dense_depth_map, cv2.COLORMAP_INFERNO)
            # cv2.imwrite(f"{save_path}/{PHONE}{container:02d}_dense_depth_view.png", colored_depth)
            plt.imsave(f"{save_path}/{PHONE}{container:02d}_dense_depth_view.png", dense_depth_map, cmap='inferno')
