# /usr/bin/python3
import os
import cv2
import numpy as np
import rosbag
import open3d as o3d
from pyrosenv.cv_bridge import CvBridge, CvBridgeError
from pyrosenv.sensor_msgs import point_cloud2
from pyrosenv.sensor_msgs.msg import PointCloud2, Image, Imu, CameraInfo
from plyfile import PlyData, PlyElement
import time


# Initialize the CvBridge class to convert ROS images to OpenCV format
bridge = CvBridge()

# Directories for saving the extracted data
output_folder = "/home/admin-anedunga/Desktop/livox_flir_data_processing/rosbags/error/"
livox_folder = os.path.join(output_folder, "livox_horizon")
blackfly_folder = os.path.join(output_folder, "flir_blackfly")
#imu_camera_param_folder = os.path.join(output_folder, "imu_camera_param")

# Create the directories if they don't exist
# os.makedirs(livox_folder, exist_ok=True)
# os.makedirs(blackfly_folder, exist_ok=True)

# Define the topics to extract from the rosbag
topics = ['/camera/camera_info', '/camera/image_color', '/camera/image_raw' '/livox/lidar', '/livox/imu']

def process_bag(bag_file):
    """Process a single rosbag file and extract data."""
    print(f"Processing {bag_file}")
    time.sleep(0.2)
    with rosbag.Bag(bag_file, 'r') as bag:
        print(f"Extracting data from {bag_file}")
        for topic, msg, t in bag.read_messages(topics=topics):
            # msg = 1
            print(t)
            # filter images by same second timestamp
            timestamp = str(t.to_sec())
            #timestamp = float(timestamp)
            #timestamp = int(timestamp/1000)
            #print("16bit:", timestamp)
            
            #print(topic, msg)
            # check if the timestamps are the same 
            #curr_tstmp = timestamp
            # Process image data from /camera/image_color
      
            if topic == '/camera/image_raw':
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8") #bgr8 or bgr16
                    image_file = os.path.join(blackfly_folder, f"{timestamp}_rgb_blackfly.png")
                    cv2.imwrite(image_file, cv_image)
                    print(f"Saved image: {image_file}")
                except CvBridgeError as e:
                    print(f"Error converting image: {e}")

            # Process point cloud data from /livox/lidar & save as PLY file
            elif topic == '/livox/lidar':
                pointcloud_file = os.path.join(livox_folder, f"{timestamp}_pcd_livox.ply")
                points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                
                # Prepare data for PlyData
                vertex = np.array([(p[0], p[1], p[2]) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

                # Save the point cloud as a PLY file
                with open(pointcloud_file, 'wb') as ply_file:
                    ply_data.write(ply_file)
                print(f"Saved point cloud: {pointcloud_file}")

                # # Save the pointcloud as a PCD file
                # pcd = o3d.io.read_point_cloud(pointcloud_file)
                # pcd_file = os.path.join(livox_folder, f"{timestamp}_pcd_livox.pcd")
                # o3d.io.write_point_cloud(pcd_file, pcd)

            # msg += 1
            # # Process IMU data from /livox/imu
            # elif topic == '/livox/imu':
            #     imu_file = os.path.join(imu_camera_param_folder, f"{timestamp}_imu.txt")
            #     print(imu_file)
            #     with open(imu_file, 'w') as imu_f:
            #         imu_f.write(str(msg))
            #     print(f"Saved IMU data: {imu_file}")

            # # Process CameraInfo from /camera/camera_info
            # elif topic == '/camera/camera_info':
            #     camera_info_file = os.path.join(imu_camera_param_folder, f"{timestamp}_camera_info.txt")
            #     with open(camera_info_file, 'w') as camera_f:
            #         camera_f.write(str(msg))
            #     print(f"Saved camera info: {camera_info_file}")

            # break from the loop after 10 messages
            # if msg > 10:
            #     break

def process_rosbags(bag_dir):
    """Process all rosbag files in the given directory."""
    for filename in os.listdir(bag_dir):
        if filename.endswith(".bag"):
            bag_file = os.path.join(bag_dir, filename)
            process_bag(bag_file)

if __name__ == '__main__':
    # Directory containing your ROS bags
    rosbag_dir = "/home/admin-anedunga/Desktop/livox_flir_data_processing/rosbags/error"
    
    # Process all rosbags in the directory
    process_rosbags(rosbag_dir)
