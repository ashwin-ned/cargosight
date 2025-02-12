import torch
import cv2
import numpy as np
from models.unidepth.utils import colorize, image_grid
from models.unidepth.models import UniDepthV1, UniDepthV2

#ARKIT Iphone Intrinsics {"fx": 768, "fy": 768, "cx": 512, "cy": 384}
# TODO: Check IPad Intrinsics 
class UniDepthInference:
    def __init__(self, model='unidepthv1', intrinsics=None, grayscale=False, view_result=False):
        self.model = model
        self.grayscale = grayscale
        self.intrinsics = intrinsics
        self.view_result = view_result
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.unidepth_model = self.load_model()
        self.unidepth_model.to(self.DEVICE)
        

    def normalize_depth(self, depth):
        normalized = 255 - (cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        normalized_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        return normalized_depth

    def load_model(self):
        if self.model == "unidepthv1":
            return UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        elif self.model == "unidepthv1ConvNext":
            return UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-cnvnxtl")
        elif self.model == "unidepthv2":
            return UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        

    def infer_depth(self, image):

      
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1) # C, H, W

        if self.intrinsics is not None:
            
            intrinsics_tensor = torch.from_numpy(self.intrinsics)
            predictions = self.unidepth_model.infer(rgb, intrinsics_tensor)
        else:
            predictions = self.unidepth_model.infer(rgb)

        # Metric Depth Estimation
        depth = predictions["depth"].squeeze().cpu().numpy()

        # Point Cloud in Camera Coordinate
        pointcloud = predictions["points"].squeeze().cpu().numpy()

        # Intrinsics Prediction
        intrinsics = predictions["intrinsics"].squeeze().cpu().numpy()
       
        # Normalized Depth
        normalized_depth = self.normalize_depth(depth)

        if self.view_result:
            cv2.imshow("UniDepth Depth", normalized_depth)
            cv2.waitKey(0)

        return depth, normalized_depth, pointcloud
    

if __name__ == "__main__":
    rgb_file = '/home/admin-anedunga/Desktop/benchmarking_data/ten-container-dataset/Scan_I_16_20_37/2024_04_11_15_46_01/frame_00000.jpg'
    rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_file = '/home/admin-anedunga/Desktop/benchmarking_data/ten-container-dataset/Scan_I_16_20_37/2024_04_11_15_46_01/depth_00000.png'
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    camera_intrinsics = np.array([[1597.4, 0.0, 960.0],  # f_x, 0, c_x
                                  [0.0, 1597.4, 720.0],  # 0, f_y, c_y
                                  [0.0, 0.0, 1.0]       # 0, 0, 1
                                  ]).astype(np.float32)

    # Supported models: metric3d_convnext_large, metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
   
    model = UniDepthInference(model='unidepthv1', view_result=True, intrinsics=camera_intrinsics)
    
    predicted_depth, normalized_depth, pcd = model.infer_depth(rgb)
    print(predicted_depth, predicted_depth.min(), predicted_depth.max())
    print(depth, depth.min(), depth.max())
 
