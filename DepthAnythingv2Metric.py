import os, sys
import numpy as np
import cv2
from PIL import Image
import torch
from models.depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

dav2_dir = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-reconstruction/models/depthanythingv2/metric_depth/"
sys.path.insert(0, dav2_dir)
os.chdir(dav2_dir)

class DepthAnythingInference:
    def __init__(self, encoder="vits", grayscale=False, view_result=False):
        self.encoder = encoder
        self.grayscale = grayscale
        self.view_result = view_result
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def inference_depth_anything(self, image):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
        
        encoder = self.encoder
        # Change as required, Pass as parameter
        max_depth = 20
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        dataset = 'hypersim'
        # 'hypersim' for metric indoor model, 'vkitti' for metric outdoor model
        model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        model.eval()
        device = torch.device(self.DEVICE)
        model.to(device)
        
        depth = model.infer_image(image)# HxW depth map in meters in numpy
        # Resize depth prediction to match the original image size
        #depth_array = np.array(depth).squeeze()
        #resized_pred = Image.fromarray(depth_array).resize((image.shape), Image.NEAREST)
        print(depth.shape, depth.dtype, depth)

       
        if self.view_result:

            normalized = 255 - (cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            norm_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            cv2.imshow("DepthAnythingv2 metric", norm_depth)
            cv2.waitKey(0)


        return depth
  

      
if __name__ == "__main__":
    rgb_file = '/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/MDE Data/PhoneDepth/Huawei/test/images/000012.jpg'
    rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(rgb, (640, 480))
    depth_anything = DepthAnythingInference(view_result=True)
    depth_anything.inference_depth_anything(rgb)