import torch
import cv2
import numpy as np
from PIL import Image

class ZeroDepthInference:
    def __init__(self, model="ZeroDepth", grayscale=False, view_result=False):
        self.model = model
        self.grayscale = grayscale
        self.view_result = view_result
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zerodepth_model = self.load_model()
        self.zerodepth_model.to(self.DEVICE)

    def load_model(self):
        if self.model == "ZeroDepth":
            return torch.hub.load("TRI-ML/vidar", self.model, pretrained=True, trust_repo=True)
        elif self.model == "PackNet":
            return torch.hub.load("TRI-ML/vidar", self.model, pretrained=True, trust_repo=True)

    def preprocess_image(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Only if no GPU
        raw_image = cv2.resize(input_image, (512, 384))
        raw_image_float = raw_image.astype(np.float32)
        return raw_image_float

    def infer_depth(self, image, camera_json):
        intrinsics = np.asarray([
            [camera_json["fx"], 0, camera_json["cx"]],
            [0, camera_json["fy"], camera_json["cy"]],
            [0, 0, 1]]).astype(np.float32)
        intrinsics = torch.tensor(intrinsics).unsqueeze(0)

        #raw_image_float = self.preprocess_image(image)
        raw_image_float = image.astype(np.float32)
        rgb = torch.tensor(raw_image_float).permute(2, 0, 1).unsqueeze(0) / 255.
        rgb = rgb.to(self.DEVICE)
        intrinsics = intrinsics.to(self.DEVICE)
        with torch.no_grad():
            depth_pred = self.zerodepth_model(rgb, intrinsics)

        t = depth_pred[0]
        new_depth = t.cpu().detach().numpy()
        new_depth = new_depth.squeeze()
        normalized = 255-(cv2.normalize(new_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        # ?
        #normalized = cv2.resize(normalized, (1024, 768))

        return new_depth, normalized

    def visualize_result(self, normalized):
        normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        cv2.imshow("Zero Depth", normalized)
        cv2.waitKey(0)

    def inference_zero_depth(self, image, camera_intrinsics):
      
        new_depth, normalized_depth = self.infer_depth(image, camera_intrinsics)

        if self.view_result:
            self.visualize_result(normalized_depth.astype(np.uint8))

        return new_depth, normalized_depth

if __name__ == "__main__":
    zero = ZeroDepthInference(view_result=True)
    zero.inference_zero_depth("./data/fraunhofer_dataset/fluid tank 2/keyframes/images/193555591334.jpg", {"fx": 768, "fy": 768, "cx": 512, "cy": 384})
