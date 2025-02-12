import torch
import cv2
from PIL import Image

class ZoeDepthInference:
    def __init__(self, model='zoe_nk', grayscale=False, view_result=False):
        self.model = model
        self.grayscale = grayscale
        self.view_result = view_result
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe_model = self.load_model()
        self.zoe_model.to(self.DEVICE)

    def load_model(self):
        if self.model == "zoe_n":
            return torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        elif self.model == "zoe_k":
            return torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)
        elif self.model == "zoe_nk":
            return torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

    def infer_depth(self, image):

        image = Image.fromarray(image)

        # Rotate Image
        #image.rotate(90)
        with torch.no_grad():
            depth_numpy = self.zoe_model.infer_pil(image)
        ## Unused output formats
        #depth_pil = self.zoe_model.infer_pil(image, output_type="pil")
        #depth_tensor = self.zoe_model.infer_pil(image, output_type="tensor")
        normalized = 255 - (cv2.normalize(depth_numpy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        
        if self.view_result:
         
            depth = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            cv2.imshow("Zoe Depth", depth)
            cv2.waitKey(0)

        return depth_numpy, normalized
    
if __name__ == "__main__":
    zoe = ZoeDepthInference(view_result=True)
    zoe.infer_depth("./data/fraunhofer_dataset/fluid tank 2/keyframes/images/193555591334.jpg")
    
    