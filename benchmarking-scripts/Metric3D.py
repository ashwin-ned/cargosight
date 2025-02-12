import sys
import os
import torch
import cv2
from PIL import Image
import numpy as np
# DEBUG
import time

try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

metric3d_dir = "/home/admin-anedunga/Desktop/mde-benchmark/mde-reconstruction/models/metric3d"
sys.path.insert(0, metric3d_dir)
os.chdir(metric3d_dir)

from models.metric3d.mono.model.monodepth_model import get_configured_monodepth_model

class Metric3DInference:
    def __init__(self, model='metric3d_convnext_large', grayscale=False, view_result=False):
        self.MODEL_TYPE = {
            'ConvNeXt-Tiny': {
                # Model Not Released, Coming Soon
            },
            'ConvNeXt-Large': {
                'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
                'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
            },
            'ViT-Small': {
                'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
                'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
            },
            'ViT-Large': {
                'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
                'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
            },
            'ViT-giant2': {
                'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
            },}
        self.model = model
        self.grayscale = grayscale
        self.view_result = view_result

        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.metric_model = self.load_model()
        #print(self.metric_model)
      
    def metric3d_convnext_large(self, pretrain=False, **kwargs):
        '''
        Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
        For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
        Args:
            pretrain (bool): whether to load pretrained weights.
        Returns:
            model (nn.Module): a Metric3D model.
        '''
        cfg_file = self.MODEL_TYPE['ConvNeXt-Large']['cfg_file']
        ckpt_file = self.MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

        cfg = Config.fromfile(cfg_file)
        model = get_configured_monodepth_model(cfg)
        if pretrain:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
            )
        return model

    def metric3d_vit_small(self, pretrain=False, **kwargs):
        '''
        Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
        For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
        Args:
            pretrain (bool): whether to load pretrained weights.
        Returns:
            model (nn.Module): a Metric3D model.
        '''
        cfg_file = self.MODEL_TYPE['ViT-Small']['cfg_file']
        ckpt_file = self.MODEL_TYPE['ViT-Small']['ckpt_file']

        cfg = Config.fromfile(cfg_file)
        model = get_configured_monodepth_model(cfg)
        if pretrain:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
            )
        return model

    def metric3d_vit_large(self, pretrain=False, **kwargs):
        '''
        Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
        For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
        Args:
            pretrain (bool): whether to load pretrained weights.
        Returns:
            model (nn.Module): a Metric3D model.
        '''
        cfg_file = self.MODEL_TYPE['ViT-Large']['cfg_file']
        ckpt_file = self.MODEL_TYPE['ViT-Large']['ckpt_file']

        cfg = Config.fromfile(cfg_file)
        model = get_configured_monodepth_model(cfg)
        if pretrain:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
            )
        return model

    def metric3d_vit_giant2(self, pretrain=False, **kwargs):
        '''
        Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
        For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
        Args:
            pretrain (bool): whether to load pretrained weights.
        Returns:
            model (nn.Module): a Metric3D model.
        '''
        cfg_file = self.MODEL_TYPE['ViT-giant2']['cfg_file']
        ckpt_file = self.MODEL_TYPE['ViT-giant2']['ckpt_file']

        cfg = Config.fromfile(cfg_file)
        model = get_configured_monodepth_model(cfg)
        if pretrain:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
            strict=False,
            )
        return model
   

    def load_model(self):
        # metric3d_convnext_large, metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
        # ConvNeXt Large is V1, Rest is V2

        if self.model == 'metric3d_convnext_large':
            #return self.metric3d_convnext_large(pretrain=True)
            return torch.hub.load('yvanyin/metric3d', 'metric3d_convnext_large', pretrain=True)
        elif self.model == 'metric3d_vit_small':
            #return self.metric3d_vit_small(pretrain=True)
            return torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
        elif self.model == 'metric3d_vit_large':
            return torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        elif self.model == 'metric3d_vit_giant2':
            return torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)


    def infer_depth(self, image, intrinsic):
        #FIXME: Some issue with Focal Length/Intrinsics or Scaling 
        
        ############# Pre-Processing #############
        ## Pre-process input size to fit pretrained model
        # keep ratio resize
        if self.model == 'metric3d_convnext_large':
            # original 
            input_size = (544, 1216)
            #input_size = (384, 512)
        else:
            # Original
            input_size = (616, 1064)
            #input_size = (384, 512)
        

        h, w = image.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)   
        image_resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        # Padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = image_resized.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        image_resized = cv2.copyMakeBorder(image_resized, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        ## Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        image_resized = torch.from_numpy(image_resized.transpose((2, 0, 1))).float()
        image_resized = torch.div((image_resized - mean), std)
        image_resized = image_resized[None, :, :, :].cuda()

        ### We are now in canonical camera sapce ###
        # Inference
        self.metric_model.cuda().eval()
        with torch.no_grad():
            predicted_depth, confidence, output_dict = self.metric_model.inference({'input': image_resized})

        ############# Post-Processing #############
        # Un-pad the results
        predicted_depth = predicted_depth.squeeze()
        predicted_depth = predicted_depth[pad_info[0] : predicted_depth.shape[0] - pad_info[1], pad_info[2] : predicted_depth.shape[1] - pad_info[3]]
      
        # Downsample to gt depth size
        predicted_depth = torch.nn.functional.interpolate(predicted_depth[None, None, :, :], image.shape[:2], mode='bilinear').squeeze()

        #### de-canonical transform
        #print("Predicted Depth Info:", predicted_depth.shape, predicted_depth.dtype, predicted_depth)
        canonical_to_real_scale = intrinsic[0] / 7000.0  # 1000.0 is the focal length of canonical camera they gave, but 5k-8k works best 
        predicted_depth = predicted_depth * canonical_to_real_scale # now the depth is metric

        # Max Depth for ARKit Data is 5m so clamp to values (0,5)
        predicted_depth = torch.clamp(predicted_depth, 0, 5)
        predicted_depth = predicted_depth.cpu().numpy()
        confidence = confidence.cpu().numpy()

        pred_depth_norm = cv2.normalize(predicted_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if self.view_result:
            # Normalize to view
            
            #print("Predicted Depth Info:", predicted_depth.shape, predicted_depth.dtype)
            #print(predicted_depth, "Predicted Depth Max/Min", predicted_depth.min(), predicted_depth.max())
            cv2.imshow("Image", image)
            cv2.imshow("Depth", pred_depth_norm)
            cv2.waitKey(0)

        # Trying to round
        predicted_depth = np.round(predicted_depth, 3)
        return predicted_depth, pred_depth_norm, confidence



    
if __name__ == "__main__":
    rgb_file = '/home/admin-anedunga/Desktop/benchmarking_data/ten-container-dataset/Scan_I_16_20_37/2024_04_11_15_46_01/frame_00000.jpg'
    rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_file = '/home/admin-anedunga/Desktop/benchmarking_data/ten-container-dataset/Scan_I_16_20_37/2024_04_11_15_46_01/depth_00000.png'
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    # IPad 12 intrinsics
    intrinsics = [1598.798, 1598.798, 949.251, 722.789]
    gt_depth_scale = 1.0


    # Supported models: metric3d_convnext_large, metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
    strt = time.time()
    model = Metric3DInference(model='metric3d_convnext_large', view_result=True)
    
    predicted_depth, normalized_depth, confidence = model.infer_depth(rgb, intrinsics)
    end = time.time()
    print(f"Time taken for Inference: {end - strt} seconds")
    print("GT Depth:", depth, depth.min(), depth.max())
    print("Predicted Depth Info:", predicted_depth.shape, predicted_depth.dtype)
