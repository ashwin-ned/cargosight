import cv2
import numpy as np
from dotenv import load_dotenv
import os, time

#from DepthAnything import DepthAnythingInference
from DepthAnythingv2Metric import DepthAnythingInference
#from ZoeDepth import ZoeDepthInference
#from ZeroDepth import ZeroDepthInference
#from UniDepth import UniDepthInference
#rom Metric3D import Metric3DInference
#from PixelFormer import PixelFormerInference


from DataLoader import DataLoaderARKit, DataLoaderNavvis
from DepthMetrics import DepthMetrics
from BenchmarkLogger import BenchmarkLogger
from navvis_depth_masking import mask_image

def normalize_depth(depth):
    # Normalize the depthmap
    scaled_depth = (depth).astype(np.float32)
    scaled_depth = scaled_depth.astype(np.uint8)

    normalized = 255 - (cv2.normalize(scaled_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    result = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return result

if __name__ == "__main__":
    # Setup Paths from config file
    load_dotenv()
    ARKIT_PATH = os.environ.get("ARKit")
    NAVVIS_PATH = os.environ.get("Navvis")
    # Get all the subfolder paths 
    arkit_paths = [os.path.join(ARKIT_PATH, folder) for folder in os.listdir(ARKIT_PATH) if os.path.isdir(os.path.join(ARKIT_PATH, folder))]
    navvis_paths = [os.path.join(NAVVIS_PATH, folder) for folder in os.listdir(NAVVIS_PATH)]
   

    # Camera Intrinsics
    camera_json = {"fx": 768, "fy": 768, "cx": 512, "cy": 384}
    camera_intrinsics = np.array([[768.0, 0.0, 512.0],  # f_x, 0, c_x
                                  [0.0, 768.0, 384.0],  # 0, f_y, c_y
                                  [0.0, 0.0, 1.0]       # 0, 0, 1
                                  ]).astype(np.float32)
    # Dataloader          
    ARKIT_DATA = True
    NAVVIS_DATA = False

    # Load the Dataset Paths & Check the loaded data
    if ARKIT_DATA:
        view = False
        # Load the model(s)
        # 1. Depth Anything
        DepthAnythingObj = DepthAnythingInference(encoder="vitb", view_result=False)
        model_name = "DepthAnythingV2-ViTb-20depth"
        # 2. Zoe Depth 
        #ZoeDepthObj = ZoeDepthInference(view_result=False)
        #model_name = "ZoeDepth"
        # 3. ZeroDepth
        #ZeroDepthObj = ZeroDepthInference(view_result=False)
        #model_name = "ZeroDepth"
        # 4. UniDepth
        #UniDepthObj = UniDepthInference(model='unidepthv2', intrinsics=camera_intrinsics, view_result=False)
        # Model Name for Logging
        # 5. Metric3D
        #model_name = "metric3d_vit_small"
        #etric3DObj = Metric3DInference(model=model_name, view_result=False)
        # 6. PixelFormer (rand)
        #model_name = "pixel_former_Oliver"
        #PixelFormerObj = PixelFormerInference(model='pixel_former', view_result=False)

  
        for parent_path in arkit_paths:
            for sub_folder in os.listdir(parent_path):
                subfolder_path = os.path.join(parent_path, sub_folder)
                data_loader_arkit = DataLoaderARKit(subfolder_path)
                # BenchMark 
                images = 0
                metrics_list = {'mae': 0.0, 'are': 0.0, 'rmse': 0.0, 'delta1': 0.0}
                for key, value in data_loader_arkit.data.items():
                    images += 1
                    image = value["image"]
                    depth = value["depth"]
                    print("Original Shapes:", image.shape, depth.shape)
                    #intrinsics = [768, 768, 512, 384]
                    intrinsics_ARKit = [1597.4, 1597.4, 960, 720]
                    # TODO: Inference all models and save metrics, add Time taken for inference
                    start_time = time.time()
                    #predicted_depth, normalized_depth, confidence = Metric3DObj.infer_depth(image, intrinsics_ARKit)
                    #predicted_depth, normalized_depth = ZeroDepthObj.inference_zero_depth(image, camera_json)
                    #predicted_depth = PixelFormerObj.infer_depth(image)
                    predicted_depth = DepthAnythingObj.inference_depth_anything(image)
                    end_time = time.time()
                    inference_time = end_time - start_time
                    print('#'*30, "DEBUG ARKIT", '#'*30)
                    print(depth, "GT Depth Max/Min", depth.min(), depth.max())

                    print(predicted_depth, "Predicted Depth Max/Min", predicted_depth.min(), predicted_depth.max())
                    print("Differnce GT & Predicted", np.abs(depth - predicted_depth).max(), np.abs(depth - predicted_depth).min())

                    # Metrics
                    Metrics = DepthMetrics(depth, predicted_depth)
                    metrics_list['mae'] += Metrics.mean_absolute_error()
                    metrics_list['are'] += Metrics.absolute_relative_error()
                    metrics_list['rmse'] += Metrics.root_mean_squared_error()
                    metrics_list['delta1'] += Metrics.threshold_accuracy()
                    print(metrics_list)
                    print("KEY", key)
                    print(sub_folder)

                    # Logger
                    logger = BenchmarkLogger(key, sub_folder, model_name, (image.shape, depth.shape), inference_time, Metrics.mean_absolute_error(), 
                                             Metrics.root_mean_squared_error(), Metrics.absolute_relative_error(), Metrics.threshold_accuracy())
                    logger.add_entry()
                    logger.save_to_csv()

                print("Total Images:", images)
                print("Average Metrics:", {key: value/images for key, value in metrics_list.items()})

            # TODO: Can't normalize depth to be smooth, artifacts present
            # TODO Later: Save graph every 3 or 5 images
            #depth_viz = normalize_depth(value["depth"])
            #if view:
                #cv2.imshow("Image Viz", image)
                #cv2.imshow("Depth Viz", depth_viz)
                #cv2.waitKey(0)

    if NAVVIS_DATA:
        view = False
        # Temp: Load 1
        data_loader_navvis = DataLoaderNavvis(navvis_paths[0])
        # Load All
        #for path in navvis_paths:
            #data_loader_navvis = DataLoaderNavvis(path)

        UnidepthObj = UniDepthInference(model='unidepthv1ConvNext', intrinsics=None, view_result=True)
        for key, value in data_loader_navvis.data.items():
            print(value["image"].shape, value["depth"].shape)
            image = value["image"]
            depth = value["depth"]

            depth = cv2.resize(depth, (1368, 912), interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image,  (1368, 912), interpolation=cv2.INTER_NEAREST)

            # Mask the the image to the size of the depth map
            masked_image = mask_image(image, depth)

            # TODO: Clean up the model class and use proper variale names
            predicted_depth, normalized_depth, pcd = UnidepthObj.infer_depth(image)
            print('#'*30, "DEBUG", '#'*30)
            print(depth.shape, depth.dtype)
            print(predicted_depth.shape, predicted_depth.dtype)

            Metrics = DepthMetrics(depth, predicted_depth)
            mae = Metrics.mean_absolute_error()
            are = Metrics.absolute_relative_error()
            rmse = Metrics.root_mean_squared_error()
            delta1 = Metrics.threshold_accuracy()

            print(f"MAE: {mae}, ARE: {are}, RMSE: {rmse}, Delta1: {delta1}")
            #plot = Metrics.plot_depth_density()
    


