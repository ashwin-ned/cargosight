import cv2
import numpy as np
import albumentations as A
import os
from matplotlib import pyplot as plt


class DataAugmenter:
    def __init__(self):
        # Define the Albumentations augmentation pipeline
        self.rgb_augmenter = A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0),
            A.GaussNoise(var_limit=(0, 50), p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, p=1.0),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, p=1.0)
        ], p=1.0)

    def augment(self, rgb, depth, augmentations):
        augmented_data = {}
        h, w = rgb.shape[:2]

        for aug in augmentations:
            if aug == 'lr':
                augmented_data['rgb_lr'] = np.fliplr(rgb)
                augmented_data['depth_lr'] = np.fliplr(depth)

            elif aug == 'crop_zoom':
                crop_h, crop_w = np.random.randint(int(0.5 * h), h), np.random.randint(int(0.5 * w), w)
                y1, x1 = np.random.randint(0, h - crop_h), np.random.randint(0, w - crop_w)
                cropped_rgb = rgb[y1:y1 + crop_h, x1:x1 + crop_w]
                cropped_depth = depth[y1:y1 + crop_h, x1:x1 + crop_w]
                augmented_data['rgb_crop_zoom'] = cv2.resize(cropped_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                augmented_data['depth_crop_zoom'] = cv2.resize(cropped_depth, (w, h), interpolation=cv2.INTER_NEAREST)

            elif aug == 'random_skew':
                angle = np.random.uniform(-15, 15)
                rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                augmented_data['rgb_random_skew'] = cv2.warpAffine(rgb, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                augmented_data['depth_random_skew'] = cv2.warpAffine(depth, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

          
            elif aug == 'channel_shuffle':
                channel_shuffle = A.ChannelShuffle(p=1.0)
                augmented_data['rgb_channel_shuffle'] = channel_shuffle(image=rgb)['image']

            elif aug == 'random_color_jitter':
                photometric_aug = A.Compose([
                    A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2, p=1.0)
                ])
                augmented_data['rgb_random_photometric'] = photometric_aug(image=rgb)['image']

            elif aug == 'gaussian':
                noise = np.random.normal(0, 5, rgb.shape).astype(np.float32)
                augmented_data['rgb_gaussian'] = np.clip(rgb + noise, 0, 255).astype(np.uint8)

            elif aug == 'defocus_blur':
                blurred_rgb = A.GaussianBlur(blur_limit=(0, 3), p=1.0)(image=rgb)['image']
                augmented_data['rgb_defocus_blur'] = blurred_rgb

            elif aug == 'shadow_simulation':
                shadow_aug = A.CoarseDropout(max_holes=5, max_height=50, max_width=50, min_holes=1, min_height=20, min_width=20, p=1.0)
                augmented_data['rgb_shadow_simulation'] = shadow_aug(image=rgb)['image']

            elif aug == 'perspective_transform':
                perspective_rgb = A.Perspective(scale=(0.05, 0.6), p=1.0)(image=rgb)['image']
                augmented_data['rgb_perspective_transform'] = perspective_rgb

            elif aug == 'affine':
                affine_rgb = A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), shear=(-5, 5), p=1.0)(image=rgb)['image']
                augmented_data['rgb_affine'] = affine_rgb

        return augmented_data


if __name__ == "__main__":
    SAVE, DEBUG = 1, 0
    container = f'{5:02d}'

    # Load file paths
    base_path = "/home/rog-nuc/Desktop/ashwin-thesis-workspace/mde-data/livox-flir-pipeline-benchmark/flir-rgb-depth"
    flir_rgb_path = f"{base_path}/C{container}_rgb.png"
    flir_depth_path = f"{base_path}/C{container}_inpainted.tiff"

    # Load RGB and depth images
    flir_rgb = cv2.cvtColor(cv2.imread(flir_rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    flir_depth = cv2.imread(flir_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    augmenter = DataAugmenter()
    augmentations = ['lr', 'crop_zoom', 'random_skew', 'random_color_jitter', 'random_photometric', 'gaussian', 'defocus_blur', 'shadow_simulation']
    augmented_data = augmenter.augment(flir_rgb, flir_depth, augmentations)

    # Save augmented images
    output_dir = f"{os.getcwd()}/data-augmentation/"
    os.makedirs(output_dir, exist_ok=True)
    if SAVE:
        for key, img in augmented_data.items():
            plt.imsave(os.path.join(output_dir, f"{container}_{key}.png"), img, cmap='inferno' if 'depth' in key else None)

    # Display visualizations
    if DEBUG:
        fig, axes = plt.subplots(1, len(augmented_data), figsize=(15, 5))
        for i, (key, img) in enumerate(augmented_data.items()):
            axes[i].imshow(img, cmap='inferno' if 'depth' in key else None)
            axes[i].axis('off')
            axes[i].set_title(key)
        plt.show()

    print("Augmentations complete!")
