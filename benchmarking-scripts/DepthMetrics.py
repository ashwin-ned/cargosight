import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

class DepthMetrics:
    def __init__(self, ground_truth, predicted, view_result=False):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.view_result = view_result
        self.mask = self.__mask_zero_values()  # Create a mask for valid depth values
        self.__check_depth_maps()

    def __check_depth_maps(self):
        if self.ground_truth.shape != self.predicted.shape:
            raise ValueError("Depth maps are not of equal size.")
        if np.isnan(self.ground_truth).any() or np.isnan(self.predicted).any():
            raise ValueError("Depth maps contain NaN values.")

    def __mask_zero_values(self):
        """Create a mask to exclude zero values in the ground truth."""
        return self.ground_truth > 0

    def mean_absolute_error(self):
        valid_diff = np.abs(self.ground_truth[self.mask] - self.predicted[self.mask])
        result = np.mean(valid_diff)
        if self.view_result:
            cv2.imshow("Mean Absolute Error", result)
            cv2.waitKey(0)
        return result

    def root_mean_squared_error(self):
        diff = self.ground_truth[self.mask] - self.predicted[self.mask]
        rmse = np.sqrt(np.mean(np.square(diff)))
        return rmse

    def absolute_relative_error(self):
        valid_gt = self.ground_truth[self.mask]
        valid_pred = self.predicted[self.mask]
        result = np.mean(np.abs((valid_gt - valid_pred) / valid_gt))
        if self.view_result:
            cv2.imshow("Absolute Relative Error", result)
            cv2.waitKey(0)
        return result

    def threshold_accuracy(self):
        threshold = 1.25
        valid_gt = self.ground_truth[self.mask]
        valid_pred = self.predicted[self.mask]
        ratio = np.maximum(valid_gt / valid_pred, valid_pred / valid_gt)
        accuracy = np.mean(ratio < threshold)
        return accuracy

    def depth_map_difference(self):
        result = np.abs(self.ground_truth - self.predicted)
        result[~self.mask] = 0  # Exclude zero-value areas
        viz = cv2.applyColorMap((result / np.max(result) * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)

        if self.view_result:
            cv2.imshow("Depth GT", self.ground_truth)
            cv2.imshow("Depth Estimation", self.predicted)
            cv2.imshow("Depth Map Difference", viz)
            cv2.waitKey(0)
        return result
    
    def plot_depth_histogram(self):

        sns.histplot(self.ground_truth.flatten(), color='green', label='Normalized Ground Truth')
        sns.histplot(self.predicted.flatten(), color='red', label='Zoe Depth')

        plt.xlim(0, 6)
        plt.xlabel('Normalized Depth valule')
        plt.ylabel('Frequency of Depth Pixel')
        plt.legend()

        plt.show()

    def plot_depth_density(self):

        sns.kdeplot(self.ground_truth.flatten(), color='green', label='Normalized Ground Truth', fill=True)
        sns.kdeplot(self.predicted.flatten(), color='red', label='Zoe Depth', fill=True)
        
        plt.xlabel('Normalized Depth valule')
        plt.ylabel('Density of Depth Pixel')
        plt.legend()
        plt.show()
    def plot_hist_density_difference(self):

        difference = abs(self.ground_truth - self.predicted)
  
        sns.histplot(difference.flatten(), color='red', label='Depth Difference (m)', kde=True)
        plt.xlabel('Depth Difference')
        plt.ylabel('Frequency of Depth Pixel')
        plt.legend()
        plt.show()

        # Plot KDE Seprately
        #sns.kdeplot(difference.flatten(), color='black', label='Depth Difference Density', fill=True)
        #plt.xlabel('Depth Difference')
        #plt.ylabel('Density of Depth Pixel')
        #plt.legend()
        #plt.show()
