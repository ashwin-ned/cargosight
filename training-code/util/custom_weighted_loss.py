# # Custom Weighted Loss Function 
# # Author: Ashwin Nedungadi

import torch
import torch.nn as nn
import kornia.losses

class DepthReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.6, gamma=0.8, lambd=0.5, 
                 huber_delta=1.0, max_depth=80.0, ssim_window=11):
        """
        Improved loss function for metric depth estimation.
        
        Args:
        - alpha (float): Weight for SIlog loss (recommended 0.5-0.7)
        - beta (float): Weight for Huber loss (recommended 0.3-0.5)
        - gamma (float): Weight for SSIM loss (recommended 0.1-0.2)
        - lambd (float): Scale invariance factor for SILog
        - huber_delta (float): Delta for Huber loss
        - max_depth (float): Maximum depth value for normalization
        - ssim_window (int): Window size for SSIM computation
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.max_depth = max_depth
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.ssim_window = ssim_window

    def silog_loss(self, pred, target):
        """Scale-Invariant Logarithmic Loss (variance formulation) from Eigen et al. (2014)"""
        diff_log = torch.log(target) - torch.log(pred)
        return torch.mean(diff_log ** 2) - self.lambd * (diff_log.mean() ** 2)

    def forward(self, pred, target, valid_mask):
        # Apply valid mask
        pred_masked = pred[valid_mask]
        target_masked = target[valid_mask]
        
        # SILog Loss (scale-invariant depth consistency)
        silog = self.silog_loss(pred_masked, target_masked)
        
        # Huber Loss (absolute depth accuracy)
        huber = self.huber_loss(pred_masked, target_masked)
        
        # SSIM Loss (structural preservation)
        pred_norm = (pred / self.max_depth).unsqueeze(1)  # Add channel dim
        target_norm = (target / self.max_depth).unsqueeze(1)
        
        ssim_map = 1 - kornia.losses.ssim_loss(
            pred_norm, target_norm, 
            window_size=self.ssim_window, 
            reduction='none'
        )
        
        # Apply valid mask and average
        valid_mask_ = valid_mask.unsqueeze(1).expand_as(ssim_map)
        ssim = (ssim_map * valid_mask_).sum() / valid_mask_.sum().clamp(min=1e-6)

        # Weighted combination
        return (
            self.alpha * silog + 
            self.beta * huber + 
            self.gamma * ssim
        )