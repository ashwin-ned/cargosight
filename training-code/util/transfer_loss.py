import torch
import torch.nn as nn
import kornia

class CombinedDepthLoss(nn.Module):
    def __init__(self, theta=0.1, max_depth=100.0):
        super().__init__()
        self.theta = theta
        self.max_depth = max_depth
        self.sobel = kornia.filters.Sobel()
        
    def gradient_loss(self, pred, target, mask):
        # Compute gradients using Sobel operator
        pred_grad = self.sobel(pred)
        target_grad = self.sobel(target)
        
        # Calculate edge differences only on valid pixels
        edge_loss = (torch.abs(pred_grad - target_grad)[mask]).mean()
        return edge_loss

    def forward(self, pred, target, valid_mask):
        # Normalize depth values to [0, 1] for SSIM
        pred_norm = pred / self.max_depth
        target_norm = target / self.max_depth
        
        # Mask valid pixels
        mask = valid_mask.detach()
        pred_masked = pred_norm[mask]
        target_masked = target_norm[mask]

        # MAE loss (depth)
        l_depth = torch.abs(pred_masked - target_masked).mean()

        # Edge loss
        l_edges = self.gradient_loss(pred_norm, target_norm, mask)

        # SSIM loss (Kornia returns 1 - SSIM)
        ssim_loss = kornia.losses.ssim_loss(
            pred_norm.unsqueeze(1), 
            target_norm.unsqueeze(1), 
            window_size=11,
            reduction='none'
        )
        l_ssim = (ssim_loss * 0.5).mean()  # Scale to match paper's formulation

        # Combine losses with weights
        total_loss = l_ssim + l_edges + self.theta * l_depth
        return total_loss