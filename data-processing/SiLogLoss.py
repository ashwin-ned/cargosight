import torch

def SiLogLoss(pred, target, valid_mask, lambd=0.6):
    """
    Computes the Scale-Invariant Logarithmic Loss (SiLogLoss).
    
    Args:
        pred (torch.Tensor): Predicted depth map (batch_size x H x W).
        target (torch.Tensor): Ground truth depth map (batch_size x H x W).
        valid_mask (torch.Tensor): Binary mask of valid pixels (same size as target).
        lambd (float): Scale-invariant penalty weight.
    
    Returns:
        torch.Tensor: The computed SiLogLoss.
    """
    # Ensure valid_mask is boolean
    valid_mask = valid_mask.detach().to(torch.bool)

    # Compute the logarithmic difference
    diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
    
    # Mean squared logarithmic difference
    mean_squared = torch.pow(diff_log, 2).mean()
    
    # Scale-invariant term
    scale_invariant_term = lambd * torch.pow(diff_log.mean(), 2)
    
    # Final SiLogLoss
    loss = torch.sqrt(mean_squared - scale_invariant_term)
    
    return loss.item()  # Return as a scalar



