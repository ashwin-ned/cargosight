# Implemented from the pixelformer paper
# Author: Ashwin Nedungadi

import torch
import torch.nn as nn

class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85): # variance focus == lambda
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0