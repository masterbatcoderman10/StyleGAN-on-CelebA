import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    
    def __init__(self):
        
        super(PixelNorm, self).__init__()
    
    def forward(self, x):
        
        pixel_sum = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x / pixel_sum