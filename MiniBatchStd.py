import torch
import torch.nn as nn

class MiniBatchStd(nn.Module):
    
    def __init__(self):
        
        super(MiniBatchStd, self).__init__()
    
    def forward(self, x):
        
        w = x.size(2)
        h = x.size(3)
        
        feature_std = torch.std(x, dim=0)
        feature_std = torch.mean(feature_std)
        feature_std = feature_std.repeat(x.size(0), 1, w, h)
        
        return torch.cat([x, feature_std], dim=1)