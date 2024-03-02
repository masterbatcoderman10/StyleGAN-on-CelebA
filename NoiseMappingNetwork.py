import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseMapper(nn.Module):

    def __init__(self, in_dim=512, out_dim=512, hidden_dim=512, depth=8):

        super(NoiseMapper, self).__init__()

        modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            )
        ])

        for _ in range(depth-2):
            modules.extend(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            )
        
        modules.extend(
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
            )
        )

        self.noise_mapper = nn.Sequential(
            *modules
        )

    
    def forward(self, z):

        w = self.noise_mapper(z)

        return w