import torch
import torch.nn as nn
import torch.nn.functional as F
from NoiseMappingNetwork import NoiseMapper
from WSConv2d import WSConv2d
import numpy as np
device = 'cuda'

class AdaIN(nn.Module):

    def __init__(self, noise_dim, in_features):

        super(AdaIN, self).__init__()
        self.scale_map = nn.Linear(noise_dim, in_features)
        self.bias_map = nn.Linear(noise_dim, in_features)

        self.instance_norm = nn.InstanceNorm2d(in_features)
    
    def forward(self, x, w):

        #x : the feature map
        #w : the latent vector

        scale = self.scale_map(w)
        scale = scale[:, :,None, None]
        bias = self.bias_map(w)
        bias = bias[:, :, None, None]

        x = self.instance_norm(x)

        return (scale * x) + bias

class StochasticVariation(nn.Module):
    '''
    Arguments:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super(StochasticVariation, self).__init__()
        self.weight = nn.Parameter( 
            # Initiate the weights for the channels from a random normal distribution
            torch.randn((1,channels,1,1))
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + (self.weight * noise) # Applies to image after multiplying by the weight for each channel

class StyleBlock(nn.Module):

    def __init__(self, in_features, out_features, noise_dim=512,upsample=True):

        super(StyleBlock, self).__init__()
        self.conv1 = WSConv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = WSConv2d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        self.adain1 = AdaIN(noise_dim, out_features)
        self.adain2 = AdaIN(noise_dim, out_features)
        self.svar1 = StochasticVariation(out_features)
        self.svar2 = StochasticVariation(out_features)
        self.relu = nn.LeakyReLU(0.2)
        self.up = upsample
    
    def upsample(self, x):

        return F.interpolate(x, scale_factor=2, mode="nearest")
    
    def forward(self, x, w):

        x = self.upsample(x) if self.up else x
        x = self.conv1(x)
        x = self.svar1(x)
        x = self.adain1(x, w)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.svar2(x)
        x = self.adain2(x, w)
        x = self.relu(x)

        return x

class FirstStyleBlock(nn.Module):
    
    def __init__(self, noise_dim, out_features):
        
        super(FirstStyleBlock, self).__init__()
        
        self.adain1 = AdaIN(noise_dim, noise_dim)
        self.adain2 = AdaIN(noise_dim, out_features)
        self.svar1 = StochasticVariation(noise_dim)
        self.svar2 = StochasticVariation(out_features)
        self.conv1 = WSConv2d(noise_dim, out_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, w):
        
        x = self.svar1(x)
        x = self.adain1(x, w)
        
        x = self.conv1(x)
        x = self.svar2(x)
        x = self.adain2(x, w)
        x = self.relu(x)
        
        return x
        
        

class SynthesisNetwork(nn.Module):

    def __init__(self, noise_dim=512, w_dim=512, hidden_dim=512, depth=8, img_channels=3):
        super(SynthesisNetwork, self).__init__()
        self.first_style_block = FirstStyleBlock(noise_dim, 512)
        self.npm = NoiseMapper(noise_dim, w_dim, hidden_dim, depth)
        self.style_blocks = nn.ModuleList([
            #8x8 
            StyleBlock(512, 512, noise_dim),
            #16x16
            StyleBlock(512, 512, noise_dim),
            #32x32
            StyleBlock(512, 512, noise_dim),
            #64x64
            StyleBlock(512, 256, noise_dim),
            #128x128
            StyleBlock(256, 128, noise_dim),
            #256x256
            StyleBlock(128, 64, noise_dim),
            #512x512
            StyleBlock(64, 32, noise_dim),
            #1024x1024
            StyleBlock(32, 16, noise_dim)
        ])

        self.rgb_blocks = nn.ModuleList([
            #produces 8x8 rgb
            WSConv2d(512, img_channels, 1, 1, 0),
            #produces 16x16 rgb
            WSConv2d(512, img_channels, 1, 1, 0),
            #produces 32x32 rgb
            WSConv2d(512, img_channels, 1, 1, 0),
            #produces 64x64 rgb
            WSConv2d(256, img_channels, 1, 1, 0),
            #produces 128x128 rgb
            WSConv2d(128, img_channels, 1, 1, 0),
            #produces 256x256 rgb
            WSConv2d(64, img_channels, 1, 1, 0),
            #produces 256x256 rgb
            WSConv2d(32, img_channels, 1, 1, 0),
            #produces 256x256 rgb
            WSConv2d(16, img_channels, 1, 1, 0),
        ])

        self.first_rgb_block = nn.Conv2d(512, img_channels, 1, 1, 0)

        self.opened_blocks = []
        self._alpha = 0
        self.c = torch.randn(1, noise_dim, 4, 4)
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):

        assert new_alpha >= 0, "Alpha must be between 0 and 1"
        self._alpha = new_alpha if new_alpha < 1 else 1
    
    def update_opened_blocks(self):

        if len(self.opened_blocks) == len(self.style_blocks):
            return
        self.opened_blocks.append(len(self.opened_blocks))
    
    def upsample(self, x):

        return F.interpolate(x, scale_factor=2, mode="nearest")
    

    def forward(self, z):

        # c = self.c.repeat(z.size(0), 1, 1, 1).to(device)
        c = torch.randn(z.size(0), z.size(1), 4, 4).to(device)
        w = self.npm(z)
        x = self.first_style_block(c, w)
    
        if len(self.opened_blocks) == 0:
            
            return self.first_rgb_block(x)

        elif len(self.opened_blocks) == 1:
            
            block = self.style_blocks[0]
            rgb_block = self.rgb_blocks[0]
            x_out = block(x, w)

            x = self.upsample(x)
            fst_img = self.first_rgb_block(x)
            x_img = rgb_block(x_out)

            return (self._alpha * x_img) + ((1 - self._alpha) * fst_img)
        
        else:

            for opened in self.opened_blocks[:-2]:
                block = self.style_blocks[opened]
                x = block(x, w)
            
            second_last_block = self.style_blocks[self.opened_blocks[-2]]
            second_last_rgb = self.rgb_blocks[self.opened_blocks[-2]]
            last_block = self.style_blocks[self.opened_blocks[-1]]
            last_rgb = self.rgb_blocks[self.opened_blocks[-1]]

            x = second_last_block(x, w)
            out = last_block(x, w)

            x_se = self.upsample(x)
            img_se = second_last_rgb(x_se)
            img_lst = last_rgb(out)
            
            return ((1-self._alpha) * img_se) + (self._alpha * img_lst)
