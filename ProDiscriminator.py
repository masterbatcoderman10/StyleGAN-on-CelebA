import torch
import torch.nn as nn
from MiniBatchStd import MiniBatchStd
from WSConv2d import WSConv2d

class LastDiscBlock(nn.Module):

    def __init__(self, img_channels=3):

        super(LastDiscBlock, self).__init__()

        self.img_2_features = WSConv2d(img_channels, 512, kernel_size=1)
        self.minibatch_std = MiniBatchStd()
        self.conv1 = WSConv2d(512+1, 512, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.conv2 = WSConv2d(512, 512, kernel_size=4)
        self.clf = nn.Linear(512, 1)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x=None, x_img=None, alpha=1):

        if x_img is not None:

            if x is None:

                features = self.leaky(self.img_2_features(x_img))
                features = self.minibatch_std(features)
                features = self.leaky(self.conv1(features))
                features = self.leaky(self.conv2(features))
                features = features.view(features.size(0), -1)
                # features = self.dropout(features)
                pred = self.clf(features)
                
                return pred
            #Both image and feature are passed in
            else:

                features_pre = self.leaky(self.img_2_features(self.downsample(x_img)))
                fm = ((1-alpha) * features_pre) + (alpha * x)
                x = self.minibatch_std(fm)
                x = self.leaky((self.conv1(x)))
                x = self.leaky((self.conv2(x)))
                x = x.view(x.size(0), -1)
                # x = self.dropout(x)
                pred = self.clf(x)

                return pred
        
        else:

            #No need to convert img to feature map, well trained feature map is passed in
            x = self.minibatch_std(x)
            x = self.leaky((self.conv1(x)))
            x = self.leaky((self.conv2(x)))
            x  = x.view(x.size(0), -1)
            # x = self.dropout(x)
            pred = self.clf(x)

            return pred

class DiscBlock(nn.Module):

    def __init__(self, in_dim, out_dim, img_channels=3):

        super(DiscBlock, self).__init__()

        self.img_2_features = WSConv2d(img_channels, in_dim, kernel_size=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.convs = nn.Sequential(
            WSConv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x=None, x_img=None, alpha=0):

        if x_img is not None:

            #Top of the stack
            if x is None:

                features = self.leaky((self.img_2_features(x_img)))
                x_out = self.convs(features)
                x_out = self.downsample(x_out)

                return x_out
            else:
            #Feature map passed in, not the top most block
                #Convert the passed in image to feature map
                features = self.leaky(self.img_2_features(self.downsample(x_img)))
                x = ((1- alpha) * features) + (alpha * x)
                x_out = self.convs(x)
                x_out = self.downsample(x_out)

                return x_out
        
        else:
        #Only features have been passed in

            x_out = self.convs(x)
            x_out = self.downsample(x_out)

            return x_out

class ProDiscriminator(nn.Module):

    def __init__(self, img_channels):

        super(ProDiscriminator, self).__init__()
        self.last_block = LastDiscBlock(img_channels) # returns prediction, takes in 512 channel tensor

        self.pro_blocks = nn.ModuleList([
            # Takes 1024x1024 image
            DiscBlock(16, 32, img_channels),
            # Takes 512x512 image and/or tensor
            DiscBlock(32, 64, img_channels),
            # Takes 256x256 image and/or tensor
            DiscBlock(64, 128, img_channels),
            # Takes 128x128 image and/or tensor
            DiscBlock(128, 256, img_channels),
            # Takes 64x64 image and/or tensor
            DiscBlock(256, 512, img_channels),
            # Takes 32x32 image and/or tensor
            DiscBlock(512, 512, img_channels),
            # Takes 16x16 image and/or tensor
            DiscBlock(512, 512, img_channels),
            # Takes 8x8 image and/or tensor
            DiscBlock(512, 512, img_channels)
        ])

        self.opened_blocks = []
    
    def update_opened_blocks(self):

        if len(self.opened_blocks) == len(self.pro_blocks):
            return
        self.opened_blocks.append(-(len(self.opened_blocks) + 1))
    
    def forward(self, x_img, alpha=1):

        if len(self.opened_blocks) == 0:
            
            #Only last block exists
            pred = self.last_block(x_img=x_img)
            return pred
        
        elif len(self.opened_blocks) == 1:

            #Only one block on top
            block = self.pro_blocks[self.opened_blocks[0]]
            features = block(x_img=x_img)
            pred = self.last_block(x_img=x_img, x=features, alpha=alpha)

            return pred

        else:

            top_block = self.pro_blocks[self.opened_blocks[-1]]
            second_block = self.pro_blocks[self.opened_blocks[-2]]

            features = top_block(x_img=x_img)
            features = second_block(x_img=x_img, x=features, alpha=alpha)

            for opened in self.opened_blocks[::-1][2:]:
                block = self.pro_blocks[opened]
                features = block(x=features)
            
            #Time for final prediction
            pred = self.last_block(x=features)

            return pred    
    