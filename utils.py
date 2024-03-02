import os
import shutil
import pathlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torchvision.datasets as datasets


def generate_images(gen, num_imgs, z_dim):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.randn(num_imgs, z_dim).to(device)
    return gen(z)

def show_images(images, nrow=None):
    
    nrow = int(len(images) / 2) if nrow is None else nrow
    if images.min() < 0:
        images = (images.detach().cpu() + 1) / 2
        
    img_grid = make_grid(images.detach().cpu(), nrow=nrow).numpy()
    img_grid = np.transpose(img_grid, (1,2,0))
    plt.imshow(img_grid)
    plt.show()

def save_images(images, fname, nrow=None):
    
    nrow = int(len(images) / 2) if nrow is None else nrow
    if images.min() < 0:
        images = (images.detach().cpu() + 1) / 2
        
    img_grid = make_grid(images.detach().cpu(), nrow=nrow).numpy()
    img_grid = np.transpose(img_grid, (1,2,0))
    plt.imshow(img_grid)
    plt.savefig(f"{fname}.png")

from scipy.stats import truncnorm

def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    #### START CODE HERE ####
    # Define the bounds for the truncation
    lower_bound = -truncation
    upper_bound = truncation
    
    # Generate truncated noise using scipy.stats.truncnorm.rvs()
    truncated_noise = truncnorm.rvs(lower_bound, upper_bound, size=(n_samples, z_dim))
    
    #### END CODE HERE ####
    return torch.Tensor(truncated_noise)

def get_gradient_means(model):
    
    # Collect gradients of model parameters and calculate their sum
    gradient_sum = 0.0
    gradient_count = 0
    for param in model.parameters():
        if param.grad is not None:
            gradient_sum += torch.sum(param.grad)
            gradient_count += 1

    # Calculate the average gradient
    mean_gradient = gradient_sum / gradient_count

    print("Mean Gradient:", mean_gradient.item())

def scale_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            c = nn.init.calculate_gain('relu')  # Calculate the normalization constant c from He's initializer
            module.weight.data /= c  # Scale the weights by dividing them with the normalization constant
        

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0, std=1)

def img_resize(img, size):
    
    img = nn.functional.interpolate(
        img,
        (size,size),
        mode="bilinear"
    )
    
    return img