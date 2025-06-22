import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import cv2
import warnings
from tqdm.notebook import tqdm
import git
from pathlib import Path
import os
warnings.filterwarnings("ignore")

from collections import defaultdict
import pickle

np.random.seed(0)

# Load pretrained AlexNet
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()  # set to evaluation mode

# Extract the first convolutional layer filters
first_conv = alexnet.features[0]
filters = first_conv.weight.data.clone().cpu().numpy()  # shape: [out_channels, in_channels, height, width]

filter_groups = {
    "single_edge": [3, 6, 10, 11, 12, 13, 14, 23, 24, 28, 29, 30, 32, 34, 43, 48, 49, 50, 55, 57], #gabor-like / edge detector
    "multi_edge": [9, 16, 18, 22, 25, 27, 33, 41, 54, 63], #complex gabor / complex edge detector
    "eye": [21, 31, 37, 39, 45, 46,], # color contrast
    "dual_color": [0, 2, 4, 5, 17, 20, 26, 38, 42, 44, 47, 56, 59], # color contrast
    "inside_out": [7, 15, 19, 35, 40, 51, 52, 53, 58], # smoothing
    "misc": [1, 8, 36, 60, 61, 62] # misc
}

def apply_learned_filters(image, filters, device='cpu'):
    """
    image: a numpy array of shape (H, W, 3) in range [0, 255] or normalized appropriately.
    filters: numpy array of shape (num_filters, in_channels, kH, kW).
    Returns: list of feature maps (one per filter).
    """
    # Convert image to a tensor and add batch dimension: shape [1, 3, H, W]
    # Assume image is in HxWxC format and normalized [0,1].
    transform = transforms.ToTensor()  # converts to range [0, 1] and shape (C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Convert filters to torch tensor and place on the same device.
    filter_tensor = torch.tensor(filters, dtype=torch.float32).to(device)
    
    # Convolve the image with all filters.
    feature_maps = F.conv2d(image_tensor, filter_tensor, bias=None, stride=1, padding='same')
    
    # feature_maps shape: [1, num_filters, H_out, W_out]
    return feature_maps.squeeze(0).cpu().detach().numpy()  # shape: (num_filters, H_out, W_out)
def apply_learned_filters_batch(images, filters, device='cpu'):
    """
    images: NumPy array of shape (N, H, W, 3) or list of images in HWC format, dtype float32 or uint8.
    filters: NumPy array of shape (num_filters, in_channels, kH, kW).
    Returns: list of feature maps for each image.
    """
    # Ensure images are float32
    images = [img.astype(np.float32) for img in images]
    feature_maps_list = []
    for img in tqdm(images, desc="Applying filters"):
        fmap = apply_learned_filters(img, filters, device=device)
        feature_maps_list.append(fmap)
    feature_maps = np.stack(feature_maps_list)  # shape: (N, num_filters, H_out, W_out)

    return feature_maps  # shape: (N, num_filters, H_out, W_out)

def apply_learned_filters_batch_at_once(images, filters, device='cpu'):
    """
    images: NumPy array of shape (N, H, W, 3) or list of images in HWC format, dtype float32 or uint8.
    filters: NumPy array of shape (num_filters, in_channels, kH, kW).
    Returns: torch tensor of shape (N, num_filters, H_out, W_out)
    """
    # Ensure images are float32
    images = [img.astype(np.float32) for img in images]
    images = torch.stack([transforms.ToTensor()(img) for img in images])  # (N, 3, H, W)
    images = images.to(device)

    # Convert filters to tensor
    filter_tensor = torch.tensor(filters, dtype=torch.float32).to(device)

    # Run batch convolution (groups=1, so apply all filters to each image)
    feature_maps = F.conv2d(images, filter_tensor, bias=None, stride=1, padding='same')

    return feature_maps.cpu().detach()


def load_images_from_directory(directory, n=None, jitter=False, normalize=False):

    all_images = os.listdir(directory)
    num_images = len(all_images)
    #n=100
    #jitter=True
    #normalize=True

    if not n:
        n = num_images

    images = []
    subset = np.random.permutation(num_images)[:n]
    
    for i in tqdm(subset, desc="Loading images"):
        filename = all_images[i]
        if filename.endswith(".npz"):
            img = np.load(os.path.join(directory, filename))["image"].astype(np.float64)
        elif filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename)).astype(np.float64)

        if jitter:
            img += np.random.uniform(-0.5, 0.5, size=img.shape)
        if normalize:
            img = (img - np.mean(img))/np.std(img)
        images.append(img)

    return np.array(images)

def transform_images(images, filters, device='cpu'):
    """
    images: list or np.array of (N, H, W, 3)
    filters: numpy array of shape (num_filters, in_channels, kH, kW)
    """
    # Apply filters in batch
    feature_maps = apply_learned_filters_batch(images, filters, device)  # (N, num_filters, H_out, W_out)

    # Flatten per-filter results across all images
    num_filters = feature_maps.shape[1]
    print(f"Feature maps shape: {feature_maps.shape}")  # (N, num_filters, H_out, W_out)
    feature_maps = np.transpose(feature_maps, (1, 0, 2, 3))  # (num_filters, N, H_out, W_out)
    flattened = feature_maps.reshape(num_filters, -1)  # Each row: all outputs of one filter

    return flattened  # shape: (num_filters, total_pixels_across_all_images)

def bootstrap_skew(data, n_bootstrap=10000, sample_size=20000):
    # From each filter distribution of around 600k coefficients, take a random sample of sample_size, and then bootstrap n_bootsrap times
    """Bootstrap the skewness of the data to compute a confidence interval."""
    if len(data) > sample_size:
        data = np.random.choice(data, size=sample_size, replace=False)  # Downsample
    bootstrapped_skews = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_skews.append(skew(sample))
    return np.percentile(bootstrapped_skews, [2.5, 97.5])

def run_skew_test_with_filters(all_group_coef, filter_groups, filters, n_bootstrap=10000, sample_size=20000):
    skewed_data = []
    nonskewed_data = []
    skewed_groups = [] 
    nonskewed_groups = []
    skewed_indices = [] # Alex Indices
    nonskewed_indices = []

    for group in all_group_coef.keys():
        for i in range(len(all_group_coef[group])):
            coefficients = all_group_coef[group][i].flatten()
            ci = bootstrap_skew(coefficients, n_bootstrap=n_bootstrap, sample_size=sample_size)

            # Test if 0 is outside the CI
            if ci[0] > 0 or ci[1] < 0:
                skewed_data.append(coefficients)
                skewed_groups.append(group)
                skewed_indices.append(filter_groups[group][i])
            else:
                nonskewed_data.append(coefficients)
                #nonskewed_labels.append(f'{label}, CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
                nonskewed_groups.append(group)
                nonskewed_indices.append(filter_groups[group][i])

    return skewed_data, nonskewed_data, skewed_groups, nonskewed_groups, skewed_indices, nonskewed_indices