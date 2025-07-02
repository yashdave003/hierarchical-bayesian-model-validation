import git
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import convolve
from scipy.stats import bootstrap
from scipy import stats
from scipy.stats import skew
import cv2
import warnings
from tqdm.notebook import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
np.set_printoptions(legacy='1.25')
np.random.seed(0)

ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F

# alexnet = torchvision.models.alexnet(pretrained=True)
# alexnet.eval()  
# first_conv = alexnet.features[0]
# filters = first_conv.weight.data.clone().cpu().numpy().transpose(0, 2, 3, 1)  # shape: [out_channels, in_channels, height, width]

filters = pd.read_pickle(os.path.join(ROOT_DIR, 'learned-filters', 'filters.pickle'))

def load_images_from_directory(directory, n=None, jitter=False, normalize=False):

    all_images = os.listdir(directory)
    num_images = len(all_images)

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
        elif filename.endswith(".tif"):
            img = np.array(cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)).astype(np.float64)
            if img.ndim == 2:  # grayscale image
                img = np.stack([img] * 3, axis=-1)

        if jitter:
            img += np.random.uniform(-0.5, 0.5, size=img.shape)
        if normalize:
            img = (img - np.mean(img))/np.std(img)
        images.append(img)

    return np.array(images)

def bootstrap_metric(x, metric=None, n_bootstrap=1000, bootstrap_size = 10000, ci=0.99, replace=True):
    metric_values = []
    for _ in tqdm(range(n_bootstrap)):
        resampled = np.random.choice(x, size=bootstrap_size, replace=replace)
        metric_values.append(metric(resampled))
        
    metric_point_estimate = metric(x)
    ci_lower = np.percentile(metric_values, (1 - ci) / 2 * 100)
    ci_upper = np.percentile(metric_values, (1 + ci) / 2 * 100)
    
    return metric_point_estimate, ci_lower, ci_upper, metric_values
