import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import Config 
import torch
import torch.nn.functional as F
from test import visualize_frames
def gaussian_filter_3d(input_tensor, kernel_size=(3, 3, 3), sigma=(1.0, 1.0, 1.0)):
    # Create a 3D Gaussian kernel
    kernel = torch.zeros(1, 1, *kernel_size)
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            for k in range(kernel_size[2]):
                kernel[0, 0, i, j, k] = torch.exp(
                    -((i - kernel_size[0] // 2) ** 2 / (2 * sigma[0] ** 2) +
                      (j - kernel_size[1] // 2) ** 2 / (2 * sigma[1] ** 2) +
                      (k - kernel_size[2] // 2) ** 2 / (2 * sigma[2] ** 2))
                )
    kernel /= torch.sum(kernel)

    # Apply 3D convolution with the Gaussian kernel
    output_tensor = F.conv3d(input_tensor.unsqueeze(0).unsqueeze(0), kernel)
    return output_tensor.squeeze(0).squeeze(0)

def gaussian_filter_2d(input_tensor, kernel_size=(3, 3), sigma=(1.0, 1.0)):
    # Create a 2D Gaussian kernel
    kernel = torch.zeros(1, 1, *kernel_size)
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            kernel[0, 0, i, j] = torch.exp(
                -((i - kernel_size[0] // 2) ** 2 / (2 * sigma[0] ** 2) +
                  (j - kernel_size[1] // 2) ** 2 / (2 * sigma[1] ** 2))
            )
    kernel /= torch.sum(kernel)

    # Apply 2D convolution with the Gaussian kernel
    output_tensor = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel)
    return output_tensor.squeeze(0).squeeze(0)
def temporal_smoothing(input_tensor, window_size=3):
    # Pad the input tensor along the temporal dimension
    padding = window_size // 2
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, 0, padding, padding))

    # Apply temporal averaging
    smoothed_tensor = torch.zeros_like(input_tensor)
    for i in range(input_tensor.shape[0]):
        smoothed_tensor[i] = torch.mean(padded_tensor[i:i+window_size], dim=0)

    return smoothed_tensor

def smooth_segmentation_maps(segmentation_maps, threshold=0.5):
    smoothed_maps = np.copy(segmentation_maps)

    for i in range(1, segmentation_maps.shape[0] - 1):
        prev_frame_diff = np.abs(segmentation_maps[i] - segmentation_maps[i - 1])
        next_frame_diff = np.abs(segmentation_maps[i] - segmentation_maps[i + 1])

        sudden_change_mask = (prev_frame_diff > threshold) & (next_frame_diff > threshold)
        smoothed_maps[i][sudden_change_mask] = (segmentation_maps[i - 1][sudden_change_mask] + segmentation_maps[i + 1][sudden_change_mask]) / 2.0

    return smoothed_maps
config = Config()

numpy_files = [f for f in os.listdir(config.dataset_path) if f.endswith('.npy')]

# Iterate over numpy files
for file_name in numpy_files:
    # Load numpy file
    data = np.load(os.path.join(config.dataset_path, file_name))
    data = np.squeeze(data)

   # Create directory to save modified frames
    save_dir = config.dataset_path+"/smoothed" # Assuming file_name ends with '.npy'
    os.makedirs(save_dir, exist_ok=True)
    all_frames=[]
    # for i, frame in enumerate(data):
    # smoothed_data = gaussian_filter_3d(data, kernel_size=(3, 3, 3), sigma=(1.0, 1.0, 1.0))
    # smoothed_data = gaussian_filter_2d(data, kernel_size=(3, 3), sigma=(1.0, 1.0))
    smoothed_data = smooth_segmentation_maps(data, threshold=0.5)
    save_path = os.path.join(save_dir, f"{file_name}.npy")
    visualize_frames(smoothed_data, num_frames=50)

    np.save(save_path, smoothed_data)


