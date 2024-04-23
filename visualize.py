import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_frames(video_data, video_data2, output_dir, num_frames=15, frame_rate=30, total_window=3):
    total_frames = video_data.shape[0]
    rolling_window = total_window*frame_rate
    interval = rolling_window//num_frames
    data_points = total_frames // rolling_window
    print(f'interval,rolling_window, total_frames, data_points{interval,rolling_window, total_frames, data_points}')

    # Create output directory if it doesn't exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    selected_frames = [ video_data[i * interval] for i in range(total_frames//interval)]
    selected_frames2 = [ video_data2[i * interval] for i in range(total_frames//interval)]

    print(len(selected_frames))
    # for idx in range(data_points):
        
        # Select frames at regular intervals within the segment
        
        # Initialize subplots
        # fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        # fig.suptitle(f'Selected Frames from Data Point {idx + 1}')

        # Plotting
    for i in range(data_points):  # Loop over each sample in the batch
        fig, axes = plt.subplots(3, 10, figsize=(15, 9))  # 3 rows, 5 columns for each sample

        # Plot images from batch_x for the current sample
        for j in range(5):
            print(i*num_frames+j)
            axes[0, j].imshow(selected_frames[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[0, j].set_title(f'True Label[{j} data point {i}]')
            axes[0, j].axis('off')
            axes[0, j+5].imshow(selected_frames2[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[0, j+5].set_title(f'True Label[{j} data point {i}]')
            axes[0, j+5].axis('off')
        for j in range(5,10):
            axes[1, j-5].imshow(selected_frames[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[1, j-5].set_title(f'True Label[{j} data point {i}]')
            axes[1, j-5].axis('off')
            axes[1, j].imshow(selected_frames2[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[1, j].set_title(f'True Label[{j} data point {i}]')
            axes[1, j].axis('off')
        
        # Plot images from batch_y for the current sample
        for j in range(10,15):
            axes[2, j-10].imshow(selected_frames[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[2, j-10].set_title(f'True Label[{j} data point {i}]')
            axes[2, j-10].axis('off')
            axes[2, j-5].imshow(selected_frames2[i*num_frames+j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[2, j-5].set_title(f'True Label[{j} data point {i}]')
            axes[2, j-5].axis('off')

        plt.tight_layout()
        # plt.suptitle(f"Loss: {loss.item():.4f}", fontsize=16)
        plt.show()

        # plt.savefig(os.path.join(visualization_folder, f"visualization_{i} batch {num_batches}.png"))  # Save the figure
        plt.close()
# Load numpy file
video_data = np.load('/Users/springy/Desktop/New Projects/YOLOPv2-1D_Coordinates/train_data/2d_maps/modified/modified/000d4f89-3bcbe37a.npy.npy.npy')

# Output directory to save visualizations
output_dir = 'visualizations'

# Visualize and save frames
visualize_frames(video_data,video_data, output_dir, num_frames=15, total_window=3)
