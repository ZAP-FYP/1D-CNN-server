import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import Config 
config = Config()

def check_arrays_identical(np_file):
    # Load the numpy file
    data = np.load(np_file, allow_pickle=True)
    # Check if the loaded data is an array
    print(data.shape)
    # Check if the loaded data is an array
    if isinstance(data, np.ndarray) and data.ndim == 4 :
        # Check if all arrays are identical to the first one
        first_array = data[0, 0]  # Get the first array
        identical = all(np.array_equal(first_array, arr[0]) for arr in data)
        return identical
    else:
        return False  # Data shape is not compatible or loaded data is not an array or not 4-dimensional


# Example usage:


# np_file = "/Users/springy/Desktop/New Projects/YOLOPv2-1D_Coordinates/train_data/2d_maps/000d35d3-41990aa4.npy"
# # arrays_identical = check_arrays_identical(np_file)
# if arrays_identical:
#     print("All arrays in the numpy file are identical.")
# else:
#     print("Not all arrays in the numpy file are identical.")


def visualize_frames(np_file, num_frames=500):
    # Load the numpy file
    data = np.load(np_file, allow_pickle=True)
    # Check if the loaded data is an array
    if isinstance(data, np.ndarray) and data.ndim == 3 :
        # Display num_frames frames
        for i in range(min(num_frames, len(data))):
            if i%100==0:
                frame = data[i]
                # frame = data[i,0]

                plt.imshow(frame, cmap='gray')
                plt.title(f'Frame {i+1}')
                plt.axis('off')
                plt.show()
    else:
        print("Data shape is not compatible or loaded data is not an array or not 4-dimensional")


# Example usage:
# visualize_frames(np_file, num_frames=500)
# import h5py

# # Load your NumPy array from the .npy file
# # numpy_array = np.load("your_numpy_array.npy")

# # Create an HDF5 file
# with h5py.File("train_data/2d_maps/0000f77c-62c2a288.npy.h5", "w") as hdf5_file:
#     # Create a dataset within the HDF5 file and write your NumPy array data to it
#     hdf5_file.create_dataset("000f77c-62c2a288", data=np_file)


# try:
#     # Open the HDF5 file in read mode
#     with h5py.File("000f77c-62c2a288.h5", "r") as hdf5_file:
#         # Access the dataset by name
#         dataset = hdf5_file["000f77c-62c2a288"]
        
#         # Convert dataset to a NumPy array
#         numpy_array = dataset[:]
        
#         # Process the array as needed
#         print(numpy_array)
        
# except IOError:
#     print("Error: Unable to open HDF5 file.")

# Function to detect top edge of drivable area
def detect_top_edge(frame):
    # Find indices of non-zero elements along the vertical axis (along columns)
    nonzero_indices = np.nonzero(frame)[0]
    if len(nonzero_indices) > 0:
        # Get the index of the top edge
        top_edge_index = np.min(nonzero_indices)
        return top_edge_index
    else:
        # If no drivable area is detected, return -1
        return -1
numpy_files = [f for f in os.listdir(config.dataset_path) if f.endswith('.npy')]

# Iterate over numpy files
for file_name in numpy_files:
    # Load numpy file
    data = np.load(os.path.join(config.dataset_path, file_name))
    data = np.squeeze(data)

   # Create directory to save modified frames
    save_dir = config.dataset_path+"/modified" # Assuming file_name ends with '.npy'
    os.makedirs(save_dir, exist_ok=True)
    all_frames=[]
    # Iterate over frames in the data
    for i, frame in enumerate(data):
        # Detect top edge of drivable area
        top_edge_index = detect_top_edge(frame)
        if top_edge_index != -1:
            # Set all pixels below the top edge to 1
            frame[top_edge_index:, :] = 1
        all_frames.append(frame)
        # Save modified frame
    save_path = os.path.join(save_dir, f"{file_name}.npy")
    np.save(save_path, all_frames)

visualize_frames(save_path, num_frames=500)
