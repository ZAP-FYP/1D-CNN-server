import numpy as np
import matplotlib.pyplot as plt

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


np_file = "/Users/springy/Desktop/New Projects/YOLOPv2-1D_Coordinates/train_data/2d_maps/000d35d3-41990aa4.npy"
arrays_identical = check_arrays_identical(np_file)
if arrays_identical:
    print("All arrays in the numpy file are identical.")
else:
    print("Not all arrays in the numpy file are identical.")


def visualize_frames(np_file, num_frames=500):
    # Load the numpy file
    data = np.load(np_file, allow_pickle=True)
    
    # Check if the loaded data is an array
    if isinstance(data, np.ndarray) and data.ndim == 4 :
        # Display num_frames frames
        for i in range(min(num_frames, len(data))):
            print(i)
            if i%100==0:
                frame = data[i, 0]
                plt.imshow(frame, cmap='gray')
                plt.title(f'Frame {i+1}')
                plt.axis('off')
                plt.show()
    else:
        print("Data shape is not compatible or loaded data is not an array or not 4-dimensional")


# Example usage:
visualize_frames(np_file, num_frames=500)
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