import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import json

class DrivableAreaDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        self.n_samples = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class SimpleFrameDataset:
    def __init__(
        self,
        frame_length,
        car_width,
        min_velocity,
        max_velocity,
        num_frames,
        window_x,
        window_y,
        num_duplicates=20,
        train_frame_start=None,
        val_frame_start=None,
        test_frame_start=None,
        val_test_frames=20,
        horizontal_velocity=1,
        vertical_velocity=1,
        acceleration=False,
    ):
        self.frame_length = frame_length
        self.car_width = car_width
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.num_frames = num_frames
        self.window_x = window_x
        self.window_y = window_y
        self.num_duplicates = num_duplicates
        self.train_frame_start = (
            train_frame_start if train_frame_start is not None else 0
        )
        self.val_frame_start = val_frame_start
        self.test_frame_start = test_frame_start
        self.horizontal_velocity = horizontal_velocity
        self.vertical_velocity = vertical_velocity
        self.acceleration = acceleration

        self.create_dataset()

    @staticmethod
    def generate_base_frame(frame_length, car_width):
        base_frame = np.zeros(frame_length)
        car_start = frame_length - car_width - 5
        base_frame[car_start : car_start + car_width] = 10
        return base_frame

    @staticmethod
    def generate_moved_frame(
        base_frame,
        horizontal_velocity,
        vertical_velocity,
        noise_factor=0.1,
        noise_flag=False,
    ):
        moved_frame = np.copy(base_frame)

        if horizontal_velocity > 0:
            if np.any(moved_frame[:-horizontal_velocity]):
                moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
            else:
                moved_frame[horizontal_velocity:] = base_frame[:-horizontal_velocity]
        elif horizontal_velocity < 0:
            if np.any(moved_frame[-horizontal_velocity:]):
                moved_frame = np.roll(base_frame, shift=-horizontal_velocity)
            else:
                moved_frame[:horizontal_velocity] = base_frame[-horizontal_velocity:]

        moved_frame += vertical_velocity
        if noise_flag:
            noise = np.random.normal(scale=noise_factor, size=moved_frame.shape)
            moved_frame += noise

        return moved_frame

    def generate_frames(self):
        base_frame = self.generate_base_frame(self.frame_length, self.car_width)
        frames = [base_frame]

        if self.acceleration:
            acceleration_frames = 300
            constant_velocity_frames = 0
            deceleration_frames = 0

            pattern = np.concatenate(
                [
                    np.linspace(
                        self.min_velocity, self.max_velocity, acceleration_frames
                    ),
                    np.full(constant_velocity_frames, self.max_velocity),
                    np.linspace(
                        self.max_velocity, self.min_velocity, deceleration_frames
                    ),
                ]
            )

            velocity_sequence = np.tile(pattern, self.num_frames // len(pattern))
        else:
            velocity_sequence = np.full(self.num_frames, self.vertical_velocity)

        for i in range(self.num_frames):
            frame = self.generate_moved_frame(
                frames[-1],
                self.horizontal_velocity,
                velocity_sequence[i],
                noise_flag=True,
            )
            frames.append(frame)

        self.frames = frames

    def split_frames(self):
        train_frames, val_frames, test_frames = self.get_split_frames()

        self.x_train, self.y_train = self.index_frames(train_frames)
        self.x_val, self.y_val = self.index_frames(train_frames[30:50])
        self.x_test, self.y_test = self.index_frames(train_frames[40:60])

    def get_split_frames(self):
        train_frames = self.frames[self.train_frame_start : self.val_frame_start]
        val_frames = self.frames[self.val_frame_start : self.test_frame_start]
        test_frames = self.frames[self.test_frame_start :]

        return train_frames, val_frames, test_frames

    def index_frames(self, frames):
        x_frames = np.array(
            [
                frames[i : i + self.window_x]
                for i in range(len(frames) - self.window_x - self.window_y)
            ]
        )
        y_frames = np.array(
            [
                frames[i + self.window_x : i + self.window_x + self.window_y]
                for i in range(len(frames) - self.window_x - self.window_y)
            ]
        )
        return x_frames, y_frames

    def create_numpy_arrays(self):
        self.x_train_duplicate = np.tile(self.x_train, (self.num_duplicates, 1, 1))
        self.y_train_duplicate = np.tile(self.y_train, (self.num_duplicates, 1, 1))

        self.flatten_y = self.y_train_duplicate.reshape(
            (len(self.y_train_duplicate), -1)
        )
        self.flatten_y_val = self.y_val.reshape((len(self.y_val), -1))
        self.flatten_y_test = self.y_test.reshape((len(self.y_test), -1))

        count, in_channels, in_seq_len = self.x_train_duplicate.shape

    def visualize_frames(
        self, save_directory="frames_plots", save_filename="frames_plot.png"
    ):
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, save_filename)
        plt.figure(figsize=(60, 25))
        for i, frame in enumerate(self.frames):
            plt.plot(frame, label=f"Frame {i + 1}")
            plt.ylim([-1, np.max(self.frames) + 1])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Sequence of Frames with Car Movement")
            plt.xlabel("Position")
            plt.ylabel("Value")
            plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def create_dataset(self):
        self.generate_frames()
        self.split_frames()
        self.create_numpy_arrays()
        self.train_dataset = DrivableDataset(self.x_train_duplicate, self.flatten_y)
        self.validation_dataset = DrivableDataset(self.x_val, self.flatten_y_val)
        self.test_dataset = DrivableDataset(self.x_test, self.flatten_y_test)


class VideoFrameDataset:
    def __init__(
        self,
        directory_path,
        split_ratio=0.80,
        test_flag=False,
        DRR=0,
        n_th_frame=True,
        frame_avg_rate=0,
        prev_frames=10,
        future_frames=5,
        threshold=0
    ):
        self.directory_path = directory_path
        self.split_ratio = split_ratio
        self.test_flag = test_flag
        self.DRR = DRR
        self.n_th_frame = n_th_frame
        self.frame_avg_rate = frame_avg_rate
        self.prev_frames = prev_frames
        self.future_frames = future_frames
        self.threshold = threshold
        self.create_dataset()

    def get_X_y(self, data, prev_frames, future_frames, threshold):
        
        window_size = prev_frames + future_frames
        X = [data[i : i + prev_frames] for i in range(len(data[:-window_size]))]
        X_frame_diffs = [self.calculate_frame_diff(sample) for sample in X]
        
        # print(X_frame_diffs[:10])
        # print("mean:",np.mean(X_frame_diffs))
        # for frame_index, frame in enumerate(X_file):
            #     if np.all(frame == 0):
            #         start_index = max(0, frame_index - 10)
            #         end_index = min(len(X_file), frame_index + 11)
            #         frames_to_visualize = X_file[start_index:end_index]
            #         self.visualize_frames(_file, frame_index, frames_to_visualize)


        if self.n_th_frame:
            y = [
                data[i + prev_frames + future_frames]
                for i in range(len(data[:-window_size]))
            ]
        else:
            y = [
                data[(i + prev_frames) : (i + prev_frames + future_frames)]
                for i in range(len(data[:-window_size]))
            ]
        filtered_X = []
        filtered_y = []
        # print("old:",np.array(X).shape)

        with open("filtered_data.txt", "w") as file:
            for i, diff in enumerate(X_frame_diffs):
                file.write(f"i: {i}, diff: {diff}\n")
                
                # Check if any element in diff is greater than threshold
                if np.any(diff > int(threshold)):
                    # Append corresponding X and y values to filtered lists
                    filtered_X.append(X[i])
                    filtered_y.append(y[i])


        # print("new:",np.array(filtered_X).shape)
        return np.array(filtered_X), np.array(filtered_y)
        # return np.array(X), np.array(y)

    def create_averaged_frames(self, data, avg_rate):
        averaged_frames = []
        # print("shape before averaging:", data.shape)
        
        for i in range(0, len(data), avg_rate):
            averaged_frames.append(data[i:i+avg_rate].mean(axis=0))
                
        return np.array(averaged_frames)
    

    def create_dataset(self):
        total_data = []
        X = []
        y = []

        filenames = [
            f for f in os.listdir(self.directory_path) if not f.startswith(".DS_Store")
        ]

        for _file in filenames:
            file = os.path.join(self.directory_path, _file)
            data_npy = np.load(file)
            total_data.extend(data_npy)

            if self.frame_avg_rate > 0:
                averaged_frames = self.create_averaged_frames(data_npy, self.frame_avg_rate)
                X_file, y_file = self.get_X_y(averaged_frames, self.prev_frames, self.future_frames,self.threshold)
                # print("averaged_frames:", averaged_frames.shape)
            elif self.frame_avg_rate == 0:
                X_file, y_file = self.get_X_y(data_npy, self.prev_frames, self.future_frames, self.threshold)
                # print("Not averaging")
            
            X.extend(X_file)
            y.extend(y_file)

        X = np.array(X)
        y = np.array(y)

        # self.visualize(X[:100], y[:100], "filtered")

        print("total data: ", np.array(total_data).shape)
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        flatten_y = y.reshape((len(y), -1))
        count, self.in_channels, self.in_seq_len = X.shape

        if not self.test_flag:
            idx = int(count)
        else:
            idx = int(count * self.split_ratio)

        val_idx = int(idx * self.split_ratio)

        if self.DRR == 0:
            self.train_dataset = DrivableDataset(X[:val_idx:], flatten_y[:val_idx:])
            self.validation_dataset = DrivableDataset(
                X[val_idx:idx:], flatten_y[val_idx:idx:]
            )
            self.test_dataset = DrivableDataset(
                X[idx ::], flatten_y[idx ::]
            )
        else:
            self.train_dataset = DrivableDataset(
                X[: val_idx : self.DRR], flatten_y[: val_idx : self.DRR]
            )
            self.validation_dataset = DrivableDataset(
                X[val_idx : idx : self.DRR], flatten_y[val_idx : idx : self.DRR]
            )
            self.test_dataset = DrivableDataset(
                X[idx :: self.DRR], flatten_y[idx :: self.DRR]
        )
        
        print(f'Train samples {len(self.train_dataset)}')
        print(f'Validation samples {len(self.validation_dataset)}')
        print(f'Test samples {len(self.test_dataset)}')

    def visualize(self, x, y, output_folder):
        num_samples, num_frames_x, frame_length_x = x.shape
        _, num_frames_y, frame_length_y = y.shape

        for sample_index in range(num_samples):
            sample_folder = os.path.join(output_folder, f"sample_{sample_index}")
            os.makedirs(sample_folder, exist_ok=True)

            plt.figure(figsize=(15, 4))

            # Plot x
            for frame_index in range(num_frames_x):
                x_frame = x[sample_index, frame_index]
                plt.plot(
                    x_frame, label=f"Sample {sample_index}, Frame {frame_index} - Input (x)"
                )

            # Plot y
            for frame_index in range(num_frames_y):
                y_frame = y[sample_index, frame_index]
                plt.plot(
                    y_frame,
                    label=f"Sample {sample_index}, Frame {frame_index} - Output (y)",
                    color="red",
                )

            plt.xlabel("Time Steps")
            plt.ylabel("Values")
            plt.legend()

            plt.tight_layout()

            # Save the figure
            plt.savefig(
                os.path.join(sample_folder, f"sample_{sample_index}_visualization.png")
            )
            plt.close()

    def visualize_frames(self, filename, frame_index, frames):
        num_frames = len(frames)
        num_cols = 3
        num_rows = (num_frames + num_cols - 1) // num_cols  # Calculate number of rows needed

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

        for i, ax in enumerate(axes.flat):
            if i < num_frames:
                frame = frames[i]
                if len(frame.shape) == 1:  # Check if the frame is 1D
                    ax.plot(frame)          # Use plot for 1D arrays
                    ax.set_title(f'Frame {frame_index - 10 + i}')
                else:
                    ax.imshow(frame, cmap='gray')
                    ax.set_title(f'Frame {frame_index - 10 + i}')
                ax.axis('off')
            else:
                ax.axis('off')  # Turn off empty subplots

        plt.tight_layout()
        save_path = os.path.join('visualization_folder', f'{filename}_{frame_index}.png')
        plt.savefig(save_path)
        plt.close()

    def calculate_frame_diff(self, data):
        # print(np.sum(np.abs(np.diff(data, axis=0)), axis=(0, 1)))
        return np.sum(np.abs(np.diff(data, axis=0)), axis=(0, 1))


class CollisionDataset:
    def __init__(
        self,
        directory_path,
        split_ratio=0.80,
        test_flag=False,
        DRR=0,
        frame_avg_rate=0,
        prev_frames=10,
        custom_loss=False
    ):
        self.directory_path = directory_path
        self.split_ratio = split_ratio
        self.test_flag = test_flag
        self.DRR = DRR
        self.frame_avg_rate = frame_avg_rate
        self.prev_frames = prev_frames
        self.custom_loss = custom_loss
        self.create_dataset()

    def get_X_y(self, data):
        X = []
        y = []
        tta = []

        for i in range(len(data) - self.prev_frames):
            # Extract the previous frames for each sample
            x_frames = [frame['numpy_coordinates'] for frame in data[i:i + self.prev_frames]]
            X.append(x_frames)

            # Calculate y based on the new criteria
            labels = [frame['prediction_label'] for frame in data[i:i + self.prev_frames]]
            if sum(labels) >= 5:
                y.append(1)
            else:
                y.append(0)

            if self.custom_loss:
                # print(data[i + self.prev_frames]['tta'])
                tta.append(data[i + self.prev_frames]['tta'])
        
        return np.array(X), np.array(y), np.array(tta)

    def create_averaged_frames(self, data, avg_rate):
        averaged_frames = []
        frame_id_counter = 0

        for i in range(0, len(data), avg_rate):
            frames_subset = [frame['numpy_coordinates'] for frame in data[i:i+avg_rate]]
            averaged_frame = np.mean(frames_subset, axis=0)

            # Average the prediction labels
            labels_subset = [frame['prediction_label'] for frame in data[i:i+avg_rate]]
            averaged_label = 1 if np.mean(labels_subset) >= 0.5 else 0

            # Assign the new frame_id
            new_frame_id = frame_id_counter
            frame_id_counter += 1

            if self.custom_loss:
                tta_subset = [frame['tta'] for frame in data[i:i+avg_rate]]
                averaged_tta = np.mean(tta_subset)
                averaged_frames.append({
                    'frame_id': new_frame_id,
                    'numpy_coordinates': averaged_frame,
                    'prediction_label': averaged_label, 
                    'tta': averaged_tta
                })
            else:
                averaged_frames.append({
                    'frame_id': new_frame_id,
                    'numpy_coordinates': averaged_frame,
                    'prediction_label': averaged_label
                })

        return averaged_frames

    def get_tta(self, data):
        # Initialize a variable to keep track of the index of the first item with prediction: 1
        first_prediction_1_index = None

        # Iterate over the items in the JSON data to find the first occurrence of prediction: 1
        for index, item in enumerate(data):
            if item.get('prediction_label') == 1:
                first_prediction_1_index = index
                break
        
        # print("first prediction index", first_prediction_1_index)

        # If no item with prediction: 1 is found, set the first_prediction_1_index to the last index
        if first_prediction_1_index is None:
            first_prediction_1_index = len(data)

        # Iterate over the items again to calculate tta
        for index, item in enumerate(data):
            if first_prediction_1_index is None or index >= first_prediction_1_index:
                item['tta'] = 0
            else:
                item['tta'] = first_prediction_1_index - index

        # Save the modified data back to the JSON file
        with open('data_with_tta.json', 'w') as f:
            json.dump(data, f, indent=4)

        return data

    def create_dataset(self):
        total_data = []
        X = []
        y = []
        tta = []

        filenames = [
            f for f in os.listdir(self.directory_path) if not f.startswith(".DS_Store")
        ]

        for _file in filenames:
            file = os.path.join(self.directory_path, _file)
            with open(file, 'r') as f:
                data = json.load(f)

            if self.custom_loss:
                data = self.get_tta(data)
            
            if self.frame_avg_rate > 0:
                averaged_frames = self.create_averaged_frames(data, self.frame_avg_rate)
                X_file, y_file, tta_file = self.get_X_y(averaged_frames)
            else:
                X_file, y_file, tta_file = self.get_X_y(data)
            
            total_data.extend(data)
            X.extend(X_file)
            y.extend(y_file)
            tta.extend(tta_file)

        X = np.array(X)
        y = np.array(y)
        tta = np.array(tta)

        unique_values, counts = np.unique(y, return_counts=True)

        for value, count in zip(unique_values, counts):
            print(f"Value: {value}, Count: {count}")

        print("total data: ", np.array(total_data).shape)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        np.save("X.npy", X)
        np.save("y.npy", y)

        # Convert X to a list of lists
        # Convert numpy arrays to lists
        data_list = []

        for i in range(10):
            x_data = X[i].tolist()  # Convert numpy array to list
            y_data = y[i].item()    # Convert numpy int64 to Python int
            if tta.size > 0:
                tta_data = tta[i].item()
                data_list.append({"X": x_data, "y": y_data, "tta": tta_data})
            data_list.append({"X": x_data, "y": y_data})
        
        # Save data as JSON
        with open("data.json", "w") as json_file:
            json.dump(data_list, json_file,indent=4)

        count, self.in_channels, self.in_seq_len = X.shape

        if not self.test_flag:
            idx = int(count)
        else:
            idx = int(count * self.split_ratio)

        val_idx = int(idx * self.split_ratio)

        if self.DRR == 0:
            self.train_dataset = DrivableDataset(X[:val_idx:], y[:val_idx:], tta[:val_idx:])
            self.validation_dataset = DrivableDataset(
                X[val_idx:idx:], y[val_idx:idx:], tta[val_idx:idx:]
            )
            self.test_dataset = DrivableDataset(
                X[idx ::], y[idx ::], tta[idx ::]
            )
        else:
            self.train_dataset = DrivableDataset(
                X[: val_idx : self.DRR], y[: val_idx : self.DRR], tta[: val_idx : self.DRR]
            )
            self.validation_dataset = DrivableDataset(
                X[val_idx : idx : self.DRR], y[val_idx : idx : self.DRR], tta[val_idx : idx : self.DRR]
            )
            self.test_dataset = DrivableDataset(
                X[idx :: self.DRR], y[idx :: self.DRR], tta[idx :: self.DRR]
            )
            self.demo_dataset = DrivableDataset(
                X[:: self.DRR], y[:: self.DRR], tta[:: self.DRR]
            )

        print(f'Train samples {len(self.train_dataset)}')
        print(f'Validation samples {len(self.validation_dataset)}')
        print(f'Test samples {len(self.test_dataset)}')

class DrivableDataset(Dataset):
    def __init__(self, X, y, tta=torch.tensor([])):
        self.X = torch.from_numpy(X).float().requires_grad_()
        self.y = torch.from_numpy(y).float().requires_grad_()
        if len(tta) > 0:
            self.tta = torch.from_numpy(tta).float().requires_grad_()
        else:
            self.tta = torch.tensor([])
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        if self.tta is not None and len(self.tta) > 0:
            return self.X[index], self.y[index], self.tta[index]
        else:
            return self.X[index], self.y[index], torch.tensor([])

    def __len__(self):
        return self.n_samples

    def visualize(self, x, y, output_folder):
        num_samples, num_frames_x, frame_length_x = x.shape

        for sample_index in range(num_samples):
            sample_folder = os.path.join(output_folder, f"sample_{sample_index}")
            os.makedirs(sample_folder, exist_ok=True)

            fig, axes = plt.subplots(2, 1, figsize=(15, 8))

            # Plot x
            for frame_index in range(num_frames_x):
                x_frame = x[sample_index, frame_index]
                axes[0].plot(x_frame, label=f"Frame {frame_index} - Input (x)")
            axes[0].set_xlabel("Time Steps")
            axes[0].set_ylabel("Values")
            axes[0].legend()

            # Plot y
            y_value = y[sample_index]
            axes[1].plot([0, num_frames_x], [y_value, y_value], color="red", label="Output (y)")
            axes[1].set_xlabel("Time Steps")
            axes[1].set_ylabel("Event")
            axes[1].legend()

            plt.tight_layout()

            # Save the figure
            plt.savefig(os.path.join(sample_folder, f"sample_{sample_index}_visualization.png"))
            plt.close()


class Conv2d_dataset(Dataset):
    def __init__(self, data, x_window_size=10, y_window_size=5, stride=1):
        self.data = data
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.stride = stride

    def __len__(self):
        return len(self.data) - (self.x_window_size + self.y_window_size) * self.stride + 1

    def __getitem__(self, idx):
        max_idx = len(self.data) - (self.x_window_size + self.y_window_size) * self.stride + 1
        if idx>max_idx:
            idx = max_idx
        x = self.data[idx:idx+self.x_window_size*self.stride:self.stride]
        y = self.data[idx+self.x_window_size*self.stride:idx+(self.x_window_size+self.y_window_size)*self.stride:self.stride]
        # x = self.data[idx:idx+self.x_window_size*self.stride:]
        # y = self.data[idx+self.x_window_size*self.stride:idx+(self.x_window_size+self.y_window_size)*self.stride:]
        x_array = np.array(x)
        y_array = np.array(y)
        return torch.tensor(x_array), torch.tensor(y_array)

