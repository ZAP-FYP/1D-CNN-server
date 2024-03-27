import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from src.config import Config 
from src.tee import Tee
import os
import sys
import matplotlib.pyplot as plt

config = Config()
checkpoint_file = "model/" + config.model_name + "/model_checkpoint.pth"

if not os.path.exists("model/" + config.model_name):
    os.makedirs("model/" + config.model_name)
    f = open("model/" + config.model_name + "/log.txt", "w")
else:
    f = open("model/" + config.model_name + "/log.txt", "a")

original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Device {device}')
# Load numpy file
data = np.load('/Users/springy/Desktop/New Projects/YOLOPv2-1D_Coordinates/train_data/2d_maps/000d35d3-41990aa4.npy')
data = np.squeeze(data)

print(f'data.shape {data.shape}')
def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def visualize( x, y, output_folder):
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
class MyDataset(Dataset):
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
        return torch.tensor(x), torch.tensor(y)




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer

        # self.fc = nn.Linear(5 * (168//8 ) * (256//8 ), 1152000)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)  # Apply pooling layer
        x = torch.relu(self.conv2(x))
        # x = self.pool(x)  # Apply pooling layer
        x = torch.relu(self.conv3(x))
        # x = self.pool(x)  # Apply pooling layer
        # print(f'after all pooling layers {x.shape}')
        # x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer
        # print(f'after reshaping {x.shape}')

        # x = self.fc(x)
        # print(f'after fcn {x.shape}')
        return x.view(x.size(0), y_window_size, 168, 256)

        # return x.view(x.size(0), -1)  # Reshape output


# Define parameters
x_window_size = 10
y_window_size = 5
stride = 6
batch_size = 64
num_epochs = 1
learning_rate = 0.001

# Assuming you have your data loaded into 'data'

# Splitting data into train and test/validation sets
data_train, data_test_val = train_test_split(data, test_size=0.2, random_state=42)

# Further splitting test/validation set into separate test and validation sets
data_validation, data_test = train_test_split(data_test_val, test_size=0.5, random_state=42)

# Creating datasets
train_dataset = MyDataset(data_train, x_window_size, y_window_size, stride)
validation_dataset = MyDataset(data_validation, x_window_size, y_window_size, stride)
test_dataset = MyDataset(data_test, x_window_size, y_window_size, stride)

print(f'Train samples {len(train_dataset)}')
print(f'Validation samples {len(validation_dataset)}')
print(f'Test samples {len(test_dataset)}')

# Creating data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CNNModel()
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    current_epoch = checkpoint["epoch"]

model.to(device)

# # Training loop
# model = model.train()
# for epoch in range(num_epochs):
#     train_epoch_loss = 0.0  # Initialize loss accumulator for the epoch
#     val_epoch_loss = 0.0  # Initialize loss accumulator for the epoch

#     num_batches = 0  # Initialize counter for the number of batches
    
#     for batch_x, batch_y in train_dataloader:
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
#         print(batch_x.shape)

#         optimizer.zero_grad()
#         output = model(batch_x.float())
#         loss = criterion(output, batch_y.float())
#         loss.backward()
#         optimizer.step()
        
#         train_epoch_loss += loss.item()  # Accumulate the loss for the batch
#         num_batches += 1  # Increment the batch counter
    
    # # Calculate average loss for the epoch
    # average_train_loss = train_epoch_loss / num_batches

    # for batch_x, batch_y in validation_dataloader:
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
    #     optimizer.zero_grad()
    #     output = model(batch_x.float())
    #     loss = criterion(output, batch_y.float())
    #     loss.backward()
    #     optimizer.step()
        
    #     val_epoch_loss += loss.item()  # Accumulate the loss for the batch
    #     num_batches += 1  # Increment the batch counter
    
    # # Calculate average loss for the epoch
    # average_val_loss = val_epoch_loss / num_batches
    
    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Average Loss: {average_train_loss:.4f}, Validation Average Loss: {average_val_loss:.4f}')
    


# Testing loop
model = model.eval()
test_epoch_loss=0
num_batches = 0
visualization_folder = "visualizations/"+config.model_name
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)

for batch_x, batch_y in test_dataloader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
    optimizer.zero_grad()
    output = model(batch_x.float())
    loss = criterion(output, batch_y.float())
    loss.backward()
    optimizer.step()
    
    test_epoch_loss += loss.item()  # Accumulate the loss for the batch
    num_batches += 1  # Increment the batch counter
    

    # Plotting
    for i in range(batch_x.shape[0]):  # Loop over each sample in the batch
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 3 rows, 5 columns for each sample

        # Plot images from batch_x for the current sample
        for j in range(10):
            axes[j // 5, j % 5].imshow(batch_x[i, j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[j // 5, j % 5].set_title(f'x[{j}]')
            axes[j // 5, j % 5].axis('off')

        for j in range(5):
            axes[2, j].imshow(batch_y[i, j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[2, j].set_title(f'y[{j}]')
            axes[2, j].axis('off')
        
        # Plot images from batch_y for the current sample
        for j in range(5):
            axes[3, j].imshow(output[i, j].detach().cpu().numpy(), cmap='gray')  # Assuming binary images (0 and 1)
            axes[3, j].set_title(f'pred[{j}]')
            axes[3, j].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Loss: {loss.item():.4f}", fontsize=16)

        plt.savefig(os.path.join(visualization_folder, f"visualization_{i}.png"))  # Save the figure
        plt.close()

# Calculate average loss for the epoch
average_test_loss = test_epoch_loss / num_batches

print(f'Test Average Loss: {average_test_loss:.4f}')