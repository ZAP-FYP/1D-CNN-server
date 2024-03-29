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
from src.dataset import Conv2d_dataset
from src.models.Conv2d import Conv2d, DeepConv2d, Conv2d_Deconv, Conv2d_Pooling, DeepConv2d_Residual, DeepConv2d_SpatialPyramidPooling, Conv2d_Residual
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define parameters
x_window_size = 10
y_window_size = 5
stride = 6
batch_size = 256
num_epochs = 1000
learning_rate = 0.001

config = Config()
checkpoint_file = "model/" + config.model_name + "/best_model_checkpoint.pth"

if not os.path.exists("model/" + config.model_name):
    os.makedirs("model/" + config.model_name)
    f = open("model/" + config.model_name + "/log.txt", "w")
else:
    f = open("model/" + config.model_name + "/log.txt", "a")

original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Device {device}')


# List all numpy files in the directory
numpy_files = [f for f in os.listdir(config.dataset_path) if f.endswith('.npy')]
train_data = []
validation_data = []
test_data = []

# Iterate over each numpy file
for file_name in numpy_files:
    # Load numpy file
    data = np.load(os.path.join(config.dataset_path, file_name))
    data = np.squeeze(data)

    # Splitting data into train and test/validation sets
    data_train, data_test_val = train_test_split(data, test_size=0.2, random_state=42)

    # Further splitting test/validation set into separate test and validation sets
    data_validation, data_test = train_test_split(data_test_val, test_size=0.5, random_state=42)

    # Append data to combined datasets
    train_data.extend(data_train)
    validation_data.extend(data_validation)
    test_data.extend(data_test)

# Create datasets from combined data
train_dataset = Conv2d_dataset(train_data, x_window_size, y_window_size, stride)
validation_dataset = Conv2d_dataset(validation_data, x_window_size, y_window_size, stride)
test_dataset = Conv2d_dataset(test_data, x_window_size, y_window_size, stride)

print(f'Combined Train samples {len(train_dataset)}')
print(f'Combined Validation samples {len(validation_dataset)}')
print(f'Combined Test samples {len(test_dataset)}')


# Load numpy file
# data = np.load(config.dataset_path)
# data = np.squeeze(data)

# print(f'data.shape {data.shape}')
def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# # Creating data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Conv2d_Deconv()
print(model)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    current_epoch = checkpoint["epoch"]

model.to(device)

# Training loop
model = model.train()
best_val_loss = float("inf")
consecutive_no_improvement = 0

for epoch in range(num_epochs):
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    model.train()  # Set the model to training mode

    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x.float())

        loss = criterion(output, batch_y.float())
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

    average_train_loss = train_epoch_loss / len(train_dataloader)

    # print(f"Epoch {epoch+1}: Train Loss: {average_train_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print(f"Epoch {epoch+1}: Train Loss: {average_train_loss:.4f}")
    num_batches = 0
    for batch_x, batch_y in validation_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
        optimizer.zero_grad()
        output = model(batch_x.float())
        loss = criterion(output, batch_y.float())
        loss.backward()
        optimizer.step()
        
        val_epoch_loss += loss.item()  # Accumulate the loss for the batch
        num_batches += 1  # Increment the batch counter
    
    # Calculate average loss for the epoch
    average_val_loss = val_epoch_loss / num_batches
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        consecutive_no_improvement = 0
        save_checkpoint(
            epoch,
            model,
            optimizer,
            "model/" + config.model_name + "/best_model_checkpoint.pth",
        )
    else:
        consecutive_no_improvement += 1

    if consecutive_no_improvement >= config.patience:
        print(f"best_val_loss {best_val_loss}")
        print(f"Early stopping at epoch {epoch+1}")
        break
    print(f"best_val_loss {best_val_loss}")
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Average Loss: {average_train_loss:.4f}, Validation Average Loss: {average_val_loss:.4f}')
    


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
    
    batch_x_cpu = batch_x.cpu().detach().numpy()
    batch_y_cpu = batch_y.cpu().detach().numpy()
    output_cpu = output.cpu().detach().numpy()

    # Plotting
    for i in range(batch_x.shape[0]):  # Loop over each sample in the batch
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 3 rows, 5 columns for each sample

        # Plot images from batch_x for the current sample
        for j in range(10):
            axes[j // 5, j % 5].imshow(batch_x_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[j // 5, j % 5].set_title(f'X[{j}]')
            axes[j // 5, j % 5].axis('off')

        for j in range(5):
            axes[2, j].imshow(batch_y_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[2, j].set_title(f'True Label[{j}]')
            axes[2, j].axis('off')
        
        # Plot images from batch_y for the current sample
        for j in range(5):
            axes[3, j].imshow(output_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
            axes[3, j].set_title(f'Predicted Frame[{j}]')
            axes[3, j].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Loss: {loss.item():.4f}", fontsize=16)

        plt.savefig(os.path.join(visualization_folder, f"visualization_{i}.png"))  # Save the figure
        plt.close()

# Calculate average loss for the epoch
average_test_loss = test_epoch_loss / num_batches

print(f'Test Average Loss: {average_test_loss:.4f}')