import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load numpy file
data = np.load('/Users/springy/Desktop/New Projects/YOLOPv2-1D_Coordinates/train_data/2d_maps/000d4f89-3bcbe37a.npy')
print(f'data.shape {data.shape}')
# Define dataset class
class MyDataset(Dataset):
    def __init__(self, data, x_window_size=10, y_window_size=5, stride=1):
        self.data = data
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.stride = stride

    def __len__(self):
        return len(self.data) - (self.x_window_size + self.y_window_size) + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.x_window_size * self.stride
        y_start = x_end
        y_end = y_start + self.y_window_size * self.stride

        x = self.data[x_start:x_end].reshape(-1, 1, 360, 640)
        y = self.data[y_start:y_end].reshape(-1, 1, 360, 640)

        return x, y




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * (360 // 8) * (640 // 8), y_window_size * 360 * 640)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), y_window_size, 1, 360, 640)



# Define parameters
x_window_size = 10
y_window_size = 5
stride = 1
batch_size = 64
num_epochs = 1000
learning_rate = 0.001

# Create dataset and dataloader
dataset = MyDataset(data, x_window_size, y_window_size, stride)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = CNNModel()
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        print(f'batch_x.shape, batch_y.shape {batch_x.shape, batch_y.shape }')
        optimizer.zero_grad()
        output = model(batch_x.float())
        loss = criterion(output, batch_y.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
# (You can use a separate validation set or split your data into train and validation sets)
