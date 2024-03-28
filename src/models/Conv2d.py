import torch.nn as nn
import torch.optim as optim
import torch

class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer

        # self.fc = nn.Linear(5 * (168//8 ) * (256//8 ), 1152000)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)  # Apply pooling layer
        x = torch.relu(self.conv2(x))
        # x = self.pool(x)  # Apply pooling layer
        x = torch.relu(self.conv3(x))

        x = self.sigmoid(x)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)

