import torch.nn as nn
import torch.optim as optim
import torch

class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
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



class Conv2d_Residual(nn.Module):
    def __init__(self):
        super(Conv2d_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer

        # self.fc = nn.Linear(5 * (168//8 ) * (256//8 ), 1152000)  

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        # x = self.pool(x)  # Apply pooling layer
        x2 = torch.relu(self.conv2(x1))
        x2 = x2 + x1  # Skip connection

        # x = self.pool(x)  # Apply pooling layer
        x3 = torch.relu(self.conv3(x2))


        x3 = self.sigmoid(x3)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)

class DeepConv2d(nn.Module):
    def __init__(self):
        super(DeepConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        
        x = self.sigmoid(x)
        
        return x.view(x.size(0), 5, 168, 256)

class DeepConv2d_Residual(nn.Module):
    def __init__(self):
        super(DeepConv2d_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial convolutional layer
        x1 = torch.relu(self.conv1(x))
        
        # Second convolutional layer with skip connection
        x2 = torch.relu(self.conv2(x1))
        x2 = x2 + x1  # Skip connection
         
        # Third convolutional layer with skip connection
        x3 = torch.relu(self.conv3(x2))
        x3 = x3 + x2  # Skip connection
        
        # Fourth convolutional layer with skip connection
        x4 = torch.relu(self.conv4(x3))
        x4 = x4 + x3  # Skip connection
        
        # Fifth convolutional layer with skip connection
        x5 = torch.relu(self.conv5(x4))
        x5 = x5 + x4  # Skip connection
        
        # Sixth convolutional layer with skip connection
        x6 = torch.relu(self.conv6(x5))
        x6 = x6 + x5  # Skip connection
        
        # Seventh convolutional layer with skip connection
        x7 = torch.relu(self.conv7(x6))
        x7 = x7 + x6  # Skip connection
        
        # Final convolutional layer
        x8 = torch.relu(self.conv8(x7))
        
        # Apply sigmoid activation
        x8 = self.sigmoid(x8)
        
        return x8.view(x8.size(0), 5, 168, 256)

class Conv2d_Pooling(nn.Module):
    def __init__(self):
        super(Conv2d_Pooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(f'shape after 1st layer {x.shape}')
        x = self.avg_pool(x)  # Apply average pooling layer
        print(f'shape after 1st pooling {x.shape}')

        
        x = torch.relu(self.conv2(x))
        print(f'shape after 2nd layer {x.shape}')

        x = self.avg_pool(x)  # Apply average pooling layer
        print(f'shape after 2nd pooling {x.shape}')

        
        x = torch.relu(self.conv3(x))
        print(f'shape after 3rd layer {x.shape}')

        x = self.avg_pool(x)  # Apply average pooling layer
        print(f'shape after 3rd pooling {x.shape}')

        x = self.sigmoid(x)  # Applying sigmoid activation

        
        return x.view(x.size(0), 5, 168, 256)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, output_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.output_sizes = output_sizes

    def forward(self, x):
        features = []
        num_channels = x.size(1)
        for output_size in self.output_sizes:
            # Apply adaptive average pooling
            pooled_features = nn.functional.adaptive_avg_pool2d(x, output_size)
            # Reshape and append to the list of features
            features.append(pooled_features.view(x.size(0), num_channels, -1))
        # Concatenate the pooled features along the channel dimension
        return torch.cat(features, dim=2)

class DeepConv2d_SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(DeepConv2d_SpatialPyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32 * 5, out_channels=5, kernel_size=1)  # Adjusted channel count
        
        self.sigmoid = nn.Sigmoid()
        self.spp = SpatialPyramidPooling(output_sizes=[(1, 1), (2, 2), (4, 4)])  # Output sizes for SPP

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        
        # Apply Spatial Pyramid Pooling
        x = self.spp(x)
        
        x = torch.relu(self.conv8(x))
        x = self.sigmoid(x)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)

class Conv2d_Deconv(nn.Module):
    def __init__(self):
        super(Conv2d_Deconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, padding=1)
        
        
        self.sigmoid = nn.Sigmoid()
        
        # Deconvolutional layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Apply deconvolutional layers for upsampling
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        
        x = torch.relu(self.conv3(x))
        x = self.sigmoid(x)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)