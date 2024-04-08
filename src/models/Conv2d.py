import torch.nn as nn
import torch.optim as optim
import torch

class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.sigmoid(x)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)

class Conv2dLSTM(nn.Module):
    def __init__(self):
        super(Conv2dLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer


        # Define LSTM layers

        self.lstm = nn.LSTM(input_size=5*168*256, hidden_size=5*256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(5 * 256, 64*5*168*256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Reshape x for LSTM

        x = x.view(x.size(0), -1)
        print(f'inpit lstm {x.unsqueeze(1).shape}')
        # LSTM layer

        out, _ = self.lstm(x.unsqueeze(1))  # Add a channel dimension for LSTM

        print(f'output lstm {out.shape}')

        # Sigmoid activation
        x = self.fc(out)
        x = self.sigmoid(x)

        return x.view(x.size(0), 5, 168, 256)  


class Conv2d_Residual(nn.Module):
    def __init__(self):
        super(Conv2d_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer

        # self.fc = nn.Linear(5 * (168//8 ) * (256//8 ), 1152000)  

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        # x = self.pool(x)  # Apply pooling layer
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2))
        print(f' before{x3.shape}')

        x3 = x3 + x1  # Skip connection
        print(f'after {x3.shape}')

        # x = self.pool(x)  # Apply pooling layer
        x4 = torch.relu(self.conv4(x3))
        print(f'after relu {x4.shape}')


        x4 = self.sigmoid(x4)  # Applying sigmoid activation
        print(f'after sigmoid {x4.shape}')

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

class Conv2d_SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(Conv2d_SpatialPyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        
        self.spp = SpatialPyramidPooling(output_sizes=[(1, 1), (2, 2), (4, 4)])  # Output sizes for SPP
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=3, padding=1)  # Adjusted channel count
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(f'after 1rd conv {x.shape}')

        # Apply Spatial Pyramid Pooling
        x = self.spp(x)
        print(f'after 1rd pool {x.shape}')

        x = torch.relu(self.conv2(x))
        print(f'after 2rd conv {x.shape}')

        x = self.sigmoid(x)  # Applying sigmoid activation
        
        return x.view(x.size(0), 5, 168, 256)

class Conv2d_Pooling_Deconv(nn.Module):
    def __init__(self):
        super(Conv2d_Pooling_Deconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Add batch normalization after conv1

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,stride = 2 , padding = 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Add batch normalization after conv2

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,stride=2, padding=2 )

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # Add batch normalization after conv3

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=5, kernel_size=4,stride=2, padding=2 )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply batch normalization before activation

        x = self.avg_pool(x)
        x = torch.relu(self.deconv1(x))
        # print(f'after 1rd deconv {x.shape}')

        x = torch.relu(self.bn2(self.conv2(x)))  # Apply batch normalization before activation

        x = self.avg_pool(x)
        x = torch.relu(self.deconv2(x))
        # print(f'after 2rd deconv {x.shape}')

        x = torch.relu(self.bn3(self.conv3(x)))  # Apply batch normalization before activation

        x = self.avg_pool(x)
        x = torch.relu(self.deconv3(x))
        # print(f'after 3rd deconv {x.shape}')
        x = self.sigmoid(x)

        return x.view(x.size(0), 5, 168, 256)


import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder Path
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv = self.conv_block(128, 256)
        
        # Decoder Path
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv1 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv2 = self.conv_block(128, 64)
        
        # Output
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Path
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(self.pool(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck_conv(self.pool(enc2))
        
        # Decoder Path
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat([enc2, dec1], dim=1)
        dec1 = self.decoder_conv1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([enc1, dec2], dim=1)
        dec2 = self.decoder_conv2(dec2)
        
        # Output
        output = self.output_conv(dec2)
        output = torch.sigmoid(output)  # Applying sigmoid to ensure output is in [0,1] range
        
        return output
import torch
import torch.nn as nn

class UNetWithRNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=256, num_layers=1, rnn_type='LSTM'):
        super(UNetWithRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Encoder Path
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv = self.conv_block(128, 256)
        
        # Recurrent Layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(64, hidden_size, num_layers, batch_first=True)
        
        # Decoder Path
        self.upconv1 = nn.ConvTranspose2d(hidden_size, 128, kernel_size=2, stride=2)
        self.decoder_conv1 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv2 = self.conv_block(128, 64)
        
        # Output
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Path
        enc1 = self.encoder_conv1(x)
        # RNN Input Preparation
        batch_size, channels, height, width = enc1.size()
        rnn_input = enc1.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape for LSTM

        # Apply RNN
        rnn_output, _ = self.rnn(rnn_input)

        rnn_output = rnn_output.permute(0, 2, 1).view(batch_size, channels, height, width)  # Reshape back
        
        enc2 = self.encoder_conv2(self.pool(rnn_output.squeeze(1)))
        
        # Bottleneck
        bottleneck = self.bottleneck_conv(self.pool(enc2))
        

        # Decoder Path
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat([enc2, dec1], dim=1)
        dec1 = self.decoder_conv1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([enc1, dec2], dim=1)
        dec2 = self.decoder_conv2(dec2)
        
        # Output
        output = self.output_conv(dec2)
        output = torch.sigmoid(output)  # Applying sigmoid to ensure output is in [0,1] range
        
        return output


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = prediction.contiguous()
        target = target.contiguous()

        intersection = (prediction * target).sum(dim=2).sum(dim=2)
        dice_coeff = (2. * intersection + self.smooth) / (prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        dice_loss = 1 - dice_coeff.mean()

        return dice_loss

class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1., class_weights=None):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, prediction, target):
        prediction = prediction.contiguous()
        target = target.contiguous()

        intersection = (prediction * target).sum(dim=2).sum(dim=2)
        dice_coeff = (2. * intersection + self.smooth) / (prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        
        if self.class_weights is not None:
            dice_coeff = dice_coeff * self.class_weights

        dice_loss = 1 - dice_coeff.mean()

        return dice_loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2
SMOOTH = 1

class FocalLossWithDiversityPenalty(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLossWithDiversityPenalty, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=SMOOTH):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # First compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        
        # Compute diversity penalty
        diversity_penalty = self.compute_diversity_penalty(inputs)
        
        # Add diversity penalty to focal loss
        total_loss = focal_loss + diversity_penalty
        
        return total_loss
    
    def compute_diversity_penalty(self, inputs):
        # Compute diversity penalty based on the standard deviation of predictions across the five output channels
        # Inputs shape: [batch_size * num_channels * height * width]
        batch_size, num_channels, _, _ = inputs.size()
        
        # Reshape inputs to [batch_size, num_channels, -1]
        inputs_reshaped = inputs.view(batch_size, num_channels, -1)
        
        # Compute mean prediction across channels for each pixel
        mean_prediction = torch.mean(inputs_reshaped, dim=1, keepdim=True)
        
        # Compute diversity penalty based on standard deviation
        diversity_penalty = torch.std(inputs_reshaped, dim=1, keepdim=True)
        diversity_penalty = torch.mean(diversity_penalty)
        
        return diversity_penalty


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

#PyTorch
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky