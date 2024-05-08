import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
from src.models.conv_lstm import ConvLSTM1D_Attention

class CollisionClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(CollisionClassifier, self).__init__()
        self.pretrained_model = pretrained_model

        # Freeze the parameters of the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Get the number of input features for the new fully connected layer
        num_ftrs = self.pretrained_model.fc.out_features  # Assuming .fc is your fully connected layer

        # Define your new classification model
        num_outputs = 1  # Change this according to your task
        # Convolutional LSTM layers
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=500, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=500, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=100, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        # Replace fc layer in pretrained model with new classifier
        self.fc = nn.Linear(100, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Output of pre-trained model
        pretrained_output = self.pretrained_model(x)
        
        x = self.conv_layer1(pretrained_output)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.fc(x)

        final_output = self.sigmoid(x)
        
        return final_output, pretrained_output



def get_classification_model(pretrained_model, checkpoint_file):
    print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    # pretrained_model = checkpoint['model_state_dict']
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])  

    print("pretrained_model:\n", pretrained_model)

    collision_model = CollisionClassifier(pretrained_model)

    return collision_model


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
from src.models.conv_lstm import ConvLSTM1D_Attention

class CollisionClassifierFull(nn.Module):
    def __init__(self, pretrained_model1, pretrained_model2):
        super(CollisionClassifierFull, self).__init__()
        self.pretrained_model1 = pretrained_model1
        self.pretrained_model2 = pretrained_model2


    def forward(self, x):
        # Output of pre-trained model
        pretrained_output = self.pretrained_model1(x)
        final_output = self.pretrained_model2(pretrained_output)

        
        return final_output, pretrained_output



def get_classification_model_full(pretrained_model1, checkpoint_file1,pretrained_model2, checkpoint_file2 ):
    print(checkpoint_file1, checkpoint_file2)
    checkpoint1 = torch.load(checkpoint_file1)
    # pretrained_model = checkpoint['model_state_dict']
    pretrained_model1.load_state_dict(checkpoint1["model_state_dict"])  
    checkpoint2 = torch.load(checkpoint_file2)
    # pretrained_model = checkpoint['model_state_dict']
    pretrained_model2.load_state_dict(checkpoint2["model_state_dict"])  
    print("pretrained_model1:\n", pretrained_model1)
    print("pretrained_model2:\n", pretrained_model2)

    collision_model = CollisionClassifierFull(pretrained_model1, pretrained_model2)

    return collision_model

class CollisionClassifierTrainable(nn.Module):
    def __init__(self):
        super(CollisionClassifierTrainable, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=500, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=500, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=500, out_channels=100, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        # Replace fc layer in pretrained model with new classifier
        self.fc = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Output of pre-trained model
        # pretrained_output = self.pretrained_model(x)
        
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.fc(x)

        final_output = self.sigmoid(x)
        
        return final_output