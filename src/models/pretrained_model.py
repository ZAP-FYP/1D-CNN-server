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
        
        # Output of the final classification layer
        x = self.fc(pretrained_output)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

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

