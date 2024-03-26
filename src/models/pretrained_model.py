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

        # Replace fc layer in pretrained model with new classifier
        self.fc = nn.Linear(num_ftrs, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x



def get_classification_model(pretrained_model, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    # pretrained_model = checkpoint['model_state_dict']
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])  

    print("pretrained_model:\n", pretrained_model)

    collision_model = CollisionClassifier(pretrained_model)

    return collision_model

