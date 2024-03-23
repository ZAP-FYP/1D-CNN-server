import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
from src.models.conv_lstm import ConvLSTM1D_Attention

# class CollisionClassifier(ConvLSTM1D_Attention):
#     def __init__(self):
#         super(CollisionClassifier, self).__init__()
#         num_ftrs = self.fc.in_features
#         self.fc = nn.Linear(num_ftrs, 2)
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         x = super().forward(x)  # Call the forward method of the parent class
#         x = self.sigmoid(x)
#         return x

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
    print(checkpoint.keys())
    # pretrained_model = checkpoint['model_state_dict']
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])  

    print("pretrained_model:\n", pretrained_model)

    collision_model = CollisionClassifier(pretrained_model)

    # # Freeze the parameters of the pre-trained model
    # for param in pretrained_model.parameters():
    #     param.requires_grad = False

    # # Get the number of input features for the new fully connected layer
    # num_ftrs = pretrained_model.fc.in_features  # Assuming .fc is your fully connected layer

    # # Define your new classification model
    # num_classes = 1  # Change this according to your task

    # # Replace fc layer in pretrained model with new classifier
    # # pretrained_model.fc = nn.Sequential(
    # #     nn.Linear(num_ftrs, num_classes),
    # #     nn.Sigmoid()
    # # )

    # pretrained_model.fc = nn.Linear(num_ftrs, num_classes)
    # pretrained_model.add_module('sigmoid', nn.Sigmoid())

    return collision_model

