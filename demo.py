from src.models.conv_lstm import ConvLSTM1D_Attention2,  ConvLSTM1D_Attention, ConvLSTM1D
from src.pipeline import train
from src.dataset import SimpleFrameDataset, VideoFrameDataset, CollisionDataset
import torch.nn as nn
import torch
from src.config import Config
from src.models.pretrained_model import get_classification_model
import os
import sys
from src.models.custom_loss import CustomLoss
from torch.utils.data import DataLoader
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

config = Config()


input_size = 100
hidden_size = 500
kernel_size = 3
num_layers = 3
learning_rate = 0.001
bidirectional = False
momentum = 0.9

model = ConvLSTM1D_Attention(input_size, hidden_size, kernel_size, num_layers, bidirectional)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device {device}')
if config.collision_flag:
    if config.pretrained_flag:
        checkpoint_file = f"model/{config.model_name}/best_model_checkpoint.pth"
        if os.path.isfile(checkpoint_file):
            model = get_classification_model(model, checkpoint_file)
            model.to(device)
            if config.custom_loss:
                criterion = CustomLoss(frame_rate=config.frame_rate/config.frame_avg_rate)
            else:
                criterion = nn.BCELoss()
            # print("Classification model:\n", model)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                model_name = config.collision_model_name
        else:
            print("pretrained model does not exist!")
            sys.exit(1)
    else:
        print("Pretrained flag is False. Shutting down...")
        sys.exit(1)
else:
    print("Collision flag is False. Shutting down...")
    sys.exit(1)


if config.dataset_type == 'collision':
    dataset = CollisionDataset(

        directory_path=config.dataset_path,
        split_ratio=0.80,
        test_flag=config.test_flag,
        DRR=config.DRR,
        frame_avg_rate=config.frame_avg_rate,
        prev_frames=config.prev_f, 
        custom_loss=config.custom_loss
    )
else:
    print("Dataset Type is not Collision. Shutting down...")


print("Dataset:",  config.dataset_type)
print("Model:", model_name, "| pretrained layers are frozen:", config.pretrained_flag)
print("loss function:", criterion)
print("threshold:", config.filtering_thresold)

model.eval()
test_loss = 0

samples_count = 0
data_loader = DataLoader(
    dataset=dataset.demo_dataset, batch_size=1, shuffle=False
)
# Create the Tkinter window
window = tk.Tk()
window.title("Video Prediction UI")

# Create labels to display past frames
past_labels = [tk.Label(window) for _ in range(10)]
for i in range(10):
    past_labels[i].grid(row=0, column=i)

# Create labels to display future frames
future_labels = [tk.Label(window) for _ in range(5)]
for i in range(5):
    future_labels[i].grid(row=1, column=i)

# Create label to display collision flag
collision_label = tk.Label(window, text="Collision Flag: ")
collision_label.grid(row=2, column=0, columnspan=5)

# Run the Tkinter event loop

window.update()

with torch.no_grad():
    for i, (images, labels, tta) in enumerate(data_loader):
        print(i)
        images = images.to(device)
        images = images.squeeze(0).to(device)
        print(labels.shape, labels)
        labels = labels.unsqueeze(1).to(device)
        # Convert tensors to PhotoImages
        past_frames = [tensor_to_photoimage(images[i]) for i in range(10)]
        future_frames = [tensor_to_photoimage(labels[i]) for i in range(10, 15)]

        test_pred_collision, test_pred_frames = model(images)
        test_pred_collision = torch.where(test_pred_collision>0.5, torch.tensor(1.0), torch.tensor(0.0))
        # Update UI elements with new images and labels
        for i in range(10):
            past_labels[i].config(image=past_frames[i])
            if i<5:
                future_labels[i].config(image=future_frames[i])
        collision_label.config(text=f'Collision Flag: {test_pred_collision} (Target: {test_pred_collision})')

        if config.custom_loss:
            tta = tta.to(device)
            batch_loss = criterion(test_pred_collision, labels, tta)
        else:
            batch_loss = criterion(test_pred_collision, labels)
            
        samples_count += labels.size(0)

        







def tensor_to_photoimage(tensor):
    tensor = tensor.cpu().detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    image = Image.fromarray((tensor * 255).astype(np.uint8))
    return ImageTk.PhotoImage(image)

