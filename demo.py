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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

config = Config()
def tensor_to_photoimage(tensor):
    # images = []
    # for x in range(tensor.shape[0]):
    x = 0
    coordinates = tensor[x].cpu().detach().numpy()
    # Create a line plot using Seaborn
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.lineplot(x=range(len(coordinates)), y=coordinates, ax=ax)
    ax.set_ylim(0, 350)

    if tensor.shape[0]==10:
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title(f'Past Frame ')
    else:
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title(f'Forecasted Future Frame ')

    # Render the figure into a buffer
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()

    # Convert the buffer to a PIL Image
    image = Image.frombytes('RGBA', canvas.get_width_height(), buf)

    # Convert the PIL Image to a PhotoImage
    photo_image = ImageTk.PhotoImage(image)
    # images.append(photo_image)
    return photo_image

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

past_labels = tk.Label(window) 
past_labels.grid(row=1, column=1, padx=10, pady=5)

future_labels = tk.Label(window) 
future_labels.grid(row=1, column=2, padx=10, pady=5)

confusion_matrix_frame = tk.Frame(window)
confusion_matrix_frame.grid(row=2, column=0, columnspan=5)

# Label for confusion matrix
confusion_matrix_label = tk.Label(window, text="Confusion Matrix",bg="white")
confusion_matrix_label.grid(row=3, column=0, columnspan=5)

tp_color = "green"
tn_color = "green"
fp_color = "red"
fn_color = "red"
window.config(bg="white")
default_bg_color = window.cget("bg")

confusion_matrix_labels = []
confusion_matrix_counts = []

tp_count = 0
tn_count = 0
fp_count = 0
fn_count = 0
# Labels for rows and columns of confusion matrix
confusion_matrix_rows = ["True Positive", "False Negative"]
confusion_matrix_columns = ["False Positive", "True Negative"]

for i in range(2):  # Assuming binary classification
    for j in range(2):
        label = tk.Label(confusion_matrix_frame, width=10, height=2)
        label.grid(row=i, column=j)
        confusion_matrix_labels.append(label)

# Create another copy of the confusion matrix to track counts
for i in range(2):  # Assuming binary classification
    for j in range(2):
        label = tk.Label(confusion_matrix_frame, width=10, height=2)
        label.grid(row=i, column=j + 3)  # Shift columns for the counts
        confusion_matrix_counts.append(label)

window.update()

with torch.no_grad():
    for i, (images, labels, tta) in enumerate(data_loader):
        print(i)
        images = images.to(device)
        test_pred_collision, test_pred_frames = model(images)

        images = images.squeeze(0).to(device)
        print(labels.shape, labels)
        print(images.shape)

        labels = labels.unsqueeze(1).to(device)

        # Convert tensors to PhotoImages
        past_frames = tensor_to_photoimage(images)

        test_pred_frames = test_pred_frames.reshape(5, 100)
        print(test_pred_frames.shape)
        future_frames = tensor_to_photoimage(test_pred_frames)

        test_pred_collision = torch.where(test_pred_collision>0.5, 1, 0)
        # Update UI elements with new images and labels
        past_labels.config(image=past_frames)
        future_labels.config(image=future_frames)

        true_positive = (labels == 1) & (test_pred_collision == 1)
        true_negative = (labels == 0) & (test_pred_collision == 0)
        false_positive = (labels == 0) & (test_pred_collision == 1)
        false_negative = (labels == 1) & (test_pred_collision == 0)
        if true_positive:
            tp_count += 1
        elif true_negative:
            tn_count += 1
        elif false_positive:
            fp_count += 1
        elif false_negative:
            fn_count += 1

        # Update confusion matrix based on true positives, true negatives, false positives, and false negatives
        confusion_matrix_labels[0].config(bg=tp_color if true_positive else default_bg_color, text="TP" )
        confusion_matrix_labels[1].config(bg=fn_color if false_negative else default_bg_color, text="FN" )
        confusion_matrix_labels[2].config(bg=fp_color if false_positive else default_bg_color, text="FP" )
        confusion_matrix_labels[3].config(bg=tn_color if true_negative else default_bg_color, text="TN" )

        # Update counts of true positives, true negatives, false positives, and false negatives
        confusion_matrix_counts[0].config(text=f"TP: {tp_count}",bg = default_bg_color)
        confusion_matrix_counts[1].config(text=f"FN: {fn_count}",bg = default_bg_color)
        confusion_matrix_counts[2].config(text=f"FP: {fp_count}",bg = default_bg_color)
        confusion_matrix_counts[3].config(text=f"TN: {tn_count}",bg = default_bg_color)

        window.update()

window.destroy()
