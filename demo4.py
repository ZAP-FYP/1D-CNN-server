from src.models.conv_lstm import ConvLSTM1D
from src.pipeline import train
from src.dataset import SimpleFrameDataset, VideoFrameDataset, CollisionDataset
import torch.nn as nn
import torch
from src.config import Config
from src.models.pretrained_model import get_classification_model, CollisionClassifierFull, CollisionClassifierTrainable, get_classification_model_full
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
    x = 0
    coordinates = tensor[x].cpu().detach().numpy()
    
    # Create a line plot using Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size
    sns.lineplot(x=range(len(coordinates)), y=coordinates, ax=ax)
    ax.set_ylim(0, 350)

    if tensor.shape[0] == 10:
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title('Current Frame')
    else:
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title('Forecasted Future Frame')

    # Save the figure to a temporary file
    temp_file_path = "temp_figure.png"
    fig.savefig(temp_file_path, bbox_inches='tight', pad_inches=0.1)
    
    # Load the temporary file as a PhotoImage
    photo_image = ImageTk.PhotoImage(file=temp_file_path)
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    return photo_image


input_size = 100
hidden_size = 500
kernel_size = 3
num_layers = 3
learning_rate = 0.001
bidirectional = False
momentum = 0.9

pretrained_drivable_model = ConvLSTM1D(input_size, hidden_size, kernel_size, num_layers, bidirectional)
pretrained_collision_model = CollisionClassifierTrainable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device {device}')
if config.collision_flag:
    if config.pretrained_flag:
        checkpoint_file1 = f"{config.model_name}/best_model_checkpoint.pth"
        checkpoint_file2 = f"{config.collision_model_name}/best_model_checkpoint.pth"

        if os.path.isfile(checkpoint_file1) and os.path.isfile(checkpoint_file2):
            model = get_classification_model_full(pretrained_drivable_model, checkpoint_file1, pretrained_collision_model, checkpoint_file2 )
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
window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()

# Set window geometry
window.geometry(f"{window_width}x{window_height}")
window.title("Drivable Area Prediction")

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

# confusion_matrix_labels = []
# confusion_matrix_counts = []

tp_count = 0
tn_count = 0
fp_count = 0
fn_count = 0
# Labels for rows and columns of confusion matrix
confusion_matrix_rows = ["True Positive", "False Negative"]
confusion_matrix_columns = ["False Positive", "True Negative"]

# for i in range(2):  # Assuming binary classification
#     for j in range(2):
#         label = tk.Label(confusion_matrix_frame, width=10, height=2)
#         label.grid(row=i, column=j)
#         confusion_matrix_labels.append(label)

# Create another copy of the confusion matrix to track counts
# for i in range(2):  # Assuming binary classification
#     for j in range(2):
#         label = tk.Label(confusion_matrix_frame, width=10, height=2)
#         label.grid(row=i, column=j + 3)  # Shift columns for the counts
#         confusion_matrix_counts.append(label)

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
        future_frames = tensor_to_photoimage(test_pred_frames)

        test_pred_collision = torch.where(test_pred_collision>0.5, 1, 0)
        image_width = past_frames.width()  # Assuming past_frames and future_frames have the same width
        window_width = window.winfo_screenwidth()
        horizontal_position = (window_width - (2 * image_width))  # Divide by 4 for center alignment

        # # Set the horizontal position for past and future frames
        past_labels.place(relx=0.25, rely=0.44, anchor=tk.CENTER)
        future_labels.place(relx=0.75, rely=0.44, anchor=tk.CENTER)

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
                
                # Create a frame to contain the labels and boxes
        frame = tk.Frame(window, bg="white", bd=2, relief=tk.GROOVE)
        frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER, relwidth=0.4)  # Adjust rely to move the frame higher

        # Create a title label for the box
        title_label = tk.Label(frame, text="Anomaly Labels", bg="white", fg="black", font=("Helvetica", 12, "bold"))
        title_label.pack(pady=5)

        # Create a subframe for labels and boxes
        subframe = tk.Frame(frame, bg="white")
        subframe.pack(pady=10)

        actual_collision_label = tk.Label(subframe, text="Actual anomaly Label", bg="white", fg="black")
        actual_collision_label.grid(row=0, column=0, padx=5)

        actual_collision_box = tk.Label(subframe, width=10, height=2, bg="white", fg="black")
        actual_collision_box.grid(row=0, column=1, padx=5)

        predicted_collision_label = tk.Label(subframe, text="Predicted anomaly Label", bg="white", fg="black")
        predicted_collision_label.grid(row=0, column=2, padx=5)

        predicted_collision_box = tk.Label(subframe, width=10, height=2, bg="white", fg="black")
        predicted_collision_box.grid(row=0, column=3, padx=5)

                # Create a frame to contain the anomaly status
        anomaly_frame = tk.Frame(window, bg="white", bd=2, relief=tk.GROOVE)
        anomaly_frame.place(relx=0.96, rely=0.9, anchor=tk.E, relwidth=0.12)  # Adjust relwidth to make the frame smaller

        # Create a red box indicating anomaly
        anomaly_box = tk.Label(anomaly_frame, width=5, height=1, bg="red", fg="black")
        anomaly_box.grid(row=0, column=0, padx=5, pady=2)

        # Caption for the red box indicating anomaly
        anomaly_caption = tk.Label(anomaly_frame, text="Anomaly", bg="white", fg="black")
        anomaly_caption.grid(row=0, column=1, padx=5, pady=2)

        # Create a green box indicating not anomaly
        not_anomaly_box = tk.Label(anomaly_frame, width=5, height=1, bg="green", fg="black")
        not_anomaly_box.grid(row=1, column=0, padx=5, pady=2)

        # Caption for the green box indicating not anomaly
        not_anomaly_caption = tk.Label(anomaly_frame, text="Not Anomaly", bg="white", fg="black")
        not_anomaly_caption.grid(row=1, column=1, padx=5, pady=2)

        if true_positive:
            actual_collision_box.config(bg="red")
            predicted_collision_box.config(bg="red")
        elif true_negative:
            actual_collision_box.config(bg="green")
            predicted_collision_box.config(bg="green")
        elif false_positive:
            actual_collision_box.config(bg="green")
            predicted_collision_box.config(bg="red")
        elif false_negative:
            actual_collision_box.config(bg="red")
            predicted_collision_box.config(bg="green")

        # Update confusion matrix based on true positives, true negatives, false positives, and false negatives
        # confusion_matrix_labels[0].config(bg=tp_color if true_positive else default_bg_color, text="TP", fg="black")
        # confusion_matrix_labels[1].config(bg=fn_color if false_negative else default_bg_color, text="FN", fg="black")
        # confusion_matrix_labels[2].config(bg=fp_color if false_positive else default_bg_color, text="FP", fg="black")
        # confusion_matrix_labels[3].config(bg=tn_color if true_negative else default_bg_color, text="TN", fg="black")

        # # Update counts of true positives, true negatives, false positives, and false negatives
        # confusion_matrix_counts[0].config(text=f"TP: {tp_count}", bg=default_bg_color, fg="black")
        # confusion_matrix_counts[1].config(text=f"FN: {fn_count}", bg=default_bg_color, fg="black")
        # confusion_matrix_counts[2].config(text=f"FP: {fp_count}", bg=default_bg_color, fg="black")
        # confusion_matrix_counts[3].config(text=f"TN: {tn_count}", bg=default_bg_color, fg="black")


        window.update()

window.destroy()
