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
    images = []
    for x in range(tensor.shape[0]):
        coordinates = tensor[x].cpu().detach().numpy()
        
        # Create a line plot using Seaborn
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.lineplot(x=range(len(coordinates)), y=coordinates, ax=ax)
        if tensor.shape[0]==10:
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_title(f'Past Frame {x}')
        else:
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_title(f'Future Frame {x}')

        # Render the figure into a buffer
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()

        # Convert the buffer to a PIL Image
        image = Image.frombytes('RGBA', canvas.get_width_height(), buf)

        # Convert the PIL Image to a PhotoImage
        photo_image = ImageTk.PhotoImage(image)
        images.append(photo_image)
    return images

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
    if i<5:
        past_labels[i].grid(row=1, column=i)
    else:
        past_labels[i].grid(row=2, column=i-5)


# Create labels to display future frames
future_labels = [tk.Label(window) for _ in range(5)]
for i in range(5):
    future_labels[i].grid(row=4, column=i)

# Create label to display collision flag
collision_label = tk.Label(window, text="")
collision_label.grid(row=5, column=0, columnspan=5)
# Create labels to display past frames
past_frame_heading = tk.Label(window, text="Past Frames")
past_frame_heading.grid(row=0, column=0, columnspan=10)

# past_frame_canvas_1 = tk.Canvas(window, width=500, height=300)
# past_frame_canvas_1.grid(row=1, column=0, columnspan=10)
# past_frame_canvas_2 = tk.Canvas(window, width=500, height=300)
# past_frame_canvas_2.grid(row=2, column=0, columnspan=10)
# Create labels to display future frames
future_frame_heading = tk.Label(window, text="Future Frames")
future_frame_heading.grid(row=3, column=0, columnspan=5)

# future_frame_canvas = tk.Canvas(window, width=1000, height=300)
# future_frame_canvas.grid(row=4, column=0, columnspan=5)

# Create label to display collision flag
# collision_label = tk.Label(window, text="Collision Flag: ")
# collision_label.grid(row=4, column=0, columnspan=5)


# Run the Tkinter event loop

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

        # past_frames = [tensor_to_photoimage(images[i]) for i in range(10)]
        test_pred_frames = test_pred_frames.reshape(5, 100)
        print(test_pred_frames.shape)
        future_frames = tensor_to_photoimage(test_pred_frames)

        # future_frames = [tensor_to_photoimage(test_pred_frames[i]) for i in range(5)]
        test_pred_collision = torch.where(test_pred_collision>0.5, 1, 0)
        # Update UI elements with new images and labels
        for i in range(10):
            past_labels[i].config(image=past_frames[i])

            if i>=5:
                pass
                # past_frame_canvas_2.create_image(100*(i-5), 0, anchor=tk.NW, image=past_frames[i])
                # past_labels[i].config(image=past_frames[i])
            else:
                # past_labels[i].config(image=past_frames[i])
                future_labels[i].config(image=future_frames[i])

                # past_frame_canvas_1.create_image(100*i, 0, anchor=tk.NW, image=past_frames[i])
                # future_frame_canvas.create_image(100*i, 0, anchor=tk.NW, image=future_frames[i])

        # for i in range(5):
        if test_pred_collision:
            window.config(bg="red")
            collision_label.config(text=f'Collision detected in the future')       
        else:
            window.config(bg="green")
            collision_label.config(text=f'No Collisions detected in the future')       

        window.update()

window.destroy()








