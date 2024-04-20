from src.models.conv_lstm import ConvLSTM1D_Attention2,  ConvLSTM1D_Attention
from src.pipeline import train
from src.dataset import SimpleFrameDataset, VideoFrameDataset, CollisionDataset
import torch.nn as nn
import torch
from src.config import Config
from src.models.pretrained_model import get_classification_model
import os
import sys
from src.models.custom_loss import CustomLoss

config = Config()


input_size = 100
hidden_size = 500
kernel_size = 3
num_layers = 3
learning_rate = 0.001
bidirectional = False
momentum = 0.9

model = ConvLSTM1D_Attention(input_size, hidden_size, kernel_size, num_layers, bidirectional)

if config.collision_flag:
    if config.pretrained_flag:
        checkpoint_file = f"model/{config.model_name}/best_model_checkpoint.pth"
        if os.path.isfile(checkpoint_file):
            model = get_classification_model(model, checkpoint_file)
            # print("Classification model:\n", model)
        else:
            print("pretrained model does not exist!")
            sys.exit(1)
    else:
        print("Collision model from scratch starting...")
        model = ConvLSTM1D_Attention2(input_size, hidden_size, kernel_size, num_layers, bidirectional)        

    if config.custom_loss:
        criterion = CustomLoss(frame_rate=config.frame_rate/config.frame_avg_rate)
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    model_name = config.collision_model_name
else:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_name = config.model_name
    

if config.dataset_type == 'simple':
    dataset = SimpleFrameDataset(
        frame_length=100,
        car_width=20,
        min_velocity=1,
        max_velocity=5,
        num_frames=100,
        window_x=10,
        window_y=5,
        num_duplicates=20,
        train_frame_start=0,
        val_frame_start=60,
        test_frame_start=80,
        horizontal_velocity=1,
        vertical_velocity=1,
        acceleration=False,
    )
elif config.dataset_type == 'video':
    dataset = VideoFrameDataset(
        directory_path=config.dataset_path,
        split_ratio=0.80,
        test_flag=config.test_flag,
        DRR=config.DRR,
        n_th_frame=config.n_th_frame,
        frame_avg_rate=config.frame_avg_rate,
        prev_frames=config.prev_f,
        future_frames=config.future_f,
        threshold=config.filtering_thresold
    )
elif config.dataset_type == 'collision':
    dataset = CollisionDataset(

        directory_path=config.dataset_path,
        split_ratio=0.80,
        test_flag=config.test_flag,
        DRR=config.DRR,
        frame_avg_rate=config.frame_avg_rate,
        prev_frames=config.prev_f, 
        custom_loss=config.custom_loss
    )


print("Dataset:",  config.dataset_type)
print("Model:", model_name, "| pretrained layers are frozen:", config.pretrained_flag)
print("loss function:", criterion)
print("threshold:", config.filtering_thresold)
train(
    dataset,
    criterion,
    optimizer,
    model,
    model_name,
    config.train_flag,
    config.test_flag,
    config.n_th_frame,
    config.future_f,
    config.visualization_flag,
    config.collision_flag,
    num_epochs=1000,
    batch_size=256,
    patience=config.patience, 
    custom_loss=config.custom_loss
)