import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from src.config import Config 
from src.tee import Tee
import os
import sys
import matplotlib.pyplot as plt
from src.dataset import Conv2d_dataset
from src.models.Conv2d import Conv2d, DeepConv2d, Conv2d_Pooling_Deconv, Conv2d_Residual,\
     DeepConv2d_Residual, Conv2d_SpatialPyramidPooling,Conv2dLSTM, UNet, DiceLoss,\
     WeightedDiceLoss, IoULoss, FocalLoss, DiceBCELoss, TverskyLoss, UNetWithRNN,\
         FocalLossWithVariencePenalty
from sklearn.metrics import f1_score, average_precision_score
from torchmetrics.classification import BinaryJaccardIndex
import torch.nn.functional as F

def calculate_positive_weight(dataset):
    num_ones = np.sum(dataset == 1)
    num_zeros = np.sum(dataset == 0)
    # Calculate the positive class ratio
    positive_ratio = num_zeros / num_ones
    return positive_ratio  

def calculate_weights(dataset):
    num_ones = np.sum(dataset == 1)
    num_zeros = np.sum(dataset == 0)

    return 1/num_zeros , 1/num_ones   
# Define parameters
x_window_size = 10
y_window_size = 5
stride = 6
batch_size = 16
num_epochs = 1000
learning_rate = 0.001

config = Config()
checkpoint_file = "model/" + config.model_name + "/best_model_checkpoint.pth"

if not os.path.exists("model/" + config.model_name):
    os.makedirs("model/" + config.model_name)
    f = open("model/" + config.model_name + "/log.txt", "w")
else:
    f = open("model/" + config.model_name + "/log.txt", "a")

original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Device {device}')
if device != "cpu":
    torch.cuda.empty_cache()


# List all numpy files in the directory
numpy_files = [f for f in os.listdir(config.dataset_path) if f.endswith('.npy')]
# numpy_files = ['/home/arvinths_19/YOLOPv2-1D_Coordinates/train_data/2d_maps/0000f77c-62c2a288.npy']

train_data = []
validation_data = []
test_data = []
positive_weights=[]
zero_freq=[]
one_freq=[]
# Iterate over each numpy file
for file_name in numpy_files:
    # Load numpy file
    data = np.load(os.path.join(config.dataset_path, file_name))
    data = np.squeeze(data)
    # positive_weight = calculate_positive_weight(data)
    # num_zeros , num_ones = calculate_weights(data)
    # if positive_weight==float('inf'):
    #     print(f"inf found in {file_name}")
    #     continue
    # break_flag = False
    # for frame in data:
    #     if np.all(frame == 0):
    #         print(file_name, "has empty frames")
    #         break_flag = True
    #         break
    # if break_flag:
    #     continue
    # positive_weights.append(positive_weight)
    # zero_freq.append(num_zeros)
    # one_freq.append(num_ones)
    # Splitting data into train and test/validation sets
    data_train, data_test_val = train_test_split(data, test_size=0.2, random_state=42)

    # Further splitting test/validation set into separate test and validation sets
    data_validation, data_test = train_test_split(data_test_val, test_size=0.5, random_state=42)

    # Append data to combined datasets
    train_data.extend(data_train)
    validation_data.extend(data_validation)
    test_data.extend(data_test)

# Create datasets from combined data
train_dataset = Conv2d_dataset(train_data, x_window_size, y_window_size, stride)
validation_dataset = Conv2d_dataset(validation_data, x_window_size, y_window_size, stride)
test_dataset = Conv2d_dataset(test_data, x_window_size, y_window_size, stride)

print(f'Combined Train samples {len(train_dataset)}')
print(f'Combined Validation samples {len(validation_dataset)}')
print(f'Combined Test samples {len(test_dataset)}')


# Load numpy file
# data = np.load(config.dataset_path)
# data = np.squeeze(data)

# print(f'data.shape {data.shape}')
def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# # Creating data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = UNetWithRNN(in_channels=x_window_size, out_channels=y_window_size)
print(model)
# print("positive weights",positive_weights)
# positive_weight = np.mean(positive_weights) 
# print(f'positive_weight {positive_weight}')
# total_one_freq = sum(one_freq)
# total_zero_freq = sum(zero_freq)
# total_freq = [total_zero_freq, total_one_freq]
# class_weights = [freq / total_zero_freq+total_one_freq for freq in total_freq]
# # Convert to tensor
# class_weights_tensor = torch.tensor(class_weights)
# # Add a new dimension
# class_weights_tensor = torch.unsqueeze(class_weights_tensor, 0)

# # Repeat the tensor along the new dimension
# class_weights_tensor = class_weights_tensor.repeat(5, 1)  # Repeat 5 times along dim 0, and 1 time along dim 1

# print(class_weights_tensor)
# Usage of WeightedDiceLoss with class weights
# criterion = WeightedDiceLoss(class_weights=class_weights_tensor)
# Define the weighted BCELoss
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
# criterion = DiceLoss()
Bce_criterion = nn.BCELoss()
Iou_criterion = IoULoss()
Iou = BinaryJaccardIndex().to(device)

criterion = FocalLoss()
Focal_criterion = FocalLoss()
# criterion = FocalLossWithDiversityPenalty()
# criterion = DiceBCELoss()
# criterion = TverskyLoss()


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.isfile(checkpoint_file):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    current_epoch = checkpoint["epoch"]

model.to(device)
if config.train_flag:
    # Training loop
    model = model.train()
    best_val_loss = float("inf")
    consecutive_no_improvement = 0

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        model.train()  # Set the model to training mode

        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x.float())
            loss = criterion(output, batch_y.float())
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            del batch_x
            del batch_y


        average_train_loss = train_epoch_loss / len(train_dataloader)

        # print(f"Epoch {epoch+1}: Train Loss: {average_train_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print(f"Epoch {epoch+1}: Train Loss: {average_train_loss:.4f}")
        num_batches = 0
        for batch_x, batch_y in validation_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
            optimizer.zero_grad()
            output = model(batch_x.float())
            loss = criterion(output, batch_y.float())
            loss.backward()
            optimizer.step()
            
            val_epoch_loss += loss.item()  # Accumulate the loss for the batch
            num_batches += 1  # Increment the batch counter
            del batch_x
            del batch_y
        
        # Calculate average loss for the epoch
        average_val_loss = val_epoch_loss / num_batches
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            consecutive_no_improvement = 0
            save_checkpoint(
                epoch,
                model,
                optimizer,
                "model/" + config.model_name + "/best_model_checkpoint.pth",
            )
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= config.patience:
            print(f"best_val_loss {best_val_loss}")
            print(f"Early stopping at epoch {epoch+1}")
            break
        print(f"best_val_loss {best_val_loss}")
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Average Loss: {average_train_loss:.4f}, Validation Average Loss: {average_val_loss:.4f}')
        

if config.test_flag:
    print("Testing Started...")
    # Testing loop
    model = model.eval()
    test_epoch_loss=0
    num_batches = 0
    visualization_folder = "visualizations/"+config.model_name
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
    f1_scores = []
    avg_precision_scores = []
    bce_scores = []
    iou_loss_scores = []
    iou_scores = [] 
    focal_scores = []
    channel_losses = {0:[],1:[],2:[],3:[],4:[]}
    for batch_x, batch_y in test_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Transfer data to CUDA
        
        output = model(batch_x.float())
        # Assuming batch_y is a tensor of shape [batch_size, num_channels, height, width]
        new_width = 100
        batch_y = F.interpolate(batch_y.float(), size=(batch_y.size(2), new_width), mode='nearest')               
        output = F.interpolate(output.float(), size=(batch_y.size(2), new_width), mode='nearest')        
        
        optimizer.zero_grad()
        
        output_flat = output.view(-1)
        batch_y_flat = batch_y.view(-1)
        loss = criterion(output, batch_y.float())
        bce = Bce_criterion(output_flat, batch_y_flat.float()).item()
        iou_loss = Iou_criterion(output_flat, batch_y_flat.float()).item()
        iou = Iou(output_flat, batch_y_flat.float()).item()

        focal = Focal_criterion(output_flat, batch_y_flat.float()).item()
        # Compute additional metrics
        output_binary = (output > 0.5).int()
        # print(f'output_binary.shape, batch_y.shape {output_binary.shape, batch_y_flat.shape}')
        f1 = f1_score(batch_y_flat.cpu().numpy(), output_binary.view(-1).cpu().numpy(),zero_division=1)
        avg_precision = average_precision_score(batch_y_flat.cpu().numpy(), output_binary.view(-1).cpu().detach().numpy())
        f1_scores.append(f1)
        avg_precision_scores.append(avg_precision)
        bce_scores.append(bce)
        iou_loss_scores.append(iou_loss)
        focal_scores.append(focal)
        iou_scores.append(iou)

        #  This channel wise loss works for focal loss

        # for sample_idx in range(batch_y.size(0)):
        #     for channel_idx in range(batch_y.size(1)):
        #         # print(f'output[sample_idx, channel_idx].shape, batch_y[sample_idx, channel_idx {output[sample_idx, channel_idx].shape, batch_y[sample_idx, channel_idx].shape}')
        #         # pred, true_label = torch.unsqueeze(output[sample_idx, channel_idx],0), torch.unsqueeze(batch_y[sample_idx, channel_idx],0) 
        #         pred, true_label = output[sample_idx, channel_idx], batch_y[sample_idx, channel_idx]
        #         # print(f'pred, true_label {pred.shape, true_label.shape}')
        #         channel_loss = criterion(pred, true_label.float())
        #         channel_losses[channel_idx].append(channel_loss.item())
    
        #  This channel wise loss works for focal loss with penalty

        # for sample_idx in range(batch_y.size(0)):
        #     for channel_idx in range(batch_y.size(1)):
        #         # print(f'output[sample_idx, channel_idx].shape, batch_y[sample_idx, channel_idx {output[sample_idx, channel_idx].shape, batch_y[sample_idx, channel_idx].shape}')
        #         # pred, true_label = torch.unsqueeze(output[sample_idx, channel_idx],0), torch.unsqueeze(batch_y[sample_idx, channel_idx],0) 

        #         pred, true_label = output[sample_idx, channel_idx].unsqueeze(0).unsqueeze(0), batch_y[sample_idx, channel_idx].unsqueeze(0).unsqueeze(0)
        #         # print(f'pred, true_label {pred.shape, true_label.shape}')
        #         channel_loss = criterion(pred, true_label.float())
        #         channel_losses[channel_idx].append(channel_loss.item())

        test_epoch_loss += loss.item()  # Accumulate the loss for the batch
        num_batches += 1  # Increment the batch counter
        
        batch_x_cpu = batch_x.cpu().detach().numpy()
        batch_y_cpu = batch_y.cpu().detach().numpy()
        output_cpu = output_binary.cpu().detach().numpy()


        # Plotting
        for i in range(batch_x.shape[0]):  # Loop over each sample in the batch
            fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 3 rows, 5 columns for each sample

            # Plot images from batch_x for the current sample
            for j in range(10):
                axes[j // 5, j % 5].imshow(batch_x_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
                axes[j // 5, j % 5].set_title(f'X[{j}] batch {num_batches}')
                axes[j // 5, j % 5].axis('off')

            for j in range(5):
                axes[2, j].imshow(batch_y_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
                axes[2, j].set_title(f'True Label[{j} batch {num_batches}]')
                axes[2, j].axis('off')
            
            # Plot images from batch_y for the current sample
            for j in range(5):
                axes[3, j].imshow(output_cpu[i, j], cmap='gray')  # Assuming binary images (0 and 1)
                axes[3, j].set_title(f'Predicted Frame[{j} batch {num_batches}]')
                axes[3, j].axis('off')

            plt.tight_layout()
            plt.suptitle(f"Loss: {loss.item():.4f}", fontsize=16)

            plt.savefig(os.path.join(visualization_folder, f"visualization_{i} batch {num_batches}.png"))  # Save the figure
            plt.close()
        # del batch_x
        # del batch_y
    # Calculate average loss for the epoch
    average_test_loss = test_epoch_loss / num_batches
    print(f"iou {iou_scores}")
    while 'nan' in iou_scores:
        iou_scores.remove('nan')
    print(f'Test Average Loss: {average_test_loss:.4f} \
        Focal loss: {sum(focal_scores) / len(focal_scores):.4f}\
        BCE loss: {sum(bce_scores) / len(bce_scores):.4f}\
        IoU loss: {sum(iou_loss_scores) / len(iou_loss_scores):.4f}\
        IoU: {sum(iou_scores) / len(iou_scores)}\
        Avg precision: {sum(avg_precision_scores) / len(avg_precision_scores):.4f} \
        F1 score: {sum(f1_scores) / len(f1_scores):.4f}')

    # print(f'Channel 1 loss: {sum(channel_losses[0]) / len(channel_losses[0]):.4f}\
    #         Channel 2 loss: {sum(channel_losses[1]) / len(channel_losses[1]):.4f}\
    #         Channel 3 loss: {sum(channel_losses[2]) / len(channel_losses[2]):.4f}\
    #         Channel 4 loss: {sum(channel_losses[3]) / len(channel_losses[3]):.4f}\
    #         Channel 5 loss: {sum(channel_losses[4]) / len(channel_losses[4]):.4f}\
    #             ')
    
