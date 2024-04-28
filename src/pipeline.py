import torch
from torch.utils.data import DataLoader
import os
import sys
from src.tee import Tee
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from PIL import Image
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score

def train(
    dataset,
    criterion,
    optimizer,
    model,
    model_name,
    train_flag,
    test_flag,
    n_th_frame,
    future_f,
    visualization_flag,
    collision_flag,
    num_epochs=1000,
    batch_size=25,
    patience = 40, 
    custom_loss = False
):
    
    checkpoint_file = "model/" + model_name + "/model_checkpoint.pth"

    if not os.path.exists("model/" + model_name):
        os.makedirs("model/" + model_name)
        f = open("model/" + model_name + "/log.txt", "w")
    else:
        f = open("model/" + model_name + "/log.txt", "a")

    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    train_loader = DataLoader(
        dataset=dataset.train_dataset, batch_size=batch_size, shuffle=False
    )
    validation_loader = DataLoader(
        dataset=dataset.validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=dataset.test_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device {device}')
    model.to(device)

    current_epoch = 0
    total_steps = 0

    if os.path.isfile(checkpoint_file):
        print("Loading saved model...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]

    # print(f"Model summary : {summary(model, (in_channels, in_seq_len))}")
    # torchinfo.summary(model, (in_channels, 10, 100), device="cpu")
    print(model)
    if train_flag:
        # Define early stopping parameters
        print("Starting training...")
        best_val_loss = float("inf")
        consecutive_no_improvement = 0
        model = model.train()
        for epoch in range(current_epoch, num_epochs):
            batch_losses = []

            train_loss = 0.0

            for i, (images, labels, tta) in enumerate(train_loader):
                images = images.to(device)

                # y_hat = model(images)

                if collision_flag:
                    labels = labels.unsqueeze(1).to(device)
                    pred_collision, pred_frames = model(images)
                    # pred_collision = torch.where(pred_collision>0.5, torch.tensor(1.0), torch.tensor(0.0))
                else:
                    labels = labels.to(device)
                    pred_frames = model(images)

                                
                if collision_flag:
                    if custom_loss:
                        tta = tta.to(device)
                        loss = criterion(pred_collision, labels, tta)
                    else:
                        loss = criterion(pred_collision, labels)
                else:
                    loss = criterion(pred_frames, labels)

                train_loss += loss.item()                    
                batch_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1

                if not (i + 1) % 400:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] Current time:{datetime.now()}"
                    )

            train_loss /= len(train_loader)

            save_checkpoint(epoch, model, optimizer, checkpoint_file)

            with torch.no_grad():
                # model.eval()
                val_loss = 0.0
                bad_samples_val =[]
                mse_threshold = 1000

                predictions = []
                true_labels = []
                
                for i, (val_images, val_labels, val_tta) in enumerate(validation_loader):
                    val_images = val_images.to(device)

                    if collision_flag:
                        val_pred_collision, val_pred_frames = model(val_images)
                        # val_pred_collision = torch.where(val_pred_collision>0.5, torch.tensor(1.0), torch.tensor(0.0))
                        val_labels = val_labels.unsqueeze(1).to(device)                    
                    else:
                        val_labels = val_labels.to(device)
                        val_pred_frames = model(val_images)

                    # print(f'Val - y_hat.shape {val_outputs.shape} labels.shape {val_labels.shape}')
                    if collision_flag:
                        if custom_loss:
                            val_tta = val_tta.to(device)
                            loss = criterion(val_pred_collision, val_labels, val_tta)
                            val_pred_collision = torch.where(val_pred_collision>0.5, torch.tensor(1.0), torch.tensor(0.0))
                            true_labels.extend(val_labels.cpu().numpy())
                            predictions.extend(val_pred_collision.cpu().numpy())
                        else:
                            loss = criterion(val_pred_collision, val_labels)
                    else:
                        loss = criterion(val_pred_frames, val_labels)

                    val_loss += loss.item()

                    batch_size = val_images.size(0)
                    # last_frames = val_images[:, -1, :]
                    # duplicated_last_frames = last_frames.repeat(1, 5)
                    # reshaped_last_frames = duplicated_last_frames.view(batch_size, -1)
                    # random = criterion(reshaped_last_frames, val_labels)
                    # print("not accurate loss(last frame):", random)

                    # if visualization_flag:
                    #     if loss.item() > mse_threshold:
                    #         for val_output, image, label in zip(val_outputs, val_images, val_labels):
                    #             bad_samples_val.append((image, label, val_output))

                    #     output_folder = "visualizations/validation/" + model_name+ "/both"
                    #     visualize(val_images[-5:], val_labels[-5:], val_outputs[-5:], output_folder, n_th_frame, future_f)

                    #     output_folder_b = "visualizations/validation/" + model_name+ "/bad"
                    #     os.makedirs(output_folder_b, exist_ok=True)
                    #     if len(bad_samples_val)>10:
                    #         samples = 5
                    #     else:
                    #         samples = len(bad_samples_val)
                    #     for idx, (image, label, y_hat) in enumerate(bad_samples_val[-samples:]):
                    #         # print(idx)
                    #         sample_folder = os.path.join(output_folder_b, f"sample_{idx}")
                    #         os.makedirs(sample_folder, exist_ok=True)
                    #         visualize(image.unsqueeze(0), label.unsqueeze(0), y_hat.unsqueeze(0), sample_folder, n_th_frame, future_f)


                val_loss /= len(validation_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

                if collision_flag:
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions, zero_division=1) # Set zero_division=1 to set precision to 1.0 when no samples are predicted
                    recall = recall_score(true_labels, predictions)
                    f1_collision = f1_score(true_labels, predictions)

                    print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))
                    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_collision:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    "model/" + model_name + "/best_model_checkpoint.pth",
                )
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= patience:
                print(f"best_val_loss {best_val_loss}")
                print(f"Early stopping at epoch {epoch+1}")
                break
            print(f"best_val_loss {best_val_loss}")

    
    if test_flag:
        print("Starting testing...")
        model.eval()
        test_loss = 0
        baseline_loss = 0.0
        test_miou, baseline_test_miou = 0.0, 0.0
        avg_precision, baseline_avg_precision = 0.0, 0.0
        test_f1, baseline_test_f1 = 0.0, 0.0

        # test_miou1, baseline_test_miou1 = 0.0, 0.0
        # avg_precision1, baseline_avg_precision1 = 0.0, 0.0
        # test_f11, baseline_test_f11 = 0.0, 0.0

        test_bce, baseline_test_bce = 0.0, 0.0
        samples_count = 0
        Iou = BinaryJaccardIndex()
        f1_scores = []
        avg_precision_scores = []
        iou_scores = []
        mse_threshold = 150000
        bad_samples = []
        good_samples = []

        test_preds = []
        test_labels = []

        with torch.no_grad():
            for i, (images, labels, tta) in enumerate(test_loader):
                images = images.to(device)
                baseline_yhat = images[:, -5:, :].view(images.size(0), -1)
                
                if collision_flag:
                    labels = labels.unsqueeze(1).to(device)
                    test_pred_collision, test_pred_frames = model(images)
                    test_pred_collision = torch.where(test_pred_collision>0.5, torch.tensor(1.0), torch.tensor(0.0))
                else:
                    labels = labels.to(device)
                    test_pred_frames = model(images)

                if collision_flag:
                    if custom_loss:
                        tta = tta.to(device)
                        batch_loss = criterion(test_pred_collision, labels, tta)
                        test_loss += batch_loss.item()
                        test_labels.extend(labels.cpu().numpy())
                        test_preds.extend(test_pred_collision.cpu().numpy())
                    else:
                        batch_loss = criterion(test_pred_collision, labels)
                        test_loss += batch_loss.item()
                else:
                    batch_loss = criterion(test_pred_frames, labels)
                    test_loss += batch_loss.item() 
                    baseline_loss += criterion(baseline_yhat, labels).item()

                    
                samples_count += labels.size(0)

                if batch_loss.item() > mse_threshold:
                    for y_pred, image, label in zip(test_pred_frames, images, labels):
                        bad_samples.append((image, label, y_pred))
                else:
                    for y_pred, image, label in zip(test_pred_frames, images, labels):
                        good_samples.append((image, label, y_pred))
                
                if not collision_flag:
                    
                    ious, precisions, f1_scores, bce_losses, ious_1, precisions_1, f1_scores_1 = get_metrics(labels.reshape(labels.size(0), future_f, 100), test_pred_frames.reshape(test_pred_frames.size(0), future_f, 100),Iou)
                    # print("ious:", ious)
                    test_miou += (sum(ious)/(labels.size(0)*future_f))
                    avg_precision += (sum(precisions)/(labels.size(0)*future_f))
                    test_f1 += (sum(f1_scores)/(labels.size(0)*future_f))
                    test_bce += (sum(bce_losses)/(labels.size(0)*future_f))

                    # test_miou1 += (sum(ious_1)/(labels.size(0)*future_f))
                    # avg_precision1 += (sum(precisions_1)/(labels.size(0)*future_f))
                    # test_f11 += (sum(f1_scores_1)/(labels.size(0)*future_f))

                    baseline_ious, baseline_precisions, baseline_f1_scores, baseline_bce_losses,  baseline_ious_1, baseline_precisions_1, baseline_f1_scores_1= get_metrics(labels.reshape(labels.size(0), future_f, 100), baseline_yhat.reshape(y_hat.size(0), future_f, 100),Iou)
                    baseline_test_miou += (sum(baseline_ious)/(labels.size(0)*future_f))
                    baseline_avg_precision += (sum(baseline_precisions)/(labels.size(0)*future_f))
                    baseline_test_f1 += (sum(baseline_f1_scores)/(labels.size(0)*future_f))
                    baseline_test_bce += (sum(baseline_bce_losses)/(labels.size(0)*future_f))

                    # baseline_test_miou1 += (sum(baseline_ious_1)/(labels.size(0)*future_f))
                    # baseline_avg_precision1 += (sum(baseline_precisions_1)/(labels.size(0)*future_f))
                    # baseline_test_f11 += (sum(baseline_f1_scores_1)/(labels.size(0)*future_f))

        mean_test_loss = test_loss / len(test_loader)

        if not(collision_flag):
            baseline_loss /= len(test_loader)

            test_miou /= len(test_loader)
            avg_precision /= len(test_loader)
            test_f1 /= len(test_loader)
            test_bce /= len(test_loader)

            baseline_test_miou /= len(test_loader)
            baseline_avg_precision /= len(test_loader)
            baseline_test_f1 /= len(test_loader)
            baseline_test_bce /= len(test_loader)

            # test_miou1 /= len(test_loader)
            # avg_precision1 /= len(test_loader)
            # test_f11 /= len(test_loader)

            # baseline_test_miou1 /= len(test_loader)
            # baseline_avg_precision1 /= len(test_loader)
            # baseline_test_f11 /= len(test_loader)
        

        if collision_flag:
            accuracy = accuracy_score(test_labels, test_preds)
            precision = precision_score(test_labels, test_preds, zero_division=1) # Set zero_division=1 to set precision to 1.0 when no samples are predicted
            recall = recall_score(test_labels, test_preds)
            f1_collision = f1_score(test_labels, test_preds)

            print("Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))
            print(f"For Test Data - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_collision:.4f}")
            if custom_loss:
                print(f"Custom Loss of test data: {mean_test_loss:.3f}")
            else:
                print(f"Mean BCE of test data: {mean_test_loss:.3f}")
        else:
            print(f"MSE of test data: {mean_test_loss:.3f}")
            print("BCE for test data:", test_bce)
            print("IOU for test data:", test_miou)
            print("AP for test data:", avg_precision)
            print("F1 for test data:", test_f1)
            # print("IOU1 for test data:", test_miou1)
            # print("AP1for test data:", avg_precision1)
            # print("F11 for test data:", test_f11)


            print(f"Baseline MSE of test data: {baseline_loss:.3f}")
            print("Baseline BCE for test data:", baseline_test_bce)
            print("Baseline IOU for test data:", baseline_test_miou)
            print("Baseline AP for test data:", baseline_avg_precision)
            print("Baseline F1 for test data:", baseline_test_f1)
            # print("Baseline IOU1 for test data:", baseline_test_miou1)
            # print("Baseline AP1 for test data:", baseline_avg_precision1)
            # print("Baseline F11 for test data:", baseline_test_f11)

                # Save visualizations
        if visualization_flag:
            output_folder_b = "visualizations/test/bad/" + model_name
            os.makedirs(output_folder_b, exist_ok=True)

            output_folder_g = "visualizations/test/good/" + model_name
            os.makedirs(output_folder_g, exist_ok=True)

            for idx, (image, label, y_pred) in enumerate(bad_samples):
                sample_folder = os.path.join(output_folder_b, f"sample_{idx}")
                os.makedirs(sample_folder, exist_ok=True)
                visualize(image.unsqueeze(0), label.unsqueeze(0), y_pred.unsqueeze(0), sample_folder, n_th_frame, future_f)
            
            for idx, (image, label, y_pred) in enumerate(good_samples):
                sample_folder = os.path.join(output_folder_g, f"sample_{idx}")
                os.makedirs(sample_folder, exist_ok=True)
                visualize(image.unsqueeze(0), label.unsqueeze(0), y_pred.unsqueeze(0), sample_folder, n_th_frame, future_f)


            

def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def visualize(viz_images, viz_labels, viz_outputs, output_folder, n_th_frame, future_f):
    labels = viz_labels.view(viz_labels.size(0), 5, 100)
    y_hat = viz_outputs.view(viz_outputs.size(0), 5, 100)

    if n_th_frame:
        outer_loop = 1
        inner_loop = 1
    else:
        outer_loop = labels.size(0)
        interval = 1
        num_iterations = outer_loop // interval
        inner_loop = future_f

    for iteration in range(num_iterations):
        i = iteration * interval
        label_frame = labels[i]
        y_hat_frame = y_hat[i]
        image_frame = viz_images[i]

        sample_folder = os.path.join(output_folder, f"sample_{i}")
        os.makedirs(sample_folder, exist_ok=True)

        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))  # Increased nrows to accommodate the new row

        for k in range(10):
            ax = axes[k // 5, k % 5]  # Accessing subplot axes
            image = image_frame[k]  # Get the image frame
            image_array = image.cpu().detach().numpy()
            ax.plot(image_array, label="Data Array")
            ax.set_title(f'Data {k+1}')  # Set title for the subplot
            ax.set_ylim(bottom=0)

        for j in range(inner_loop):
            label_array = label_frame[j].cpu().detach().numpy()
            y_hat_array = y_hat_frame[j].cpu().detach().numpy()

            # Create a new row of subplots for inner_loop
            ax = axes[-1, j]  # Accessing subplot axes in the new row
            ax.plot(label_array, label="Label Array")
            ax.plot(y_hat_array, label="y_hat Array")
            ax.set_title(f"Frame {j}")
            ax.legend()
            ax.set_ylim(bottom=0)
         
        y_max = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(0, y_max)

        plt.tight_layout()  # Adjust layout

        # Save the plot as an image file
        output_file = os.path.join(sample_folder, f"sample_{i}.png")
        try:
            plt.savefig(output_file)
            # print(f"Image saved successfully: {output_file}")
        except Exception as e:
            print(f"Error saving image: {output_file}")
            print(e)
        finally:
            plt.close()  # Close the plot after saving


def get_metrics(labels, yhats, Iou):
    # gives the segmentation matrices for a given array
    ious = []
    precisions = []
    f1_scores = []
    bce_losses = []
    ious_1 = []
    precisions_1 = []
    f1_scores_1 = []

    labels_max_value = torch.max(torch.max(labels.reshape(-1, 100), dim=1)[0], dim=0)[0].item()
    yhats_max_value = torch.max(torch.max(yhats.reshape(-1, 100), dim=1)[0], dim=0)[0].item()
    ultimate_max_value = max(labels_max_value, yhats_max_value)
    # print("Ultimate max value:", ultimate_max_value)

    for i in range(labels.size(0)):
        # print(labels[i].shape)
        for j in range(labels[i].size(0)):
            label_seg_matrix = get_segmentation_matrix(labels[i][j], ultimate_max_value)
            yhat_seg_matrix = get_segmentation_matrix(yhats[i][j], ultimate_max_value)
            

            iou = calculate_iou(label_seg_matrix, yhat_seg_matrix)
            precision = calculate_precision(label_seg_matrix, yhat_seg_matrix)
            recall = calculate_recall(label_seg_matrix, yhat_seg_matrix)
            f1 = calculate_f1_score(precision, recall)
            
            iou_1 = Iou(yhat_seg_matrix.cpu(), label_seg_matrix.cpu()).item()
            f1_1 = f1_score(label_seg_matrix.cpu().numpy(), yhat_seg_matrix.cpu().numpy(),average='micro')
            avg_precision_1 = average_precision_score(label_seg_matrix.cpu().numpy(), yhat_seg_matrix.cpu().numpy())
        

            bce = calculate_bce(label_seg_matrix, yhat_seg_matrix)
            # print("bce", bce)
            
            ious_1.append(iou_1)
            precisions_1.append(avg_precision_1)
            f1_scores_1.append(f1_1)

            ious.append(iou)
            precisions.append(precision)
            f1_scores.append(f1)
            bce_losses.append(bce)
    
    return ious, precisions, f1_scores, bce_losses, ious_1, precisions_1, f1_scores_1


def get_segmentation_matrix1(array, max_size):
    # Create an empty binary matrix
    max_size = round(max_size * 2)
    binary_matrix = torch.zeros((max_size, 100), dtype=torch.int)

    # Iterate through each row and column
    for i in range(max_size):
        for j in range(100):
            # Check if row index is greater than the corresponding value in data array
            if i > array[j]:
                binary_matrix[i, j] = 0
            else:
                binary_matrix[i, j] = 1
    
    # Plot the binary matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(binary_matrix, cmap='binary', interpolation='nearest')
    # plt.colorbar(label='Binary Value')
    # plt.title('Binary Matrix Visualization')
    # plt.xlabel('Column Index')
    # plt.ylabel('Row Index')
    # plt.gca().invert_yaxis()
    # plt.savefig("seg_matrix_for_iou.jpg")

    return binary_matrix

def get_segmentation_matrix(array, max_size):
    max_size = max_size + 2 
    row_indices = torch.arange(max_size, dtype=torch.int, device='cuda:0')
    array = array.to('cuda:0')
    binary_matrix = torch.where(row_indices.unsqueeze(1) <= array.unsqueeze(0), 1, 0)
    return binary_matrix

def calculate_iou(y_true, y_pred):
    intersection = torch.logical_and(y_pred, y_true).sum()
    union = torch.logical_or(y_pred, y_true).sum()
    iou = intersection.float() / union.float()
    return iou.item() 


def calculate_precision(y_true, y_pred):
    TP = (y_true * y_pred).sum().item()
    FP = ((1 - y_true) * y_pred).sum().item()
    FN = (y_true * (1 - y_pred)).sum().item()

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision

def calculate_recall(y_true, y_pred):
    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    TP = (y_true * y_pred).sum().item()
    FP = ((1 - y_true) * y_pred).sum().item()
    FN = (y_true * (1 - y_pred)).sum().item()

    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall

def calculate_f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1_score

def calculate_bce(y_true, y_pred):
    criterion = nn.BCELoss()
    loss = criterion(y_pred.float(), y_true.float())  # Ensure y_true is of type float for the loss calculation
    return loss.item()