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
    patience = 40
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
            train_loss = 0.0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                if collision_flag:
                    labels = labels.unsqueeze(1).to(device)
                else:
                    labels = labels.to(device)

                y_hat = model(images)
                # print(f'y_hat.shape {y_hat.shape} labels.shape {labels.shape}')
                
                loss = criterion(y_hat, labels)
                train_loss += loss.item()

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
                
                for i, (val_images, val_labels) in enumerate(validation_loader):
                    val_images = val_images.to(device)

                    if collision_flag:
                        val_labels = val_labels.unsqueeze(1).to(device)
                    else:
                        val_labels = val_labels.to(device)

                    val_outputs = model(val_images)
                    # print(f'Val - y_hat.shape {val_outputs.shape} labels.shape {val_labels.shape}')
                    loss = criterion(val_outputs, val_labels)

                    val_loss += loss.item()

                    # _, predicted = torch.max(val_outputs, 1)
                    true_labels.extend(val_labels.cpu().numpy())
                    predictions.extend(np.where(val_outputs.cpu().numpy() > 0.5, 1, 0))

                    # print("original loss:",loss)

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
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                if collision_flag:
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions, zero_division=1) # Set zero_division=1 to set precision to 1.0 when no samples are predicted
                    recall = recall_score(true_labels, predictions)
                    f1 = f1_score(true_labels, predictions)

                    print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))
                    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                
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
        samples_count = 0
        mse_threshold = 6
        bad_samples = []
        good_samples = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                
                if collision_flag:
                    labels = labels.unsqueeze(1).to(device)
                else:
                    labels = labels.to(device)

                y_hat = model(images)
                batch_loss = criterion(y_hat, labels)

                test_preds = np.where(y_hat.cpu().numpy() > 0.5, 1, 0)

                test_loss += batch_loss.item() * labels.size(0)
                samples_count += labels.size(0)

                if batch_loss.item() > mse_threshold:
                    for y_pred, image, label in zip(y_hat, images, labels):
                        bad_samples.append((image, label, y_pred))
                else:
                    for y_pred, image, label in zip(y_hat, images, labels):
                        good_samples.append((image, label, y_pred))

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

        mean_test_loss = test_loss / samples_count

        if collision_flag:
            accuracy = accuracy_score(labels, test_preds)
            precision = precision_score(labels, test_preds, zero_division=1) # Set zero_division=1 to set precision to 1.0 when no samples are predicted
            recall = recall_score(labels, test_preds)
            f1 = f1_score(labels, test_preds)

            print("Confusion Matrix:\n", confusion_matrix(labels, y_hat))
            print(f"For Test Data - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            print(f"Mean BCE of test data: {mean_test_loss:.3f}")
        else:
            print(f"MSE of test data: {mean_test_loss:.3f}")

            

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