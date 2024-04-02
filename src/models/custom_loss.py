import torch
import torch.nn as nn
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, tta):
        batch_size = y_pred.size(0)  # Get batch size
        
        frame_rate = 1
        tta = tta / frame_rate

        # Calculate binary cross-entropy loss
        bce_loss = nn.BCELoss(reduction='none')
        loss = bce_loss(y_pred, y_true)

        total_loss = 0
        # Apply the exponential term for each item in the batch
        for i in range(batch_size):
            exp = -torch.exp(torch.max(torch.tensor(0, dtype=torch.float32), tta[i].clone().detach()))
            exp_loss = -torch.sum(loss[i] * exp)
            total_loss += exp_loss
                
        # Total loss computation
        total_loss /= batch_size  # Average loss over the batch

        return total_loss