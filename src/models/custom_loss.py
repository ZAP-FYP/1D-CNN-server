import torch
import torch.nn as nn
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, frame_rate):
        super(CustomLoss, self).__init__()
        self.frame_rate = frame_rate

    def forward(self, y_pred, y_true, tta):
        batch_size = y_pred.size(0)  # Get batch size
        
        tta = tta / self.frame_rate

        total_loss = 0
        # Apply the exponential term for each item in the batch
        for i in range(batch_size):
            exp = -torch.exp(torch.max(torch.tensor(0, dtype=torch.float32), tta[i].clone().detach()))
            exp_loss = -torch.sum(y_true[i] * exp * y_pred[i])
            total_loss += exp_loss
                
        # Total loss computation
        total_loss /= batch_size  # Average loss over the batch

        return total_loss