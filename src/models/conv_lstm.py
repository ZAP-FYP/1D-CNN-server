
# class ConvLSTM1D(nn.Module):
#     def __init__(
#         self, input_size, hidden_size, kernel_size, num_layers, bidirectional=False
#     ):
#         super(ConvLSTM1D, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional

#         # Convolutional LSTM layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=10, out_channels=1, kernel_size=5, stride=1, padding=2
#             ),
#             nn.ReLU(),
#         )
#         self.conv_lstm = nn.LSTM(
#             input_size,
#             hidden_size,
#             num_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#         )

#         # Adjust the size of the fully connected layer output accordingly
#         fc_input_size = 2 * hidden_size if bidirectional else hidden_size
#         self.fc = nn.Linear(fc_input_size, 500)

#     def forward(self, x):
#         # Input shape: (batch_size, sequence_length, input_size)

#         # print("Input Shape:", x.shape)

#         # Initialize hidden and cell states
#         batch_size, _, _ = x.size()
#         # Initialize hidden and cell states
#         num_directions = 2 if self.bidirectional else 1
#         h0 = torch.zeros(
#             self.num_layers * num_directions, batch_size, self.hidden_size
#         ).to(x.device)
#         c0 = torch.zeros(
#             self.num_layers * num_directions, batch_size, self.hidden_size
#         ).to(x.device)
#         # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         input_x = x
#         selected_array = input_x[:, -1:, :]     # Get the 9th array in the middle dimension
#         # print("selected_array", selected_array.shape)

#         x = self.conv_layers(x)


#         # ConvLSTM forward pass
#         lstm_out, _ = self.conv_lstm(x, (h0, c0)) 

#         # Adding residual connection to lstm layer
#         # concatenated_input = torch.cat([x, selected_array], dim=1)
#         # lstm_out, _ = self.conv_lstm(concatenated_input, (h0, c0))

#         # print("LSTM Output Shape:", lstm_out.shape)


#         # Take the output of the last time step
#         lstm_last_output = lstm_out[:, -1, :] 
#         # print(lstm_last_output.shape)


#         # Fully connected layer
#         # output = self.fc(lstm_last_output) 

#         # Adding residual connection to fcn
#         reshaped = selected_array.repeat(1, 5, 1)        # Replicate it 5 times along the specified dimension (dimension 1 in this case)
#         reshaped = torch.flatten(reshaped, 1)
#         output = self.fc(lstm_last_output) + reshaped

#         # print("Output Shape:", output.shape)

#         return output


import torch
import torch.nn as nn

class ConvLSTM1D(nn.Module):
    def __init__(
        self, input_size, hidden_size, kernel_size, num_layers, bidirectional=False
    ):
        super(ConvLSTM1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Convolutional LSTM layers
        self.conv_layer = nn.Conv1d(
            in_channels=10, out_channels=1, kernel_size=5, stride=1, padding=2
        )
        self.relu = nn.ReLU()
        self.conv_lstm = nn.LSTM(
            1,  # Change input size to 1
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Adjust the size of the fully connected layer output accordingly
        fc_input_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size * input_size, 500)  # Adjusting input size

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)

        # Initialize hidden and cell states
        batch_size, _, _ = x.size()
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size
        ).to(x.device)

        # Process each channel sequentially
        outputs = []
        for i in range(x.size(2)):
            x_channel = x[:, :, i].unsqueeze(2)  # Select one channel
            x_channel = self.conv_layer(x_channel)  # Apply convolution
            x_channel = self.relu(x_channel)  # Apply ReLU
            x_channel = x_channel.permute(0, 2, 1)  # Reshape for LSTM
            lstm_out, _ = self.conv_lstm(x_channel, (h0, c0))  # ConvLSTM forward pass
            lstm_last_output = lstm_out[:, -1, :]  # Take output of last time step
            outputs.append(lstm_last_output)

        # Concatenate outputs from all channels
        output = torch.cat(outputs, dim=1)

        # Reshape output to match the input size of fully connected layer
        output = output.view(batch_size, -1)

        # Fully connected layer
        output = self.fc(output)

        return output

