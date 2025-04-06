import torch
import torch.nn as nn
import torch.nn.functional as F 

class EEGModel(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden_size=100):
        super(EEGModel, self).__init__()

        # --- Convolutional Block ---
        # Each layer processes each time window (frame) individually, with each frame is a 32x4 matrix (32 channels, 4 frequency band magnitudes).
        # We first add a channel dimension so each frame becomes (1, 32, 4).
        # Normalize Batch &Add Pooling layer to reduce dimensionality
        # Need to test the kernel size (eeg channels, freq bands), tentively set to capture the relationship betwen 3 EEG channels (independantly from EEG channels)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding=(1, 0)) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Calculate the flattened feature size:
        # After two poolings on the channel dimension: 32 -> 16 -> 8. Flattened size per frame = 64 channels * 8 * 4 = 2048.
        self.flattened_size = 64 * 8 * 4
        
        # --- LSTM  ---
        # This layer processes the sequence of frames (each represented by a 1536-dimensional feature vector) to capture temporal dynamics.
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=lstm_hidden_size, batch_first=True)
        
        # --- Fully Connected Layers ---
        # These layers take the final output of the LSTM and linearly propagates them for a softmax classification task.
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, T, 32, 4)
        # where T is the number of time windows (each representing 1/2 second of EEG data after FFT)
        batch_size, T, H, W = x.shape
        
        # --- Step 1: Prepare Each Frame for Convolution ---
        # Reshape to combine batch and time dimensions so each frame is processed identically.
        # New shape: (batch_size * T, 1, 32, 4)
        x = x.view(batch_size * T, 1, H, W)
        
        # --- Step 2: Convolutional Processing ---
        # First convolutional layer with BatchNorm and ReLU activation.
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Second convolutional layer.
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # --- Step 3: Flatten Each Frame's Features ---
        # Reshape back so that we have one feature vector per frame.
        # New shape: (batch_size, T, flattened_size)
        x = x.view(batch_size, T, -1)
        
        # --- Step 4: LSTM for Temporal Dynamics ---
        # The LSTM processes the sequence of feature vectors.
        lstm_out, (hn, cn) = self.lstm(x)
        # We take the output of the final time step as the representation for the entire sequence.
        final_feature = lstm_out[:, -1, :]
        
        # --- Step 5: Fully Connected Layers for Classification ---
        x = F.relu(self.fc1(final_feature))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # Final output layer provides logits for each class.
        x = self.fc3(x)
        
        return x

# Example instantiation and summary:
if __name__ == "__main__":
    # Simulate a batch of data: batch_size=8, T=10 time windows, 32 channels, 4 frequency magnitudes
    sample_input = torch.randn(8, 10, 32, 4)
    model = EEGModel(num_classes=5, lstm_hidden_size=100)
    output = model(sample_input)
    print("Output shape:", output.shape)  
