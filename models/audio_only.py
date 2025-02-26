import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class TDNN(nn.Module):
    def __init__(self, input_dim=40, output_dim=1):
        """
        TDNN model for binary classification (Active Speaker Detection).
        :param input_dim: Number of Mel filterbanks (default: 40).
        :param output_dim: Binary classification (1 output neuron).
        """
        super(TDNN, self).__init__()
        
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.tdnn1(x))
        x = self.relu(self.tdnn2(x))
        x = self.relu(self.tdnn3(x))
        x = self.relu(self.tdnn4(x))
        x = self.relu(self.tdnn5(x))

        x = x.mean(dim=2)  # Global temporal pooling
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x


