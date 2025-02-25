import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from msdwild import MSDWildBase  # Assuming your dataset is inside a module `msdwild`

# Define a simple Audio-Only Active Speaker Dataset
class AudioOnlyDataset(Dataset):
    def __init__(self, data_path, partition, segment_duration=1.0):
        self.dataset = MSDWildBase(data_path, partition)
        self.segment_duration = segment_duration
        self.mel_transform = AT.MelSpectrogram(sample_rate=16000, n_mels=64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        audio_stream, labels = self.get_audio_segment(index)

        if audio_stream is None:
            return None

        mel_spectrogram = self.mel_transform(audio_stream)  # Convert to Mel-Spectrogram
        label = 1 if "active_speaker" in labels else 0  # Binary classification

        return mel_spectrogram, torch.tensor(label, dtype=torch.float)

    def get_audio_segment(self, index):
        _, audio_stream, labels, _ = self.dataset[index]
        if audio_stream is None:
            return None, None

        waveform, sample_rate = torchaudio.load(audio_stream)
        segment_length = int(self.segment_duration * sample_rate)

        if waveform.shape[1] < segment_length:
            return None, None  # Skip if too short

        start_sample = random.randint(0, waveform.shape[1] - segment_length)
        segment = waveform[:, start_sample : start_sample + segment_length]

        return segment, labels

# Simple Binary Classifier for Active Speaker Detection
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjust based on spectrogram size
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)  # Add channel dimension

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Load dataset
data_path = "path/to/data"
train_dataset = AudioOnlyDataset(data_path, "few_train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x if None not in x else None)

# Initialize model, loss, and optimizer
model = AudioClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer)
