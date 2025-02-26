from datasets.MSDWild import MSDWildBase, MSDWildFrames
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.audio_only import TDNN

class AudioOnlyDataset(Dataset):
    def __init__(self, msd_dataset):
        """
        Creates an audio-only dataset using the MSDWildFrames dataloader.
        :param msd_dataset: Instance of MSDWildFrames.
        """
        self.msd_dataset = msd_dataset
        self.mel_transform = AT.MelSpectrogram(sample_rate=16000, n_mels=40)

    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, index):
        anchor, _, _ = self.msd_dataset[index]  # Ignore positive/negative pairs
        _, audio_segment, label, _ = anchor  # Extract audio & label

        if audio_segment is None:
            return None  # Skip missing audio

        # Convert to Mel-Spectrogram
        mel_spectrogram = self.mel_transform(audio_segment)

        # Convert label from binary vector to scalar (if needed)
        label = 1 if label.sum() > 0 else 0  # Active speaker: 1, Silent: 0

        return mel_spectrogram, torch.tensor(label, dtype=torch.float32)
    

msd_dataset = MSDWildFrames("/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample", "sample", transforms=None)
audio_dataset = AudioOnlyDataset(msd_dataset)
sample_idx = 0
mel_spec, label = audio_dataset[sample_idx]

print(f"Sample {sample_idx} - Mel-Spectrogram Shape: {mel_spec.shape}")
print(f"Sample {sample_idx} - Label: {label}")
train_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x if None not in x else None)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TDNN().to(device)
criterion = nn.BCELoss()  # Binary Classification Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            if batch is None:
                print("⚠️ Found a None batch! Check dataset processing.")
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs).to(device).squeeze(1)  # Shape: [batch, 40, T]
            labels = torch.tensor(labels, dtype=torch.float32).to(device).unsqueeze(1)  # Shape: [batch, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Run training
train(model, train_loader, criterion, optimizer, num_epochs=10)


