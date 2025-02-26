from datasets.MSDWild import MSDWildBase, MSDWildFrames
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from models.audio_only import TDNN

class AudioOnlyDataset(Dataset):
    def __init__(self, msd_dataset):
        """
        Creates an audio-only dataset using the MSDWildFrames dataloader.
        :param msd_dataset: Instance of MSDWildFrames.
        """
        self.msd_dataset = msd_dataset
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop_length=160,
            win_length=400,
            pad_mode="reflect"
        )

    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, index):
        anchor, _, _ = self.msd_dataset[index]  # Ignore positive/negative pairs
        _, audio_segment, label, _ = anchor  # Extract audio & label

        if audio_segment is None:
            return None  # Skip missing audio

        # ✅ Convert stereo to mono if needed
        if audio_segment.shape[-1] == 2:
            audio_segment = audio_segment.mean(dim=-1, keepdim=True)

        audio_segment = audio_segment.squeeze(-1)  # Shape: [1, 1024]

        # ✅ Compute Mel-Spectrogram
        mel_spectrogram = self.mel_transform(audio_segment).squeeze(0)  # Shape: [40, T]

        # ✅ Convert label to binary
        label = torch.tensor(1 if label.sum() > 0 else 0, dtype=torch.float32)

        return mel_spectrogram, label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    inputs, labels = zip(*batch)

    # ✅ Convert list of tensors into a padded tensor along time axis
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Shape: [batch, max_T, 40]

    # ✅ Ensure the input has 3D shape: [batch, channels, time]
    if inputs.dim() == 3:  # [batch, time, 40]
        inputs = inputs.permute(0, 2, 1)  # Convert to [batch, 40, time]

    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: [batch, 1]

    return inputs, labels

# ✅ Initialize dataset and DataLoader
msd_dataset = MSDWildFrames("/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample", "sample", transforms=None)
audio_dataset = AudioOnlyDataset(msd_dataset)

# ✅ Check a sample
sample_idx = 0
mel_spec, label = audio_dataset[sample_idx]

print(f"Sample {sample_idx} - Mel-Spectrogram Shape: {mel_spec.shape}")
print(f"Sample {sample_idx} - Label: {label}")

train_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# ✅ Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TDNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            if batch is None:
                print("⚠️ Found a None batch! Skipping.")
                continue

            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# ✅ Run training
train(model, train_loader, criterion, optimizer, num_epochs=10)
