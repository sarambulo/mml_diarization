from datasets.MSDWild import MSDWildFrames
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from utils.metrics import calculate_metrics_for_dataset
from data import rttm_to_annotations

class AudioOnlyTDNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512):
        super().__init__()
        
        # Add padding to preserve temporal dimension
        self.tdnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding='same')
        self.tdnn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding='same')
        self.tdnn3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3, padding='same')
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Add shape printing for debugging
        print(f"Input shape: {x.shape}")
        
        x = F.relu(self.tdnn1(x))
        print(f"After TDNN1: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.tdnn2(x))
        print(f"After TDNN2: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.tdnn3(x))
        print(f"After TDNN3: {x.shape}")
        x = self.dropout(x)
        
        # Global mean pooling
        x = torch.mean(x, dim=2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc2(x))
        return x

class AudioOnlyDataset(Dataset):
    def __init__(self, msd_dataset):
        self.msd_dataset = msd_dataset
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop_length=80,  # Reduced hop length to get more time steps
            win_length=400,
            pad_mode="reflect"
        )

    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, index):
        try:
            anchor, _, _, label = self.msd_dataset[index]
            _, audio_segment, _ = anchor

            if audio_segment is None:
                return None

            # Handle stereo to mono conversion
            if audio_segment.shape[-1] == 2:
                audio_segment = audio_segment.mean(dim=-1, keepdim=True)
            
            # Ensure 2D shape [1, samples]
            if audio_segment.dim() == 1:
                audio_segment = audio_segment.unsqueeze(0)
            audio_segment = audio_segment.squeeze(-1)

            # Convert to mel spectrogram
            mel_spectrogram = self.mel_transform(audio_segment)
            
            # Ensure consistent channel dimension
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)
            elif mel_spectrogram.shape[0] > 1:
                mel_spectrogram = mel_spectrogram.mean(dim=0, keepdim=True)

            # Convert label to binary (active speaker or not)
            label = torch.tensor(1 if label.sum() > 0 else 0, dtype=torch.float32)

            return mel_spectrogram, label
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None

def collate_fn(batch):
    # Remove None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    inputs, labels = zip(*batch)
    
    # Stack inputs and ensure correct shape
    inputs = torch.stack(inputs)  # [batch, channels, freq, time]
    inputs = inputs.squeeze(1)    # [batch, freq, time]
    
    # Stack labels
    labels = torch.stack(labels).unsqueeze(1)  # [batch, 1]

    return inputs, labels

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

def main():
    # Initialize dataset with None transforms to skip face transformations
    data_path = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample"
    msd_dataset = MSDWildFrames(data_path, "few_train", transforms=None)
    audio_dataset = AudioOnlyDataset(msd_dataset)
    
    # Create data loader with reduced number of workers to avoid potential issues
    batch_size = 32
    train_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Changed from 2 to 0 to avoid multiprocessing issues
    )
    
    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioOnlyTDNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save the trained model
    torch.save(model.state_dict(), 'audio_tdnn_model.pth')

if __name__ == "__main__":
    main()
