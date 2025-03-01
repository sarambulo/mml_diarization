from datasets.MSDWild import MSDWildFrames
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class AudioOnlyTDNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512, num_speakers=10):  # Adjust num_speakers as needed
        super().__init__()
        
        self.tdnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, dilation=1, padding='same')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=3, padding='same')
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_speakers)  # Output for each speaker
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.dropout(x)
        
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.dropout(x)
        
        x = self.bn3(F.relu(self.tdnn3(x)))
        x = self.dropout(x)
        
        x = self.bn4(F.relu(self.tdnn4(x)))
        x = self.dropout(x)
        
        # Statistical pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        x = torch.cat([mean, std], dim=1)
        
        x = self.bn_fc1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.bn_fc2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for multi-label output
        return x

class AudioOnlyDataset(Dataset):
    def __init__(self, msd_dataset, sample_rate=2):
        self.msd_dataset = msd_dataset
        self.sample_rate = sample_rate
        self.used_indices = list(range(0, len(msd_dataset), sample_rate))
        
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop_length=160,
            win_length=400,
            pad_mode="reflect"
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.used_indices)

    def __getitem__(self, index):
        try:
            actual_index = self.used_indices[index]
            anchor, _, _, label = self.msd_dataset[actual_index]
            audio_segment = anchor[1]
            
            if audio_segment is None:
                return None

            mel_spectrogram = self.mel_transform(audio_segment)
            
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)
            elif mel_spectrogram.shape[0] > 1:
                mel_spectrogram = mel_spectrogram.mean(dim=0, keepdim=True)

            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).float()
            
            return mel_spectrogram, label
            
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    inputs, labels = zip(*batch)
    
    # Find max length in the batch
    max_length = max([x.shape[2] for x in inputs])
    
    # Pad each input to max_length
    padded_inputs = []
    for inp in inputs:
        pad_length = max_length - inp.shape[2]
        if pad_length > 0:
            padded_inp = F.pad(inp, (0, pad_length))
            padded_inputs.append(padded_inp)
        else:
            padded_inputs.append(inp)
    
    inputs = torch.stack(padded_inputs)
    inputs = inputs.squeeze(1)
    
    # Convert labels to tensors if they're not already
    processed_labels = []
    for label in labels:
        if isinstance(label, (int, np.integer)):
            # Convert integer to one-hot encoding
            one_hot = torch.zeros(10)  # Using fixed size 10 for now
            one_hot[int(label)] = 1
            processed_labels.append(one_hot)
        elif isinstance(label, np.ndarray):
            processed_labels.append(torch.from_numpy(label).float())
        else:
            processed_labels.append(label)
    
    labels = torch.stack(processed_labels)
    
    return inputs, labels

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    best_loss = float('inf')
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            if epoch < num_epochs // 2:
                noise = torch.randn_like(inputs) * 0.1
                inputs = inputs + noise
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}')
        
        scheduler.step(epoch_loss)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f'New best model saved with loss: {best_loss:.4f}')

def main():
    data_path = "data"
    train_dataset = MSDWildFrames(data_path, "few_train")
    
    # Let's print some debug info about the labels
    print("Debugging first few samples:")
    for i in range(5):
        sample_data = train_dataset[i]
        _, _, _, label = sample_data
        print(f"Sample {i} label type: {type(label)}")
        print(f"Sample {i} label: {label}")
        if isinstance(label, np.ndarray):
            print(f"Label shape: {label.shape}")
    
   
    max_speakers = 10  
    print(f"Using fixed number of speakers: {max_speakers}")
    
    train_audio_dataset = AudioOnlyDataset(train_dataset)
    
    batch_size = 32
    train_loader = DataLoader(
        train_audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioOnlyTDNN(input_dim=40, num_speakers=max_speakers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    train_model(model, train_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
