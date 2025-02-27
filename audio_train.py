from datasets.MSDWild import MSDWildFrames
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

def create_rttm_file(predictions, file_ids, timestamps, output_path):
    """
    Create RTTM file from model predictions
    predictions: list of binary predictions (0/1)
    file_ids: list of file IDs corresponding to each prediction
    timestamps: list of (start_time, end_time) for each prediction
    """
    with open(output_path, 'w') as f:
        for pred, file_id, (start, end) in zip(predictions, file_ids, timestamps):
            if pred == 1:  # Active speaker
                f.write(f"SPEAKER {file_id} 1 {start:.3f} {end-start:.3f} <NA> <NA> SPEAKER <NA> <NA>\n")

def evaluate_model(model, test_loader, output_rttm_path="predictions.rttm"):
    model.eval()
    all_predictions = []
    all_file_ids = []
    all_timestamps = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            
            inputs, file_ids, timestamps = batch
            inputs = inputs.to(next(model.parameters()).device)
            
            outputs = model(inputs)
            predictions = (outputs >= 0.5).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_file_ids.extend(file_ids)
            all_timestamps.extend(timestamps)
    
    # Create RTTM file
    create_rttm_file(all_predictions, all_file_ids, all_timestamps, output_rttm_path)
    
    # Calculate metrics
    preds = rttm_to_annotations(output_rttm_path)
    targets = rttm_to_annotations("data/test/many.val.rttm")
    
    metrics = calculate_metrics_for_dataset(preds, targets)
    
    print("\nEvaluation Metrics:")
    print(f"DER: {metrics['DER']:.4f}")
    print(f"JER: {metrics['JER']:.4f}")
    print(f"Missed Speech Rate: {metrics['MSR']:.4f}")
    print(f"False Alarm Rate: {metrics['FAR']:.4f}")
    print(f"Speaker Error Rate: {metrics['SER']:.4f}")
    
    return metrics

class AudioOnlyTDNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512):
        super().__init__()
        
        # Batch normalization after each layer
        self.tdnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, dilation=1, padding='same')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=2, padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=3, padding='same')
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.tdnn4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for statistical pooling
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
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
        
        x = torch.sigmoid(self.fc3(x))
        return x

class AudioOnlyDataset(Dataset):
    def __init__(self, msd_dataset, augment=True):
        self.msd_dataset = msd_dataset
        self.augment = augment
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=400,
            hop_length=80,
            win_length=400,
            pad_mode="reflect"
        )
        
        # Audio augmentation transforms
        self.time_stretch = AT.TimeStretch()
        self.freq_mask = AT.FrequencyMasking(freq_mask_param=15)
        self.time_mask = AT.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, index):
        try:
            anchor, _, _, label = self.msd_dataset[index]
            file_id, audio_segment, timestamp = anchor

            if audio_segment is None:
                return None

            if audio_segment.shape[-1] == 2:
                audio_segment = audio_segment.mean(dim=-1, keepdim=True)
            
            if audio_segment.dim() == 1:
                audio_segment = audio_segment.unsqueeze(0)
            audio_segment = audio_segment.squeeze(-1)

            mel_spectrogram = self.mel_transform(audio_segment)
            
            if self.augment:
                if torch.rand(1) < 0.5:
                    mel_spectrogram = self.freq_mask(mel_spectrogram)
                if torch.rand(1) < 0.5:
                    mel_spectrogram = self.time_mask(mel_spectrogram)
            
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)
            elif mel_spectrogram.shape[0] > 1:
                mel_spectrogram = mel_spectrogram.mean(dim=0, keepdim=True)

            label = torch.tensor(1 if label.sum() > 0 else 0, dtype=torch.float32)

            return mel_spectrogram, label
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    inputs = inputs.squeeze(1)
    labels = torch.stack(labels).unsqueeze(1)

    return inputs, labels

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    best_loss = float('inf')
    
    # Create directory for model checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Add noise for regularization
            if epoch < num_epochs // 2:
                noise = torch.randn_like(inputs) * 0.1
                inputs = inputs + noise
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # L2 regularization
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        scheduler.step(epoch_loss)
        
        # Save checkpoint for each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
        
        # Save best model separately
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f'New best model saved with loss: {best_loss:.4f}')

def main():
    # Initialize dataset with None transforms to skip face transformations
    data_path = "data"
    msd_dataset = MSDWildFrames(data_path, "few_train", transforms=None)
    audio_dataset = AudioOnlyDataset(msd_dataset)
    
    # Create data loader with reduced number of workers to avoid potential issues
    batch_size = 256
    train_loader = DataLoader(
        train_audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioOnlyTDNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
