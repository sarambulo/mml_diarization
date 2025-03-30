import os
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.metrics import rttm_to_annotations, calculate_metrics_for_dataset
from pathlib import Path

from datasets.MSDWild import MSDWildChunks
import torch.nn.functional as F

from torchinfo import summary  

class DiarizationLoss(nn.Module):
    def __init__(self, triplet_lambda=0.7, bce_lambda=0.3):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
        self.bce_loss = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss to handle logits directly
        self.triplet_lambda = triplet_lambda
        self.bce_lambda = bce_lambda
        
    def forward(self, anchors, positives, negatives, logits, labels):
        triplet_loss = self.triplet_loss(anchors, positives, negatives)
        bce_loss = self.bce_loss(logits, labels)
        return self.triplet_lambda * triplet_loss + self.bce_lambda * bce_loss


class AudioActiveSpeakerModel(nn.Module):
    def __init__(self, base_model, embedding_dim=256, num_classes=1):  # Using num_classes=1 for binary classification
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Linear(embedding_dim, num_classes)  # Single output for binary classification

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return embedding, logits.squeeze(-1)  # Squeeze to get scalar output for binary classification

    @torch.no_grad()
    def predict_frame(self, x):
        self.eval()
        embedding, logits = self.forward(x)
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        return embedding, probs


# Modified AudioTripletDataset to include labels
class AudioTripletDatasetWithLabels(Dataset):
    def __init__(self, msd_dataset, augment=True):
        self.msd_dataset = msd_dataset
        self.augment = augment
        
        # Ultra small FFT window for very short segments
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=40,
            n_fft=16,         # Tiny FFT window
            hop_length=8,     # Tiny hop length
            win_length=16,    # Tiny window length
            pad_mode="constant",
            center=False      # Disable center padding
        )

    def __len__(self):
        return len(self.msd_dataset)

    def process_audio(self, audio_segment):
        # Same as your original process_audio method
        if audio_segment is None:
            return None

        # Get original shape for debugging
        original_shape = audio_segment.shape
        
        # For 30-channel audio with 22 samples, average across channels first
        if audio_segment.shape[0] == 30 and audio_segment.shape[1] == 22:
            # Average all 30 channels to get a single channel
            audio_segment = audio_segment.mean(dim=0, keepdim=True)
            
        # Handle other shapes - normalize dimensions
        elif audio_segment.shape[-1] == 2:  # Stereo
            audio_segment = audio_segment.mean(dim=-1, keepdim=True)
        
        # Ensure we have a 2D tensor [channels, samples]
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
            
        # Make sure it's [channels, samples] not [channels, samples, 1]
        if audio_segment.dim() > 2:
            audio_segment = audio_segment.squeeze(-1)
            
        # Skip very short audio
        if audio_segment.shape[-1] < 16:
            print(f"Audio too short: {audio_segment.shape}")
            return None
                
        # For short audio, repeat it to get more samples
        if audio_segment.shape[-1] < 64:
            repeats_needed = max(1, 64 // audio_segment.shape[-1])
            # Repeat the audio multiple times to get more frames
            audio_segment = audio_segment.repeat(1, repeats_needed)
        
        # Apply mel spectrogram transform
        try:
            mel = self.mel_transform(audio_segment)
            
            # For spectrograms with few time frames, repeat to get to target size
            if mel.shape[-1] < 3:
                repeats_needed = max(1, 3 // mel.shape[-1])
                mel = mel.repeat(1, 1, repeats_needed)
                print(f"Repeated mel spectrogram in time: {mel.shape}")
        except Exception as e:
            print(f"Error in mel transform: {e}")
            return None

        # Clip or pad to (40, 64)
        target_shape = (40, 64)
        
        # Create a zero tensor with the target shape
        padded = torch.zeros(1, *target_shape)
        
        # Calculate how much of the original mel spectrogram we can use
        use_F = min(mel.shape[-2], target_shape[0])
        use_T = min(mel.shape[-1], target_shape[1])
        
        # Copy only what we can from the original mel spectrogram
        padded[:, :use_F, :use_T] = mel[:, :use_F, :use_T]
        
        # If the time dimension is still very small, repeat it to fill the target shape
        if use_T < 5:
            # Tile the available columns across the target
            for t in range(use_T, target_shape[1]):
                padded[:, :use_F, t] = padded[:, :use_F, t % use_T]
        
        mel = padded

        if self.augment:
            # Increased augmentation
            time_mask_param = max(1, min(20, mel.shape[-1] - 1))  # More aggressive time masking
            freq_mask_param = max(1, min(10, mel.shape[-2] - 1))  # More aggressive freq masking

            time_mask = AT.TimeMasking(time_mask_param=time_mask_param)
            freq_mask = AT.FrequencyMasking(freq_mask_param=freq_mask_param)

            # Apply multiple masks with higher probability
            if torch.rand(1).item() < 0.7:  # Increased from 0.5
                mel = freq_mask(mel)
                if torch.rand(1).item() < 0.5:
                    mel = freq_mask(mel)  # Second freq mask
                    
            if torch.rand(1).item() < 0.7:  # Increased from 0.5
                mel = time_mask(mel)
                if torch.rand(1).item() < 0.5:
                    mel = time_mask(mel)  # Second time mask

        return mel

    def __getitem__(self, index):
        try:
            # Get both audio triplet and is_speaking label
            _, audio_triplet, is_speaking = self.msd_dataset[index]
            anchor_audio, positive_audio, negative_audio = audio_triplet

            anchor_mel = self.process_audio(anchor_audio)
            positive_mel = self.process_audio(positive_audio)
            negative_mel = self.process_audio(negative_audio)

            if any(x is None for x in [anchor_mel, positive_mel, negative_mel]):
                return None

            # Convert is_speaking to float tensor for BCE loss
            is_speaking_tensor = torch.tensor(float(is_speaking), dtype=torch.float32)
            
            return anchor_mel, positive_mel, negative_mel, is_speaking_tensor
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None


# Modified collate function to handle labels
def triplet_collate_fn_with_labels(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    anchors, positives, negatives, labels = zip(*batch)
    return (
        torch.stack(anchors).squeeze(1),
        torch.stack(positives).squeeze(1),
        torch.stack(negatives).squeeze(1),
        torch.stack(labels)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Added dropout
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Added dropout
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)  # Apply dropout
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)  # Apply dropout
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Reduced parameter version with ~8M parameters and improved regularization
class CompactAudioEmbedding(nn.Module):
    def __init__(self, input_dim=40, embedding_dim=256, dropout_rate=0.3):
        super().__init__()
        
        # Initial convolutional layers - reduced filters
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate/2),  # Add dropout
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual blocks with moderate channel counts and dropout
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, dropout_rate=dropout_rate),
            ResidualBlock(64, 64, dropout_rate=dropout_rate),
            SqueezeExcitation(64)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, dropout_rate=dropout_rate),
            SqueezeExcitation(128)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, dropout_rate=dropout_rate),
            SqueezeExcitation(256)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            SqueezeExcitation(512)
        )
        
        # Adaptive pooling to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Flatten and project
        self.flatten_dim = 512 * 2 * 2
        
        # Final embedding layers with increased dropout
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Handle 3D input (batch, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and flattening
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final embedding
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalization for embedding
        x = F.normalize(x, p=2, dim=1)
        
        return x

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Training function with DiarizationLoss
def train_with_diarization_loss(model, train_loader, optimizer, device, triplet_lambda=0.7, bce_lambda=0.3, num_epochs=20):
    # Initialize the combined loss function
    criterion = DiarizationLoss(triplet_lambda=triplet_lambda, bce_lambda=bce_lambda)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # Reduce LR by half when plateauing
        patience=2,   # Wait 2 epochs before reducing
        verbose=True
    )

    # Count total batches for reporting
    total_batches = len(train_loader)
    print(f"Dataset contains {len(train_loader.dataset)} triplets")
    print(f"Using batch size of {train_loader.batch_size}, resulting in {total_batches} batches")
    print(f"Loss weighting: triplet={triplet_lambda}, bce={bce_lambda}")
    
    # Initialize counters for tracking progress
    total_processed = 0
    skipped_batches = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                skipped_batches += 1
                continue
                
            batch_count += 1
            anchor, positive, negative, labels = batch
            batch_size = anchor.size(0)
            total_processed += batch_size
            
            # Print batch information more frequently
            if batch_idx % 2 == 0:  # Print every other batch
                print(f"Batch {batch_idx}/{total_batches}: anchor shape={anchor.shape}")
            
            anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

            # Forward pass through model
            anchor_embeddings, anchor_logits = model(anchor)
            positive_embeddings = model.encoder(positive)
            negative_embeddings = model.encoder(negative)

            # Compute combined loss
            loss = criterion(
                anchor_embeddings, 
                positive_embeddings, 
                negative_embeddings, 
                anchor_logits,
                labels
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            
            # Print occasional updates
            if batch_idx % 2 == 0:
                print(f"  Batch {batch_idx}/{total_batches}, Loss: {loss.item():.6f}")

        # Calculate actual average loss based on batches processed
        avg_loss = epoch_loss / max(batch_count, 1)  # Avoid division by zero
        
        # Update learning rate scheduler based on epoch loss
        scheduler.step(avg_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        print(f"Processed {batch_count} batches, Skipped {skipped_batches} empty batches")
        print(f"Total triplets processed so far: {total_processed}")
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss: {best_loss:.6f}")
            # Save best model
            torch.save(model.state_dict(), 'best_diarization_model.pth')
            print("Saved best model")
        
        # Reset counter for next epoch
        skipped_batches = 0
        
        # Early stopping criteria
        if current_lr < 1e-6:  # If learning rate gets too small
            print("Learning rate too small. Early stopping.")
            break
    
    if total_processed == 0:
        print("WARNING: No data was processed during training! Check your dataset filtering.")
    else:
        print(f"Training complete. Processed {total_processed} triplets in total.")
        print(f"Best loss achieved: {best_loss:.6f}")


def main():
    data_path = "../preprocessed"  # Root directory where video folders like 00001, 00002 exist
    partition_file = "../data_sample/few_train.rttm"  # Should list video IDs like: 00001, 00002...

    msd_dataset = MSDWildChunks(data_path=data_path, partition_path=partition_file, subset=1.0)
        
    # Create the audio dataset with labels
    audio_dataset = AudioTripletDatasetWithLabels(msd_dataset)
    sample = audio_dataset[0]
    if sample:
        anchor_mel, positive_mel, negative_mel, is_speaking = sample
        print("Anchor mel shape:", anchor_mel.shape)  # Should be [1, 40, 64]
        print("Is speaking:", is_speaking.item())

    # Create data loader with the modified collate function
    train_loader = DataLoader(
        audio_dataset, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=triplet_collate_fn_with_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create base embedding model
    base_model = CompactAudioEmbedding(input_dim=40, embedding_dim=256, dropout_rate=0.3).to(device)
    
    # Create the complete model with classifier
    model = AudioActiveSpeakerModel(
        base_model=base_model,
        embedding_dim=256,
        num_classes=1  # Binary classification
    ).to(device)
    
    # Count and print the number of parameters
    param_count = count_parameters(model)
    print(f"\nModel has {param_count:,} trainable parameters")
    
    # Optimizer with increased weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print("\nModel Summary:")
    summary(model, input_size=(1, 1, 40, 64))  
    
    # Train with the combined loss
    triplet_lambda = 0.7  # Weight for triplet loss
    bce_lambda = 0.3     # Weight for BCE loss
    train_with_diarization_loss(
        model, 
        train_loader, 
        optimizer, 
        device,
        triplet_lambda=triplet_lambda,
        bce_lambda=bce_lambda,
        num_epochs=20
    )


if __name__ == "__main__":
    main()