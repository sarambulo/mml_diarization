import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
import torchaudio.transforms as AT
import sys
import os

# Get the parent directory of Add_embeds
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from datasets.MSDWild import MSDWildFrames
from audio_train import AudioOnlyTDNN  # Import audio model from wherever it's defined

### Load the trained AudioOnlyTDNN model
def load_audio_model(checkpoint_path=f"{parent_dir}/checkpoints/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioOnlyTDNN().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode

    print(f"Loaded AudioOnlyTDNN model from {checkpoint_path}")
    return model

### Dataset to Extract Audio Embeddings from MSDWildFrames
class AudioEmbeddingDataset(Dataset):
    def __init__(self, msd_dataset, audio_model):
        self.msd_dataset = msd_dataset
        self.audio_model = audio_model
        self.device = next(audio_model.parameters()).device
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000, n_mels=40, n_fft=400, hop_length=80, win_length=400, pad_mode="reflect"
        )

    def __len__(self):
        return len(self.msd_dataset)

    def __getitem__(self, index):
        try:
            anchor, _, _, label = self.msd_dataset[index]
            file_id, audio_segment, timestamp = anchor

            if audio_segment is None:
                return None

            if audio_segment.shape[-1] == 2:
                audio_segment = audio_segment.mean(dim=-1, keepdim=True)  # Convert stereo to mono

            if audio_segment.dim() == 1:
                audio_segment = audio_segment.unsqueeze(0)

            audio_segment = audio_segment.squeeze(-1)
            mel_spectrogram = self.mel_transform(audio_segment).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.audio_model(mel_spectrogram).cpu()

            label = torch.tensor(1 if label.sum() > 0 else 0, dtype=torch.float32)

            return embedding.squeeze(0), label  # Return audio embedding instead of raw data

        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None

### Collate function to handle None values in DataLoader
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels).unsqueeze(1)

    return inputs, labels

### Multimodal Dataset (Audio + Visual)
class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings, visual_embeddings, labels):
        self.audio_embeddings = audio_embeddings
        self.visual_embeddings = visual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.visual_embeddings[idx], self.labels[idx]

### Multimodal Model
class AddSimpleMultimodalModel(nn.Module):
    def __init__(self, embedding_dim=512, fusion_dim=512, num_speakers=10):
        super(AddSimpleMultimodalModel, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.BatchNorm1d(2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * embedding_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(fusion_dim, num_speakers)

    def forward(self, audio_embedding, visual_embedding):
        combined_embedding = torch.cat((audio_embedding, visual_embedding), dim=1)
        fused_embedding = self.fusion_layer(combined_embedding)
        return fused_embedding

    def classify(self, fused_embedding):
        return self.classifier(fused_embedding)

    def predict_speakers(self, audio_embedding, visual_embedding):
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(audio_embedding, visual_embedding).cpu().numpy()
        cluster = AgglomerativeClustering(n_clusters=self.num_speakers)
        labels = cluster.fit_predict(embeddings)
        return labels

### Training Function
def train(model, dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for audio_emb, visual_emb, labels in dataloader:
            audio_emb = audio_emb.to(device)
            visual_emb = visual_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            fused_emb = model(audio_emb, visual_emb)
            logits = model.classify(fused_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

### Function to Train Multimodal Model
def when_classify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Audio Model
    audio_model = load_audio_model()

    # Load Dataset
    data_path = f"{parent_dir}/data_sample"
    msd_dataset = MSDWildFrames(data_path, "few_train", transforms=None, subset=1)
    audio_dataset = AudioEmbeddingDataset(msd_dataset, audio_model)

    # DataLoader
    train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Extract embeddings dynamically
    audio_embeddings, labels = [], []
    for batch in train_loader:
        if batch is None:
            continue
        emb, lbl = batch
        audio_embeddings.append(emb)
        labels.append(lbl)

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Generate Random Visual Embeddings (For Testing)
    visual_embeddings = torch.randn(audio_embeddings.shape)

    # Multimodal Dataset
    dataset = MultimodalDataset(audio_embeddings, visual_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize Model
    model = AddSimpleMultimodalModel(embedding_dim=512, fusion_dim=512, num_speakers=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train(model, dataloader, criterion, optimizer, epochs=20, device=device)

### Function to Perform Clustering
def when_cluster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Audio Model
    audio_model = load_audio_model()

    # Load Dataset
    data_path = "data"
    msd_dataset = MSDWildFrames(data_path, "few_train", transforms=None, subset=0.01)
    audio_dataset = AudioEmbeddingDataset(msd_dataset, audio_model)

    # DataLoader
    train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Extract Embeddings
    audio_embeddings = []
    for batch in train_loader:
        if batch is None:
            continue
        emb, _ = batch
        audio_embeddings.append(emb)

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    visual_embeddings = torch.randn(audio_embeddings.shape)

    # Perform Clustering
    model = AddSimpleMultimodalModel(embedding_dim=512, fusion_dim=512, num_speakers=10)
    speaker_labels = model.predict_speakers(audio_embeddings, visual_embeddings)
    print("Predicted Speaker Labels:", speaker_labels)

if __name__ == "__main__":
    print("Classification:")
    when_classify()

    print("Clustering:")
    when_cluster()
