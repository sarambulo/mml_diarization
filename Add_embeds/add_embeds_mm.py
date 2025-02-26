import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering

# Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings, visual_embeddings, labels):
        self.audio_embeddings = audio_embeddings
        self.visual_embeddings = visual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.visual_embeddings[idx], self.labels[idx]
    

# Simple Multimodal Model
class AddSimpleMultimodalModel(nn.Module):
    def __init__(self, embedding_dim=512, fusion_dim=512, num_speakers=10):
        super(AddSimpleMultimodalModel, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),    #2*512 (1024) -> 2*512 (1024)
            nn.BatchNorm1d(2 * embedding_dim),  #2*512 (1024)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * embedding_dim, fusion_dim),   #2*512 (1024) -> 512
            nn.BatchNorm1d(fusion_dim), #512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),  #512-> 512
            nn.ReLU()
        )
        self.num_speakers = num_speakers
        self.classifier = nn.Linear(fusion_dim, self.num_speakers)

    def forward(self, audio_embedding, visual_embedding):    #get embed
        combined_embedding = torch.cat((audio_embedding, visual_embedding), dim=1)  #concatenate both embeds ((batch_size, 512) + (batch_size, 512) -> (batch_size, 1024))
        fused_embedding = self.fusion_layer(combined_embedding)
        return fused_embedding

    def classify(self, fused_embedding):     #when classify
        logits = self.classifier(fused_embedding)
        return logits

    def predict_speakers(self, audio_embedding, visual_embedding):   #when cluster
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(audio_embedding, visual_embedding).cpu().numpy()
        cluster = AgglomerativeClustering(n_clusters=self.num_speakers)
        labels = cluster.fit_predict(embeddings)
        return labels

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

def when_classify(audio_embeddings = torch.randn(100, 512), visual_embeddings = torch.randn(100, 512)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = torch.randint(0, 10, (100,))

    dataset = MultimodalDataset(audio_embeddings, visual_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AddSimpleMultimodalModel(embedding_dim=512, fusion_dim=512, num_speakers=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, epochs=20, device=device)

def when_cluster(audio_embeddings = torch.randn(100, 512), visual_embeddings = torch.randn(100, 512)):
    # Predict (speaker diarization)
    model = AddSimpleMultimodalModel(embedding_dim=512, fusion_dim=512, num_speakers=10)
    speaker_labels = model.predict_speakers(audio_embeddings, visual_embeddings)
    print("Predicted Speaker Labels:", speaker_labels)

print("Classification: ")
when_classify()

print("Clustering")
when_cluster()