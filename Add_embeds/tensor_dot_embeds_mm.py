import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
from torchsummary import summary


class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings, visual_embeddings, labels):
        self.audio_embeddings = audio_embeddings
        self.visual_embeddings = visual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.visual_embeddings[idx], self.labels[idx]


class TensorDotMultimodalModel(nn.Module):
    def __init__(self, embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10):
        super(TensorDotMultimodalModel, self).__init__()

        self.projection = nn.Linear(embedding_dim, reduced_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(reduced_dim * reduced_dim, 2 * embedding_dim), #16384 -> 2*512 (1024)
            nn.BatchNorm1d(2 * embedding_dim), #1024
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * embedding_dim, fusion_dim), #1024 -> 512
            nn.BatchNorm1d(fusion_dim), #512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim), #512 -> 512
            nn.ReLU()
        )
        self.num_speakers = num_speakers
        self.classifier = nn.Linear(fusion_dim, self.num_speakers)

    def forward(self, audio_embedding, visual_embedding):   #get embed
        audio_embedding = self.projection(audio_embedding)  # (batch_size, 128)
        visual_embedding = self.projection(visual_embedding)
        combined_embedding = torch.bmm(audio_embedding.unsqueeze(2), visual_embedding.unsqueeze(1)) #mat mul both embeds ((batch_size, 128, 1) + (batch_size, 1, 128) -> (batch_size, 128, 128))
        combined_embedding = combined_embedding.view(combined_embedding.size(0), -1) #reshape into (batch_size, 16384)
        fused_embedding = self.fusion_layer(combined_embedding)
        return fused_embedding

    def classify(self, fused_embedding):    #when classify
        logits = self.classifier(fused_embedding)
        return logits

    def predict_speakers(self, audio_embedding, visual_embedding):  #when cluster
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(audio_embedding, visual_embedding).cpu().numpy()
        cluster = AgglomerativeClustering(n_clusters=self.num_speakers)
        labels = cluster.fit_predict(embeddings)
        return labels


# Training 
def train(model, dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for audio_emb, visual_emb, labels in dataloader:
            audio_emb, visual_emb, labels = audio_emb.to(device), visual_emb.to(device), labels.to(device)

            optimizer.zero_grad()
            fused_emb = model(audio_emb, visual_emb)
            logits = model.classify(fused_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


def when_classify():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_embeddings = torch.randn(100, 512)
    visual_embeddings = torch.randn(100, 512)
    labels = torch.randint(0, 10, (100,))

    dataset = MultimodalDataset(audio_embeddings, visual_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TensorDotMultimodalModel(embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10)
    summary(model, [(512,), (512,)], batch_size=16, device=str(device))
    # summary(model, input_size=[(16, 512), (16, 512)], device=str(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, epochs=20, device=device)


def when_cluster(audio_embeddings=torch.randn(100, 512), visual_embeddings=torch.randn(100, 512)):
    model = TensorDotMultimodalModel(embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10)
    speaker_labels = model.predict_speakers(audio_embeddings, visual_embeddings)
    print("Predicted Speaker Labels:", speaker_labels)


# Example 
if __name__ == "__main__":
    print("Classification tensor:")
    when_classify()

    print("Clustering tensor:")
    when_cluster()
