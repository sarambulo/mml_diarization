import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
from torchsummary import summary
from add_embeds_mm import AddSimpleMultimodalModel
from tensor_dot_embeds_mm import TensorDotMultimodalModel


class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings, visual_embeddings, labels):
        self.audio_embeddings = audio_embeddings
        self.visual_embeddings = visual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.visual_embeddings[idx], self.labels[idx]


# class AddSimpleMultimodalModel(nn.Module):
#     def __init__(self, embedding_dim=512, fusion_dim=512, num_speakers=10):
#         super(AddSimpleMultimodalModel, self).__init__()
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(2 * embedding_dim, 2 * embedding_dim),
#             nn.BatchNorm1d(2 * embedding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(2 * embedding_dim, fusion_dim),
#             nn.BatchNorm1d(fusion_dim),
#             nn.ReLU(),
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.ReLU()
#         )
#         self.classifier = nn.Linear(fusion_dim, num_speakers)

#     def forward(self, audio_embedding, visual_embedding):
#         combined_embedding = torch.cat((audio_embedding, visual_embedding), dim=1)
#         fused_embedding = self.fusion_layer(combined_embedding)
#         return fused_embedding

#     def classify(self, fused_embedding):
#         return self.classifier(fused_embedding)


# class TensorDotMultimodalModel(nn.Module):
#     def __init__(self, embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10):
#         super(TensorDotMultimodalModel, self).__init__()

#         self.projection = nn.Linear(embedding_dim, reduced_dim)

#         self.fusion_layer = nn.Sequential(
#             nn.Linear(reduced_dim * reduced_dim, 2 * embedding_dim), #16384 -> 2*512 (1024)
#             nn.BatchNorm1d(2 * embedding_dim), #1024
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(2 * embedding_dim, fusion_dim), #1024 -> 512
#             nn.BatchNorm1d(fusion_dim), #512
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(fusion_dim, fusion_dim), #512 -> 512
#             nn.ReLU()
#         )
#         self.num_speakers = num_speakers
#         self.classifier = nn.Linear(fusion_dim, self.num_speakers)

#     def forward(self, audio_embedding, visual_embedding):   #get embed
#         audio_embedding = self.projection(audio_embedding)  # (batch_size, 128)
#         visual_embedding = self.projection(visual_embedding)
#         combined_embedding = torch.bmm(audio_embedding.unsqueeze(2), visual_embedding.unsqueeze(1)) #mat mul both embeds ((batch_size, 128, 1) + (batch_size, 1, 128) -> (batch_size, 128, 128))
#         combined_embedding = combined_embedding.view(combined_embedding.size(0), -1) #reshape into (batch_size, 16384)
#         fused_embedding = self.fusion_layer(combined_embedding)
#         return fused_embedding

#     def classify(self, fused_embedding):
#         return self.classifier(fused_embedding)


class BaggingMultimodalEnsemble(nn.Module):
    def __init__(self, num_models=5, embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10):
        super(BaggingMultimodalEnsemble, self).__init__()
        self.models = nn.ModuleList(
            [AddSimpleMultimodalModel(embedding_dim, fusion_dim, num_speakers) if i % 2 == 0
             else TensorDotMultimodalModel(embedding_dim, reduced_dim, fusion_dim, num_speakers) for i in range(num_models)]
        )

    def forward(self, audio_embedding, visual_embedding):
        outputs = [model(audio_embedding, visual_embedding) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)  # Average predictions

    def classify(self, fused_embedding):
        outputs = [model.classify(fused_embedding) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)  # Average class scores
    
    def predict_speakers(self, audio_embedding, visual_embedding):
        """ Use each model to predict speakers, then return majority voting results """
        predictions = torch.stack([torch.tensor(model.predict_speakers(audio_embedding, visual_embedding)) for model in self.models])
        majority_vote = torch.mode(predictions, dim=0).values  # Majority voting
        return majority_vote.numpy()
    
    def train_bag(self, dataloader, criterion, optimizer, epochs=10, device='cuda'):
        self.to(device)
        self.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch} started...')
            total_loss = 0
            for audio_emb, visual_emb, labels in dataloader:
                audio_emb, visual_emb, labels = audio_emb.to(device), visual_emb.to(device), labels.to(device)

                optimizer.zero_grad()
                fused_emb = self(audio_emb, visual_emb)
                logits = self.classify(fused_emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


class BoostingMultimodalEnsemble(nn.Module):
    def __init__(self, num_models=5, embedding_dim=512, reduced_dim=128, fusion_dim=512, num_speakers=10):
        super(BoostingMultimodalEnsemble, self).__init__()
        self.models = nn.ModuleList(
            [AddSimpleMultimodalModel(embedding_dim, fusion_dim, num_speakers) if i % 2 == 0
             else TensorDotMultimodalModel(embedding_dim, reduced_dim, fusion_dim, num_speakers) for i in range(num_models)]
        )
        self.weights = torch.ones(num_models) / num_models  # Equal weights initially

    def forward(self, audio_embedding, visual_embedding):
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(self.weights[i] * model(audio_embedding, visual_embedding))
        return torch.sum(torch.stack(outputs), dim=0)  # Weighted sum of outputs

    def classify(self, fused_embedding):
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(self.weights[i] * model.classify(fused_embedding))
        return torch.sum(torch.stack(outputs), dim=0)  # Weighted sum of class scores
    
    def predict_speakers(self, audio_embedding, visual_embedding):
        """ Use weighted sum of predictions for boosting """
        weighted_outputs = []
        for i, model in enumerate(self.models):
            weighted_outputs.append(self.weights[i] * torch.tensor(model.predict_speakers(audio_embedding, visual_embedding), dtype=torch.float32))
        weighted_prediction = torch.sum(torch.stack(weighted_outputs), dim=0)  # Sum weighted votes
        return torch.round(weighted_prediction).numpy().astype(int)  # Convert to integer labels
    
    def update_weights(self, losses):
        """ Update model weights based on their performance (lower loss = higher weight) """
        losses = torch.tensor(losses, dtype=torch.float32)  
        self.weights = torch.exp(-losses)  # Higher loss -> Lower weight
        self.weights /= torch.sum(self.weights)  # Normalize to sum to 1
    
    def train_boost(self, dataloader, criterion, optimizer, epochs=10, device='cuda'):
        self.to(device)
        self.train()

        for epoch in range(epochs):
            print(f'Epoch {epoch} started...')
            total_losses = []

            for audio_emb, visual_emb, labels in dataloader:
                audio_emb, visual_emb, labels = (
                    audio_emb.to(device), visual_emb.to(device), labels.to(device)
                )

                optimizer.zero_grad()
                model_losses = []

                for i, model in enumerate(self.models):
                    fused_emb = model(audio_emb, visual_emb)
                    logits = model.classify(fused_emb)
                    loss = criterion(logits, labels)
                    model_losses.append(loss.item())

                # Update weights based on losses
                self.update_weights(model_losses)

                total_losses.append(sum(model_losses) / len(model_losses))

            avg_loss = sum(total_losses) / len(total_losses)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')



# **Bagging Training**
def when_classify_bagging():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_embeddings = torch.randn(100, 512)
    visual_embeddings = torch.randn(100, 512)
    labels = torch.randint(0, 10, (100,))

    dataset = MultimodalDataset(audio_embeddings, visual_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BaggingMultimodalEnsemble(num_models=5)
    summary(model, [(512,), (512,)], batch_size=16, device=str(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train_bag(dataloader, criterion, optimizer, epochs=10, device=device)


# **Boosting Training**
def when_classify_boosting():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_embeddings = torch.randn(100, 512)
    visual_embeddings = torch.randn(100, 512)
    labels = torch.randint(0, 10, (100,))

    dataset = MultimodalDataset(audio_embeddings, visual_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BoostingMultimodalEnsemble(num_models=5)
    summary(model, [(512,), (512,)], batch_size=16, device=str(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train_boost(dataloader, criterion, optimizer, epochs=10, device=device)


# **Clustering (Speaker Diarization) for Bagging**
def when_cluster_bagging():
    model = BaggingMultimodalEnsemble(num_models=5)
    audio_embeddings = torch.randn(100, 512)
    visual_embeddings = torch.randn(100, 512)

    speaker_labels = model.predict_speakers(audio_embeddings, visual_embeddings)
    print("Predicted Speaker Labels:", speaker_labels)


# **Clustering (Speaker Diarization) for Boosting**
def when_cluster_boosting():
    model = BoostingMultimodalEnsemble(num_models=5)
    audio_embeddings = torch.randn(100, 512)
    visual_embeddings = torch.randn(100, 512)

    speaker_labels = model.predict_speakers(audio_embeddings, visual_embeddings)
    print("Predicted Speaker Labels:", speaker_labels)


if __name__ == "__main__":
    print("\n\nBagging Classification:")
    when_classify_bagging()

    # print("\n\nBoosting Classification:")
    # when_classify_boosting()

    print("\n\nBagging Clustering:")
    when_cluster_bagging()

    # print("\n\nBoosting Clustering:")
    # when_cluster_boosting()
