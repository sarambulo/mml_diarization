import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchensemble import BaggingClassifier, GradientBoostingClassifier
from torchensemble.utils.logging import set_logger


# ----------------------------
# Data Preparation
# ----------------------------
class MultimodalDataset(Dataset):
    def __init__(self, num_samples=1000, embedding_dim=512, num_classes=10):
        self.audio_embeddings = torch.randn(num_samples, embedding_dim)
        self.visual_embeddings = torch.randn(num_samples, embedding_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.visual_embeddings[idx], self.labels[idx]


def get_dataloaders(batch_size=32):
    train_dataset = MultimodalDataset()
    test_dataset = MultimodalDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ----------------------------
# Base Model
# ----------------------------
class MultimodalMLP(nn.Module):
    def __init__(self, embedding_dim=512, fusion_dim=256, num_classes=10):
        super(MultimodalMLP, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, audio_embedding, visual_embedding):
        combined_embedding = torch.cat((audio_embedding, visual_embedding), dim=1)
        return self.fusion_layer(combined_embedding)


# ----------------------------
# Training Function (Generic)
# ----------------------------
def train_ensemble(model, train_loader, test_loader, epochs=20):
    # Set Loss Function
    criterion = nn.CrossEntropyLoss()
    model.set_criterion(criterion)

    # Set Optimizer
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Train Model
    model.fit(
        train_loader,
        epochs=epochs,
        test_loader=test_loader,
    )

    return model


# ----------------------------
# Train Bagging Model
# ----------------------------
def train_bagging(train_loader, test_loader):
    logger = set_logger("multimodal_bagging")
    print("\n🔹 Training Bagging Model...")

    bagging_model = BaggingClassifier(
        estimator=MultimodalMLP,
        n_estimators=10,
        cuda=torch.cuda.is_available(),
    )

    return train_ensemble(bagging_model, train_loader, test_loader)


# ----------------------------
# Train Boosting Model
# ----------------------------
def train_boosting(train_loader, test_loader):
    logger = set_logger("multimodal_boosting")
    print("\n🔹 Training Boosting Model...")

    boosting_model = GradientBoostingClassifier(
        estimator=MultimodalMLP,
        n_estimators=10,
        shrinkage_rate=0.1,  # Learning rate for boosting
        cuda=torch.cuda.is_available(),
    )

    return train_ensemble(boosting_model, train_loader, test_loader)


# ----------------------------
# Model Evaluation
# ----------------------------
def evaluate_model(model, test_loader, model_name):
    acc = model.evaluate(test_loader)
    print(f"✅ {model_name} Accuracy: {acc:.4f}")


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=32)

    # Train and Evaluate Bagging Model
    bagging_model = train_bagging(train_loader, test_loader)
    evaluate_model(bagging_model, test_loader, "Bagging")

    # Train and Evaluate Boosting Model
    boosting_model = train_boosting(train_loader, test_loader)
    evaluate_model(boosting_model, test_loader, "Boosting")
