from transformers import AutoFeatureExtractor, ResNetModel
import torch
import numpy as np
import torch.nn.functional as F


class VisualEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-34")

    def forward(self, x):
        pixel_values = self.image_process(x)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            embeddings = outputs.last_hidden_state
            embeddings = F.adaptive_avg_pool2d(embeddings, (1, 1))
            embeddings = embeddings.squeeze(-1).squeeze(-1)
        return embeddings

    def image_process(self, bbox):
        if bbox.min() < 0 or bbox.max() > 1:
            bbox = (bbox - bbox.min()) / (bbox.max() - bbox.min())
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixel_values = (bbox_tensor - mean) / std
        return pixel_values


class VisualFullModel(torch.nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.visual_encoder = VisualEmbedding()
        self.classifier = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedding = self.visual_encoder(x)
        logits = self.classifier(embedding)
        logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        return embedding, probs


##TESTING
# data_path = '../preprocessed/00001/Chunk_1/face_0.npy'
# face_batch = np.load(data_path)  # Shape: [18, 3, 112, 112]

# model = VisualEmbedding()
# test_output = model(face_batch)
# full_model = VisualFullModel()
# test_full = full_model(face_batch)

# print(test_full[1].size())
# print(test_full[1])
