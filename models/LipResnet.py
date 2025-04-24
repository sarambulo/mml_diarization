# from transformers import ResNetModel
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class VisualEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim = 1024):
        super().__init__()
        # self.model = ResNetModel.from_pretrained("microsoft/resnet-34")
        
        base_model = models.resnet34(pretrained=True)
        
        # Modify first convolutional layer to accept 3-channel input
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        # ResNet backbone
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Global average pooling
        self.avgpool = base_model.avgpool
        #proj to 1024 to match with face model
        self.projection = torch.nn.Linear(512, embedding_dim)

    def forward(self, x):
        pixel_values = self.image_process(x)
        pixel_values = pixel_values.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)
        # print(pixel_values.dtype)
        # Forward through ResNet backbone
        x = self.conv1(pixel_values)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Get embeddings through average pooling
        embeddings = self.avgpool(x)
        embeddings = torch.flatten(embeddings, 1)
        
        # with torch.no_grad():
        #     outputs = self.model(pixel_values=pixel_values)
        #     embeddings = outputs.last_hidden_state
        #     embeddings = F.adaptive_avg_pool2d(embeddings, (1, 1))
        #     embeddings = embeddings.squeeze(-1).squeeze(-1)
        
        embeddings = self.projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def image_process(self, bbox):
        if bbox.min() < 0 or bbox.max() > 1:
            bbox = (bbox - bbox.min()) / (bbox.max() - bbox.min())
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).to('cuda')
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')
        pixel_values = (bbox_tensor - mean) / std
        return pixel_values

class VisualLipModel(torch.nn.Module):
   def __init__(self, embedding_dim=1024):
      super().__init__()
      self.visual_encoder = VisualEmbedding(embedding_dim)
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
# full_model = VisualLipModel()
# test_full = full_model(face_batch)
# print(test_output.size())
# print(test_full[1].size())
# print(test_full[1])
