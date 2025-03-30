import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
import torchaudio.transforms as AT
import torch.nn.functional as F
import sys
import os
import tqdm
from models.VisualOnly import ResNet34
from models.audio_model import CompactAudioEmbedding


##get_audio_embedding, get_video_embedding need to be updated based on final unimodal implementation


class ConcatenationFusionModel(nn.Module):
    # can pass pretrained audio or visual module instead here otherwise it creates an instance of the same encoder architecture
    def __init__(
        self,
        audio_model=None,
        visual_model=None,
        fusion_dim=512,
        embedding_dim=256,
        fusion_type="concat",
        tensor_fusion_dim = 64
    ):
        super(ConcatenationFusionModel, self).__init__()
        if audio_model is None:
            self.audio_encoder = CompactAudioEmbedding(input_dim=40, embedding_dim=512, dropout_rate=0.3)
        else:
            self.audio_encoder = audio_model
        if visual_model is None:
            self.visual_encoder = ResNet34(embedding_dims=512)
        else:
            self.visual_encoder = visual_model

        audio_dim = 512  # self.audio_encoder.fc2.out_features
        visual_dim = 512  # self.visual_encoder.model[-3].out_channels

        if fusion_type == "concat":
            input_dim = audio_dim + visual_dim
        elif fusion_type == "tensor":
            # UPDATE BASED ON ACTUAL TENSOR FUSION IMPLEMENTATION
            self.audio_projector = nn.Linear(audio_dim, tensor_fusion_dim)
            self.visual_projector = nn.Linear(visual_dim, tensor_fusion_dim)
            
            input_dim = (tensor_fusion_dim + 1) * (tensor_fusion_dim + 1) #+ 1 for bias
        elif fusion_type == "additive":
            assert audio_dim == visual_dim
            input_dim = audio_dim
        else:
            raise ValueError("choose 'concat' or 'tensor' for fusion type")

        # fusion model
        self.fusion_linear = nn.Linear(input_dim, fusion_dim)
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.fusion_embedding = nn.Linear(fusion_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(0.25)

        # freeze the parameters of the audio and visual encoders
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        for _, param in self.visual_encoder.named_parameters():
            param.requires_grad = False

        self.fusion_type = fusion_type

    def __call__(self, audio, video):
        return self.forward(audio, video)

    def get_audio_embedding(self, audio_input):
        return self.audio_encoder.forward(audio_input)
        # batch_size, num_bands, time_steps = audio_input.shape
        # return torch.rand((batch_size, 512))

    def get_video_embedding(self, visual_input):
        return self.visual_encoder.forward(visual_input)
        # return torch.rand((batch_size, 512))

    def unfreeze_audio_encoder(self, unfreeze_all):
        if unfreeze_all:
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            for name, param in self.audio_encoder.named_parameters():
                if any(x in name for x in ["fc2", "bn_fc2"]):
                    param.requires_grad = True

    def unfreeze_visual_encoder(self, unfreeze_all):
        if unfreeze_all:
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            total_layers = len(self.visual_encoder.visual_encoder.model)
            layers_to_unfreeze = max(1, int(total_layers * 0.1))
            for name, param in self.visual_encoder.named_parameters():
                if any(
                    f"model.{i}" in name
                    for i in range(total_layers - layers_to_unfreeze, total_layers)
                ):
                    param.requires_grad = True

    def tensor_fusion(self, audio_emb, vis_emb):
        batch_size = vis_emb.size(0)
        
        projected_audio = self.audio_projector(audio_emb)
        projected_visual = self.visual_projector(vis_emb)  
        bias_ones = torch.ones(batch_size, 1) 
        
        #add ones for bias
        audio_full_emb = torch.cat([bias_ones, projected_audio], dim=1)
        vis_full_emb = torch.cat([bias_ones, projected_visual], dim=1)
        
        audio_full_emb = audio_full_emb.unsqueeze(2)  # (batch_size, audio_dim+1, 1)
        vis_full_emb = vis_full_emb.unsqueeze(1)  # (batch_size, 1, visual_dim+1)
        
        #outerproduct by batch
        fusion_tensor = torch.bmm(audio_full_emb, vis_full_emb)  # (batch_size, audio_dim+1, visual_dim+1)
        fusion_vector = fusion_tensor.reshape(batch_size, -1) #flatten everything besides batch_size
        
        return fusion_vector

    def forward(self, audio_input, visual_input):
        audio_embedding = self.get_audio_embedding(audio_input)
        video_embedding = self.get_video_embedding(visual_input)

        # combine based on fusion type
        if self.fusion_type == "concat":
            combined_embedding = torch.cat((audio_embedding, video_embedding), dim=1)
        elif self.fusion_type == "tensor":
            combined_embedding = self.tensor_fusion(audio_embedding, video_embedding)
        elif self.fusion_type == "additive":
            combined_embedding = audio_embedding + video_embedding
        else:
            raise ValueError("choose 'concat', 'tensor', or 'additive' for fusion type")

        # fusion model
        fusion = self.fusion_linear(combined_embedding)
        fusion = F.relu(self.bn(fusion))
        fusion = self.dropout(fusion)

        fusion_embedding = self.fusion_embedding(fusion)
        fusion_embedding = F.normalize(fusion_embedding, p=2, dim=1)

        # classification layer
        classfier_output = self.classifier(fusion_embedding)
        # print(f"classfier_output: {classfier_output}")
        probability = torch.sigmoid(classfier_output)
        # print(f"Probability: {probability}")

        return audio_embedding, video_embedding, fusion_embedding, probability

    # def process_triplet(self, audio_data, visual_data):
    #     anchor_v, pos_v, neg_v = visual_data[:, 0], visual_data[:, 1], visual_data[:, 2]
    #     anchor_a, pos_a, neg_a = audio_data[:, 0], audio_data[:, 1], audio_data[:, 2]
        

    #     _, _, anchor_emb, logits = self.forward(anchor_a, anchor_v)

    #     pos_emb = self.forward(pos_a, pos_v)[2]
    #     neg_emb = self.forward(neg_a, neg_v)[2]

    #     triplet_emb = torch.stack([anchor_emb, pos_emb, neg_emb], dim=1)
    #     return triplet_emb, logits.flatten()
