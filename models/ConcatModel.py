import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
import torchaudio.transforms as AT
import torch.nn.functional as F
import sys
import os
import tqdm
from VisualOnly import ResNet34
from audio_train import AudioOnlyTDNN


##get_audio_embedding, get_video_embedding need to be updated based on final unimodal implementation

class ConcatenationFusionModel(nn.Module):
     #can pass pretrained audio or visual module instead here otherwise it creates an instance of the same encoder architecture
    def __init__(self, audio_model = None, visual_model = None, fusion_dim = 512, embedding_dim =256, fusion_type = 'concat'):
        super(ConcatenationFusionModel, self).__init__()
        if audio_model is None:
            self.audio_encoder = AudioOnlyTDNN(input_dim=40, hidden_dim=512)
        else: 
            self.audio_encoder = audio_model
        if visual_model is None:
            self.visual_encoder = ResNet34(embeddin_dims=512)
        else: 
            self.visual_encoder = audio_model
        
        audio_dim = self.audio_encoder.fc2.out_features
        visual_dim = self.visual_encoder.visual_encoder.model[-3].out_channels
        
        if fusion_type == 'concat':
            input_dim = audio_dim + visual_dim
        elif fusion_type == 'tensor': 
            input_dim = audio_dim * visual_dim #UPDATE BASED ON ACTUAL TENSOR FUSION IMPLEMENTATION
        elif fusion_type == 'additive':
            assert(audio_dim == visual_dim)
            input_dim = audio_dim
        else:
            raise ValueError("choose 'concat' or 'tensor' for fusion type")
        
        #fusion model 
        self.fusion_linear = nn.Linear(input_dim, fusion_dim)
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.fusion_embedding = nn.Linear(fusion_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        self.droput = nn.Dropout(.25)
        
        #freeze the parameters of the audio and visual encoders
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        for _, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        
        self.fusion_type = fusion_type

    def get_audio_embedding(self, audio_input):
        x = self.audio_encoder.bn1(F.relu(self.audio_encoder.tdnn1(audio_input)))
        x = self.audio_encoder.dropout(x)
    
        x = self.audio_encoder.bn2(F.relu(self.audio_encoder.tdnn2(x)))
        x = self.audio_encoder.dropout(x)
         
        x = self.audio_encoder.bn3(F.relu(self.audio_encoder.tdnn3(x)))
        x = self.audio_encoder.dropout(x)
        
        x = self.audio_encoder.bn4(F.relu(self.audio_encoder.tdnn4(x)))
        x = self.audio_encoder.dropout(x)
        
        # Statistical pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        x = torch.cat([mean, std], dim=1)
        
        x = self.audio_encoder.bn_fc1(F.relu(self.audio_encoder.fc1(x)))
        x = self.audio_encoder.dropout(x)
        
        x = self.audio_encoder.bn_fc2(F.relu(self.audio_encoder.fc2(x)))
        
        return x
    
    def get_video_embedding(self, visual_input):
        x = self.visual_encoder.forward(visual_input)
        return x
    
    def unfreeze_audio_encoder(self, unfreeze_all):
        if unfreeze_all:
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            for name, param in self.audio_encoder.named_parameters():
                if any(x in name for x in ['fc2', 'bn_fc2']):
                    param.requires_grad = True
                    
    def unfreeze_visual_encoder(self, unfreeze_all):
        if unfreeze_all:
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            total_layers = len(self.visual_encoder.visual_encoder.model)
            layers_to_unfreeze = max(1, int(total_layers * 0.1))
            for name, param in self.visual_encoder.named_parameters():
                if any(f'model.{i}' in name for i in range(total_layers - layers_to_unfreeze, total_layers)):
                    param.requires_grad = True
                    
    def tensor_fusion(self, audio_emb, vis_emb):
        pass        
    
    def forward(self, audio_input, visual_input):
        audio_embedding = self.get_audio_embedding(audio_input)
        video_embedding = self.get_video_embedding(visual_input)
        
        #combine based on fusion type
        if self.fusion_type == 'concat':
            combined_embedding = torch.cat((audio_embedding, video_embedding), dim=1)
        elif self.fusion_type == 'tensor':
            combined_embedding = self.tensor_fusion(audio_embedding, video_embedding)
        else:
            raise ValueError("choose 'concat' or 'tensor' for fusion type")
        
        #fusion model
        fusion = self.fusion_linear(combined_embedding)
        fusion = F.relu(self.bn(fusion))
        fusion = self.dropout(fusion)
        
        fusion_embedding = self.embedding_fc(fusion)
        fusion_embedding = F.normalize(fusion_embedding, p=2, dim=1) ### CHECK DIM BASED ON SHAPE
        
        #classification layer
        probability = torch.sigmoid(self.classifier(fusion_embedding))
        
        return audio_embedding, video_embedding, fusion_embedding, probability
    
    def process_triplet(self, visual_data, audio_data):
        anchor_v, pos_v, neg_v = visual_data[:,0],visual_data[:,1], visual_data[:,2]
        anchor_a, pos_a, neg_a = audio_data[:,0],audio_data[:,1], audio_data[:,2]
        
        _, _, anchor_emb, logits = self.forward(anchor_a, anchor_v)
        
        pos_emb = self.forward(pos_a, pos_v)[2]
        neg_emb = self.forward(neg_a, neg_v)[2]
        
        triplet_emb = torch.stack([anchor_emb, pos_emb, neg_emb], dim=1)
        return triplet_emb, logits