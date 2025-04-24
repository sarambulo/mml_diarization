import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from models.melbert_model import MelHuBERTConfig, MelHuBERTModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.MSDWild import MSDWildChunks

class MelHuBERTModule(nn.Module):
    """
    A PyTorch module that wraps the MelHuBERT model for feature extraction
    from mel spectrograms of custom dimensions.
    """
    def __init__(self, time_dim=30, freq_dim=22, embedding_dim=None):
        """
        Initialize the MelHuBERT module.
        
        Args:
            time_dim (int): Time dimension of input mel spectrograms
            freq_dim (int): Frequency dimension of input mel spectrograms
            embedding_dim (int, optional): Desired embedding dimension (None to use MelHuBERT's hidden size)
        """
        super().__init__()
        
        # Store input dimensions
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.expected_freq_dim = 80  # MelHuBERT expects 40 frequency bins
        
        # Define frequency adapter layer if needed
        if freq_dim != self.expected_freq_dim:
            self.freq_adapter = nn.Linear(freq_dim, self.expected_freq_dim)
            self.needs_freq_adapter = True
        else:
            self.needs_freq_adapter = False
        
        # MelHuBERT model (to be loaded in load_pretrained)
        self.model = None
        self.config = None
        self.hidden_size = None
        
        # Define embedding projection if needed
        self.embedding_dim = embedding_dim
        self.projection = None  # Will be defined after loading pretrained model
        
        print(f"MelHuBERTModule initialized for input shape: [batch_size, {time_dim}, {freq_dim}]")
    
    def load_pretrained(self, checkpoint_path, device='cuda'):
        """
        Load pretrained weights from a MelHuBERT checkpoint.
        
        Args:
            checkpoint_path (str): Path to the MelHuBERT checkpoint
            device (str): Device to run the model on
            
        Returns:
            self: The module instance
        """
    
        # Load checkpoint
        print(f"Loading pretrained weights from {checkpoint_path}")
        all_states = torch.load(checkpoint_path, map_location="cpu", weights_only = False)
        upstream_config = all_states["Upstream_Config"]["hubert"]
        print(all_states.keys())
        
        # Initialize model configuration
        self.config = MelHuBERTConfig(upstream_config)
        self.hidden_size = self.config.encoder_embed_dim
        
        # Initialize model
        self.model = MelHuBERTModel(self.config)
        
        # Load weights
        self.model.load_state_dict(all_states["model"])
        
        # Define embedding projection if needed
        if self.embedding_dim is not None and self.embedding_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.embedding_dim)
        else:
            self.embedding_dim = self.hidden_size
            self.projection = nn.Identity()
        
        # Move model to device
        self.to(device)
        self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Successfully loaded model with {total_params} parameters")
        print(f"Model hidden size: {self.hidden_size}")
        print(f"Output embedding dimension: {self.embedding_dim}")
        
        return self
    
    def adapt_input(self, mel_specs):
        """
        Adapt the input mel spectrograms to the format expected by MelHuBERT.
        
        Args:
            mel_specs (torch.Tensor): Mel spectrograms of shape [batch_size, time, freq]
            
        Returns:
            torch.Tensor: Adapted mel spectrograms
            torch.Tensor: Padding mask
        """
        batch_size, time_dim, freq_dim = mel_specs.shape
        
        # Create padding mask (all ones since we're using full sequences)
        pad_mask = torch.ones(batch_size, time_dim, device=mel_specs.device)
        
        # Adapt frequency dimension if needed
        if self.needs_freq_adapter:
            mel_specs = self.freq_adapter(mel_specs)  # [batch_size, time, expected_freq]
        
        return mel_specs, pad_mask
    
    def mean_pool(self, features, pad_mask):
        """
        Apply mean pooling to get a fixed-size embedding per sequence.
        
        Args:
            features (torch.Tensor): Features of shape [batch_size, time, hidden_size]
            pad_mask (torch.Tensor): Padding mask of shape [batch_size, time]
            
        Returns:
            torch.Tensor: Pooled embeddings of shape [batch_size, hidden_size]
        """
        # Mean pooling (considering padding)
        mask_expanded = pad_mask.unsqueeze(-1).expand_as(features)
        sum_embeddings = torch.sum(features * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        return sum_embeddings / sum_mask
    
    def forward(self, mel_specs, layer=-1, pool_method='mean'):
        """
        Forward pass through the model.
        
        Args:
            mel_specs (torch.Tensor): Mel spectrograms of shape [batch_size, time, freq]
            layer (int): Which transformer layer to extract (-1 for last layer)
            pool_method (str): How to pool sequence embeddings ('mean', 'max', 'cls')
            
        Returns:
            torch.Tensor: Embeddings of shape [batch_size, embedding_dim]
        """
        # Check if model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained first.")
        
        # Adapt input
        adapted_specs, pad_mask = self.adapt_input(mel_specs)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(adapted_specs, pad_mask, get_hidden=True, no_pred=True)
        
        # Get features from the specified layer
        if layer == -1:
            # Get last layer features
            features = outputs[0]  # [batch_size, time, hidden_size]
        else:
            # Get hidden states from specific layer
            hidden_states = outputs[5]  # list of tensors
            features = hidden_states[layer]  # [batch_size, time, hidden_size]
        
        # Pool features to get fixed-size embedding
        if pool_method == 'mean':
            pooled_features = self.mean_pool(features, pad_mask)
        elif pool_method == 'max':
            # Max pooling (with masking)
            mask_expanded = pad_mask.unsqueeze(-1).expand_as(features)
            features = features * mask_expanded - 1e10 * (1 - mask_expanded)
            pooled_features = torch.max(features, dim=1)[0]
        elif pool_method == 'cls':
            # Use [CLS] token or first token embedding
            pooled_features = features[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")
        
        # Apply projection if needed
        embeddings = self.projection(pooled_features)
        
        return embeddings

##TESTING
# data_path = '../preprocessed/00001/Chunk_1/melspectrogram.npy'
# mel_numpy = np.load(data_path)
# print(mel_numpy.shape)
# mel = torch.tensor(mel_numpy, dtype = torch.float32)

# data_path = "./preprocessed"
# partition_path = "./data_sample/few_train.rttm"
# dataset = MSDWildChunks(data_path=data_path, partition_path=partition_path, subset=1.0)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# # encoder = AudioASTEncoder()  # returns [B, 768]
# # encoder.eval()
# # encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
# print(type(dataloader))

# model = MelHuBERTModule(time_dim=30, freq_dim=22)
# model.load_pretrained('./models/960_stage2_20ms.ckpt', device='cpu')

# all_embeddings = []
# all_labels = []

# with torch.no_grad():
#     for sample in tqdm(dataloader):        
#         _, audio_batch, label = sample
        
#         anchors   = audio_batch[:, 0, :, :]
#         positives = audio_batch[:, 1, :, :]
#         negatives = audio_batch[:, 2, :, :] 
#         all_audios = torch.cat([anchors, positives, negatives], dim=0)
#         print(all_audios.shape)
#         embeddings = model(all_audios)
#         print(embeddings.shape)
            


# model.eval()
# with torch.no_grad():

    

# model.eval()

# with torch.no_grad():
#     embeddings = model(mel)
    
# print(f"Output embeddings shape: {embeddings.shape}")

# embeddings = ast(mel)
print(embeddings.shape)

# test_output = model(mel)

# print(test_output.size())
# print(test_full[1])
    