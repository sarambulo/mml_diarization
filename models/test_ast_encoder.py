from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from ast_models import ASTModel  # assuming ASTModel is available
from ast_encoder import AudioASTEncoder  # your encoder wrapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.metrics import rttm_to_annotations, calculate_metrics_for_dataset
from pathlib import Path
from datasets.MSDWild import MSDWildChunks
from audio_model import AudioTripletDatasetWithLabels  # your existing dataset

def generate_ast_embeddings_from_msdwild():
    data_path = "../preprocessed"
    partition_path = "../data_sample/few_train.rttm"
    msd_dataset = MSDWildChunks(data_path=data_path, partition_path=partition_path, subset=1.0)
    
    dataset = AudioTripletDatasetWithLabels(msd_dataset, augment=False) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    encoder = AudioASTEncoder()  # returns [B, 768]
    encoder.eval()
    encoder.cuda() if torch.cuda.is_available() else encoder.cpu()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(dataloader):
            if sample is None or sample[0] is None:
                continue
            anchor, _, _, label = sample
            anchor = anchor.squeeze(0)  # [1, 40, 64] → [40, 64]
            anchor = anchor.T  # to [64, 40] → pad to [1024, 128] if needed

            # resize to [1024, 128] (required for AST)
            padded = torch.zeros((1024, 128))
            anchor = anchor.squeeze()  # makes it [F, T]
            if anchor.dim() != 2:
                raise ValueError(f"Expected 2D mel spectrogram, got shape {anchor.shape}")
            T, F = anchor.shape

            padded[:min(1024, T), :min(128, F)] = anchor[:min(1024, T), :min(128, F)]
            padded = padded.unsqueeze(0)  # [1, 1024, 128]

            embedding = encoder(padded).cpu()
            all_embeddings.append(embedding.squeeze(0))
            all_labels.append(label.item())

    torch.save({
        "embeddings": torch.stack(all_embeddings),
        "labels": torch.tensor(all_labels)
    }, "msdwild_ast_embeddings.pt")

    print(f"Saved {len(all_embeddings)} embeddings to msdwild_ast_embeddings.pt")

generate_ast_embeddings_from_msdwild()