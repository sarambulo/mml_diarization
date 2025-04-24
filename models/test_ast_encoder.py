from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
# from models.ast_models import ASTModel  # assuming ASTModel is available
from models.ast_encoder import AudioASTEncoder  # your encoder wrapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.metrics import rttm_to_annotations, calculate_metrics_for_dataset
from pathlib import Path
from datasets.MSDWild import MSDWildChunks
from models.audio_model import AudioTripletDatasetWithLabels  # your existing dataset

def generate_ast_embeddings_from_msdwild():
    data_path = "./preprocessed"
    partition_path = "./data_sample/few_train.rttm"
    dataset = MSDWildChunks(data_path=data_path, partition_path=partition_path, subset=1.0)
    
    # dataset = AudioTripletDatasetWi?thLabels(msd_dataset, augment=False) 
    dataloader = DataLoader(dataset, batch_size= 32, shuffle=False)

    encoder = AudioASTEncoder()  # returns [B, 768]
    encoder.eval()
    encoder.cuda() if torch.cuda.is_available() else encoder.cpu()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(dataloader):
            if sample is None or sample[0] is None:
                continue
            
            _, audio_batch, label = sample
            
            anchors   = audio_batch[:, 0, :, :]
            positives = audio_batch[:, 1, :, :]
            negatives = audio_batch[:, 2, :, :] 
            all_audios = torch.cat([anchors, positives, negatives], dim=0)
            print(all_audios.shape)
            # anchor, _, _, label = sample

            embedding = encoder(all_audios).cpu()
            all_embeddings.append(embedding.squeeze(0))
            all_labels.append(label.item())

    torch.save({
        "embeddings": torch.stack(all_embeddings),
        "labels": torch.tensor(all_labels)
    }, "msdwild_ast_embeddings.pt")

    print(f"Saved {len(all_embeddings)} embeddings to msdwild_ast_embeddings.pt")

generate_ast_embeddings_from_msdwild()