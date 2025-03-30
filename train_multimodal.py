import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.cluster import AgglomerativeClustering
import torchaudio.transforms as AT
import torch.nn.functional as F
import sys
import os
import tqdm
from losses.DiarizationLoss import DiarizationLoss
from models.ConcatModel import ConcatenationFusionModel
from models.VisualOnly import ResNet34
from models.audio_model import CompactAudioEmbedding
from datasets.MSDWild import MSDWildChunks
from torchinfo import summary


def collate_fn(batch):
    # Extract each feature: do the zip thing
    video_data, audio_data, is_speaking = list(zip(*batch))
    # Padding: NOTE: Not necessary
    # Stack:
    video_data = torch.stack(video_data)
    audio_data = torch.stack(audio_data)
    is_speaking = torch.tensor(is_speaking)
    # Return tuple((N, video_data, melspectrogram), (N, video_data, melspectrogram), (N, video_data, melspectrogram))
    # (N, C, H, W), (N, Bands, T) x3 (ask Prachi)
    batch_data = {
        "video_data": video_data,
        "audio_data": audio_data,
        "labels": is_speaking,
    }
    return batch_data


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    epoch_loss = 0.0

    for batch in train_loader:

        if batch is None:
            continue

        batch_size = batch["video_data"].shape[0]
        video_data = batch["video_data"].to(device)
        audio_data = batch["audio_data"].to(device)
        labels = batch["labels"].to(device)
        
        anchor_v, pos_v, neg_v = video_data[:, 0], video_data[:, 1], video_data[:, 2]
        anchor_a, pos_a, neg_a = audio_data[:, 0], audio_data[:, 1], audio_data[:, 2]
        
        all_images  = torch.cat([anchor_v, pos_v, neg_v], dim=0).to(device)
        all_audios  = torch.cat([anchor_a, pos_a, neg_a], dim=0).to(device)
        
        _, _, triplet_emb, triplet_probs = model(all_audios, all_images)
        
        anchor_emb = triplet_emb[:batch_size]
        pos_emb = triplet_emb[batch_size: 2*batch_size]
        neg_emb = triplet_emb[2*batch_size: 3*batch_size]
        
        probs = triplet_probs[:batch_size]
       
        # labels = labels.long()
        loss = criterion(anchor_emb, pos_emb, neg_emb, probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = (probs > 0.5).float().flatten()
        print(probs.shape)
        print(f"train_probs: {probs}")
        print(f"train_predicted: {predicted}")
        print(f"train_labels: {labels}")
        
        correct += (predicted == labels).sum().item()
        print(f"correct: {correct}")
        total += labels.size(0)
        print(f"total: {total}")
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            video_data = batch["video_data"].to(device)
            audio_data = batch["audio_data"].to(device)
            labels = batch["labels"].to(device)
            
            batch_size = video_data.shape[0]
                        
            anchor_v, pos_v, neg_v = video_data[:, 0], video_data[:, 1], video_data[:, 2]
            anchor_a, pos_a, neg_a = audio_data[:, 0], audio_data[:, 1], audio_data[:, 2]
            
            all_images  = torch.cat([anchor_v, pos_v, neg_v], dim=0).to(device)
            all_audios  = torch.cat([anchor_a, pos_a, neg_a], dim=0).to(device)

            _, _, triplet_emb, triplet_probs = model(all_audios, all_images)
        
            anchor_emb = triplet_emb[:batch_size]
            pos_emb = triplet_emb[batch_size: 2*batch_size]
            neg_emb = triplet_emb[2*batch_size: 3*batch_size]
            
            probs = triplet_probs[:batch_size]
            # labels = labels.long()
            loss = criterion(anchor_emb, pos_emb, neg_emb, probs, labels)
            # print(logits.shape)
            predicted = (probs > 0.5).int().flatten()
            # print(f"val_logits: {logits}")
            # print(f"val_predicted: {predicted}")
            # print(f"val_labels: {labels}")
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(val_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def train_fusion_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=5,
    unfreeze_schedule=None,
):

    model.to(device)
    best_loss = float("inf")

    if unfreeze_schedule is None:
        unfreeze_schedule = {2: "audio_last", 2: "video_last", 3: "all"}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if epoch in unfreeze_schedule:
            u = unfreeze_schedule[epoch]
            if u == "audio_last":
                model.unfreeze_audio_encoder(unfreeze_all=False)
            elif u == "video_last":
                model.unfreeze_visual_encoder(unfreeze_all=False)
            elif u == "all":
                model.unfreeze_audio_encoder(unfreeze_all=True)
                model.unfreeze_visual_encoder(unfreeze_all=True)
            print("Update Schedule to", u)

            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=0.0001,
                weight_decay=0.01,
            )

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"training: loss: {train_loss:.4f}, acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"val: loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        scheduler.step(val_loss)
        # TO-DO: ADD MODEL SAVING/CHECKPOINT DIRECTORY
        if val_loss < best_loss:
            best_loss = val_loss

    return model, best_loss


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##LOAD CHECKPOINTS
    # audio_model = CompactAudioEmbedding()
    # audio_model.load_state_dict(torch.load("audio_model.pth", map_location=DEVICE))
    # visual_model = ResNet34()
    # visual_model.load_state_dict(torch.load("visual_model.pth", map_location=DEVICE))

    fusion_model = ConcatenationFusionModel(
        audio_model=None,
        visual_model=None,
        fusion_dim=512,
        embedding_dim=256,
        fusion_type="tensor",
    ).to(DEVICE)

    train_rttm_path = "data_sample/all.rttm"
    train_data_path = "preprocessed"

    train_dataset_full = MSDWildChunks(
        data_path=train_data_path, rttm_path=train_rttm_path, subset=0.8
    )

    # split few_train into train + val
    train_size = int(0.8 * train_dataset_full.length)
    val_size = train_dataset_full.length - train_size

    train_subset, val_subset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(69),
    )

    batch_size = 2
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    unfreeze_schedule = {
        2: "audio_last",  # after 2 epochs - unfreeze last layers of audio encoder
        2: "visual_last",  # same as above for visual
        3: "all",  # unfreeze everything
    }

    optimizer = optim.AdamW(
        [
            {"params": fusion_model.fusion_linear.parameters()},
            {"params": fusion_model.bn.parameters()},
            {"params": fusion_model.fusion_embedding.parameters()},
            {"params": fusion_model.classifier.parameters()},
        ],
        lr=0.001,
        weight_decay=0.01,
    )

    criterion = DiarizationLoss(triplet_lambda=0.3, bce_lambda=0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)

    trained_model, best_val_loss = train_fusion_model(
        fusion_model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        DEVICE,
        num_epochs=1,
        unfreeze_schedule=unfreeze_schedule,
    )


if __name__ == "__main__":
    main()
