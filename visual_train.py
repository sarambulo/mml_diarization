from torch.utils.data import DataLoader
import sys

sys.path.insert(0, "datasets")
from MSDWild import MSDWildChunks
import numpy as np
from pathlib import Path
import random
from torchsummaryX import summary
import torch
import pandas as pd
import torch.nn.functional as F
from models.VisualOnly import VisualOnlyModel
from sklearn.cluster import AgglomerativeClustering
from losses.DiarizationLoss import DiarizationLogitsLoss
from tqdm import tqdm
from torch.utils.data import Subset
from pairs.config import S3_BUCKET_NAME, S3_VIDEO_DIR
import os

CHECKPOINT_PATH = "model_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def get_metrics(logits, labels):
    # pred_labels = torch.argmax(logits, dim=-1)
    accuracy = (pred_labels == labels).float().mean()
    return accuracy


def save_model(model, metrics, epoch, path):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    print("Saving to", path)
    torch.save(checkpoint, path)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=5,
    )
    avg_loss = 0
    avg_accuracy = 0
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        visual_batch = batch[0].to(DEVICE)
        labels = batch[2].to(DEVICE)
        anchors = visual_batch[:, 0, :, :, :]
        positives = visual_batch[:, 1, :, :, :]
        negatives = visual_batch[:, 2, :, :, :]
        optimizer.zero_grad()
        batch_size = labels.shape[0]
        all_images = torch.cat([anchors, positives, negatives], dim=0).to(DEVICE)
        with torch.amp.autocast(DEVICE):
            embeddings, logits = model((None, None, all_images))
            anchors = embeddings[:batch_size]
            positive_pairs = embeddings[batch_size : 2 * batch_size]
            negative_pairs = embeddings[2 * batch_size : 3 * batch_size]
            logits = logits[:batch_size]
            probs = torch.sigmoid(logits)
            pred_labels = (probs >= 0.5).float()
            labels = labels.float()
            loss = criterion(anchors, positive_pairs, negative_pairs, logits, labels)
        loss.backward()
        optimizer.step()
        accuracy = get_metrics(logits, labels)
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_accuracy = (avg_accuracy * i + accuracy) / (i + 1)
        batch_bar.set_postfix(
            acc="{:.04%} ({:.04%})".format(accuracy, avg_accuracy),
            loss="{:.04f} ({:.04f})".format(loss.item(), avg_loss),
            lr="{:.06f}".format(float(optimizer.param_groups[0]["lr"])),
        )
        batch_bar.update()
    batch_bar.close()
    return avg_accuracy * 100, avg_loss


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    avg_loss = 0.0
    avg_accuracy = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        visual_batch = batch[0].to(DEVICE)
        labels = batch[2].to(DEVICE)
        B = labels.shape[0]
        anchors = visual_batch[:, 0, :, :, :]
        positives = visual_batch[:, 1, :, :, :]
        negatives = visual_batch[:, 2, :, :, :]
        all_images = torch.cat([anchors, positives, negatives], dim=0).to(DEVICE)
        embeddings, logits = model((None, None, all_images))
        anchors_emb = embeddings[0:B]
        pos_emb = embeddings[B : 2 * B]
        neg_emb = embeddings[2 * B : 3 * B]
        logits = logits[:B]
        labels = labels.float()
        probs = torch.sigmoid(logits)
        loss = criterion(anchors_emb, pos_emb, neg_emb, logits, labels)
        accuracy = get_metrics(logits, labels)
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_accuracy = (avg_accuracy * i + accuracy) / (i + 1)
        count += 1
    return avg_accuracy * 100, avg_loss


def main():
    data_path = "preprocessed"
    partition_path_train = "few.train.rttm"
    partition_path_val = "few.train.rttm"
    batch_size = 64
    full_dataset = MSDWildChunks(
        data_path=S3_VIDEO_DIR,
        data_bucket=S3_BUCKET_NAME,
        partition_path=partition_path_train,
        subset=0.025,
        refresh_fileset=False,
    )

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=full_dataset.build_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=full_dataset.build_batch,
    )
    print(f"Train size: {len(train_subset)}   Val size: {len(val_subset)}")
    model = VisualOnlyModel(embedding_dims=512, num_classes=2)
    model = model.float().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = DiarizationLogitsLoss(0.3, 0.7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=2, threshold=0.01
    )
    start_epoch = 0
    final_epoch = 10
    metrics = {}
    best_valid_acc = 0
    for epoch in range(start_epoch, final_epoch):
        print("\nEpoch {}/{}".format(epoch + 1, final_epoch))
        curr_lr = float(scheduler.get_last_lr()[0])
        metrics.update({"lr": curr_lr})
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        print(
            "\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, final_epoch, train_acc, train_loss, curr_lr
            )
        )
        metrics.update({"train_cls_acc": train_acc, "train_loss": train_loss})
        valid_acc, valid_loss = evaluate_epoch(model, val_loader, criterion)
        print(
            "Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(
                valid_acc, valid_loss
            )
        )
        metrics.update({"valid_cls_acc": valid_acc, "valid_loss": valid_loss})
        if epoch % 5 == 4:
            epoch_ckpt_path = Path(CHECKPOINT_PATH, f"epoch_{epoch+1}.pth")
            save_model(model, metrics, epoch, epoch_ckpt_path)
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            save_model(model, metrics, epoch, Path(CHECKPOINT_PATH, "best_visual.pth"))
        if scheduler is not None:
            scheduler.step(valid_loss)
    save_model(model, metrics, epoch, Path(CHECKPOINT_PATH, "last_visual.pth"))
