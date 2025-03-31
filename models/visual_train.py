from torch.utils.data import DataLoader
import sys
# sys.path.insert(0, 'datasets')
from datasets.MSDWild import MSDWildChunks
import numpy as np
from pathlib import Path
import random
# from torchsummaryX import summary
import torch
import pandas as pd
import torch.nn.functional as F
from .VisualOnly import VisualOnlyModel
from sklearn.cluster import AgglomerativeClustering
from losses.DiarizationLoss import DiarizationLoss
from tqdm import tqdm
from torch.utils.data import Subset

CHECKPOINT_PATH = 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



@torch.no_grad()
def get_metrics(logits, labels):
    # print(logits)
    # print(labels)
    pred_labels = torch.argmax(logits, dim=-1)
    # print(pred_labels)
    # n = labels.shape[0]
    # print(labels)
    accuracy = (pred_labels == labels).float().mean()
    # accuracy = n_correct / n
    return accuracy

def save_model(model, metrics, epoch, path):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
    avg_loss = 0
    avg_accuracy = 0
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        visual_batch=batch[0]
        labels=batch[2]
        # print(visual_batch.shape)
        # print(len(labels))
        anchors   = visual_batch[:, 0, :, :, :]  # shape (B, 3, H, W)
        positives = visual_batch[:, 1, :, :, :]  # shape (B, 3, H, W)
        negatives = visual_batch[:, 2, :, :, :] 
        optimizer.zero_grad() # Zero gradients
        batch_size = labels.shape[0]
        # Join all inputs
        
        # features = list(zip(anchors, positive_pairs, negative_pairs))
        # print(len(features))
        # # feature: [(batch_size, ...), (batch_size, ...), (batch_size, ...)]
        # # send to cuda
        # for index, feature in enumerate(features):
        #     features[index] = torch.concat(feature, dim=0).to(DEVICE)
        #     # feature: (batch_size * 3, ...)
        # anchors   = anchors.to(DEVICE)   # shape [B, 3, H, W]
        # positives = positives.to(DEVICE) # shape [B, 3, H, W]
        # negatives = negatives.to(DEVICE)
        # labels = labels.to(DEVICE)
        all_images = torch.cat([anchors, positives, negatives], dim=0).to(DEVICE)
        # print(anchors.shape)
        # print(positives.shape)
        # print(negatives.shape)
        # forward
        with torch.amp.autocast(DEVICE):  # This implements mixed precision. Thats it!
            embeddings, logits = model((None, None,all_images))
            anchors        = embeddings[            :   batch_size]
            positive_pairs = embeddings[  batch_size: 2*batch_size]
            negative_pairs = embeddings[2*batch_size: 3*batch_size]
            logits         = logits[:batch_size]
            probs = torch.sigmoid(logits)
            pred_labels = (probs >= 0.5).float()
            # print(logits)
            # Use the type of output depending on the loss function you want to use
            labels=labels.float()
            loss = criterion(anchors, positive_pairs, negative_pairs, probs, labels)

        loss.backward() # This is a replacement for loss.backward()
        optimizer.step() # This is a replacement for optimizer.step()
        # print(logits)
        # print(pred_labels)
        accuracy = get_metrics(pred_labels, labels)
        # print(accuracy)
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_accuracy = (avg_accuracy * i + accuracy) / (i + 1)
        
        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc   = "{:.04%} ({:.04%})".format(accuracy, avg_accuracy),
            loss  = "{:.04f} ({:.04f})".format(loss.item(), avg_loss),
            lr    = "{:.06f}".format(float(optimizer.param_groups[0]['lr']))
        )

        batch_bar.update() # Update tqdm bar

    batch_bar.close()

    return avg_accuracy*100, avg_loss


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion):
    model.eval()

    avg_loss = 0.0
    avg_accuracy = 0.0
    count = 0

    for i, batch in enumerate(dataloader):
        visual_batch = batch[0]
        labels       = batch[2]

        B = labels.shape[0]
        anchors   = visual_batch[:, 0, :, :, :]  # (B, 3, H, W)
        positives = visual_batch[:, 1, :, :, :]
        negatives = visual_batch[:, 2, :, :, :]

        all_images = torch.cat([anchors, positives, negatives], dim=0).to(DEVICE)

        embeddings, logits = model((None, None, all_images))
        
        anchors_emb = embeddings[0:B]
        pos_emb     = embeddings[B:2*B]
        neg_emb     = embeddings[2*B:3*B]
        logits      = logits[:B]
        labels = labels.float().to(DEVICE)
        probs = torch.sigmoid(logits)
        loss = criterion(anchors_emb, pos_emb, neg_emb, probs, labels)

        # Accuracy
        accuracy = get_metrics(logits, labels)

        # Running average
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_accuracy = (avg_accuracy * i + accuracy) / (i + 1)

        count += 1

    return avg_accuracy*100, avg_loss


data_path = "preprocessed"
partition_path_train = "data_sample/few_train.rttm"
partition_path_val = "data_sample/few_train.rttm"
# train_dataset = MSDWildChunks(data_path=data_path, partition_path=partition_path_train, subset=1.0)
# val_dataset= MSDWildChunks(data_path=data_path, partition_path=partition_path_val, subset=1.0)

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.build_batch)
# val_loader=DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.build_batch)


full_dataset = MSDWildChunks(
    data_path=data_path, 
    partition_path=partition_path_train,  
    subset=1.0
)

dataset_size = len(full_dataset)
indices = list(range(dataset_size))
random.shuffle(indices)

# 80/20 split
split = int(0.8 * dataset_size)
train_indices = indices[:split]
val_indices   = indices[split:]

train_subset = Subset(full_dataset, train_indices)
val_subset   = Subset(full_dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True,  # we can still shuffle
    collate_fn=full_dataset.build_batch
)

val_loader = DataLoader(
    val_subset,
    batch_size=4,
    shuffle=False,
    collate_fn=full_dataset.build_batch
)


print(f"Train size: {len(train_subset)}   Val size: {len(val_subset)}")

for batch_idx, (visual_data, audio_data, is_speaking) in enumerate(train_loader):
    print(f"\n--- Batch {batch_idx} ---")
    print(f"visual_data shape: {visual_data.shape}")
    print(f"audio_data shape:  {audio_data.shape}")
    print(f"is_speaking shape: {is_speaking.shape}")
    
    # If you just want to check the first batch, break after printing:
    break

model = VisualOnlyModel(embedding_dims=512, num_classes=2) 

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = DiarizationLoss(0.5, 0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=2, threshold=0.01
    )
start_epoch = 0
final_epoch = 10
metrics = {}
best_valid_acc = 0
for epoch in range(start_epoch, final_epoch):
        print("\nEpoch {}/{}".format(epoch+1, final_epoch))
        # train
        curr_lr = float(scheduler.get_last_lr()[0])
        metrics.update({'lr': curr_lr})
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        print("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(epoch + 1, final_epoch, train_acc, train_loss, curr_lr))
        metrics.update({
            'train_cls_acc': train_acc,
            'train_loss': train_loss,
        })
        # validation
        valid_acc, valid_loss = evaluate_epoch(model, val_loader, criterion)
        print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_acc, valid_loss))
        metrics.update({
            'valid_cls_acc': valid_acc,
            'valid_loss': valid_loss,
        })
        epoch_ckpt_path = Path(CHECKPOINT_PATH, f"epoch_{epoch+1}.pth")
        save_model(model, metrics, epoch, epoch_ckpt_path)
        print(f"Saved checkpoint for epoch {epoch+1}")

        # save best model
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            model_path = Path(CHECKPOINT_PATH, f'best_visual.pth')
            save_model(model, metrics, epoch, model_path)
            print("Saved best model")

        
        # You may want to call some schedulers inside the train function. What are these?
        if scheduler is not None:
            scheduler.step(valid_loss)

    # save last model
model_path = Path(CHECKPOINT_PATH, f'last_visual.pth')
save_model(model, metrics, epoch, model_path)
print("Saved best model")
