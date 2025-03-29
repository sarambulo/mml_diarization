import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
import torchaudio.transforms as AT
import torch.nn.functional as F
import sys
import os
import tqdm
from .losses.DiarizationLoss import DiarizationLoss
from ConcatModel import ConcatenationFusionModel

def train_epoch(model, train_loader, optimizer, criterion, device=DEVICE):
    model.train()
    correct = 0
    total = 0
    epoch_loss = 0.0
    
    for batch in train_loader:
        
        if batch is None: continue
        
        video_data = batch['video_data'].to(device)
        audio_data = batch['audio_data'].to(device)
        labels = batch['labels'].to(device)
        
        triplet_emb, logits = model.process_triplet(video_data, audio_data)
        anchor_emb, pos_emb, neg_emb = triplet_emb[:, 0], triplet_emb[:, 1], triplet_emb[:, 2]
        labels = labels.long()
        loss = criterion(anchor_emb, pos_emb, neg_emb, logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted==labels).sum().item()
        total += labels.size(0)
        epoch_loss += loss.item()
    
    epoch_loss = epoch_loss / len(train_loader)
    epoch_accuracy = correct/total 
    return epoch_loss, epoch_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            video_data = batch['video_data'].to(device)
            audio_data = batch['audio_data'].to(device)
            labels = batch['labels'].to(device)
            
            triplet_emb, logits = model.process_triplet(video_data, audio_data)
            anchor_emb, pos_emb, neg_emb = triplet_emb[:,0], triplet_emb[:,1], triplet_emb[:,2]
            labels = labels.long()
            loss = criterion(anchor_emb, pos_emb, neg_emb, logits, labels)
            _, predicted = torch.max(logits,1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            
    epoch_loss = epoch_loss / len(val_loader)
    epoch_accuracy = correct/total 
    return epoch_loss, epoch_accuracy
            
        

def train_fusion_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=5, unfreeze_schedule = None):
    
    model.to(device)
    best_loss = float('inf')
    
    ##IMPLEMENT UNFREEZING SCHEDULE TO FIRST TRAIN FUSION ONLY THEN OPEN TO AUDIO/VIDEO
    
    if unfreeze_schedule is None:
        unfreeze_schedule = {2: 'audio_last', 2: 'video_last', 3: 'all'}
        
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if epoch in unfreeze_schedule:
            u = unfreeze_schedule[epoch]
            if u == 'audio_last':
                model.unfreeze_audio_encoder(unfreeze_all = False)
            elif u == 'video_last':
                model.unfreeze_visual_encoder(unfreeze_all = False)
            elif u == 'all':
                model.unfreeze_audio_encoder(unfreeze_all = True)
                model.unfreeze_visual_encoder(unfreeze_all = True)
            
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001, weight_decay=0.01)
            
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"training: loss: {train_loss:.4f}, acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"val: loss: {val_loss:.4f}, acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        #TO-DO: ADD MODEL SAVING/CHECKPOINT DIRECTORY


def main():
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #ADAMW OPTIMIZER - look up param info - update with unfreezing schedule
    criterion = DiarizationLoss(triplet_lambda=0.3, cross_entropy_lambda=0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)
    
    ##LOAD CHECKPOINTS
    audio_model = None
    visual_model = None
    
    fusion_model = ConcatenationFusionModel(
        audio_model=audio_model,
        visual_model = visual_model,
        fusion_dim = 512,
        embedding_dim = 256)