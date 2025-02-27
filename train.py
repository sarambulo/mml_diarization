from tqdm import tqdm
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

def get_metrics(logits, labels):
    pred_labels = torch.argmax(logits, dim=-1)
    n = labels.shape[0]
    n_correct = pred_labels[pred_labels==labels].shape[0]
    accuracy = n_correct / n

def train_epoch(model, dataloader, optimizer, criterion, get_metrics, lr_scheduler, scaler, device, config):
    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, batch in enumerate(dataloader):
        anchors, positive_pairs, negative_pairs, labels = batch
        optimizer.zero_grad() # Zero gradients

        # Join all inputs
        batch_size = labels.shape[0]
        features = list(zip([anchors, positive_pairs, negative_pairs, labels]))

        # send to cuda
        for feature in features:
            feature = torch.concat(feature, dim=0).to(device)
        labels = labels.to(device)

        # forward
        with torch.amp.autocast(DEVICE):  # This implements mixed precision. Thats it!
            embedding, logit = model(features)

            # Use the type of output depending on the loss function you want to use
            loss = criterion(logit, labels)

            # Get performance metrics
            metrics = get_metrics()

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update()

        # tqdm lets you add some details so you can monitor training as you train.
        # batch_bar.set_postfix(
        #     acc   = "{:.04f}% ({:.04f})".format(acc, acc_m.avg),
        #     loss  = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
        #     lr    = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

    # You may want to call some schedulers inside the train function. What are these?
    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return metrics, loss

@torch.no_grad()
def evaluate(model, dataloader, device, config):

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val Cls.', ncols=5)

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs['out'], labels)

        # metrics
        acc = accuracy(outputs['out'], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc         = "{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg))

        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg

def main():
    start_epoch = epoch
    final_epoch = epoch + config['epochs']
    for epoch in range(start_epoch, final_epoch):
        # epoch
        print("\nEpoch {}/{}".format(epoch+1, final_epoch))

        # train
        curr_lr = float(scheduler.get_last_lr()[0])
        metrics.update({'lr': curr_lr})
        train_cls_acc, train_loss = train_epoch(model, cls_train_loader, optimizer, scheduler, scaler, DEVICE, config)
        print("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr))
        metrics.update({
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
        })
        # classification validation
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, cls_val_loader, DEVICE, config)
            print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })

        # retrieval validation
        valid_ret_metrics = valid_epoch_ver(model, ver_val_loader, DEVICE, config)
        valid_ret_acc = valid_ret_metrics['ACC']
        print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
        metrics.update({
            'valid_ret_acc': valid_ret_acc, 'valid_ret_eer': valid_ret_metrics['EER']
        })

        # save best model
        if eval_cls:
            if valid_cls_acc >= best_valid_cls_acc:
                best_valid_cls_acc = valid_cls_acc
                model_filename = os.path.join(config['checkpoint_dir'], 'best_cls.pth')
                save_model(model, optimizer, scheduler, metrics, epoch, model_filename)
                wandb.save(model_filename)
                print("Saved best classification model")

        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            model_filename = os.path.join(config['checkpoint_dir'], 'best_ret.pth')
            save_model(model, optimizer, scheduler, metrics, epoch, model_filename)
            wandb.save(model_filename)
            print("Saved best retrieval model")

        # log to tracker
        if run is not None:
            run.log(metrics)