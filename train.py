from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from audio_train import create_rttm_file, evaluate_model
from datasets.MSDWild import MSDWildFrames, MSDWildVideos
from models.VisualOnly import VisualOnlyModel
from pathlib import Path
from losses.DiarizationLoss import DiarizationLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints"


@torch.no_grad()
def get_metrics(logits, labels):
    pred_labels = torch.argmax(logits, dim=-1)
    n = labels.shape[0]
    n_correct = pred_labels[pred_labels == labels].shape[0]
    accuracy = n_correct / n
    return accuracy


def save_model(model, metrics, epoch, path):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()

    # Progress Bar
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
        anchors, positive_pairs, negative_pairs, labels = batch
        optimizer.zero_grad()  # Zero gradients

        # Join all inputs
        batch_size = labels.shape[0]
        features = list(zip(anchors, positive_pairs, negative_pairs))

        # feature: [(batch_size, ...), (batch_size, ...), (batch_size, ...)]
        # send to cuda
        for index, feature in enumerate(features):
            features[index] = torch.concat(feature, dim=0).to(DEVICE)
            # feature: (batch_size * 3, ...)
        labels = labels.to(DEVICE)

        # forward
        with torch.amp.autocast(DEVICE):  # This implements mixed precision. Thats it!
            embeddings, logits = model(features)
            anchors = embeddings[:batch_size]
            positive_pairs = embeddings[batch_size : 2 * batch_size]
            negative_pairs = embeddings[2 * batch_size : 3 * batch_size]
            logits = logits[:batch_size]
            # Use the type of output depending on the loss function you want to use

            loss = criterion(anchors, positive_pairs, negative_pairs, logits, labels)

        loss.backward()  # This is a replacement for loss.backward()
        optimizer.step()  # This is a replacement for optimizer.step()

        accuracy = get_metrics(logits, labels)
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_accuracy = (avg_accuracy * i + accuracy) / (i + 1)
        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04%} ({:.04%})".format(accuracy, avg_accuracy),
            loss="{:.04f} ({:.04f})".format(loss.item(), avg_loss),
            lr="{:.06f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()

    return accuracy, loss


def main(model_name, epochs, batch_size, learning_rate, subset):
    print("Device: ", DEVICE)
    # Configuration
    if model_name == "VisualOnlyModel":
        model = VisualOnlyModel(512, 2)
        model = model.to(DEVICE)
    else:
        raise ValueError(f"Invalid model name {model_name}")
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = DiarizationLoss(0.5, 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=2, threshold=0.01
    )
    # Load data
    train_dataset = MSDWildFrames("data", "few_train", None, subset)
    val_dataset = MSDWildVideos("data", "many_val", None, subset)
    train_dataloader = DataLoader(
        train_dataset, batch_size, True, collate_fn=train_dataset.build_batch
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size, False, collate_fn=val_dataset.build_batch
    )
    # Training process
    start_epoch = 0
    final_epoch = epochs
    metrics = {}
    best_valid_acc = 0
    for epoch in range(start_epoch, final_epoch):
        print("\nEpoch {}/{}".format(epoch + 1, final_epoch))
        # train
        curr_lr = float(scheduler.get_last_lr()[0])
        metrics.update({"lr": curr_lr})
        train_acc, train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion
        )
        print(
            "\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, final_epoch, train_acc, train_loss, curr_lr
            )
        )
        metrics.update(
            {
                "train_cls_acc": train_acc,
                "train_loss": train_loss,
            }
        )
        # validation
        valid_acc, valid_loss = 0, 0  # TODO: evaluate_model(model, val_dataloader)
        print(
            "Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(
                valid_acc, valid_loss
            )
        )
        metrics.update(
            {
                "valid_cls_acc": valid_acc,
                "valid_loss": valid_loss,
            }
        )

        # save best model
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            model_path = Path(CHECKPOINT_PATH, f"best_{model_name}.pth")
            save_model(model, metrics, epoch, model_path)
            print("Saved best model")

        # You may want to call some schedulers inside the train function. What are these?
        if scheduler is not None:
            scheduler.step(valid_loss)

    # save last model
    model_path = Path(CHECKPOINT_PATH, f"last_{model_name}.pth")
    save_model(model, metrics, epoch, model_path)
    print("Saved best model")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train procedure")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Maximum number of epochs"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "-s", "--subset", type=float, default=0.01, help="Subset of the data to use"
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset=args.subset,
    )
