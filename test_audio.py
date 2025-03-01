import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate, JaccardErrorRate
from datasets.MSDWild import MSDWildFrames
from audio_train import AudioOnlyTDNN, AudioOnlyDataset, collate_fn
from utils.metrics import calculate_metrics_for_video

# File paths
PREDICTIONS_RTTM = "predictions.rttm"
TARGETS_RTTM = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample/all.rttm"
MODEL_PATH = "checkpoints/best_model.pth"
DATA_PATH = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample/"
OUTPUT_DIR = "predictions"

# Convert binary predictions into pyannote.core.Annotation format
def convert_to_annotation(predictions, frame_duration=0.02):
    annotation = Annotation()
    num_frames, num_speakers = predictions.shape

    for frame_idx in range(num_frames):
        start_time = frame_idx * frame_duration
        end_time = (frame_idx + 1) * frame_duration
        active_speakers = np.where(predictions[frame_idx] == 1)[0]

        for speaker in active_speakers:
            annotation[Segment(start_time, end_time)] = str(speaker)

    return annotation

# Load RTTM annotations from file
def rttm_to_annotations(path):
    annotations = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 10 and fields[0] == "SPEAKER":
                file_id, start, duration, speaker = (
                    fields[1],
                    float(fields[3]),
                    float(fields[4]),
                    fields[7],
                )
                if file_id not in annotations:
                    annotations[file_id] = Annotation()
                annotations[file_id][Segment(start, start + duration)] = speaker
    return annotations

# Run inference and collect model predictions
def collect_predictions_and_labels(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            inputs, labels = batch
            inputs = inputs.to(device)

            # Get model predictions
            outputs = model(inputs)  # Shape: [batch_size, num_speakers]
            predictions = (outputs >= 0.5).cpu().numpy()  # Convert probabilities to binary decisions

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    return np.array(all_predictions), np.array(all_labels)

def main():
    print("Loading test dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_dataset = MSDWildFrames(DATA_PATH, "all")
    test_audio_dataset = AudioOnlyDataset(test_dataset)

    # Create dataloader
    test_loader = DataLoader(
        test_audio_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load trained model
    print(f"Loading model from {MODEL_PATH}")
    model = AudioOnlyTDNN(input_dim=40, num_speakers=10).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run inference
    print("Running inference...")
    predictions, ground_truth = collect_predictions_and_labels(model, test_loader, device)

    # Convert predictions and ground truth to pyannote Annotation format
    preds_annotation = convert_to_annotation(predictions)
    targets_annotation = rttm_to_annotations(TARGETS_RTTM)

    print("Calculating metrics...")
    metrics = calculate_metrics_for_video(preds_annotation, targets_annotation)

    # Print results
    print("\nDiarization Results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Save predictions in RTTM format
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions_file = os.path.join(OUTPUT_DIR, "predictions.rttm")

    with open(predictions_file, "w") as f:
        for segment, speaker in preds_annotation.itertracks():
            f.write(f"SPEAKER test {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

    print(f"\nPredictions saved to {predictions_file}")

if __name__ == "__main__":
    main()
