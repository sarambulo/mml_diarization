import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pyannote.core import Annotation, Segment
from datasets.MSDWild import MSDWildFrames
from audio_train import AudioOnlyTDNN, AudioOnlyDataset, collate_fn

# File paths
MODEL_PATH = "checkpoints/best_model.pth"
DATA_PATH = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample"
OUTPUT_DIR = "predictions"
FRAME_DURATION = 0.02  # Assuming each frame represents 20ms of audio


def convert_to_annotation(predictions, video_id, frame_duration=FRAME_DURATION):
    """
    Convert binary predictions into pyannote.core.Annotation format.
    """
    annotation = Annotation()
    num_frames, num_speakers = predictions.shape

    for frame_idx in range(num_frames):
        start_time = frame_idx * frame_duration
        end_time = (frame_idx + 1) * frame_duration
        active_speakers = np.where(predictions[frame_idx] == 1)[0]

        for speaker in active_speakers:
            annotation[Segment(start_time, end_time)] = str(speaker)

    return annotation


def collect_predictions(model, test_loader, device):
    """
    Run inference and collect predictions.
    """
    model.eval()
    video_predictions = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue

            inputs, video_ids = batch  # Ensure test_loader provides video_ids for each batch
            inputs = inputs.to(device)

            # Get model predictions
            outputs = model(inputs)  # Shape: [batch_size, num_speakers]
            predictions = (outputs >= 0.5).cpu().numpy()  # Convert probabilities to binary decisions

            for i, video_id in enumerate(video_ids):
                if video_id not in video_predictions:
                    video_predictions[video_id] = []
                video_predictions[video_id].append(predictions[i])

    # Convert lists to numpy arrays
    for video_id in video_predictions:
        video_predictions[video_id] = np.vstack(video_predictions[video_id])

    return video_predictions


def save_rttm(predictions_dict, output_dir):
    """
    Save predictions in RTTM format.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_id, predictions in predictions_dict.items():
        annotation = convert_to_annotation(predictions, video_id)
        output_file = os.path.join(output_dir, f"{video_id}.rttm")

        with open(output_file, "w") as f:
            for segment, speaker in annotation.itertracks():
                f.write(f"SPEAKER {video_id} {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

        print(f"Saved RTTM file: {output_file}")


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
    predictions = collect_predictions(model, test_loader, device)

    # Save RTTM files
    print("Saving RTTM files...")
    save_rttm(predictions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
