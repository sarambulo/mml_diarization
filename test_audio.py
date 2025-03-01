import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from datasets.MSDWild import MSDWildFrames
from audio_train import AudioOnlyTDNN, AudioOnlyDataset, collate_fn
from utils.data import rttm_to_annotations  # Ensure correct import
from pathlib import Path


def speaker_vector_to_rttm(speaker_vectors, frame_duration, file_id):
    """
    Converts speaker activation vectors to RTTM format.

    Args:
        speaker_vectors (np.ndarray): Array of shape [num_frames, num_speakers] with binary speaker activations.
        frame_duration (float): Duration of each frame in seconds.
        file_id (str): Identifier for the video.

    Returns:
        list of str: RTTM formatted lines.
    """
    rttm_lines = []
    num_frames, num_speakers = speaker_vectors.shape

    for speaker_id in range(num_speakers):
        active_segments = []
        active = False
        start_time = 0

        for frame_idx in range(num_frames):
            if speaker_vectors[frame_idx, speaker_id] == 1:
                if not active:
                    start_time = frame_idx * frame_duration
                    active = True
            else:
                if active:
                    end_time = frame_idx * frame_duration
                    active_segments.append((start_time, end_time - start_time))
                    active = False

        # Handle case where the last frame is active
        if active:
            active_segments.append((start_time, num_frames * frame_duration - start_time))

        # Convert to RTTM format
        for start_time, duration in active_segments:
            rttm_line = f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
            rttm_lines.append(rttm_line)

    return rttm_lines


def collect_predictions_and_labels(model, test_loader, device, frame_duration, output_dir):
    """Run inference, generate RTTM files for each video"""
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue

            inputs, labels = batch
            inputs = inputs.to(device)

            # Get model predictions
            outputs = model(inputs)  # Shape: [batch_size, num_speakers]
            predictions = (outputs >= 0.5).cpu().numpy()  # Convert to binary (1 for active speaker, 0 otherwise)

            # Extract file_id (assumes dataset has an ordered mapping)
            file_id = test_loader.dataset.msd_dataset.video_names[batch_idx]

            # Convert to RTTM format
            rttm_lines = speaker_vector_to_rttm(predictions, frame_duration, file_id)

            # Save RTTM file
            rttm_file = os.path.join(output_dir, f"{file_id}.rttm")
            with open(rttm_file, "w") as f:
                f.write("\n".join(rttm_lines))

            print(f"Saved RTTM: {rttm_file}")


def main():
    # Setup
    data_path = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample"
    model_path = "checkpoints/best_model.pth"
    output_dir = "predictions_rttm"
    frame_duration = 0.02  # 20ms per frame (example, adjust based on dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = MSDWildFrames(data_path, "all")
    test_audio_dataset = AudioOnlyDataset(test_dataset)

    # Create test dataloader
    test_loader = DataLoader(
        test_audio_dataset,
        batch_size=32,
        shuffle=False,  # Keep original order
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load model
    model = AudioOnlyTDNN(input_dim=40, num_speakers=10).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nGenerating RTTM files...")
    collect_predictions_and_labels(model, test_loader, device, frame_duration, output_dir)

    print(f"\nRTTM files saved in: {output_dir}")


if __name__ == "__main__":
    main()
