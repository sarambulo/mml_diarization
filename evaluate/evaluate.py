from inference import create_rttm_file
from utils.metrics import rttm_to_annotations, calculate_metrics_for_dataset
import torch
from models.VisualOnly import VisualOnlyModel
from datasets.MSDWild import MSDWildVideos
from torch.utils.data import DataLoader
from pathlib import Path


def evaluate(model, test_loader, output_rttm_path, ground_truth_path):
    model.eval()
    device = next(model.parameters()).device
    with Path(output_rttm_path).open("w") as output_file:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if batch is None:
                    continue
                # all_video_frames, all_audio_segments, all_labels, all_faces, all_timestamps, self.video_names[index]
                all_video_frames, all_audio_segments, all_faces = (
                    batch[0],
                    batch[1],
                    batch[3],
                )
                all_labels = batch[2]
                timestamps = batch[4]
                all_video_frames = [
                    video_frame.to(device) for video_frame in all_video_frames
                ]
                all_audio_segments = [
                    audio_segment.to(device) for audio_segment in all_audio_segments
                ]
                all_faces = [[face.to(device) for face in faces] for faces in all_faces]
                all_labels = [labels.to(device) for labels in all_labels]
                file_ids = batch[5]
                rttm_lines = model.predict_to_rttm_full(
                    (all_video_frames, all_audio_segments, all_faces), file_ids
                )
                # Log output
                output_file.writelines(rttm_lines)

    # Calculate metrics
    preds = rttm_to_annotations(output_rttm_path)
    targets = rttm_to_annotations(ground_truth_path)

    metrics = calculate_metrics_for_dataset(preds, targets)

    print("\nEvaluation Metrics:")
    print(f"DER: {metrics['DER']:.4f}")
    print(f"JER: {metrics['JER']:.4f}")
    print(f"Missed Speech Rate: {metrics['MSR']:.4f}")
    print(f"False Alarm Rate: {metrics['FAR']:.4f}")
    print(f"Speaker Error Rate: {metrics['SER']:.4f}")

    return metrics


if __name__ == "__main__":
    # Setup model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisualOnlyModel(512, 2)
    model = model.to(DEVICE)
    model_filename = "best_VisualOnlyModel.pth"
    model_path = Path("checkpoints", model_filename)
    checkpoint = torch.load(model_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Data
    val_dataset = MSDWildVideos("data_sample", "many_val", None, 1, max_frames=30)
    # val_dataloader = DataLoader(val_dataset, 1, False)
    metrics = evaluate(
        model=model,
        test_loader=val_dataset,
        output_rttm_path=Path("predictions", "output.rttm"),
        ground_truth_path=Path("data_sample/few_val.rttm"),
    )
    print(metrics)
