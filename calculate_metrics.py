import pandas as pd
import numpy as np
from glob import glob
from os.path import basename
from utils.rttm import rttm_to_annotations
from utils.metrics import calculate_metrics_for_dataset
from json import dump as dump_json
from pathlib import Path

FRAME_SIZE = 0.25  # 4 frames per second

PATH_TO_PREDS = "predictions"
PATH_TO_TARGETS = "data/few_val.rttm"


def main():
    root_predictions = Path(PATH_TO_PREDS)
    # --- Main logic ---
    model_folders = {
        "AWS Transcribe": root_predictions / "aws_transcribe_rttms",
        "Diaper": root_predictions / "diaper_rttms",
        "NVIDIA NeMo": root_predictions / "NEMO_rttms",
        "Powerset": root_predictions / "powerset_rttms",
        "Pyannote": root_predictions / "pyannote_rttms",
    }

    # Load ground truth
    targets_by_video = rttm_to_annotations(PATH_TO_TARGETS)

    # Load models and compare with ground truth
    metrics_by_model = {}
    for model_name in model_folders:
        print(f"\n*********** Evaluating model: {model_name} ***********")
        # Load predictions
        model_folder = model_folders[model_name]
        matched_files = 0
        preds_by_video = {}
        for pred_file in glob(f"{model_folder}/*.rttm"):
            # Add the annotations for this file
            file_id = basename(pred_file).replace(".rttm", "").zfill(5)
            if file_id in targets_by_video:
                preds_by_video.update(rttm_to_annotations(pred_file))
                matched_files += 1
            else:
                print(f"Skipping: No matching ground truth for {file_id}")
                continue

        # Get dictionary of metrics
        model_metrics = calculate_metrics_for_dataset(
            preds_dict=preds_by_video,
            targets_dict=targets_by_video,
        )

        # Dictionary for model results
        metrics_by_model[model_name] = model_metrics

    with open("intrinsic_results.txt", "w") as f:
        dump_json(metrics_by_model, f, indent=2)
    print(metrics_by_model)


def load_rttm(file_path):
    """Load RTTM file into a dataframe."""
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=[
            "Type",
            "FileID",
            "Channel",
            "Start",
            "Duration",
            "NA1",
            "NA2",
            "Speaker",
            "NA3",
            "NA4",
        ],
    )
    df["End"] = df["Start"] + df["Duration"]
    return df


def rttm_to_speaker_counts(df, frame_size=FRAME_SIZE):
    """Convert RTTM to per-frame speaker count for each file."""
    file_to_counts = {}
    for file_id, group in df.groupby("FileID"):
        end_time = group["End"].max()
        num_frames = int(np.ceil(end_time / frame_size))
        counts = np.zeros(num_frames)
        for _, row in group.iterrows():
            start_idx = int(np.floor(row["Start"] / frame_size))
            end_idx = int(np.ceil(row["End"] / frame_size))
            counts[start_idx:end_idx] += 1
        file_to_counts[file_id] = counts
    return file_to_counts


def compute_rmse(gt_counts, pred_counts):
    length = min(len(gt_counts), len(pred_counts))
    mse = np.mean((gt_counts[:length] - pred_counts[:length]) ** 2)
    return np.sqrt(mse)


def compute_temporal_metrics(counts_dict):
    """Compute speaker turn count, overlap ratio, and temporal coverage."""
    metrics = {}
    for file_id, counts in counts_dict.items():
        binary_activity = (counts > 0).astype(int)
        turn_count = np.sum(np.abs(np.diff(binary_activity)))
        overlap_ratio = np.mean(counts > 1)
        coverage = np.mean(counts > 0)
        metrics[file_id] = {
            "turns": turn_count,
            "overlap_ratio": overlap_ratio,
            "coverage": coverage,
        }
    return metrics


def average_metrics(metrics_dict):
    """Average across all files for a model."""
    if not metrics_dict:
        return {"turns": np.nan, "overlap_ratio": np.nan, "coverage": np.nan}
    turns = np.mean([m["turns"] for m in metrics_dict.values()])
    overlap = np.mean([m["overlap_ratio"] for m in metrics_dict.values()])
    coverage = np.mean([m["coverage"] for m in metrics_dict.values()])
    return {"turns": turns, "overlap_ratio": overlap, "coverage": coverage}


def compute_missed_speech(gt_counts_dict, pred_counts_dict):
    """Compute Missed Speech (MS) for each file."""
    ms_values = []
    for file_id in gt_counts_dict:
        gt_counts = gt_counts_dict[file_id]
        pred_counts = pred_counts_dict.get(file_id, np.zeros_like(gt_counts))
        length = min(len(gt_counts), len(pred_counts))
        gt_speech = (gt_counts[:length] > 0).astype(int)
        pred_speech = (pred_counts[:length] > 0).astype(int)
        missed = np.logical_and(gt_speech == 1, pred_speech == 0).sum()
        total_gt_speech = gt_speech.sum()
        if total_gt_speech > 0:
            ms = missed / total_gt_speech
            ms_values.append(ms)
    return np.mean(ms_values) if ms_values else float("nan")


if __name__ == "__main__":
    main()
