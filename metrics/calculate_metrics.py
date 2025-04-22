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
PATH_TO_TARGETS = "rmse/few.val.rttm"



def main():
    root_predictions = Path(PATH_TO_PREDS)
    model_folders = {
        "visual": root_predictions / "visual_rttms",
        "audio": root_predictions / "audio_rttms",
        "MM Concat": root_predictions / "mm_concat_rttms",
        "MM MatMul": root_predictions / "mm_tensor_rttms",
        "MM Additive": root_predictions / "mm_additive_rttms",
        "AWS Transcribe": root_predictions / "aws_transcribe_rttms",
        "Diaper": root_predictions / "diaper_rttms",
        "NVIDIA NeMo": root_predictions / "NEMO_rttms",
        "Powerset": root_predictions / "powerset_rttms",
        "Pyannote": root_predictions / "pyannote_rttms",
    }

    # Load and normalize ground truth
    raw_gt = rttm_to_annotations(PATH_TO_TARGETS)
    targets_by_video = {str(k).zfill(5): v for k, v in raw_gt.items()}

    # print(f"\nLoaded {len(targets_by_video)} ground truth entries")
    # print("Sample GT FileIDs:", list(targets_by_video.keys())[:10])

    # print("\nGround truth keys after normalization:")
    # for gt_id in list(targets_by_video.keys()):
    #     print(f"- {gt_id}")
    metrics_by_model = {}

    for model_name, model_folder in model_folders.items():
        print(f"\n*********** Evaluating model: {model_name} ***********")
        preds_by_video = {}
        matched_files = 0

        for pred_file in glob(f"{model_folder}/*.rttm"):
            pred_ann_dict = rttm_to_annotations(pred_file)

            # print(f"Checking predictions in file: {pred_file}")
            # print("FileIDs in prediction file:", list(pred_ann_dict.keys()))

            for file_id, ann in pred_ann_dict.items():
                norm_file_id = str(file_id).zfill(5)

                if norm_file_id in targets_by_video:
                    preds_by_video[norm_file_id] = ann
                    matched_files += 1
                else:
                    print(f"Skipping: No matching ground truth for {norm_file_id}")

        print(f"Matched {matched_files} files for {model_name}")

        model_metrics = calculate_metrics_for_dataset(
            preds_dict=preds_by_video,
            targets_dict=targets_by_video,
        )
        metrics_by_model[model_name] = model_metrics

    with open("intrinsic_results.txt", "w") as f:
        dump_json(metrics_by_model, f, indent=2)

    print("\n=== Final Evaluation Summary ===")
    for model, metrics in metrics_by_model.items():
        print(f"\n{model} Metrics:")
        for metric_name, value in metrics.items():
            if metric_name == "metricsByVideo":
                continue  # skip per-video details
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()


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
