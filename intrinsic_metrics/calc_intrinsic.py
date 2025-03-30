import pandas as pd
import numpy as np
from glob import glob
from os.path import basename

FRAME_SIZE = 0.25  # 4 frames per second

def load_rttm(file_path):
    """Load RTTM file into a dataframe."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None,
                     names=["Type", "FileID", "Channel", "Start", "Duration",
                            "NA1", "NA2", "Speaker", "NA3", "NA4"])
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
            "coverage": coverage
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


# --- Main logic ---

model_folders = [
    "aws_transcribe_rttms",
    "diaper_rttms",
    "NEMO_output_rttms",
    "powerset",
    "pyannote_rttm_output/few_val"
]

gt_df = load_rttm("few.val.rttm")
gt_df["FileID"] = gt_df["FileID"].apply(lambda x: str(x).zfill(5))
gt_counts_dict = rttm_to_speaker_counts(gt_df)
gt_metrics = compute_temporal_metrics(gt_counts_dict)
avg_gt_metrics = average_metrics(gt_metrics)

print("\n=== Ground Truth Metrics ===")
print(f"Avg Speaker Turns: {avg_gt_metrics['turns']:.2f}")
print(f"Avg Overlap Ratio: {avg_gt_metrics['overlap_ratio']:.4f}")
print(f"Avg Temporal Coverage: {avg_gt_metrics['coverage']:.4f}")

model_results = {}

for model_folder in model_folders:
    model_name = model_folder.strip("/")
    print(f"\n*********** Evaluating model: {model_name} ***********")
    rmses = []
    matched_files = 0
    pred_counts_dict = {}

    if model_name == "powerset":
        pred_df = load_rttm(f"{model_folder}/powerset.rttm")
        pred_df["FileID"] = pred_df["FileID"].apply(lambda x: str(x).zfill(5))
        pred_counts_dict = rttm_to_speaker_counts(pred_df)
    else:
        for pred_file in glob(f"{model_folder}/*.rttm"):
            file_id = basename(pred_file).replace(".rttm", "").zfill(5)
            if file_id not in gt_counts_dict:
                print(f"Skipping: No matching GT for {file_id}")
                continue
            pred_df = load_rttm(pred_file)
            pred_df["FileID"] = pred_df["FileID"].apply(lambda x: str(x).zfill(5))
            single_df_counts = rttm_to_speaker_counts(pred_df)
            pred_counts_dict[file_id] = single_df_counts.get(file_id, np.zeros_like(gt_counts_dict[file_id]))

    for file_id in pred_counts_dict:
        if file_id in gt_counts_dict:
            rmse = compute_rmse(gt_counts_dict[file_id], pred_counts_dict[file_id])
            if not np.isnan(rmse):
                rmses.append(rmse)
                matched_files += 1

    avg_rmse = np.mean(rmses) if rmses else float("nan")
    pred_metrics = compute_temporal_metrics(pred_counts_dict)
    avg_pred_metrics = average_metrics(pred_metrics)
    avg_missed_speech = compute_missed_speech(gt_counts_dict, pred_counts_dict)

    model_results[model_name] = {
        "rmse": avg_rmse,
        "turns": avg_pred_metrics["turns"],
        "overlap": avg_pred_metrics["overlap_ratio"],
        "coverage": avg_pred_metrics["coverage"],
        "missed_speech": avg_missed_speech
    }

    print(f"Avg RMSE: {avg_rmse:.4f}")
    print(f"Avg Speaker Turns: {avg_pred_metrics['turns']:.2f}")
    print(f"Avg Overlap Ratio: {avg_pred_metrics['overlap_ratio']:.4f}")
    print(f"Avg Temporal Coverage: {avg_pred_metrics['coverage']:.4f}")
    print(f"Avg Missed Speech (MS): {avg_missed_speech:.4f}")


with open("intrinsic_results.txt", "w") as f:
    f.write("=== Average RMSE and Temporal metrics for each model ===\n")
    f.write(f"Ground Truth - Turns: {avg_gt_metrics['turns']:.2f}, Overlap: {avg_gt_metrics['overlap_ratio']:.4f}, Coverage: {avg_gt_metrics['coverage']:.4f}\n\n")
    for model, metrics in model_results.items():
        f.write(f"{model}:\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  Speaker Turns: {metrics['turns']:.2f}\n")
        f.write(f"  Overlap Ratio: {metrics['overlap']:.4f}\n")
        f.write(f"  Temporal Coverage: {metrics['coverage']:.4f}\n")
        f.write(f"  Missed Speech (MS): {metrics['missed_speech']:.4f}\n\n")
