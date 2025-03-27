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
    """Compute RMSE for a single file."""
    length = min(len(gt_counts), len(pred_counts))
    mse = np.mean((gt_counts[:length] - pred_counts[:length]) ** 2)
    return np.sqrt(mse)


# --- Main logic ---

model_folders = [
    "aws_transcribe_rttms",
    "diaper_rttms",
    "NEMO_output_rttms",
    "powerset",
    "pyannote_rttm_output/few_val"
]

# Load and normalize GT RTTM
gt_df = load_rttm("few.val.rttm")
gt_df["FileID"] = gt_df["FileID"].apply(lambda x: str(x).zfill(5))
gt_counts_dict = rttm_to_speaker_counts(gt_df)

# print("Ground truth file IDs loaded:", list(gt_counts_dict.keys())[:5], "...")

model_rmse = {}

for model_folder in model_folders:
    model_name = model_folder.strip("/")
    print(f"\n*********** Evaluating model: {model_name} ***********")
    rmses = []
    matched_files = 0

    if model_name == "powerset":
        pred_df = load_rttm(f"{model_folder}/powerset.rttm")
        pred_df["FileID"] = pred_df["FileID"].apply(lambda x: str(x).zfill(5))
        pred_counts_dict = rttm_to_speaker_counts(pred_df)

        for file_id in pred_counts_dict:
            if file_id in gt_counts_dict:
                rmse = compute_rmse(gt_counts_dict[file_id], pred_counts_dict[file_id])
                # print(f"File {file_id}: RMSE {rmse:.4f}")
                rmses.append(rmse)
                matched_files += 1
            else:
                print(f"Skipping: No GT for {file_id}")

    else:
        for pred_file in glob(f"{model_folder}/*.rttm"):
            file_id = basename(pred_file).replace(".rttm", "")
            file_id = file_id.zfill(5)

            # print(f"Checking file: {file_id}")

            if file_id not in gt_counts_dict:
                print(f"Skipping: No matching GT for {file_id}")
                continue

            pred_df = load_rttm(pred_file)
            pred_df["FileID"] = pred_df["FileID"].apply(lambda x: str(x).zfill(5))
            pred_counts_dict = rttm_to_speaker_counts(pred_df)

            pred_counts = pred_counts_dict.get(file_id, np.zeros_like(gt_counts_dict[file_id]))

            rmse = compute_rmse(gt_counts_dict[file_id], pred_counts)
            if np.isnan(rmse):
                print(f"File {file_id} has NaN RMSE (likely empty RTTM or frame mismatch)")
            else:
                rmses.append(rmse)
                matched_files += 1
    

    avg_rmse = np.mean(rmses) if rmses else float("nan")
    model_rmse[model_name] = avg_rmse

    print(f"Avg RMSE for {model_name}: {avg_rmse:.4f} over {matched_files} files")

# summary
print("\n=== Average RMSE for each model ===")
for model, score in model_rmse.items():
    print(f"{model}: {score:.4f}")

    pyannote_files = glob("pyannote_rttm_output/few_val/*.rttm")
    pyannote_ids = [basename(f).replace(".rttm", "") for f in pyannote_files]

with open("rmse_results.txt", "w") as f:
    f.write("\n=== Average RMSE for each model ===\n")
    for model, score in model_rmse.items():
        f.write(f"{model}: {score:.4f}\n")
