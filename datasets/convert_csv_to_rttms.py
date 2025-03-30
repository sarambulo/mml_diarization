import os
import pandas as pd
from pathlib import Path

FRAME_DURATION = 0.25  # seconds per frame

def convert_csvs_to_rttm(root_dir, output_rttm="ground_truth.rttm"):
    """
    Traverse the preprocessed directory structure and convert all is_speaking.csv files to RTTM lines.
    """
    root = Path(root_dir)
    rttm_lines = []

    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name

        for chunk_dir in sorted(video_dir.glob("Chunk_*")):
            chunk_id = chunk_dir.name
            csv_path = chunk_dir / "is_speaking.csv"

            if not csv_path.exists():
                print(f"Skipping {csv_path} (missing)")
                continue

            df = pd.read_csv(csv_path)
            df = df[df["is_speaking"] == 1]  # Filter active speech frames

            for speaker_id in df["face_id"].unique():
                speaker_frames = sorted(df[df["face_id"] == speaker_id]["frame_id"].tolist())

                # Group consecutive frames into segments
                segments = []
                if speaker_frames:
                    start = speaker_frames[0]
                    end = start
                    for idx in speaker_frames[1:]:
                        if idx == end + 1:
                            end = idx
                        else:
                            segments.append((start, end))
                            start = idx
                            end = idx
                    segments.append((start, end))  # Final segment

                    for start, end in segments:
                        start_time = start * FRAME_DURATION
                        duration = (end - start + 1) * FRAME_DURATION
                        rttm_line = f"SPEAKER {video_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker{speaker_id} <NA> <NA>"
                        rttm_lines.append(rttm_line)

    # Save to RTTM file
    with open(output_rttm, "w") as f:
        for line in rttm_lines:
            f.write(line + "\n")

    print(f"✅ Ground truth RTTM saved to: {output_rttm}")

convert_csvs_to_rttm(root_dir="../test_preprocessed", output_rttm="ground_truth.rttm")
