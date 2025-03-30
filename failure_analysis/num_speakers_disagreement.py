import os
import statistics
from collections import defaultdict

rttm_dirs = [
    "NEMO_rttms",
    "pyannote_rttms",
    "diaper_rttms",
    "powerset_rttms",
    "aws_transcribe_rttms",
]
top_n = 20
if __name__ == "__main__":
    video_counts = defaultdict(list)

    # Process each model's directory
    for model_dir in rttm_dirs:
        model_dir = os.path.join("rmse", model_dir)
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".rttm"):
                    file_path = os.path.join(root, file)
                    speakers = set()
                    video_id = None
                    # print(file_path)
                    # Parse RTTM file
                    with open(file_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8 and parts[0] == "SPEAKER":
                                # Extract video ID from second column
                                video_id = parts[1]
                                # Extract speaker ID from eighth column
                                speakers.add(parts[7])

                    if video_id and speakers:
                        video_counts[video_id].append(len(speakers))

    # Calculate variance for each video
    discrepancies = []
    for vid, counts in video_counts.items():
        if len(counts) > 1:
            try:
                var = statistics.variance(counts)
                discrepancies.append((vid, var, counts))
            except statistics.StatisticsError:
                continue

    # Sort by descending variance
    discrepancies.sort(key=lambda x: x[1], reverse=True)
    top_results = discrepancies[:top_n]

    for idx, (vid, var, counts) in enumerate(top_results, 1):
        print(f"{idx}. {vid}: Variance {var:.2f} (Counts: {counts})")
