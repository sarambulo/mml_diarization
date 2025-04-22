import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from intervaltree import IntervalTree


def parse_rttm(rttm_path):
    """Parse RTTM file into structured intervals"""
    intervals = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            intervals.append((start, end, speaker))
    return intervals


def find_overlap_conflicts(model_dirs, window_size=1.0, min_overlap=0.5):
    """
    Find video clips with maximum disagreement in overlapping speech detection

    Args:
        model_dirs: List of directories containing RTTM files from different models
        window_size: Analysis window in seconds (default: 1s)
        min_overlap: Minimum overlapping duration to consider (default: 0.5s)

    Returns:
        DataFrame of conflicting clips sorted by disagreement score
    """
    conflict_scores = defaultdict(list)

    # Process each model's RTTM files
    for model_idx, model_dir in enumerate(model_dirs):
        model_dir = os.path.join("rmse", model_dir)
        for rttm_file in glob.glob(
            os.path.join(model_dir, "**/*.rttm"), recursive=True
        ):
            video_id = os.path.basename(rttm_file).split(".")[0]
            intervals = parse_rttm(rttm_file)

            # Build interval tree for efficient overlap queries
            tree = IntervalTree()
            for start, end, speaker in intervals:
                tree.addi(start, end, speaker)

            # Analyze overlapping regions
            if intervals:
                timeline = np.arange(
                    0, max(end for _, end, _ in intervals), window_size
                )
                for window_start in timeline:
                    window_end = window_start + window_size
                    overlaps = tree.overlap(window_start, window_end)

                    # Count concurrent speakers
                    speaker_counts = defaultdict(int)
                    for interval in overlaps:
                        overlap_start = max(window_start, interval.begin)
                        overlap_end = min(window_end, interval.end)
                        if (overlap_end - overlap_start) >= min_overlap:
                            speaker_counts[interval.data] += 1

                    # Calculate overlap complexity score
                    score = 0
                    if len(speaker_counts) > 0:
                        score = len(speaker_counts) * (
                            sum(speaker_counts.values()) / len(speaker_counts)
                        )
                    conflict_scores[(video_id, window_start, window_end)].append(score)

    # Calculate disagreement metrics
    results = []
    for (vid, start, end), scores in conflict_scores.items():
        if len(scores) < len(model_dirs):
            continue  # Skip incomplete data

        disagreement = {
            "video_id": vid,
            "start_time": start,
            "end_time": end,
            "score_variance": np.var(scores),
            "max_score_diff": max(scores) - min(scores),
            "avg_speaker_count": np.mean(scores),
            "model_disagreement": len(set(np.round(scores, 1))) / len(scores),
            "results": scores,
        }
        results.append(disagreement)

    df = pd.DataFrame.from_records(results)
    return df.sort_values("score_variance", ascending=False)


def main():
    # Example usage
    model_directories = [
        "NEMO_rttms",
        "pyannote_rttms",
        "diaper_rttms",
        "powerset_rttms",
        "aws_transcribe_rttms",
    ]
    conflict_df = find_overlap_conflicts(model_directories)
    top_conflicts = conflict_df.head(10)

    # Save results
    top_conflicts.to_csv("overlap_conflicts.csv", index=False)
    print("Top conflicting clips:")
    print(top_conflicts)


if __name__ == "__main__":
    main()
