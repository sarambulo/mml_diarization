import matplotlib.pyplot as plt
from overlapping_speakers_confusion import parse_rttm
import numpy as np
import os


def plot_speaker_timeline_clamped(rttm_path, start_time, end_time, output_file=None):
    """
    Visualize speaker segments from RTTM file with timeline clamped to a specific time range.

    Args:
        rttm_path (str): Path to RTTM file.
        start_time (float): Start time of the desired range (in seconds).
        end_time (float): End time of the desired range (in seconds).
        output_file (str): Optional path to save visualization.
    """
    # Read RTTM using speechbrain's built-in function
    segments = parse_rttm(rttm_path)

    # Filter segments based on the specified time range
    clamped_segments = [
        (max(start, start_time), min(end, end_time), speaker)
        for start, end, speaker in segments
        if end > start_time and start < end_time
    ]

    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(15, 3))

    # Track y positions and speaker colors
    speakers = list({seg[2] for seg in clamped_segments})
    colors = plt.cm.get_cmap("viridis", len(speakers))

    # Plot each segment
    for idx, (start, end, speaker) in enumerate(clamped_segments):
        y_pos = speakers.index(speaker)
        ax.broken_barh(
            [(start, end - start)],
            (y_pos - 0.4, 0.8),
            facecolors=colors(y_pos / len(speakers)),
            edgecolor="black",
            linewidth=0.5,
        )

    # Format plot
    ax.set_xlim(start_time, end_time)  # Clamp x-axis to the specified range
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors(i / len(speakers)))
        for i in range(len(speakers))
    ]
    ax.legend(handles, speakers, bbox_to_anchor=(1.05, 1), loc="upper left")

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")

    else:
        plt.show()


def visualize_all_models(videoId, start, end):
    model_directories = [
        "true",
        "NEMO_rttms",
        "pyannote_rttms",
        "diaper_rttms",
        "powerset_rttms",
        "aws_transcribe_rttms",
    ]
    os.makedirs(f"data/failures/{videoId}_{start}_{end}", exist_ok=True)
    for model_dir in model_directories:
        # Example usage
        plot_speaker_timeline_clamped(
            f"rmse/{model_dir}/{videoId}.rttm",
            start_time=start,
            end_time=end,
            output_file=f"data/failures/{videoId}_{start}_{end}/{model_dir}.png",
        )


visualize_all_models("02749", start=0, end=22)
