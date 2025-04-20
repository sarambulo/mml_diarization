from typing import List, Dict
import matplotlib.pyplot as plt
from .overlapping_speakers_confusion import parse_rttm
import numpy as np
import os
from pathlib import Path


def plot_speaker_timeline_clamped(
    rttm_path,
    start_time,
    end_time,
    num_speakers,
    speaker_mapping=None,
    output_file=None,
):
    """
    Visualize speaker segments from RTTM file with timeline clamped to a specific time range.

    Args:
        rttm_path (str): Path to RTTM file.
        start_time (float): Start time of the desired range (in seconds).
        end_time (float): End time of the desired range (in seconds).
        output_file (str): Optional path to save visualization.
    """
    # Extract intervals from rttm
    segments = parse_rttm(rttm_path)

    # Filter segments based on the specified time range
    clamped_segments = []
    true_num_speakers = num_speakers
    for start, end, speaker in segments:
        if end > start_time and start < end_time:
            if speaker_mapping:
                if speaker in speaker_mapping:
                    speaker = speaker_mapping[speaker]
                else:
                    num_speakers += 1
                    speaker = num_speakers
            clamped_segments.append(
                (max(start, start_time), min(end, end_time), speaker)
            )

    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(2, 4))

    # Track y positions and speaker colors
    speakers = list(range(num_speakers))
    colors = plt.cm.get_cmap("viridis", num_speakers)

    # Plot each segment
    for idx, (start, end, speaker) in enumerate(clamped_segments):
        y_pos = int(speaker)
        ax.broken_barh(
            [(start, end - start)],
            (y_pos - 0.4, 0.8),
            facecolors=colors(y_pos) if int(speaker) < true_num_speakers else 'gray',
            edgecolor="black",
            linewidth=0.5,
        )

    # Format plot
    ax.set_xlim(start_time, end_time)  # Clamp x-axis to the specified range
    ax.set_yticks(range(num_speakers))
    ax.set_yticklabels(range(num_speakers))
    ax.set_ylim(-0.6, num_speakers - 0.2)
    ax.set_ylabel("Speaker ID")
    ax.set_xlabel("Time (seconds)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Add legend
    # handles = [
    #     plt.Rectangle((0, 0), 1, 1, color=colors(i / len(speakers)))
    #     for i in range(len(speakers))
    # ]
    # ax.legend(handles, speakers, bbox_to_anchor=(1.05, 1), loc="upper left")

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_all_models(
    videoId: str,
    start: float,
    end: float,
    num_speakers: int,
    rttms_paths: Dict[str, str],
    output_path,
    speaker_mappings: dict = None,
):
    """
    rttms_paths: Dictionary with model_name as key and path to rttms as value
    """
    for model_name in rttms_paths:
        rttm_path = Path(rttms_paths[model_name], f"{videoId}.rttm")
        output_file = Path(output_path, f"{model_name}.png")
        plot_speaker_timeline_clamped(
            rttm_path=rttm_path,
            start_time=start,
            end_time=end,
            num_speakers=num_speakers,
            speaker_mapping=speaker_mappings[model_name],
            output_file=output_file,
        )
