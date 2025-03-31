import numpy as np
import os
from utils.rttm import greedy_speaker_matching
from failure_analysis.visualize import visualize_all_models
from pathlib import Path
from shutil import rmtree

def main():
    ground_truth_label = 'Ground Truth'
    ground_truth_root_path = Path('rmse', "true")
    predictions_root_path = Path('predictions')
    predictions_dirs = {
        ground_truth_label: ground_truth_root_path,
        'NEMO': predictions_root_path / "NEMO_rttms",
        'pyannote': predictions_root_path / "pyannote_rttms",
        'diaper': predictions_root_path / "diaper_rttms",
        'powerset': predictions_root_path / "powerset_rttms",
        'aws_transcribe': predictions_root_path / "aws_transcribe_rttms",
    }
    # Clip = video_name, start_time, end_time
    clips = load_clips(clips_path=str(Path('failure_analysis', 'clips.txt')))

    # Clear output path
    output_root = Path('failure_analysis', 'outputs')
    if output_root.exists():
        rmtree(str(output_root))

    # For each selected clip
    for video_name, start_time, end_time, num_speakers in clips:
        # Map the speakers using the greedy proceedure
        speaker_mappings = {}
        for model in predictions_dirs:
            if model != ground_truth_label:
                speaker_mapping = greedy_speaker_matching(
                    reference_rttm_path=predictions_dirs[ground_truth_label]/f'{video_name}.rttm',
                    predicted_rttm_path=predictions_dirs[model]/f'{video_name}.rttm',
                )
                speaker_mappings[model] = speaker_mapping
            else:
                speaker_mappings[model] = None
        # Create output path
        output_path = output_root/f"{video_name}_{start_time:.0f}_{end_time:.0f}"
        os.makedirs(output_path, exist_ok=True)
        # Create plots
        visualize_all_models(
            videoId=video_name, start=start_time, end=end_time, num_speakers=num_speakers,
            rttms_paths=predictions_dirs, output_path=str(output_path),
            speaker_mappings=speaker_mappings,
        )
            
def load_clips(clips_path: str):
    with open(clips_path, 'r') as f:
        headers = f.readline().split(',')
        clips = [line.split(',') for line in f.readlines()]
        clips = [(clip[0], float(clip[1]), float(clip[2]), int(clip[3])) for clip in clips]
    return clips

if __name__=='__main__':
    main()