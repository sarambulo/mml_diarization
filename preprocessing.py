import argparse
import glob
import os
from time import time
from utils.build_chunks import build_chunks
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Create chunks from all .mp4 videos in a directory.")
    parser.add_argument('-d', "--data_dir", type=str, required=True,
                        help="Directory containing .mp4 videos and corresponding bounding box CSVs (e.g., 00001.mp4, 00001.csv, etc.).")
    parser.add_argument('-r', "--rttm_path", type=str, required=True,
                        help="Path to RTTM file.")
    parser.add_argument('-o', "--output_path", type=str, default='preprocessed',
                    help="Path to the preprocessed output data.")
    parser.add_argument("--seconds", type=int, default=5,
                        help="Duration of each chunk in seconds.")
    parser.add_argument("--total-seconds", type=int, default=300,
                    help="Maximum number of seconds to read from each video.")
    parser.add_argument("--downsampling_factor", type=int, default=8,
                        help="Factor by which to downsample frames.")
    parser.add_argument("--img_height", type=int, default=112,
                        help="Desired height of output frames.")
    parser.add_argument("--img_width", type=int, default=112,
                        help="Desired width of output frames.")
    parser.add_argument("--scale", type=bool, default=True,
                        help="True if pixel values should be centered and scaled.")
    args = parser.parse_args()

    video_files = sorted(glob.glob(os.path.join(args.data_dir, "*.mp4")))

    MAX_CHUNKS = args.total_seconds // args.seconds

    # Clean the output directory
    if Path(args.output_path).exists():
        shutil.rmtree(args.output_path)
    
    # Process videos
    start_time = time()
    video_counter = 0
    chunk_index = 0
    for video_file in tqdm(video_files, desc='Processing videos'):
        video_counter += 1
        # Create arguments for build_chunks
        base_name = os.path.basename(video_file)      
        video_id = os.path.splitext(base_name)[0]      
        bounding_boxes_path = os.path.join(args.data_dir, f"{video_id}.csv")
        if not os.path.isfile(bounding_boxes_path):
            print(f"Warning: No CSV found for {video_file} (expected {bounding_boxes_path}). Skipping.")
            continue
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        base_dir = os.path.join(args.output_path, video_name)
        os.makedirs(base_dir, exist_ok=True)

        # Generate the chunks
        chunks = build_chunks(
            video_path=video_file,
            bounding_boxes_path=bounding_boxes_path,
            rttm_path=args.rttm_path,
            seconds=args.seconds,
            downsampling_factor=args.downsampling_factor,
            img_height=args.img_height,
            img_width=args.img_width,
            scale=args.scale,
            max_chunks=MAX_CHUNKS
        )

        # Store the chunks
        all_is_speaking = []
        for chunk in chunks:
            # Update chunk index
            chunk_index +=1
            faces, melspectrogram, mfcc, is_speaking = chunk
            chunk_dir = os.path.join(base_dir, f"Chunk_{chunk_index}")
            os.makedirs(chunk_dir, exist_ok=True)

            # Labels
            csv_path = os.path.join(chunk_dir, "is_speaking.csv")
            is_speaking['video_id'] = int(video_id)
            is_speaking["chunk_id"] = chunk_index
            is_speaking.to_csv(csv_path, index=False)
            all_is_speaking.append(is_speaking)

            # Audio
            mel_file = os.path.join(chunk_dir, "melspectrogram.npy")
            np.save(mel_file, melspectrogram)
            mfcc_file = os.path.join(chunk_dir, "mfcc.npy")
            np.save(mfcc_file, mfcc)

            # Video
            for speaker_id, face_dict in faces.items():
                # Bounding boxes
                bbox_array = face_dict 
                bbox_file = os.path.join(chunk_dir, f"face_{speaker_id}.npy")
                np.save(bbox_file, bbox_array)
        
        # Video level labels
        if all_is_speaking:
            final_csv_path = os.path.join(base_dir, "is_speaking.csv")
            final_speaking_df = pd.concat(all_is_speaking, ignore_index=True)
            final_speaking_df.to_csv(final_csv_path, index=False)

    end_time = time()
    print(f"Created {chunk_index + 1} chunks across {video_counter + 1} videos in {end_time - start_time:.0f} seconds")

if __name__ == "__main__":
    main()
