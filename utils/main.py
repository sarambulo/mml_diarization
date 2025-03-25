import argparse
import glob
import os
from .build_chunks import build_chunks

def main():
    parser = argparse.ArgumentParser(description="Create chunks from all .mp4 videos in a directory.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .mp4 videos and corresponding bounding box CSVs (e.g., 00001.mp4, 00001.csv, etc.).")
    parser.add_argument("--rttm_path", type=str, required=True,
                        help="Path to RTTM file.")
    parser.add_argument("--seconds", type=int, default=3,
                        help="Duration of each chunk in seconds.")
    parser.add_argument("--downsampling_factor", type=int, default=5,
                        help="Factor by which to downsample frames.")
    parser.add_argument("--img_height", type=int, default=112,
                        help="Desired height of output frames.")
    parser.add_argument("--img_width", type=int, default=112,
                        help="Desired width of output frames.")
    parser.add_argument("--no_scale", dest="scale", action="store_false",
                        help="Disable scaling of pixel values.")
    parser.set_defaults(scale=True)

    args = parser.parse_args()

    video_files = sorted(glob.glob(os.path.join(args.data_dir, "*.mp4")))

    chunk_count = 0
    for video_file in video_files:
        base_name = os.path.basename(video_file)      
        stem = os.path.splitext(base_name)[0]      
        bounding_boxes_path = os.path.join(args.data_dir, f"{stem}.csv")
       
        if not os.path.isfile(bounding_boxes_path):
            print(f"Warning: No CSV found for {video_file} (expected {bounding_boxes_path}). Skipping.")
            continue

  
        chunk_count = build_chunks(
            video_path=video_file,
            bounding_boxes_path=bounding_boxes_path,
            rttm_path=args.rttm_path,
            seconds=args.seconds,
            downsampling_factor=args.downsampling_factor,
            img_height=args.img_height,
            img_width=args.img_width,
            scale=args.scale,
            chunk_idx_start=chunk_count
        )

    print(f"Total number of chunks created across all videos: {chunk_count}")

if __name__ == "__main__":
    main()
