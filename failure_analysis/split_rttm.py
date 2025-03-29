def split_rttm_by_video_id(input_rttm_path):
    """
    Splits an RTTM file into separate files based on the video ID (2nd column).
    Each output file is named after the video ID.

    Args:
        input_rttm_path (str): Path to the input RTTM file.
    """
    import os

    # Dictionary to store lines grouped by video ID
    video_id_to_lines = {}

    # Read the input RTTM file
    with open(input_rttm_path, "r") as file:
        for line in file:
            # Skip empty lines or lines that don't have enough columns
            if not line.strip():
                continue

            # Split the line into columns
            columns = line.strip().split()

            # Ensure the line has at least 2 columns
            if len(columns) < 2:
                raise ValueError(f"Invalid RTTM line: {line}")

            # Extract the video ID (2nd column)
            video_id = columns[1]

            # Add the line to the corresponding video ID group
            if video_id not in video_id_to_lines:
                video_id_to_lines[video_id] = []
            video_id_to_lines[video_id].append(line)

    # Create separate RTTM files for each video ID
    for video_id, lines in video_id_to_lines.items():
        output_file_name = f"rmse/true/{video_id}.rttm"
        with open(output_file_name, "w") as output_file:
            output_file.writelines(lines)

    print(f"RTTM file has been split into {len(video_id_to_lines)} files.")


split_rttm_by_video_id("data/all.rttm")
