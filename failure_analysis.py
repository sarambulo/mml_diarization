import os

if __name__ == "__main__":
    video_id_to_lines = {}

    # Read the input RTTM file
    with open("rmse/powerset/powerset.rttm", "r") as file:
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
        output_file_name = f"rmse/powerset_rttms/{video_id}.rttm"
        with open(output_file_name, "w") as output_file:
            output_file.writelines(lines)

    print(f"RTTM file has been split into {len(video_id_to_lines)} files.")
