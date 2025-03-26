import os
import matplotlib.pyplot as plt


def create_numbered_file(dir, base_name, extension):
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_name}.{extension}"
        else:
            file_name = f"{base_name}_{counter}.{extension}"

        if not os.path.exists(os.path.join(dir, file_name)):
            return file_name
        counter += 1


def get_speaking_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        # Check if the current directory is exactly one level below the root
        if root.count(os.sep) - directory.count(os.sep) == 1:
            for file in files:
                if file == "is_speaking.csv":
                    csv_files.append((root, file))
    return csv_files


def visualize_mel_spectrogram(mel, output_dir):
    plt.figure(figsize=(10, 6))
    plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.savefig(os.path.join(output_dir, "melspectrogram.png"))
