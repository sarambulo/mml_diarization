import os
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_mel_spectrogram(mel, output_dir, saveas="audio.png"):
    print(output_dir, mel.shape)
    mel = mel.reshape(30, 440)
    plt.figure(figsize=(10, 6))
    plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.savefig(os.path.join(output_dir, saveas))
    plt.close()


def visualize_visual_triplet(images, dir, name):
    os.makedirs(os.path.join(dir, "images"), exist_ok=True)
    images_rgb = np.transpose(
        images, axes=(0, 2, 3, 1)
    )  # Transpose to (num_images, height, width, channels)

    # Plot the images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Three RGB Images Side by Side")

    names = ["Anchor", "Positive", "Negative"]

    for i in range(3):
        axs[i].imshow(images_rgb[i])  # Display each image
        axs[i].set_title(names[i])
        axs[i].axis("off")  # Turn off axis labels for cleaner display

    plt.tight_layout()
    plt.savefig(os.path.join(dir, "images", f"{name}.png"))
    plt.close()
