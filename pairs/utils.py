import os
import matplotlib.pyplot as plt
import numpy as np
import boto3
import io

S3 = boto3.client("s3")
paginator = S3.get_paginator("list_objects_v2")

import numpy as np
from PIL import Image


def plot_images_from_array(array, path):

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(min(18, array.shape[0])):
        img_array = array[i]  # shape (3, 32, 64)
        img_array = np.transpose(img_array, (1, 2, 0))  # shape (32, 64, 3)
        # img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        axes[i].imshow(img_array)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.savefig(path)


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


def list_s3_files(bucket_name, prefix=""):
    """
    List all files in an S3 bucket or a specific directory (prefix).

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Directory path in the bucket to list files from (optional).

    Returns:
        list: A list of file keys in the specified bucket and prefix.
    """
    file_keys = []

    # Use paginator for efficient listing
    paginator = S3.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in response_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                file_keys.append(obj["Key"])

    return file_keys


def upload_npz(bucket, key, face_data, lip_data, audio_data, metadata):
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        face_data=face_data,
        lip_data=lip_data,
        audio_data=audio_data,
        metadata=metadata,
    )
    buffer.seek(0)
    S3.upload_fileobj(buffer, Bucket=bucket, Key=key)


def s3_save_numpy(array, bucket_name, key):
    byte_stream = io.BytesIO()
    np.save(byte_stream, array, allow_pickle=False)
    byte_stream.seek(0)
    S3.upload_fileobj(byte_stream, bucket_name, key)


def s3_load_numpy(bucket_name, file_key):
    response = S3.get_object(Bucket=bucket_name, Key=file_key)
    data = response["Body"].read()
    return np.load(io.BytesIO(data))


def get_speaking_csv_files_s3(bucket, directory, filename):
    csv_files = []

    # Initialize the paginator
    paginator = S3.get_paginator("list_objects_v2")

    # Create a PageIterator from the paginator
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=directory)

    # Iterate through each page
    for page in page_iterator:
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) == 3 and parts[-1] == filename:
                csv_files.append((f"s3://{bucket}/{parts[0]}/{parts[1]}", parts[2]))

    return csv_files


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


def visualize_audio_triplet(audio, dir, name="audio"):
    os.makedirs(os.path.join(dir, "images"), exist_ok=True)
    # Plot the images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Audio Triplet")

    names = ["Anchor", "Positive", "Negative"]

    for i in range(3):
        axs[i].imshow(audio[i])  # Display each image
        axs[i].set_title(names[i])
        axs[i].axis("off")  # Turn off axis labels for cleaner display

    plt.tight_layout()
    plt.savefig(os.path.join(dir, "images", f"{name}.png"))
    plt.close()


def visualize_visual_triplet(images, dir, name):
    os.makedirs(os.path.join(dir, "images"), exist_ok=True)
    images_rgb = np.transpose(
        images, axes=(0, 2, 3, 1)
    )  # Transpose to (num_images, height, width, channels)

    # Plot the images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Visual Triplet")

    names = ["Anchor", "Positive", "Negative"]

    for i in range(3):
        axs[i].imshow(images_rgb[i])  # Display each image
        axs[i].set_title(names[i])
        axs[i].axis("off")  # Turn off axis labels for cleaner display

    plt.tight_layout()
    plt.savefig(os.path.join(dir, "images", f"{name}.png"))
    plt.close()
