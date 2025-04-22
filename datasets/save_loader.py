from datasets.MSDWild import MSDWildChunks
from torch.utils.data import DataLoader
from pairs.config import S3_BUCKET_NAME, S3_VIDEO_DIR
import os
import torch
from torch.utils.data import DataLoader
import boto3
import io
from tqdm import tqdm

# train_rttm_path = "sample.rttm"
# save_path = "sample_loader"

subset = 0.025

train_rttm_path = "few.train.rttm"
save_path = f"few_train_dataset_{str(subset).replace('.', '')}"

train_data_path = os.path.join("s3://", S3_BUCKET_NAME, S3_VIDEO_DIR)


dataset = MSDWildChunks(
    data_path=S3_VIDEO_DIR,
    data_bucket=S3_BUCKET_NAME,
    partition_path=train_rttm_path,
    subset=subset,
    refresh_fileset=False,
)


def collate_fn(batch):
    # Extract each feature: do the zip thing
    video_data, audio_data, is_speaking = list(zip(*batch))
    # Padding: NOTE: Not necessary
    # Stack:
    video_data = torch.stack(video_data)
    audio_data = torch.stack(audio_data)
    is_speaking = torch.tensor(is_speaking)
    # Return tuple((N, video_data, melspectrogram), (N, video_data, melspectrogram), (N, video_data, melspectrogram))
    # (N, C, H, W), (N, Bands, T) x3 (ask Prachi)
    batch_data = {
        "video_data": video_data,
        "audio_data": audio_data,
        "labels": is_speaking,
    }
    return batch_data


import multiprocessing as mp

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=mp.cpu_count(),
    pin_memory=True,
)

print("Completed Initial Loader")
for i in loader:
    print(i.keys())
    print(i["video_data"].shape)
    print(i["audio_data"].shape)
    print(i["labels"])
    break

# Save the entire model
buffer = io.BytesIO()
torch.save(dataset, buffer)
buffer.seek(0)

# Upload to S3
s3_client = boto3.client("s3")
s3_client.put_object(
    Bucket=S3_BUCKET_NAME, Key=f"loaders/{save_path}.pth", Body=buffer.getvalue()
)


# # Download from S3
s3_client = boto3.client("s3")
response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"loaders/{save_path}.pth")
model_data = response["Body"].read()
buffer = io.BytesIO(model_data)
dataset2 = torch.load(buffer, weights_only=False)

# print("Completed Saving Loader")
# for i in loader:
#     print(i.keys())
#     print(i["video_data"].shape)
#     print(i["audio_data"].shape)
#     print(i["labels"])
#     break
