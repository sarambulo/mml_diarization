from torch.utils.data import DataLoader
from MSDWild import MSDWildChunks
import numpy as np
import random
import torch
import pandas as pd
# import torchvision.transforms.v2 as ImageTransforms
def debug_dataset(dataset, num_samples: int = 1):
    """
    Prints information from the dataset by loading and inspecting `num_samples` examples.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to inspect.
        num_samples (int): How many samples to print out.
    """
    print(f"Dataset size: {len(dataset)}")

    for idx in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {idx} ---")

        sample = dataset[idx]

        if isinstance(sample, tuple):
            for i, item in enumerate(sample):
                if isinstance(item, torch.Tensor):
                    print(f"Item {i}: Tensor of shape {item.shape}, dtype={item.dtype}")
                elif isinstance(item, pd.DataFrame):
                    print(f"Item {i}: DataFrame with shape {item.shape}, columns={item.columns.tolist()}")
                elif isinstance(item, list):
                    print(f"Item {i}: List with length {len(item)}")
                else:
                    print(f"Item {i}: {type(item)}: {item}")
        else:
            print(f"Sample: {type(sample)} - {sample}")


data_path = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/preprocessed"  # base directory
partition_path = "/Users/AnuranjanAnand/Desktop/MML/mml_diarization/data_sample/few_train.rttm"  # your rttm partition file

dataset = MSDWildChunks(data_path=data_path, partition_path=partition_path, subset=1.0)
debug_dataset(dataset, num_samples=3)