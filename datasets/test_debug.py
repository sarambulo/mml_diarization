from torch.utils.data import DataLoader
from MSDWild import TestDataset

def debug_test_loader(root_dir):
    """
    
    Args:
      root_dir (str): The root directory of your preprocessed data (e.g., "Preprocessed_data")
    """
    # Import your dataset class. Make sure SingleFrameTestDataset is in scope.
    dataset = TestDataset(root_dir, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Total samples in dataset: {len(dataset)}")
    print("Iterating over a few samples...\n")
    
    # Iterate over first few batches
    for i, (face_tensor, audio_segment, label, metadata) in enumerate(loader):
        print(f"Sample {i}:")
        print(f"  Face shape: {face_tensor.shape}")           # Expected: [C, H, W]
        print(f"  Audio segment shape: {audio_segment.shape}")  # Expected: [n_mels, segment_length]
        print(f"  Label: {label}")                              # 0 or 1
        print(f"  Metadata: {metadata}\n")                     # dict with video_id, chunk_id, speaker_id, frame_idx, total_frames
        if i >= 2:  # Debug only first 3 samples
            break

if __name__ == "__main__":
    # Replace "Preprocessed_data" with your actual directory path.
    debug_test_loader("test_preprocessed")
