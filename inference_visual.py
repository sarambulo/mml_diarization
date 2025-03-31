import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
# Replace with your actual imports, e.g.:
# from your_dataset_module import SingleFrameTestDataset
# from your_model_module import VisualOnlyModel
from datasets.LoaderTest import TestDataset  
from models.VisualOnly import VisualOnlyModel  # assuming file name

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path, embedding_dims=512):
    """
    Loads the trained visual model checkpoint.
    """
    model = VisualOnlyModel(embedding_dims=embedding_dims, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def run_inference(model, test_loader, output_dir="predictions_per_video_visual"):
    """
    Runs inference and saves predictions in separate CSVs for each video_id.
    """
    from collections import defaultdict
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store predictions grouped by video_id
    predictions_by_video = defaultdict(list)

    with torch.no_grad():
        for face_tensor, audio_segment, label, metadata in tqdm(test_loader, desc="Inference"):
            input_face = face_tensor.to(DEVICE)  # [1, C, H, W]
            features = (None, None, input_face)
            embedding, logits = model(features)
            prob = torch.sigmoid(logits)
            is_speaking_pred = (prob >= 0.5).int().item()

            meta = metadata[0] if isinstance(metadata, list) else metadata
            video_id = str(meta["video_id"][0]) if isinstance(meta["video_id"], list) else str(meta["video_id"])

            prediction = {
                "video_id": video_id,
                "chunk_id": str(meta["chunk_id"]),
                "speaker_id": str(meta["speaker_id"]),
                "frame_idx": int(meta["frame_idx"]),
                "is_speaking_pred": is_speaking_pred,
                "probability": float(prob.item()) 
            }

            predictions_by_video[video_id].append(prediction)

    for video_id, records in predictions_by_video.items():
        df = pd.DataFrame(records)
        filename = f"{video_id}_visual_inference.csv"
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved predictions for video {video_id} to {path}")


def main():
    # Define paths
    test_data_dir = "test_preprocessed"  # Adjust to your test preprocessed data directory
    checkpoint_path = Path("checkpoints", "best_visual.pth")  # or your desired checkpoint
    
    # Load the test dataset
    test_dataset = TestDataset(root_dir=test_data_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Load the trained model
    model = load_model(checkpoint_path, embedding_dims=512)
    
    # Run inference and save predictions
    run_inference(model, test_loader, output_dir="predictions_per_video_visual")

if __name__ == "__main__":
    main()
