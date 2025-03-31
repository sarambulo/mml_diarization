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

def run_inference(model, test_loader, output_csv="predictions.csv"):
    """
    Runs inference on the test_loader and saves predictions with metadata to a CSV file.
    Each sample yields: (face_tensor, audio_segment, label, metadata).
    The model is assumed to require a tuple input of the form (None, None, x),
    where x is the face tensor with a batch dimension.
    """
    predictions = []
    
    with torch.no_grad():
        for face_tensor, audio_segment, label, metadata in tqdm(test_loader, desc="Inference"):
            # face_tensor is of shape [C, H, W]. Add a batch dimension:
            input_face = face_tensor.to(DEVICE)  # shape: [1, C, H, W]
            
            # Our model expects a tuple: (None, None, input_tensor)
            features = (None, None, input_face)
            embedding, logits = model(features)  # logits: shape [1]
            prob = torch.sigmoid(logits)         # probability in [0,1]
            is_speaking_pred = (prob >= 0.5).int().item()
            
            # Each metadata is a dict; since batch_size=1, extract the first element.
            meta = metadata[0] if isinstance(metadata, list) else metadata
            video_id    = str(meta["video_id"][0])
            chunk_id    = str(meta["chunk_id"][0])
            speaker_id  = (meta["speaker_id"]).item()
            frame_idx   = int(meta["frame_idx"]) 
            
            predictions.append({
                "video_id": video_id,
                "chunk_id": chunk_id,
                "speaker_id": speaker_id,
                "frame_idx": frame_idx,
                "is_speaking_pred": is_speaking_pred,
                "probability": float(prob.item()) 
            })
    
    # Save predictions to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    # Define paths
    test_data_dir = "test_preprocessed"  # Adjust to your test preprocessed data directory
    checkpoint_path = Path("checkpoints", "epoch_50.pth")  # or your desired checkpoint
    
    # Load the test dataset
    test_dataset = TestDataset(root_dir=test_data_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Load the trained model
    model = load_model(checkpoint_path, embedding_dims=512)
    
    # Run inference and save predictions
    run_inference(model, test_loader, output_csv="predictions.csv")

if __name__ == "__main__":
    main()
