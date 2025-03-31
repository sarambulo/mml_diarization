import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio_model(checkpoint_path, embedding_dim=256):
    """
    Loads the trained audio-only model checkpoint.
    This function checks for common key names ('state_dict' or 'model_state_dict').
    Adjust the import paths as needed.
    """
    from models.audio_model import AudioActiveSpeakerModel  # adjust import if necessary
    from models.audio_model import CompactAudioEmbedding         # adjust import if necessary
    
    base_model = CompactAudioEmbedding(input_dim=40, embedding_dim=embedding_dim, dropout_rate=0.3).to(DEVICE)
    model = AudioActiveSpeakerModel(base_model=base_model, embedding_dim=embedding_dim, num_classes=1).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "state_dict" in checkpoint:
        print("Loading checkpoint using key 'state_dict'")
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        print("Loading checkpoint using key 'model_state_dict'")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Checkpoint keys:", checkpoint.keys())
        print("Assuming checkpoint itself is the state dict")
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    return model

def run_audio_inference(model, test_loader, output_csv="audio_predictions.csv"):
    """
    Runs inference on the audio data from the test_loader.
    Each sample from the test_loader is assumed to be a tuple:
      (face_tensor, audio_segment, label, metadata)
    We ignore the face_tensor and use the audio_segment.
    The audio_segment is expected to have shape [n_mels, 22] (e.g. [30, 22]).
    We add a channel dimension so the model gets input of shape [B, 1, n_mels, 22].
    
    Predictions (is_speaking) and associated metadata are saved to CSV.
    """
    predictions = []
    
    with torch.no_grad():
        for face_tensor, audio_segment, label, metadata in tqdm(test_loader, desc="Audio Inference"):
            # audio_segment is returned with shape [n_mels, 22] per sample.
            # test_loader batch_size is 1, so its shape is [1, n_mels, 22] if collated.
            # To be safe, ensure batch dimension exists and then add channel dimension.
            if audio_segment.ndim == 2:
                # Add batch dimension
                audio_input = audio_segment.unsqueeze(0)
            else:
                audio_input = audio_segment  # assume already batched
            
            # Add channel dimension so input becomes [B, 1, n_mels, 22]
            audio_input = audio_input.unsqueeze(1).to(DEVICE)
            
            # Pass the audio input through the model
            # (Assuming your audio model's forward method accepts input of shape [B, 1, n_mels, 22])
            embedding, logits = model(audio_input)
            logits = logits.squeeze()  # remove extra dimensions if present
            prob = torch.sigmoid(logits)
            # For binary classification threshold at 0.5
            pred_label = (prob >= 0.5).int().item()
            
           
            
            # Convert each field to standard Python types
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
                "is_speaking_pred": int(pred_label),
                "probability": float(prob.item())
            })
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Audio predictions saved to {output_csv}")

def main():
    # Set paths (adjust these as needed)
    test_data_dir = "test_preprocessed"  # Directory containing your preprocessed test data (chunks)
    checkpoint_path = Path("checkpoints", "12.pth")  # Path to your audio model checkpoint
    
    # Import your test dataset from your MSDWild module.
    from datasets.LoaderTest import TestDataset  # adjust import if needed
    test_dataset = TestDataset(test_data_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Load the trained audio model
    model = load_audio_model(checkpoint_path, embedding_dim=512)
    
    # Run inference on audio and save predictions to CSV
    run_audio_inference(model, test_loader, output_csv="audio_predictions.csv")

if __name__ == "__main__":
    main()
