from inference import create_rttm_file
from utils.metrics import rttm_to_annotations, calculate_metrics_for_dataset
import torch
from models.VisualOnly import VisualOnlyModel
from datasets.MSDWild import MSDWildVideos
from torch.utils.data import DataLoader
from pathlib import Path

def evaluate(model, test_loader, output_rttm_path="predictions"):
   model.eval()
   all_predictions = []
   all_file_ids = []
   all_timestamps = []
   
   with torch.no_grad():
      for i, batch in enumerate(test_loader):
         if batch is None:
            continue
         
         features = (batch[0], batch[1], batch[3])
         labels = batch[2]
         timestamps = batch[4]
         features = features.to(next(model.parameters()).device)
         
         speaker_ids, labels = model.predict_video(features)
         file_ids = range(batch * i, len(batch) * (i + 1))
         file_names = [test_loader.dataset.file_names[file_id] for file_id in file_ids]
         
   
   
   # Calculate metrics
   preds = rttm_to_annotations(output_rttm_path)
   targets = rttm_to_annotations("data/test/few_val.rttm")
   
   metrics = calculate_metrics_for_dataset(preds, targets)
   
   print("\nEvaluation Metrics:")
   print(f"DER: {metrics['DER']:.4f}")
   print(f"JER: {metrics['JER']:.4f}")
   print(f"Missed Speech Rate: {metrics['MSR']:.4f}")
   print(f"False Alarm Rate: {metrics['FAR']:.4f}")
   print(f"Speaker Error Rate: {metrics['SER']:.4f}")
   
   return metrics

if __name__=='__main__':
   # Setup model
   DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = VisualOnlyModel(512, 2)
   model_filename = 'best_VisualOnlyModel.pth'
   model_path = Path('checkpoints', model_filename)
   checkpoint = torch.load(model_path, weights_only=True, map_location=DEVICE)
   model.load_state_dict(checkpoint['model_state_dict'])

   # Data
   val_dataset = MSDWildVideos('data', 'many_val', None, 0.1)
   val_dataloader = DataLoader(val_dataset, 1, False)
   metrics = evaluate(model=model, test_loader=val_dataset, output_rttm_path=Path('predictions'))
   print(metrics)
