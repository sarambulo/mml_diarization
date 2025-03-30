from matplotlib.pylab import f
import torch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from losses.DiarizationLoss import DiarizationLoss


class CNNBlock(torch.nn.Module):

   def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super().__init__()
      self.layers = torch.nn.Sequential(
         torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
         torch.nn.BatchNorm2d(out_channels),
         torch.nn.PReLU(),
      )

   def forward(self, X):
      return self.layers(X)
   

class ResNetBlock(torch.nn.Module):

   def __init__(self, in_channels, out_channels, kernel_size, stride):
      super().__init__()
      padding = kernel_size // 2
      self.stride = stride
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.cnn_1 = CNNBlock(in_channels, out_channels, kernel_size, stride, padding)
      self.cnn_2 = CNNBlock(out_channels, out_channels, kernel_size, 1, padding)
      self.linear = None
      if in_channels != out_channels:
         self.linear = torch.nn.Linear(in_channels, out_channels)
      
   def forward(self, X):
      # X: (N,  C_in,  H_in,  W_in)
      # Z: (N, C_out, H_out, W_out)
      Z = self.cnn_1(X)
      Z = self.cnn_2(Z)
      # If C_in != C_out, we need to apply a linear transform to C
      if self.linear:
         X = torch.transpose(X, 1, 3)
         X = self.linear(X) # Move channels to the end
         X = torch.transpose(X, 3, 1) # Bring channels to the second dim
      # If (H_in, W_in) != (H_out, W_out) we need to downsample the result
      if self.stride > 1:
         X = X[:, :, ::self.stride, ::self.stride]
      return Z + X
   

class ResNet34(torch.nn.Module):
   def __init__(self, embedding_dims):
      super().__init__()
      self.model = torch.nn.Sequential(
         # Initial block: 3 --> 64 channels
         CNNBlock(3, 64, kernel_size=7, stride=2, padding=3),
         torch.nn.MaxPool2d(kernel_size=3, stride=2),
         # Block 1: 64 --> 128 channels
         ResNetBlock(64, 64, kernel_size=3, stride=1),
         ResNetBlock(64, 64, kernel_size=3, stride=1),
         ResNetBlock(64, 64, kernel_size=3, stride=1),
         # Block 2: 128 --> 256 channels
         ResNetBlock(64, 128, kernel_size=3, stride=2),
         ResNetBlock(128, 128, kernel_size=3, stride=1),
         ResNetBlock(128, 128, kernel_size=3, stride=1),
         ResNetBlock(128, 128, kernel_size=3, stride=1),
         # Block 3: 256 --> 512 channels
         ResNetBlock(128, 256, kernel_size=3, stride=2),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         # Block 4: 512 --> 1024 channels
         ResNetBlock(256, 512, kernel_size=3, stride=2),
         ResNetBlock(512, 512, kernel_size=3, stride=1),
         ResNetBlock(512, 512, kernel_size=3, stride=1),
         ResNetBlock(512, embedding_dims, kernel_size=3, stride=1),
         # Flattening
         torch.nn.AdaptiveAvgPool2d((1,1)),
         torch.nn.Flatten()
      )
   def forward(self, X):
      return self.model(X)

class VisualOnlyModel(torch.nn.Module):
   def __init__(self, embedding_dims, num_classes):
      super().__init__()
      self.visual_encoder = ResNet34(embedding_dims)
      self.classifier = torch.nn.Linear(embedding_dims, 1)

   def forward(self, features):
      X = features[2]
      embedding = self.visual_encoder(X)
      logits = self.classifier(embedding)
      logits = logits.squeeze(1)
      return embedding, logits

   @torch.no_grad()
   def predict_frame(self, X):
      # Set eval mode
      self.eval()
      # Call forward with no gradients and in inference mode
      with torch.inference_mode():
         embedding, active_speaker = self.forward(X)
      return embedding, active_speaker
   
   
   def agg_clustering(self, embeddings, n_faces):
      clustering = AgglomerativeClustering(
         n_clusters = n_faces,
         metric ='euclidean',
         linkage='ward'
      )   
      
      cluster_labels = clustering.fit_predict(embeddings)
      return cluster_labels
      

   @torch.no_grad()
   def predict_video(self, X):
      # Set eval mode
      self.eval()
      num_frames = len(X[2])
      
      embeddings = []
      face_indices = []
      frame_indices = []
      active_speakers = []
      
      max_faces = max(len(faces) for faces in X[2])
      
      for t in range(num_frames):
         features = X[0][t], X[1][t], torch.stack(X[2][t], dim=0)         
         frame_embeddings, active_logits = self.predict_frame(features) 
         faces = features[2][::]
         for face_idx, face in enumerate(faces):
            is_active = torch.argmax(active_logits[face_idx]).item()
            if frame_embeddings is not None:
               embeddings.append(frame_embeddings)
               face_indices.append(face_idx)
               frame_indices.append(t)
               active_speakers.append(is_active)
            
      embeddings_array = torch.concat(embeddings, dim=0).cpu().numpy()
      
      cluster_labels = self.agg_clustering(embeddings_array, max_faces)
      
      face_to_speaker = {}
      
      for i, (frame_idx, face_idx, speaker_id, is_active) in enumerate(zip(frame_indices, face_indices, cluster_labels, active_speakers)):
         if frame_idx not in face_to_speaker:
            face_to_speaker[frame_idx] = {}
         face_to_speaker[frame_idx][face_idx] = (speaker_id, is_active)
         
      results = {
         'speaker_mapping':face_to_speaker,
         'num_speakers':max_faces,
         'num_frames':len(features[0])
      }
      return results
   
   def active_frames_by_speaker_id(self, results):
      num_speakers = results['num_speakers']
      speaker_active_frames = {speaker_id: [] for speaker_id in range(num_speakers)}
      
      mapping_dict = results['speaker_mapping']
      
      for frame_idx in range(results['num_frames']):
         if frame_idx not in mapping_dict:
            continue
         
         for face_idx in mapping_dict[frame_idx]:
            speaker_id, is_active = mapping_dict[frame_idx][face_idx]
            if is_active:
               speaker_active_frames[speaker_id].append(frame_idx)
      
      return speaker_active_frames
   
   def create_utterances(self, speaker_active_frames, fps=25, min_gap_frames=10):
      utterances = {}
      
      for speaker_id, frames in speaker_active_frames.items():
         frames = sorted(frames)
         speaker_utterances = []
         if not frames:
            utterances[speaker_id] = speaker_utterances
            continue
         current_utterance = {
            'start_frame':frames[0],
            'end_frame': frames[0]
         }
         for i in range(0, len(frames)):
            current_frame = frames[i]
            previous_frame = frames[i-1]
            
            if current_frame <= previous_frame + min_gap_frames:
               #push end forward if it's within the minimum gap
               current_utterance['end_frame'] = current_frame
            else:
               #process + add utterance
               start_time = current_utterance['start_frame'] / fps
               end_time = current_utterance['end_frame'] / fps
               duration = end_time - start_time
            
               current_utterance['start_time'] = start_time
               current_utterance['end_time'] = end_time
               current_utterance['duration'] = duration
               speaker_utterances.append(current_utterance)
               
               #reset
               current_utterance = {
                  'start_frame':current_frame,
                  'end_frame':current_frame
               }
            
            #process last utternace
            start_time = current_utterance['start_frame'] / fps
            end_time = current_utterance['end_frame'] / fps
            duration = end_time - start_time
        
            current_utterance['start_time'] = start_time
            current_utterance['end_time'] = end_time
            current_utterance['duration'] = duration
        
            speaker_utterances.append(current_utterance)
            
            #filter out lil blips
            min_duration = 0.0  
            speaker_utterances = [
                  utterance for utterance in speaker_utterances
                  if utterance['duration'] >= min_duration
            ]
            
            utterances[speaker_id] = speaker_utterances
      
      return utterances
   
   
#    def utterances_to_rttm(self, utterances, file_id):
#       rttm_lines = []
#       if len(utterances) == 0:
#          return []
#       for speaker_id, speaker_utterances in utterances.items():
#          for utterance in speaker_utterances:
#                start_time = f"{utterance['start_time']:.6f}"
#                duration = f"{utterance['duration']:.6f}"
#                speaker = f"{speaker_id}"  
               
#                rttm_line = f"SPEAKER {file_id:05d} {"0"} {start_time} {duration} {"NA"} {"NA"} {speaker} {"NA"} {"NA"}"
#                rttm_lines.append(rttm_line)
      
#       #sort by start_time
#       rttm_lines.sort(key=lambda x: float(x.split()[3]))
      
#       return rttm_lines
   
   
   def predict_to_rttm_full(self, X, file_id):
      results = self.predict_video(X)
      active_frames = self.active_frames_by_speaker_id(results)
      utterances = self.create_utterances(active_frames)
      rttm_lines = self.utterances_to_rttm(utterances, file_id)
      return rttm_lines
         
               
         
         
         
   
   
               
      
   
      
            
               
         
      
      
   
   
