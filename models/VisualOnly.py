import torch
from sklearn.cluster import AgglomerativeClustering

class CNNBlock(torch.nn.Module):

   def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super(CNNBlock).__init__()
      self.layers = torch.nn.Sequential(
         torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
         torch.nn.BatchNorm2d(out_channels),
         torch.nn.PReLU(),
      )

   def forward(self, X):
      return self.layers(X)
   

class ResNetBlock(torch.nn.Module):

   def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(ResNetBlock).__init__()
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
         print(f'Z: {Z.shape}')
         print(X.shape)
         X = torch.transpose(X, 1, 3)
         X = self.linear(X) # Move channels to the end
         X = torch.transpose(X, 3, 1) # Bring channels to the second dim
         print(X.shape)
      # If (H_in, W_in) != (H_out, W_out) we need to downsample the result
      if self.stride > 1:
         print(self.stride)
         #X = torch.nn.functional.interpolate(X, scale_factor=1/self.stride)
         print(X.shape)
      return Z + X
   

class ResNet34(torch.nn.Module):

   def __init__(self):
      super(ResNet34).__init__()
      self.layers = torch.nn.Sequential(
         # Initial block: 3 --> 64 channels
         CNNBlock(3, 64, kernel_size=7, stride=2, padding=3),
         torch.nn.MaxPool2d(kernel_size=3, stride=2),
         # Block 1: 64 --> 128 channels
         ResNetBlock(64, 64, kernel_size=3, stride=1),
         ResNetBlock(64, 64, kernel_size=3, stride=1),
         ResNetBlock(64, 128, kernel_size=3, stride=1),
         # Block 2: 128 --> 256 channels
         ResNetBlock(128, 128, kernel_size=3, stride=2),
         ResNetBlock(128, 128, kernel_size=3, stride=1),
         ResNetBlock(128, 128, kernel_size=3, stride=1),
         ResNetBlock(128, 256, kernel_size=3, stride=1),
         # Block 3: 256 --> 512 channels
         ResNetBlock(256, 256, kernel_size=3, stride=2),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 256, kernel_size=3, stride=1),
         ResNetBlock(256, 512, kernel_size=3, stride=1),
         # Block 4: 512 --> 1024 channels
         ResNetBlock(512, 512, kernel_size=3, stride=2),
         ResNetBlock(512, 512, kernel_size=3, stride=1),
         ResNetBlock(512, 1024, kernel_size=3, stride=1),
         # Flattening
         torch.nn.AdaptiveAvgPool2d((1,1)),
         torch.nn.Flatten()
      )

   def forward(self, X):
      return self.layers(X)

class VisualOnlyModel(torch.nn.Module):
   def __init__(self):
      super(VisualOnlyModel).__init__()
      self.visual_encoder = ResNet34()
      # TODO: implement classifier
      self.classifier = None

   def forward(self, X):
      embedding = self.visual_encoder(X)
      active_speaker = self.classifier(embedding)
      return embedding, active_speaker

   @torch.no_grad()
   def predict_frame(self, X):
      # Set eval mode
      self.eval()
      # Call forward with no gradients and in inference mode
      with torch.inference_mode():
         embedding, active_speaker = self.forward(X)
      return embedding, active_speaker

   @torch.no_grad()
   def predict_video(self, X):
      # Set eval mode
      self.eval()
      num_frames = X.shape[0]
      embeddings = []
      for t in range(num_frames):
         frame = X[t]
         embedding, active_speaker = self.predict_frame(frame)
         embeddings.append(embedding)
         # Perform agglomerative clustering
         # TODO
         pass
