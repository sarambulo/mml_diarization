import torch
import torch.nn as nn
from LipNet import LipCoordNet
import dlib
import cv2
import numpy as np


class FaceLipEmbedding(nn.Module):
    def __init__(
        self,
        pretrained_path="lipnet_pretrained.pt",
        coord_input_dim=40,
        final_embedding_dim=512,
    ):
        super(FaceLipEmbedding, self).__init__()

        self.base_model = LipCoordNet()
        checkpoint = torch.load(pretrained_path)
        self.base_model.load_state_dict(
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        # coordinate layers
        self.coord_fc = nn.Linear(coord_input_dim, 128)
        self.coord_relu = nn.ReLU()
        self.coord_dropout = nn.Dropout(0.5)

        # detection head with combined features
        self.detection_head = nn.Sequential(
            nn.Linear(final_embedding_dim, 1), nn.Sigmoid()
        )

    def forward(self, x, coords=None):
        x = self.base_model.conv1(x)
        x = self.base_model.relu(x)
        x = self.base_model.dropout3d(x)
        x = self.base_model.pool1(x)

        x = self.base_model.conv2(x)
        x = self.base_model.relu(x)
        x = self.base_model.dropout3d(x)
        x = self.base_model.pool2(x)

        x = self.base_model.conv3(x)
        x = self.base_model.relu(x)
        x = self.base_model.dropout3d(x)
        x = self.base_model.pool3(x)

        # flatten except batch_size
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        if coords is not None:
            coord_features = self.coord_fc(coords)
            coord_features = self.coord_relu(coord_features)
            coord_features = self.coord_dropout(coord_features)

            # concat coordinates
            combined_features = torch.cat((x_flat, coord_features), dim=1)
        else:
            combined_features = x_flat

        return combined_features


class VisualOnlyLipModel(torch.nn.Module):
    def __init__(self, embedding_dims, num_classes):
        super().__init__()
        self.visual_encoder = FaceLipEmbedding(final_embedding_dim=embedding_dims)
        self.classifier = torch.nn.sequential(
            torch.nn.Linear(embedding_dims, 1), nn.Sigmoid()
        )

    def forward(self, features):
        X = features[2]
        embedding = self.visual_encoder(X)
        preds = self.classifier(embedding)
        preds = preds.squeeze(1)
        return embedding, preds


def extract_lip_coordinates(
    bbox, predictor_path="shape_predictor_68_face_landmarks.dat"
):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Convert to grayscale
    gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        return torch.zeros(40)

    shape = predictor(gray, faces[0])

    # Extract lip landmarks (points 48-67)
    lip_points = []
    for i in range(48, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        lip_points.append(x)
        lip_points.append(y)

    # Normalize coordinates to 0-1 range
    h, w = bbox.shape[:2]
    lip_points = np.array(lip_points, dtype=np.float32)
    lip_points[0::2] /= w  # x coordinates
    lip_points[1::2] /= h  # y coordinates

    return torch.tensor(lip_points).float()
