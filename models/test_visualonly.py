import torch
import numpy as np
from models.VisualOnly import CNNBlock, ResNetBlock, ResNet34, VisualSpeakerEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score

MOCK_INPUT = torch.ones((5, 3, 6, 6))


class TestCNNBlock:
    def test_forward(self):
        block = CNNBlock(3, 2, 3, 1, 1)
        Z = block(MOCK_INPUT)
        assert (5, 2, 6, 6) == Z.shape

    def test_stride(self):
        block = CNNBlock(3, 2, 3, 2, 1)
        Z = block(MOCK_INPUT)
        assert (5, 2, 3, 3) == Z.shape


class TestResnetBlock:
    def test_forward(self):
        block = ResNetBlock(3, 2, 3, 1)
        Z = block(MOCK_INPUT)
        assert (5, 2, 6, 6) == Z.shape

    def test_stride(self):
        block = ResNetBlock(3, 2, 3, 2)
        Z = block(MOCK_INPUT)
        assert (5, 2, 3, 3) == Z.shape


class TestResnet34:
    def test_init(self):
        model = ResNet34(512)


class TestVisualSpeakerEncoder:
    def test_init(self):
        model = VisualSpeakerEncoder(512, None)

    def test_from_pretrained(self):
        model = VisualSpeakerEncoder(1024)

    def test_return_value(self):
        model = VisualSpeakerEncoder(512, None)
        x = torch.rand((2, 3, 112, 112))
        y = model(x)
        assert y.shape == (2, 512)

    def test_accuracy(self):
        model = VisualSpeakerEncoder(1024)
        # Load speaker 0 face crops
        faces_0 = np.load("preprocessed/00001/Chunk_1/face_0.npy")
        faces_0 = torch.tensor(faces_0)
        target_0 = torch.full((len(faces_0),), 0)
        # Load speaker 1 face crops
        faces_1 = np.load("preprocessed/00001/Chunk_1/face_1.npy")
        faces_1 = torch.tensor(faces_1)
        target_1 = torch.full((len(faces_1),), 1)
        # Join faces
        faces = torch.concat((faces_0, faces_1))
        target = torch.concat((target_0, target_1))
        # Predict
        embeddings = model(faces)
        clustering = AgglomerativeClustering(
            n_clusters=2, metric="euclidean", linkage="ward"
        )
        cluster_labels = clustering.fit_predict(embeddings.detach().numpy())
        # Align
        speaker_0_id = round(cluster_labels[: len(faces_0)].mean())
        if speaker_0_id == 1:
            cluster_labels = np.where(cluster_labels == 1, 0, 1)
        # Compare
        accuracy = accuracy_score(y_true=target.detach().numpy(), y_pred=cluster_labels)
        assert accuracy >= 0.95
