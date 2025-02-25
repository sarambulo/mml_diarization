import torch
import numpy as np
from VisualOnly import CNNBlock, ResNetBlock, ResNet34

MOCK_INPUT = torch.ones((5, 3, 6, 6))

class TestCNNBlock():
   def test_forward(self):
      block = CNNBlock(3, 2, 3, 1, 1)
      Z = block(MOCK_INPUT)
      assert (5, 2, 6, 6) == Z.shape
   def test_stride(self):
      block = CNNBlock(3, 2, 3, 2, 1)
      Z = block(MOCK_INPUT)
      assert (5, 2, 3, 3) == Z.shape

class TestResnetBlock():
   def test_forward(self):
      block = ResNetBlock(3, 2, 3, 1)
      Z = block(MOCK_INPUT)
      assert (5, 2, 6, 6) == Z.shape
   def test_stride(self):
      block = ResNetBlock(3, 2, 3, 2)
      Z = block(MOCK_INPUT)
      assert (5, 2, 3, 3) == Z.shape