import torch
import qsdvr
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
from models import RenderGrid
from datasets import Camera

model = RenderGrid()
rotation = torch.Tensor([1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0])
scale = torch.Tensor([1.0, 1.0, 1.0])
translation = torch.Tensor([0.0, 0.0, -2.5])
camera = Camera(1008, 756, rotation, translation, scale, 800, 800, -503.5, -377.5, 1.7, 2.5)
