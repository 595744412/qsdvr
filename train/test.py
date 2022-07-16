import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import torch
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from models import RenderGrid
from qsdvr import CameraArgs
from datasets import NerfDataset

model = RenderGrid(128, 1)
Loss = torch.nn.L1Loss(reduction='sum')
torch.cuda.synchronize()
dataset = NerfDataset("data/nerf_synthetic/chair", 1, 5, [0.7, 0.7, 0.7],
                      [0, 0, 0], 0.05, 128)
print("begin")
begin = time.time()
for input, gt in dataset:
    pred = model.forward(input)
    loss = Loss(pred, gt)
    print("loss:", loss.item())
end = time.time()
print(str(100 / (end - begin)) + "fps")
img = dataset.get_whole_image(
    model.forward(dataset[0][0]).cpu().detach().numpy(), 0)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("output/render_" + str(0) + ".jpg", img)
