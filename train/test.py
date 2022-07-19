import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import torch
import sys
import numpy as np
import imageio
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from models import RenderGrid, RenderGridTorch
from qsdvr import CameraArgs
from datasets import NerfDataset

grid_reso = 128
model = RenderGrid(grid_reso, 0.01)
model_torch = RenderGridTorch(grid_reso, 1)
Loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.2)
optimizer_torch = torch.optim.SGD(params=model_torch.parameters(), lr=0.1)
torch.cuda.synchronize()
dataset = NerfDataset("data/nerf_synthetic/chair", 1, 5, [0.7, 0.7, 0.7],
                      [0, 0, 0], 0.1, grid_reso)
print("begin")
begin = time.time()
for i in range(300):
    input, gt = dataset[i % 100]
    pred = model.forward(input)
    loss = Loss(pred, gt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    print("loss:", loss.item())
end = time.time()
print(str(100 / (end - begin)) + "fps")
img = dataset.get_whole_image(
    model.forward(dataset[0][0]).cpu().detach().numpy(), 0)
imageio.imwrite("output/render_" + str(0) + ".png",
                (img * 255).clip(0, 255).astype(np.uint8))
