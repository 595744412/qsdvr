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
model = RenderGrid(grid_reso, 0.1)
model_torch = RenderGridTorch(grid_reso, 1)
Loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
optimizer_torch = torch.optim.SGD(params=model_torch.parameters(), lr=0.1)
torch.cuda.synchronize()
dataset = NerfDataset("data/nerf_synthetic/chair", 1, 5, [1.0, 1.0, 1.0],
                      [0, 0, 0], 0.04, grid_reso)
print("begin")
begin = time.time()
for i in range(20):
    optimizer.param_groups[0]["lr"] = 0.01
    model.set_logisticCoef(0.1 * 1.2**(i + 1))
    for input, gt in dataset:
        pred = model.forward(input)
        loss = Loss(pred, gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        print("loss:", loss.item())
with torch.no_grad():
    for i in range(100):
        input, gt = dataset[i]
        img = dataset.get_whole_image(
            model.forward(input).cpu().detach().numpy(), i)
        imageio.imwrite("output/render_" + str(i) + ".png",
                        (img * 255).clip(0, 255).astype(np.uint8))
        torch.cuda.empty_cache()

end = time.time()
print(str(end - begin) + "s")
