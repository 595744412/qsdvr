import time
import torch
import numpy as np
import imageio
from tqdm import tqdm
import argparse

from models import RenderGrid, RenderGridTorch
from datasets import NerfDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", type=str,default="D:/data/nerf_synthetic/nerf_synthetic/chair")
    args = parser.parse_args()

    grid_reso = 64
    model = RenderGrid(grid_reso, 0.1)
    model_torch = RenderGridTorch(grid_reso, 1)
    Loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    optimizer_torch = torch.optim.SGD(params=model_torch.parameters(), lr=0.1)
    torch.cuda.synchronize()
    dataset = NerfDataset(args.data_path, 1, 5, [1.0, 1.0, 1.0],
                        [0, 0, 0], 0.04, grid_reso)
    print("training")
    begin = time.time()
    for i in tqdm(range(20)):
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
    print("testing")
    with torch.no_grad():
        for i in tqdm(range(100)):
            input, gt = dataset[i]
            img = dataset.get_whole_image(
                model.forward(input).cpu().detach().numpy(), i)
            imageio.imwrite("output/render_" + str(i) + ".png",
                            (img * 255).clip(0, 255).astype(np.uint8))
            torch.cuda.empty_cache()

    end = time.time()
    print("run time: " + str(end - begin) + "s")
