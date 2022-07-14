import time
import torch
import sys
import numpy as np
import os
import cv2

sys.path.append(os.path.dirname(sys.path[0]))
from models import RenderGrid
from qsdvr import CameraArgs
from datasets import Camera

args = CameraArgs()
args.width = 1008
args.height = 756
args.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
args.translation = [0.0, 0.0, -2.5]
args.scale = [1.0, 1.0, 1.0]
args.fx = 800
args.fy = 800
args.cx = -503.5
args.cy = -377.5
args.near = 1.7
args.far = 2.6
camera = Camera(args, 0.03, 128)
camera.GenerateRayPoints()
[
    renderPointList, renderIndexList, sdfPointList, sdfIndexList, depthList,
    viewDirList, rayList
] = camera.GetRayPoints()
model = RenderGrid(128, 2)
torch.cuda.synchronize()
begin = time.time()
for i in range(50):
    model.forward(renderPointList, renderIndexList, sdfPointList, sdfIndexList,
                  depthList, viewDirList, rayList)
    torch.cuda.synchronize()
end = time.time()
print(50 / (end - begin))
img = model.forward(renderPointList, renderIndexList, sdfPointList,
                    sdfIndexList, depthList, viewDirList,
                    rayList).cpu().detach().numpy().reshape((756, 1008, 3))
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("render.jpg", img)
