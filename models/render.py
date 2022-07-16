from typing import Any
from torch import nn, autograd
import torch
import qsdvr


class GridInterpolation(autograd.Function):

    @staticmethod
    def forward(ctx: Any, dataGrid: torch.Tensor, pointList: torch.Tensor,
                indexList: torch.Tensor):
        pointCount = pointList.size(0)
        dataCount = dataGrid.size(3)
        reso = dataGrid.size(0)
        out = torch.empty((pointCount, dataCount),
                          dtype=torch.float32,
                          device="cuda")
        qsdvr.GridInterpolationForward(dataGrid, pointList, indexList, out,
                                       reso)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        return torch.Tensor()


class Shader(autograd.Function):

    @staticmethod
    def forward(ctx: Any, normalList: torch.Tensor, viewDirList: torch.Tensor,
                dataList: torch.Tensor):
        pointCount = normalList.size(0)
        out = torch.empty((pointCount, 3), dtype=torch.float32, device="cuda")
        qsdvr.ShaderForward(out, normalList, viewDirList, dataList)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        return torch.Tensor()


class RayAggregate(autograd.Function):

    @staticmethod
    def forward(ctx: Any, rgbList: torch.Tensor, sdfList: torch.Tensor,
                rayList: torch.Tensor, logisticCoef):
        pixelCount = rayList.size(0)
        out = torch.empty((pixelCount, 3), dtype=torch.float32, device="cuda")
        qsdvr.RayAggregateForward(out, rgbList, sdfList, rayList, logisticCoef)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        ctx.rgbList
        return torch.Tensor()


class LayerToGrid(autograd.Function):

    @staticmethod
    def forward(ctx: Any, xLayer: torch.Tensor, yLayer: torch.Tensor,
                zLayer: torch.Tensor, offset: torch.Tensor):
        reso = xLayer.size(0)
        out = torch.empty((reso, reso, reso, 7),
                          dtype=torch.float32,
                          device="cuda")
        qsdvr.LayerToGridForward(out, xLayer, yLayer, zLayer, offset)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        return torch.Tensor()


class SampleSDF(autograd.Function):

    @staticmethod
    def forward(ctx: Any, sdfGrid: torch.Tensor, pointList: torch.Tensor,
                indexList: torch.Tensor):
        pointCount = pointList.size(0)
        out = torch.empty(pointCount, dtype=torch.float32, device="cuda")
        qsdvr.SampleSDFForward(out, sdfGrid, pointList, indexList)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        return torch.Tensor()


class SampleNormal(autograd.Function):

    @staticmethod
    def forward(ctx: Any, sdfGrid: torch.Tensor, pointList: torch.Tensor,
                indexList: torch.Tensor):
        pointCount = pointList.size(0)
        out = torch.empty((pointCount, 3), dtype=torch.float32, device="cuda")
        qsdvr.SampleNormalForward(out, sdfGrid, pointList, indexList)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        return torch.Tensor()


class QuadricGrid(nn.Module):

    def __init__(self, reso: int = 128):
        super().__init__()
        self.reso = reso
        self.xLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cpu"))
        self.yLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cpu"))
        self.zLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cpu"))
        self.offset = torch.zeros(4, dtype=torch.float32, device="cpu")
        self.init_as_sphere()
        self.offset = nn.Parameter(self.offset)

    def init_as_sphere(self, radius=1.0):
        radius *= self.reso
        c = 1 - self.reso
        self.offset[0] = 2 * c
        self.offset[1] = 2 * c
        self.offset[2] = 2 * c
        self.offset[3] = 3 * c * c - radius * radius

    def forward(
        self,
        renderPointList: torch.Tensor,
        renderIndexList: torch.Tensor,
        sdfPointList: torch.Tensor,
        sdfIndexList: torch.Tensor,
    ):
        sdfGrid = LayerToGrid.apply(self.xLayer, self.yLayer, self.zLayer,
                                    self.offset)
        sdfList = SampleSDF.apply(sdfGrid, sdfPointList, sdfIndexList)
        normalList = SampleNormal.apply(sdfGrid, renderPointList,
                                        renderIndexList)
        return sdfList, normalList


class RenderGrid(nn.Module):

    def __init__(self, reso: int = 128, logisticCoef: float = 5):
        super().__init__()
        self.quadricGrid = QuadricGrid(reso)
        self.reso = reso
        self.renderData = torch.empty((reso + 1, reso + 1, reso + 1, 32),
                                      dtype=torch.float32,
                                      device="cuda")
        self.renderData.fill_(0.5)
        self.renderData = nn.Parameter(self.renderData)
        self.logisticCoef = logisticCoef

    def forward(self, input):
        datalist = GridInterpolation.apply(self.renderData,
                                           input["renderPointList"],
                                           input["renderIndexList"])
        sdfList, normalList = self.quadricGrid.forward(
            input["renderPointList"], input["renderIndexList"],
            input["sdfPointList"], input["sdfIndexList"])
        rgbList = Shader.apply(normalList, input["viewDirList"], datalist)
        return RayAggregate.apply(rgbList, sdfList, input["rayList"],
                                  self.logisticCoef)
