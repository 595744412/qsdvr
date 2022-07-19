import math
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
        ctx.save_for_backward(pointList, indexList)
        ctx.reso = reso
        ctx.dataCount = dataCount
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        pointList, indexList = ctx.saved_tensors
        dataGradGrid = torch.zeros(
            (ctx.reso, ctx.reso, ctx.reso, ctx.dataCount),
            dtype=torch.float32,
            device="cuda")
        qsdvr.GridInterpolationBackward(dataGradGrid, pointList, indexList,
                                        gradOutput, ctx.reso)
        return dataGradGrid, None, None


class Shader(autograd.Function):

    @staticmethod
    def forward(ctx: Any, normalList: torch.Tensor, viewDirList: torch.Tensor,
                dataList: torch.Tensor):
        pointCount = normalList.size(0)
        out = torch.empty((pointCount, 3), dtype=torch.float32, device="cuda")
        specularList = torch.empty((pointCount, 3),
                                   dtype=torch.float32,
                                   device="cuda")
        qsdvr.ShaderForward(out, normalList, viewDirList, dataList,
                            specularList)
        ctx.save_for_backward(normalList, viewDirList, dataList, specularList)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        normalList, viewDirList, dataList, specularList = ctx.saved_tensors
        pointCount = normalList.size(0)
        normalGradList = torch.empty((pointCount, 3),
                                     dtype=torch.float32,
                                     device="cuda")
        dataGradList = torch.empty((pointCount, 32),
                                   dtype=torch.float32,
                                   device="cuda")
        qsdvr.ShaderBackward(gradOutput, normalList, viewDirList, dataList,
                             specularList, normalGradList, dataGradList)
        return normalGradList, None, dataGradList


class RayAggregate(autograd.Function):

    @staticmethod
    def forward(ctx: Any, rgbList: torch.Tensor, sdfList: torch.Tensor,
                rayList: torch.Tensor, logisticCoef):
        pixelCount = rayList.size(0)
        renderCount = rgbList.size(0)
        out = torch.empty((pixelCount, 3), dtype=torch.float32, device="cuda")
        alphaList = torch.empty(renderCount,
                                dtype=torch.float32,
                                device="cuda")
        TList = torch.empty(renderCount, dtype=torch.float32, device="cuda")
        qsdvr.RayAggregateForward(out, rgbList, sdfList, rayList, alphaList,
                                  TList, logisticCoef)
        ctx.save_for_backward(rgbList, sdfList, rayList, alphaList, TList)
        ctx.logisticCoef = logisticCoef
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        rgbList, sdfList, rayList, alphaList, TList = ctx.saved_tensors
        sdfCount = sdfList.size(0)
        renderCount = rgbList.size(0)
        rgbGradList = torch.empty((renderCount, 3),
                                  dtype=torch.float32,
                                  device="cuda")
        sdfGradList = torch.empty(sdfCount, dtype=torch.float32, device="cuda")
        qsdvr.RayAggregateBackward(gradOutput, rgbList, sdfList, rayList,
                                   alphaList, TList, rgbGradList, sdfGradList,
                                   ctx.logisticCoef)
        return rgbGradList, sdfGradList, None, None


class LayerToGrid(autograd.Function):

    @staticmethod
    def forward(ctx: Any, xLayer: torch.Tensor, yLayer: torch.Tensor,
                zLayer: torch.Tensor, offset: torch.Tensor):
        reso = xLayer.size(0)
        out = torch.empty((reso, reso, reso, 7),
                          dtype=torch.float32,
                          device="cuda")
        qsdvr.LayerToGridForward(out, xLayer, yLayer, zLayer, offset)
        ctx.reso = reso
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        reso = ctx.reso
        xLayer = torch.empty(reso, dtype=torch.float32, device="cpu")
        yLayer = torch.empty(reso, dtype=torch.float32, device="cpu")
        zLayer = torch.empty(reso, dtype=torch.float32, device="cpu")
        offset = torch.empty(4, dtype=torch.float32, device="cpu")
        qsdvr.LayerToGridBackward(gradOutput, xLayer, yLayer, zLayer, offset)
        return xLayer, yLayer, zLayer, offset


class SampleSDF(autograd.Function):

    @staticmethod
    def forward(ctx: Any, sdfGrid: torch.Tensor, pointList: torch.Tensor,
                indexList: torch.Tensor):
        pointCount = pointList.size(0)
        out = torch.empty(pointCount, dtype=torch.float32, device="cuda")
        qsdvr.SampleSDFForward(out, sdfGrid, pointList, indexList,
                               sdfGrid.size(0))
        ctx.save_for_backward(sdfGrid, pointList, indexList)
        ctx.reso = sdfGrid.size(0)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        sdfGrid, pointList, indexList = ctx.saved_tensors
        sdfGradGrid = torch.zeros((ctx.reso, ctx.reso, ctx.reso, 7),
                                  dtype=torch.float32,
                                  device="cuda")
        qsdvr.SampleSDFBackward(gradOutput, sdfGrid, pointList, indexList,
                                sdfGradGrid, ctx.reso)
        return sdfGradGrid, None, None


class SampleNormal(autograd.Function):

    @staticmethod
    def forward(ctx: Any, sdfGrid: torch.Tensor, pointList: torch.Tensor,
                indexList: torch.Tensor):
        pointCount = pointList.size(0)
        out = torch.empty((pointCount, 3), dtype=torch.float32, device="cuda")
        qsdvr.SampleNormalForward(out, sdfGrid, pointList, indexList)
        ctx.save_for_backward(sdfGrid, pointList, indexList)
        ctx.reso = sdfGrid.size(0)
        return out

    @staticmethod
    def backward(ctx: Any, gradOutput: Any):
        sdfGrid, pointList, indexList = ctx.saved_tensors
        sdfGradGrid = torch.zeros((ctx.reso, ctx.reso, ctx.reso, 7),
                                  dtype=torch.float32,
                                  device="cuda")
        qsdvr.SampleNormalBackward(gradOutput, sdfGrid, pointList, indexList,
                                   sdfGradGrid)
        return sdfGradGrid, None, None


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
        c = -self.reso
        self.offset[0] = -2 * c
        self.offset[1] = -2 * c
        self.offset[2] = -2 * c
        self.offset[3] = math.sqrt(3 * c * c)

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
        self.renderData = torch.zeros((reso + 1, reso + 1, reso + 1, 32),
                                      dtype=torch.float32,
                                      device="cuda")
        self.logisticCoef = logisticCoef
        self.renderData[..., 31] = 1
        self.renderData[..., 30] = 0.1
        self.renderData[..., 27:30] = 1
        self.renderData[..., 0:3] = 0.5
        self.renderData = nn.Parameter(self.renderData)

    def forward(self, input):
        sdfList, normalList = self.quadricGrid.forward(
            input["renderPointList"], input["renderIndexList"],
            input["sdfPointList"], input["sdfIndexList"])
        datalist = GridInterpolation.apply(self.renderData,
                                           input["renderPointList"],
                                           input["renderIndexList"])
        rgbList = Shader.apply(normalList, input["viewDirList"], datalist)
        return RayAggregate.apply(rgbList, sdfList, input["rayList"],
                                  self.logisticCoef)

    def set_logisticCoef(self, logisticCoef):
        self.logisticCoef = logisticCoef
