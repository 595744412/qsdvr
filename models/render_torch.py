from typing import Any
from torch import nn, autograd
import torch
import qsdvr


class QuadricGridTorch(nn.Module):

    def __init__(self, reso: int = 128):
        super().__init__()
        self.reso = reso
        self.xLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cuda"))
        self.yLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cuda"))
        self.zLayer = nn.Parameter(
            torch.ones(reso, dtype=torch.float32, device="cuda"))
        self.offset = torch.zeros(4, dtype=torch.float32, device="cuda")
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
        reso = self.reso
        sdfGrid = torch.empty((reso, reso, reso, 7), device="cuda")
        xLayer_a3 = torch.empty(reso, device="cuda")
        yLayer_a4 = torch.empty(reso, device="cuda")
        zLayer_a5 = torch.empty(reso, device="cuda")
        xLayer_a6 = torch.empty(reso, device="cuda")
        yLayer_a6 = torch.empty(reso, device="cuda")
        zLayer_a6 = torch.empty(reso, device="cuda")
        xLayer_a3[0] = self.offset[0]
        yLayer_a4[0] = self.offset[1]
        zLayer_a5[0] = self.offset[2]
        xLayer_a6[0] = 0
        yLayer_a6[0] = 0
        zLayer_a6[0] = 0
        for i in range(1, reso):
            xLayer_a3[i] = 2 * self.xLayer[
                i - 1] + 2 * self.xLayer[i] + xLayer_a3[i - 1]
            yLayer_a4[i] = 2 * self.yLayer[
                i - 1] + 2 * self.yLayer[i] + yLayer_a4[i - 1]
            zLayer_a5[i] = 2 * self.zLayer[
                i - 1] + 2 * self.zLayer[i] + zLayer_a5[i - 1]
        for i in range(1, reso):
            xLayer_a6[i] = 3 * self.xLayer[i - 1] + self.xLayer[
                i] + 2 * xLayer_a3[i - 1] + xLayer_a6[i - 1]
            yLayer_a6[i] = 3 * self.yLayer[i - 1] + self.yLayer[
                i] + 2 * yLayer_a4[i - 1] + yLayer_a6[i - 1]
            zLayer_a6[i] = 3 * self.zLayer[i - 1] + self.zLayer[
                i] + 2 * zLayer_a5[i - 1] + zLayer_a6[i - 1]
        sdfGrid[..., 6] = self.offset[3]
        for i in range(0, reso):
            sdfGrid[:, :, i, 0] = self.xLayer[i]
            sdfGrid[:, i, :, 1] = self.yLayer[i]
            sdfGrid[i, :, :, 2] = self.zLayer[i]
            sdfGrid[:, :, i, 3] = xLayer_a3[i]
            sdfGrid[:, i, :, 4] = yLayer_a4[i]
            sdfGrid[i, :, :, 5] = zLayer_a5[i]
            sdfGrid[:, :, i, 6] += xLayer_a6[i]
            sdfGrid[:, i, :, 6] += yLayer_a6[i]
            sdfGrid[i, :, :, 6] += zLayer_a6[i]
        self.sdfGrid = sdfGrid.view(-1)
        index = sdfIndexList * 7
        a0 = self.sdfGrid[index] * sdfPointList[:, 0]
        a1 = self.sdfGrid[index + 1] * sdfPointList[:, 1]
        a2 = self.sdfGrid[index + 2] * sdfPointList[:, 2]
        a3 = self.sdfGrid[index + 3]
        a4 = self.sdfGrid[index + 4]
        a5 = self.sdfGrid[index + 5]
        a6 = self.sdfGrid[index + 6]
        sdfList = ((a0 + a3) * sdfPointList[:, 0] +
                   (a1 + a4) * sdfPointList[:, 1] +
                   (a2 + a5) * sdfPointList[:, 2] +
                   a6) / torch.sqrt((2 * a0 + a3) * (2 * a0 + a3) +
                                    (2 * a1 + a4) * (2 * a1 + a4) +
                                    (2 * a2 + a5) * (2 * a2 + a5)) / reso
        index = renderIndexList * 7
        a = torch.cat([
            2 * self.sdfGrid[index] * renderPointList[:, 0] +
            self.sdfGrid[index + 3],
            2 * self.sdfGrid[index + 1] * renderPointList[:, 1] +
            self.sdfGrid[index + 4],
            2 * self.sdfGrid[index + 2] * renderPointList[:, 2] +
            self.sdfGrid[index + 5]
        ]).reshape(3, -1).transpose(0, 1)
        normalList = torch.nn.functional.normalize(a)
        return sdfList, normalList


class RenderGridTorch(nn.Module):

    def __init__(self, reso: int = 128, logisticCoef: float = 5):
        super().__init__()
        self.quadricGrid = QuadricGridTorch(reso)
        self.reso = reso
        self.renderData = nn.Parameter(
            torch.full(((reso + 1) * (reso + 1) * (reso + 1), 32),
                       0.5,
                       dtype=torch.float32,
                       device="cuda"))
        self.logisticCoef = logisticCoef

    def forward(self, input):
        renderIndexList = input["renderIndexList"].long()
        sdfIndexList = input["sdfIndexList"].long()
        renderPointList = input["renderPointList"]
        viewList = input["viewDirList"]
        rayList = input["rayList"]
        self.sdfList, self.normalList = self.quadricGrid.forward(
            renderPointList, renderIndexList, input["sdfPointList"],
            sdfIndexList)
        reso = self.reso
        logisticCoef = self.logisticCoef
        i000 = renderIndexList
        i010 = i000 + reso
        i100 = i000 + reso * reso
        i110 = i100 + reso
        dataGrid = self.renderData
        a00 = self.Interpolation1D(dataGrid[i000, :], dataGrid[i000 + 1, :],
                                   renderPointList[:, 0])
        a01 = self.Interpolation1D(dataGrid[i010, :], dataGrid[i010 + 1, :],
                                   renderPointList[:, 0])
        a0 = self.Interpolation1D(a00, a01, renderPointList[:, 1])
        a10 = self.Interpolation1D(dataGrid[i100, :], dataGrid[i100 + 1, :],
                                   renderPointList[:, 0])
        a11 = self.Interpolation1D(dataGrid[i110, :], dataGrid[i110 + 1, :],
                                   renderPointList[:, 0])
        a1 = self.Interpolation1D(a10, a11, renderPointList[:, 1])
        self.dataList = self.Interpolation1D(a0, a1, renderPointList[:, 2])
        ao = self.dataList[:, 31:32]
        diffuse = self.dataList[:, 27:30]
        metallic = self.dataList[:, 30:31]
        specular = self.dataList[:, :27].reshape((-1, 9, 3))
        vdotn = (viewList * self.normalList).sum(dim=1).reshape((-1, 1))
        reflect = viewList - self.normalList * (2.0 * vdotn)
        specularL = self.GetSH3Irradiance(reflect, specular)
        color = diffuse * (1 - metallic) + specularL * metallic
        self.rgbList = (color * ao)
        self.image = torch.empty((rayList.size(0), 3), device="cuda")

        for info in rayList:
            exp_SDF_i = torch.exp(-logisticCoef * self.sdfList[info[1]])
            T = torch.FloatTensor([1]).cuda()[0]
            color = torch.Tensor([0.0, 0.0, 0.0]).cuda()
            for i in range(0, info[2]):
                exp_SDF_i_1 = torch.exp(-logisticCoef *
                                        self.sdfList[info[1] + i + 1])
                alpha = torch.clamp(
                    (exp_SDF_i_1 - exp_SDF_i) / (1.0 + exp_SDF_i_1), 0.0, 10.0)
                color = color + self.rgbList[info[3] + i] * alpha * T
                exp_SDF_i = exp_SDF_i_1
                T = T * (1 - alpha)
            self.image[0] = color
        return self.image

    def Interpolation1D(self, a, b, c):
        c = c.reshape([-1, 1])
        return (1 - c) * a + b * c

    def GetSH3Irradiance(self, v, coef):
        g_SHFactor = torch.Tensor([
            0.2820947917, 0.4886025119, 0.4886025119, 0.4886025119,
            1.0925484305, 1.0925484305, 0.3153915652, 1.0925484305,
            0.5462742152
        ])
        x_4 = v[:, 0:1] * v[:, 2:3]
        zz = v[:, 2:3] * v[:, 2:3]
        xx = v[:, 0:1] * v[:, 0:1]
        x_5 = v[:, 0:1] * v[:, 1:2]
        x_6 = 2.0 * v[:, 1:2] * v[:, 1:2] - zz - xx
        x_7 = v[:, 1:2] * v[:, 2:3]
        x_8 = zz - xx
        return g_SHFactor[0] * coef[:, 0, :] + g_SHFactor[
            1] * coef[:, 1, :] * v[:, 0:1] + g_SHFactor[
                2] * coef[:, 2, :] * v[:, 1:2] + g_SHFactor[
                    3] * coef[:, 3, :] * v[:, 2:3] + g_SHFactor[
                        4] * coef[:, 4, :] * x_4 + g_SHFactor[
                            5] * coef[:, 5, :] * x_5 + g_SHFactor[
                                6] * coef[:, 6, :] * x_6 + g_SHFactor[
                                    7] * coef[:, 7, :] * x_7 + g_SHFactor[
                                        8] * coef[:, 8, :] * x_8
