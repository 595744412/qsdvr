import torch
from qsdvr import CameraArgs, RayCut, GenerateRayPoints


class Camera(object):

    def __init__(self,
                 args: CameraArgs,
                 interval: float = 0.03,
                 reso: int = 128):
        self.args = args
        self.args.pixelCount = args.width * args.height
        self.interval = interval
        self.reso = reso

    def GetRayPoints(self):
        dataList = {}
        dataList["rayList"] = self.rayList.cuda()
        dataList["sdfPointList"] = torch.zeros((self.args.sdfCount, 3),
                                               dtype=torch.float32,
                                               device="cuda")
        dataList["sdfIndexList"] = torch.zeros((self.args.sdfCount),
                                               dtype=torch.int32,
                                               device="cuda")
        dataList["renderPointList"] = torch.zeros((self.args.renderCount, 3),
                                                  dtype=torch.float32,
                                                  device="cuda")
        dataList["renderIndexList"] = torch.zeros((self.args.renderCount),
                                                  dtype=torch.int32,
                                                  device="cuda")
        dataList["viewDirList"] = torch.zeros((self.args.renderCount, 3),
                                              dtype=torch.float32,
                                              device="cuda")
        GenerateRayPoints(self.args, dataList["sdfPointList"],
                          dataList["sdfIndexList"],
                          dataList["renderPointList"],
                          dataList["renderIndexList"],
                          dataList["viewDirList"], dataList["rayList"],
                          self.originList.cuda(), self.rangeList.cuda(),
                          self.dirList.cuda(), self.interval, self.reso)
        return dataList

    def GenerateRayPoints(self, mask):
        self.rayList = torch.zeros((self.args.pixelCount, 4),
                                   dtype=torch.int32,
                                   device="cuda")
        self.originList = torch.zeros((self.args.pixelCount, 3),
                                      dtype=torch.float32,
                                      device="cuda")
        self.rangeList = torch.zeros((self.args.pixelCount, 2),
                                     dtype=torch.float32,
                                     device="cuda")
        self.dirList = torch.zeros((self.args.pixelCount, 3),
                                   dtype=torch.float32,
                                   device="cuda")
        RayCut(self.args, mask, self.rayList, self.originList, self.rangeList,
               self.dirList, self.interval)
        self.rayList = self.rayList.cpu()[mask, ...]
        self.originList = self.originList.cpu()[mask, ...]
        self.rangeList = self.rangeList.cpu()[mask, ...]
        self.dirList = self.dirList.cpu()[mask, ...]
