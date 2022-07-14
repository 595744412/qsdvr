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
        return self.renderPointList, self.renderIndexList, self.sdfPointList, self.sdfIndexList, self.depthList, self.viewDirList, self.rayList

    def GenerateRayPoints(self):
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
        RayCut(self.args, self.rayList, self.originList, self.rangeList,
               self.dirList, self.interval)
        self.sdfPointList = torch.zeros((self.args.sdfCount, 3),
                                        dtype=torch.float32,
                                        device="cuda")
        self.sdfIndexList = torch.zeros((self.args.sdfCount),
                                        dtype=torch.int32,
                                        device="cuda")
        self.renderPointList = torch.zeros((self.args.renderCount, 3),
                                           dtype=torch.float32,
                                           device="cuda")
        self.renderIndexList = torch.zeros((self.args.renderCount),
                                           dtype=torch.int32,
                                           device="cuda")
        self.depthList = torch.zeros((self.args.renderCount),
                                     dtype=torch.float32,
                                     device="cuda")
        self.viewDirList = torch.zeros((self.args.renderCount, 3),
                                       dtype=torch.float32,
                                       device="cuda")
        GenerateRayPoints(self.args, self.sdfPointList, self.sdfIndexList,
                          self.renderPointList, self.renderIndexList,
                          self.depthList, self.viewDirList, self.rayList,
                          self.originList, self.rangeList, self.dirList,
                          self.interval, self.reso)
