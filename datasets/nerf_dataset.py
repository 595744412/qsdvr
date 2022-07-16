import json
from os import path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import imageio
from datasets.utils import Camera
from qsdvr import CameraArgs


class NerfDataset(Dataset):

    def __init__(self,
                 root,
                 near=0.5,
                 far=2.5,
                 scene_scale=[1.0, 1.0, 1.0],
                 scene_offset=[0.0, 0.0, 0.0],
                 interval: float = 0.03,
                 reso: int = 128):
        super().__init__()

        self.interval = interval
        self.reso = reso
        self.near = near
        self.far = far

        all_c2w = []
        all_gt = []

        split_name = "train"
        data_path = path.join(root, split_name)
        data_json = path.join(root, "transforms_" + split_name + ".json")

        print("load dataset", data_path)

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))

        for frame in tqdm(j["frames"]):
            fpath = path.join(data_path,
                              path.basename(frame["file_path"]) + ".png")
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath)
            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
        self.focal = float(0.5 * all_gt[0].shape[1] /
                           np.tan(0.5 * j["camera_angle_x"]))
        self.c2w = np.stack(all_c2w)
        self.c2w[:, :3, 3] += np.array(scene_offset)
        self.c2w[:, :3, 3] *= np.array(scene_scale)
        all_gt = torch.stack(all_gt).float() / 255.0
        self.n_images, self.height, self.width, _ = all_gt.shape
        all_gt = all_gt.reshape((self.n_images, -1, 4))

        self.cx = -self.width / 2
        self.cy = -self.height / 2

        print("init raypoints")
        self.gt = []
        self.mask_index = []
        self.cameras = []
        for c2w, img in tqdm(zip(self.c2w, all_gt)):
            mask = img[..., 3] != 0
            self.mask_index.append(mask.nonzero().numpy().reshape(-1))
            self.gt.append(img[mask, :3])
            camera = self.create_camera(c2w)
            camera.GenerateRayPoints(mask)
            self.cameras.append(camera)

    def create_camera(self, c2w):
        args = CameraArgs()
        args.width = self.width
        args.height = self.height
        args.rotation = c2w[:3, :3].reshape(-1).tolist()
        args.translation = c2w[:3, 3].reshape(-1).tolist()
        args.fx = self.focal
        args.fy = self.focal
        args.cx = self.cx
        args.cy = self.cy
        args.near = self.near
        args.far = self.far
        return Camera(args, self.interval, self.reso)

    def get_whole_image(self, sparseImage, index):
        mask_index = self.mask_index[index]
        img = np.zeros((self.width * self.height, 3))
        img[mask_index] = sparseImage
        return img.reshape(self.height, self.width, 3)

    def __getitem__(self, index):
        return self.cameras[index].GetRayPoints(), self.gt[index].cuda()

    def __len__(self):
        return self.n_images