# coding:utf-8
# __author__ = yuan
# __time__ = 2020/3/31
# __file__ = facade
# __desc__ =
import os
import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset


class Aligneddataset(Dataset):
    def __init__(self, img_root, opt,phase, transforms=None):
        """
        :param opt: 命令行传过来的需要解析 该数据集全是paired
        """
        self.img_root = os.path.join(img_root,phase)
        self.imgs = os.listdir(self.img_root)
        self.opt = opt
        self.transf = transforms
        self.direction = opt.direction
        self.resize = opt.crop_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_root, img_name)
        # print(img_path)
        # AB = Image.open(img_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        #
        # X = AB.crop((0, 0, w2, h))
        # Y = AB.crop((w2, 0, w, h))
        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        middle = w//2
        X = image[:, :middle, :]
        Y = image[:, middle:, :]
        if self.direction == "BtoA":
            X, Y = Y, X

        # assert X.shape == Y.shape

        if self.transf:
            X,Y = self.transf(resize=self.resize)(X,Y)

        return X,Y

class Unaligneddataset(Dataset):
    def __init__(self, img_root, opt,phase, transforms=None):
        self.img_root = img_root
        self.dirA = os.path.join(img_root,f"{phase}A")
        self.dirB = os.path.join(img_root,f"{phase}B")
        self.imgAs = os.listdir(self.dirA)
        self.imgBs = os.listdir(self.dirB)
        self.opt = opt
        self.transf = transforms
        self.direction = opt.direction
        self.resize = opt.crop_size

    def __len__(self):
        return len(self.imgAs)

    def __getitem__(self, idx):
        imgA_name = self.imgAs[idx]
        imgB_name = self.imgBs[idx]
        imgA_path = os.path.join(self.dirA, imgA_name)
        imgB_path = os.path.join(self.dirB, imgB_name)
        X = cv2.imread(imgA_path)
        Y = cv2.imread(imgB_path)

        if self.direction == "BtoA":
            X, Y = Y, X
        assert X.shape == Y.shape
        if self.transf:
            X, Y = self.transf(resize=self.resize)(X, Y)
        return X, Y
