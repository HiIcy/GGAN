# coding:utf-8
# __author__ = yuan
# __time__ = 2020/3/31
# __file__ = augment
# __desc__ =
import collections
import math
import numbers
import random
import torchvision.transforms.functional as F
import torch
import numpy as np
import cv2
import torchvision.transforms.transforms as tf


class Transforms:
    def __init__(self, resize=None, **kwargs):
        self.resize = resize
        self.kwargs = kwargs
        self.transforms = [
            # RandomCrop(self.resize),
            # Flip(p=0.9),
            # Distort(),
            ToTensor(),
            Normalize(),
        ]

    def __call__(self, X, Y):
        rX = X.copy()
        rY = Y.copy()
        for transform in self.transforms:
            rX, rY = transform(rX, rY)
        return rX, rY


class ValTransforms:
    def __init__(self, resize=None, **kwargs):
        self.resize = resize
        self.kwargs = kwargs
        self.transforms = [
            ToTensor(),
            Normalize(),
        ]

    def __call__(self, X, Y):
        rX = X.copy()
        rY = Y.copy()
        for transform in self.transforms:
            rX, rY = transform(rX, rY)
        return rX, rY


class RandomCropV1:
    def __init__(self, resize, scale=(0., 0.2)):
        self.resize = resize
        self.scale = scale
        assert self.scale is not None

    def __call__(self, X, Y):
        oh, ow, _ = X.shape
        if max(oh, ow) < self.resize or min(oh, ow) < self.resize:
            edge = min(oh, ow)
            if oh == edge:
                ratio = self.resize / oh
            else:
                ratio = self.resize / ow
        else:
            ratio = 1.
        X = cv2.resize(X, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        h, w, _ = X.shape
        y = int(random.uniform(0, h - self.resize))  # 不会越界
        x = int(random.uniform(0, w - self.resize))

        cX = X.copy()[y:y + self.resize, x:x + self.resize, :]
        # Y = cv2.resize(Y, (w, h), interpolation=cv2.INTER_CUBIC)
        cY = Y.copy()[y:y + self.resize, x:x + self.resize, :]
        return cX, cY


class RandomCrop:
    # 专为watermask写的randomcrop
    def __init__(self, resize, scale=(0., 0.2)):
        self.resize = resize
        self.scale = scale
        assert self.scale is not None

    def __call__(self, X, Y):
        oh, ow, _ = X.shape
        if ow == self.resize:
            return (X, Y)
        w = self.resize
        ratio = self.resize / ow
        return (cv2.resize(X, (0, 0), fx=ratio, fy=ratio),
                cv2.resize(Y, (0, 0), fx=ratio, fy=ratio))


class Normalize:
    def __init__(self):
        # hard code
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
    def __call__(self, X: torch.FloatTensor, Y):
        c = X.shape[0]
        for i in range(c):
            X[i, :, :].sub_(self.mean[i]).div_(self.std[i])
            Y[i, :, :].sub_(self.mean[i]).div_(self.std[i])
        return X, Y
        # for t,m,s in zip(XX,self.mean,self.std):
        #     t.sub_(m).div_(s)


class ToTensor:

    def __call__(self, X, Y):
        return F.to_tensor(X), F.to_tensor(Y)


class RandomRotate:
    # TODO:这个项目不能旋转，否则无法对应上
    def __init__(self, rotate=(-10, 10)):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X, Y):
        if random.random() > self.p:
            X = cv2.flip(X, 1)
            Y = cv2.flip(Y, 1)
        return X, Y


class Distort:
    def _convert(self, image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def _enhance(self, X):
        image = X.copy()
        # brightness distortion
        if random.randrange(2):
            self._convert(image, beta=random.uniform(0, 8))

        # contrast distortion
        if random.randrange(2):
            self._convert(image, alpha=random.uniform(0.2, 1.2))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #
        # # saturation distortion
        # if random.randrange(2):
        #     self._convert(image[:, :, 1], alpha=random.uniform(0.1, 1.1))
        #
        # # hue distortion
        # if random.randrange(2):
        #     tmp = image[:, :, 0].astype(int) + random.randint(-1, 10)
        #     tmp %= 180
        #     image[:, :, 0] = tmp
        #
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image

    def __call__(self, X, Y):
        rX = self._enhance(X)
        rY = self._enhance(Y)
        return rX, rY


if __name__ == '__main__':
    def reverse_norm(X, Y):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        c = X.shape[0]
        for i in range(c):
            X[i, :, :] = X[i, :, :] * std[i] + mean[i]
            Y[i, :, :] = Y[i, :, :] * std[i] + mean[i]
        X = X * 255
        Y = Y * 255
        X = np.transpose(X, (1, 2, 0))
        Y = np.transpose(Y, (1, 2, 0))
        X = X.astype(np.uint8)
        Y = Y.astype(np.uint8)
        return X,Y

    p = r"F:\Resources\DataSets\OCRID\enhance\front_pos\000d0143a5b442bdb8dd5d92e49f3b81.jpg"
    p = r"E:\deeper\New\Another\pytorch-CycleGAN-and-pix2pix\datasets\facades\train\8.jpg"
    img = cv2.imread(p)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    middle = w // 2
    X = image[:, :middle, :]
    Y = image[:, middle:, :]
    X, Y = Y, X
    trans = Transforms(256)
    cX, cY = trans(X, Y)

    cX,cY = reverse_norm(cX.cpu().numpy(),cY.cpu().numpy())
    cv2.imshow('1', X)
    cv2.imshow('2', Y)
    cv2.imshow('3', cX)
    cv2.imshow('4', cY)
    cv2.waitKey(2000)
