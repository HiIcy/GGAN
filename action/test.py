# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/13
# __file__ = test
# __desc__ =

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from option.base_option import TestOption
from data import facade
from torch.utils.data import DataLoader
from action import pix2pixac,cycleganac
from data.augment import Transforms,ValTransforms
import torch.nn as nn


opt = TestOption().get_arg()

datamode:str = opt.data_mode
datamode = datamode.title()
dc_name = f"{datamode}"
datasetclass:facade.Unaligneddataset = getattr(facade,f"{dc_name}dataset")

testset = datasetclass(opt.data_root,opt,"test",ValTransforms)
testloader = DataLoader(testset,opt.test_batch,num_workers=opt.worker,shuffle=False)

netname:str = opt.netname
if netname == "pix2pix":
    actionclass = getattr(pix2pixac,f"{netname.title()}testac")
else:
    actionclass = getattr(cycleganac,f"{netname.title()}testac")

reader:cycleganac.Cyclegantestac = actionclass(opt)

# reader.inference(testloader)
# img=r'/data/soft/javad/COCO/horse2zebra/testA/n02381460_2050.jpg'
# reader.single_if(img)
reader.inference(testloader)