# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/1
# __file__ = train
# __desc__ =

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
from option.base_option import TrainOption
from data import facade
from torch.utils.data import DataLoader
from action import pix2pixac,cycleganac
from data.augment import Transforms,ValTransforms
import torch.nn as nn


opt = TrainOption().get_arg()
if not os.path.exists(opt.model_save):
    os.mkdir(opt.model_save)
if not os.path.exists(opt.log_dir):
    os.mkdir(opt.log_dir)
if not os.path.exists(opt.preview_dir):
    os.mkdir(opt.preview_dir)

datamode:str = opt.data_mode
datamode = datamode.title()
dc_name = f"{datamode}"
datasetclass:facade.Unaligneddataset = getattr(facade,f"{dc_name}dataset")

trainset = datasetclass(opt.data_root,opt,"train",Transforms)
trainloader = DataLoader(trainset,opt.batch,shuffle=True,num_workers=opt.worker)
valset = datasetclass(opt.data_root,opt,"val",ValTransforms)
valloader = DataLoader(valset,opt.val_batch,shuffle=False,num_workers=opt.worker)

netname:str = opt.netname
if netname == "pix2pix":
    actionclass = getattr(pix2pixac,f"{netname.title()}ac")
else:
    actionclass = getattr(cycleganac,f"{netname.title()}ac")

reader = actionclass(opt)


bef_loss = float("inf")
for epoch in range(opt.start_epoch,opt.nb_epoch):
    try:
        reader.train(trainloader,epoch)
        #cur_loss = reader.val(valloader,criterionGAN,criterionL1,epoch)
        #if cur_loss < bef_loss:
        #    bef_loss = cur_loss
        #    reader.save(bef_loss)
        reader.adjust_lr(epoch)
    except KeyboardInterrupt:
        break

presurfix=time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time()))
reader.save(presurfix)


