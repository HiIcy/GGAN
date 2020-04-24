# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/1
# __file__ = pix2pixac
# __desc__ =
import os
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from net import pix2pix
import torch.nn.functional as F
from torch.optim import adam, lr_scheduler
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torchvision.utils as vutils
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils.loss import GANLoss
import torchvision.transforms.transforms as tf
from PIL.Image import Image
import matplotlib.pyplot as plt
from utils.init import init_weights
plt.switch_backend('agg')
class Baseac:
    def __init__(self,opt):
        self.opt = opt
        self.writer = SummaryWriter(opt.log_dir)
        self.true = True
        self.fake = False
        self.device = f"cuda:{opt.gpuids[0]}" if torch.cuda.is_available() and len(opt.gpuids) > 0 else "cpu"

    def train(self):
        pass

    def val(self):
        pass

    def inference(self):
        pass


class Pix2Pixac(Baseac):
    def __init__(self, opt):
        super().__init__(opt)
        self.G = pix2pix.Generator(nic=opt.nin,ngf=opt.ngf, noc=opt.nc)
        # self.G = pix2pix.UnetGenerator(3,3,8,use_dropout=True)
        # self.D = pix2pix.PatchGAN(3*2, 64)
        self.D = pix2pix.NLayerDiscriminator(6,64)
        if torch.cuda.is_available() and len(opt.gpuids) > 0:
            self.G = DataParallel(self.G, device_ids=opt.gpuids).cuda(opt.gpuids[0])
            self.D = DataParallel(self.D, device_ids=opt.gpuids).cuda(opt.gpuids[0])
        init_weights(self.G)
        init_weights(self.D)
        self.optimizerD = adam.Adam(self.D.parameters(),betas=(0.5,0.99), lr=opt.lr)
        self.optimizerG = adam.Adam(self.G.parameters(),betas=(0.5,0.99), lr=opt.lr)
        self.global_step = 0
        # self.criterionGAN = nn.BCELoss()
        # self.criterionGAN = nn.BCEWithLogitsLoss()
        # 用的mse，所以判别器没有sigmoid
        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        self.schedulerG = lr_scheduler.LambdaLR(self.optimizerG, lr_lambda=lambda_rule)
        self.schedulerD = lr_scheduler.LambdaLR(self.optimizerD, lr_lambda=lambda_rule)

    def adjust_lr(self, epoch):
        # lr = self.opt.lr * (0.1 ** (epoch // 100))
        # if epoch>=650:
        #     lr=0.
        # for param_group in self.optimizerD.param_groups:
        #     param_group['lr'] = lr
        # for param_group in self.optimizerG.param_groups:
        #     param_group['lr'] = lr
        self.schedulerG.step()
        self.schedulerD.step()
        lr = self.optimizerD.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        self.writer.add_scalar("lr", lr, global_step=epoch)

    # TODO:持久化训练
    def train(self, dataloader, epoch):
        print(f"start train {epoch}")
        # self.G.train(True)
        # self.D.train(True)
        lossD = 0.
        lossG_gan = 0.
        lossG_l1 = 0.
        # numloader=len(dataloader)
        # self.global_step=epoch*numloader
        for i, (X, Y) in enumerate(dataloader):

            ##########
            # 先训练判别器的真数据，再训练假数据
            ##########
            N = Y.shape[0]
            X = X.to(self.device)  # 需要转化的数据
            Y = Y.to(self.device)  # 目标数据
            fake_Y = self.G(X)

            self.set_requires_grad(self.D,True)
            self.optimizerD.zero_grad()

            # 假样本前向传递
            # 这里就把bce损失前后部分都算上了
            fake_XY = torch.cat([X, fake_Y],1)
            pred_fake = self.D(fake_XY.detach())  # REW:注意这里detach妙用，反向传播时就不影响生成器了

            # fixme:先概率平均再算？
            errD_fake = self.criterionGAN(pred_fake, self.fake)

            # 判别器是成对喂入
            pred_real = self.D(torch.cat([X,Y],dim=1))
            errD_real = self.criterionGAN(pred_real, self.true)

            errD = (errD_real + errD_fake) * 0.5
            # errD_real.backward()  # real的梯度
            errD.backward()
            # lossD += errD.item()
            self.optimizerD.step()

            # self.writer.add_scalar("errD_fake",errD_fake.item(),global_step=self.global_step+i)
            # self.writer.add_scalar("errD_real",errD_real.item(),global_step=self.global_step+i)

            ###### 计算生成器的梯度
            self.set_requires_grad(self.D,False)  # 手动控制判别器不梯度传播,其实我也感觉不用设置，D一开始就会清零梯度
            self.optimizerG.zero_grad()

            pred_fake_2 = self.D(torch.cat([X,fake_Y],dim=1))
            # output = output.squeeze(1)
            errG_fake = self.criterionGAN(pred_fake_2, self.true)
            errG_l1 = self.criterionL1(fake_Y, Y)*self.opt.alpha

            errG = errG_fake + errG_l1
            # lossG_gan += errG_fake.item()
            # lossG_l1 += errG_l1.item()
            errG.backward()
            self.optimizerG.step()

            # self.writer.add_scalar("errG_fake",errG_fake.item(),global_step=self.global_step+i)
            # self.writer.add_scalar("errG_l1",errG_l1.item(),global_step=self.global_step+i)
            # print(float(errG_fake),float(errG_l1),float(errD_real),float(errD_fake),sep=' ')
            # input('go on \n> ')
            if (i + 1) % self.opt.show_interval == 0:
                self.writer.add_scalar("errG_fake", errG_fake.item(), global_step=self.global_step)
                self.writer.add_scalar("errG_l1",errG_l1.item(),global_step=self.global_step)
                self.writer.add_scalar("errD_fake", errD_fake.item(), global_step=self.global_step)
                self.writer.add_scalar("errD_real",errD_real.item(),global_step=self.global_step)

                # mean_lossG_gan = lossG_gan / self.opt.show_interval
                # mean_lossG_l1 = lossG_l1 / self.opt.show_interval
                # mean_lossD = lossD / self.opt.show_interval
                print(f"[{epoch}/{self.opt.nb_epoch - self.opt.start_epoch}]"
                      f"[{i + 1}/{len(dataloader)}]"
                      f"LossG_gan: {float(errG_fake)} "
                      f"LossG_l1: {float(errG_l1)} "
                      f"LossD_real: {float(errD_real)} "
                      f"LossD_fake:　{float(errD_fake)}")
                # lossD = 0.
                # lossG_l1 = 0.
                # lossG_gan = 0.
                self.global_step += 1
            if (epoch+1) % 5 ==0 and (i+1) % 100 == 0:
                self.G.eval()
                self.D.eval()

                fake_Y = self.G(X)[0].data
                fake_Y = fake_Y.cpu().float().numpy()
                # fake_Y = np.squeeze(fake_Y, 0)
                fake_Y = self.reverse_norm(fake_Y)

                cv2.imwrite(f"{self.opt.preview_dir}/{epoch}_{i}_.png",fake_Y)
                self.G.train()
                self.D.train()
            # if (i+1) % self.opt.vis_interval == 0:
            #     vutils.make_grid()

    def save(self, loss):
        path = f"{self.opt.model_save}/ckpt_{loss}_.tar"
        if torch.cuda.is_available() and len(self.opt.gpuids) > 0:
            torch.save({
                'modelG_state_dict': self.G.module.state_dict(),
                'modelD_state_dict': self.D.module.state_dict(),
                'optimizerG_state_dict': self.optimizerG.state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict()
            }, path)
        else:
            torch.save({
                'modelG_state_dict': self.G.state_dict(),
                'modelD_state_dict': self.D.state_dict(),
                'optimizerG_state_dict': self.optimizerG.state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict()
            }, path)

    def reverse_norm(self,X:np.ndarray):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        # c = X.shape[0]
        # for i in range(c):
        #     X[i,:,:] = X[i,:,:]*std[i]+mean[i]
        X = ((np.transpose(X,(1,2,0))+ 1) / 2.0) * 255.
        # X = X * 255
        # X = np.transpose(X,(1,2,0))
        X = X.astype(np.uint8)
        # X = 0.5*X+0.5
        return X

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Pix2Pixtestac:
    def __init__(self, opt):
        self.opt = opt
        self.G = pix2pix.Generator(nic=opt.nin, ngf=opt.ngf, noc=opt.nc)
        # self.G = pix2pix.UnetGenerator(3,3,8,use_dropout=True)
        self.device = f"cuda:{opt.gpuids[0]}" if torch.cuda.is_available() and len(opt.gpuids) > 0 else "cpu"

        if not os.path.exists(opt.model_save):
            raise FileExistsError("模型保存目录不存在")
        # 默认获取最新tar文件
        tars = [os.path.join(opt.model_save,tar) for tar in os.listdir(opt.model_save)]
        if len(tars)==0:
            print("Not find any tar file")
            exit(-1)
        tar_times = [os.path.getmtime(tar) for tar in tars]
        idx = tar_times.index(max(tar_times))
        tar_file = tars[idx]
        print(f"load file: {tar_file}")
        checkpoint = torch.load(tar_file,map_location=self.device)
        self.G.load_state_dict(checkpoint['modelG_state_dict'])

        if torch.cuda.is_available() and len(opt.gpuids) > 0:
            self.G = DataParallel(self.G, device_ids=opt.gpuids).cuda(opt.gpuids[0])
        if not os.path.exists(opt.result_dir):
            os.mkdir(opt.result_dir)

    def reverse_norm(self,X:np.ndarray):

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # REW:
        X = ((np.transpose(X, (1, 2, 0)) + 1) / 2.0) * 255.
        X = X.astype(np.uint8)
        return X

    def inference(self,dataloader):
        self.G.eval()
        # self.D.eval()
        dataset = dataloader.dataset
        if self.opt.data_mode=="aligned":
            imgs = dataset.imgs
        else:
            imgs = dataset.imgAs
        result_dir = self.opt.result_dir
        for i,(X,Y) in enumerate(dataloader):
            X = X.to(self.device)  # 需要转化的数据
            Y = Y.to(self.device)  # 目标数据
            fake_Y = self.G(X)
            img = imgs[i]
            name = img.rsplit(".")[0]
            realA_name = name+"_realA.png"
            realB_name = name+"_realB.png"
            fakeB_name = name+"_fakeB.png"

            with ThreadPoolExecutor() as exe:
                X = X.cpu().float().numpy()
                Y = Y.cpu().float().numpy()
                fake_Y = fake_Y.cpu().float().data.numpy()
                X = np.squeeze(X,0)
                Y = np.squeeze(Y,0)
                fake_Y = np.squeeze(fake_Y,0)
                X = self.reverse_norm(X)
                Y = self.reverse_norm(Y)
                fake_Y = self.reverse_norm(fake_Y)
                exe.map(cv2.imwrite,[os.path.join(result_dir,fakeB_name),
                                     os.path.join(result_dir,realA_name),
                                     os.path.join(result_dir,realB_name)],
                                    [fake_Y,
                                     X,
                                     Y])
        print('inference done!')



