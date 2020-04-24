# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/14
# __file__ = cycleganac
# __desc__ =
import os
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from net import cyclegan
import torch.nn.functional as F
from torch.optim import adam
import torch.nn as nn
from itertools import chain
from utils.image_pool import ImagePool
from torchvision.transforms import transforms
from utils.loss import GANLoss
from torch.nn.parallel import DataParallel
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils import mkdirs
from utils.init import init_weights


class Cycleganac():
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.G_XtoY = cyclegan.Generateor(opt.nin, opt.nc,use_dropout=True)
        self.G_YtoX = cyclegan.Generateor(opt.nin, opt.nc,use_dropout=True)

        # 判断domainX的
        self.D_X = cyclegan.PatchGAN(opt.nin, ndf=64)
        self.D_Y = cyclegan.PatchGAN(opt.nin, ndf=64)

        self.fake_Y_pool = ImagePool(opt.pool_size)
        self.fake_X_pool = ImagePool(opt.pool_size)

        self.nets = [self.G_XtoY, self.G_YtoX, self.D_X, self.D_Y]
        self.writer = SummaryWriter(opt.log_dir)
        self.true = True
        self.fake = False
        self.device = f"cuda:{opt.gpuids[0]}" if torch.cuda.is_available() and len(opt.gpuids) > 0 else "cpu"

        if torch.cuda.is_available() and len(opt.gpuids) > 0:
            self.G_XtoY = DataParallel(self.G_XtoY, device_ids=opt.gpuids).cuda(opt.gpuids[0])
            self.G_YtoX = DataParallel(self.G_YtoX, device_ids=opt.gpuids).cuda(opt.gpuids[0])
            self.D_X = DataParallel(self.D_X, device_ids=opt.gpuids).cuda(opt.gpuids[0])
            self.D_Y = DataParallel(self.D_Y, device_ids=opt.gpuids).cuda(opt.gpuids[0])

        # fixme:betas用参数传递
        # REW:注意这里的共享参数的实现方式
        self.optimizerD = adam.Adam(chain(self.D_X.parameters(),self.D_Y.parameters()), betas=(0.5, 0.999), lr=opt.lr)
        self.optimizerG = adam.Adam(chain(self.G_XtoY.parameters(),self.G_YtoX.parameters()), betas=(0.5, 0.999), lr=opt.lr)

        self.optimizers = [self.optimizerD,self.optimizerG]

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        self.schedulerG = lr_scheduler.LambdaLR(self.optimizerG, lr_lambda=lambda_rule)
        self.schedulerD = lr_scheduler.LambdaLR(self.optimizerD, lr_lambda=lambda_rule)

        self.global_step = 0

        self.criterion_cyc = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()
        # FIXME:由对数似然损失改为均方误差损失
        self.criterion_gan = GANLoss(opt.gan_mode).to(self.device)

    def adjust_lr(self, epoch):
        self.schedulerG.step()
        self.schedulerD.step()
        lr = self.optimizerG.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        self.writer.add_scalar("lr", lr, global_step=epoch)

    def train(self, loader, epoch):
        print(f"start train {epoch}")
        for net in self.nets:
            net.train(True)

        num = len(loader)
        self.global_step = num * epoch
        for i, (X, Y) in enumerate(loader):
            N = Y.shape[0]
            X = X.to(self.device)  # 需要转化的数据
            Y = Y.to(self.device)
            #### X to Y 再 to X
            """
            1.forward：两个生成器四步forward
            2.禁止判别器更新,更新生成器(包括循环一致性损失，identity loss)
            3.允许判别器，更新生成器
            """
            fake_Y = self.G_XtoY(X)
            fake_recX = self.G_YtoX(fake_Y)
            fake_X = self.G_YtoX(Y)
            fake_recY = self.G_XtoY(fake_X)
            #####   生成器损失 3部分组成
            self.set_requires_grad([self.D_X, self.D_Y], False)

            self.optimizerG.zero_grad()

            # 循环一致损失
            errG_recX = self.criterion_cyc(fake_recX, X)*self.opt.lambda_X
            errG_recY = self.criterion_cyc(fake_recY, Y)*self.opt.lambda_Y
            # identity loss
            if self.opt.lambda_idt>0:
                idtY = self.G_XtoY(Y)
                errG_idtY = self.criterion_idt(idtY,Y)*self.opt.lambda_Y*self.opt.lambda_idt
                idtX = self.G_YtoX(X)
                errG_idtX = self.criterion_idt(idtX,X)*self.opt.lambda_X*self.opt.lambda_idt
            else:
                errG_idtX=0.
                errG_idtY=0.
            # GAN loss D_Y(G_X2Y(X))
            pred_fake_Y = self.D_Y(fake_Y)
            errG_X = self.criterion_gan(pred_fake_Y,self.true)
            # GAN loss D_X(G_Y2X(Y))
            pred_fake_X = self.D_X(fake_X)
            errG_Y = self.criterion_gan(pred_fake_X,self.true)
            errG = errG_recX+errG_recY+errG_idtX+errG_idtY+errG_X+errG_Y

            errG.backward()
            self.optimizerG.step()

            #####   判别器损失
            self.set_requires_grad([self.D_X, self.D_Y], True)
            self.optimizerD.zero_grad()
            # 这里和pix2pix 判别器损失差不多
            pred_real_X = self.D_X(X)
            errD_X_real = self.criterion_gan(pred_real_X,self.true)
            fake_X = self.fake_X_pool.query(fake_X)
            pred_fake_X2 = self.D_X(fake_X.detach())
            errD_X_fake = self.criterion_gan(pred_fake_X2,self.fake)
            errD_X = (errD_X_real + errD_X_fake) * 0.5
            errD_X.backward()

            pred_real_Y = self.D_Y(Y)
            errD_Y_real = self.criterion_gan(pred_real_Y, self.true)
            fake_Y = self.fake_Y_pool.query(fake_Y)
            pred_fake_Y2 = self.D_Y(fake_Y.detach()) # detach，不影响生成器梯度
            errD_Y_fake = self.criterion_gan(pred_fake_Y2, self.fake)
            errD_Y = (errD_Y_real + errD_Y_fake) * 0.5
            errD_Y.backward()

            self.optimizerD.step()

            self.writer.add_scalar("Loss_Dx",float(errD_X),self.global_step+i)
            self.writer.add_scalar("Loss_Dy",float(errD_Y),self.global_step+i)
            self.writer.add_scalar("Loss_Gx",float(errG_X),self.global_step+i)
            self.writer.add_scalar("Loss_Gy",float(errG_Y),self.global_step+i)
            self.writer.add_scalar("Loss_Idtx",float(errG_idtX),self.global_step+i)
            self.writer.add_scalar("Loss_Idty",float(errG_idtY),self.global_step+i)
            self.writer.add_scalar("Loss_cycx",float(errG_recX),self.global_step+i)
            self.writer.add_scalar("Loss_cycy",float(errG_recY),self.global_step+i)

            if (i+1) % self.opt.show_interval ==0:
                print(f"[{epoch}/{self.opt.nb_epoch - self.opt.start_epoch}]"
                      f"[{i}/{len(loader)}]"
                      f"Loss_Dx:{float(errD_X)}"
                      f"Loss_Dy:{float(errD_Y)}"
                      f"Loss_Gx:{float(errG_X)}"
                      f"Loss_Gy:{float(errG_Y)}"
                      f"Loss_Idtx:{float(errG_idtX)}"
                      f"Loss_Idty:{float(errG_idtY)}"
                      f"Loss_Grecx:{float(errG_recX)}"
                      f"Loss_Grecy:{float(errG_recY)}")


            if (epoch+1) % 10 ==0 and (i+1) % 100 == 0:
                self.G_XtoY.eval()
                self.G_YtoX.eval()

                fake_Y = self.G_XtoY(X)[0].detach()
                fake_Y = fake_Y.cpu().float().numpy()
                fake_Y = self.reverse_norm(fake_Y)

                fake_X = self.G_YtoX(Y)[0].detach()
                fake_X = fake_X.cpu().float().numpy()
                fake_X = self.reverse_norm(fake_X)
                cv2.imwrite(f"{self.opt.preview_dir}/{epoch}_{i}_X2Y_.png",fake_Y)
                cv2.imwrite(f"{self.opt.preview_dir}/{epoch}_{i}_Y2X_.png",fake_X)
                self.G_XtoY.train()
                self.G_YtoX.train()

    def save(self, loss):
        path = f"{self.opt.model_save}/ckpt_{loss}_.tar"
        if torch.cuda.is_available() and len(self.opt.gpuids) > 0:
            torch.save({
                'modelG_XtoY_state_dict': self.G_XtoY.module.state_dict(),
                'modelD_YtoX_state_dict': self.G_YtoX.module.state_dict(),
                'modelD_X_state_dict': self.D_X.module.state_dict(),
                'modelD_Y_state_dict': self.D_Y.module.state_dict(),
                'optimizerG_state_dict': self.optimizerG.state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict()
            }, path)
        else:
            torch.save({
                'modelG_XtoY_state_dict': self.G_XtoY.state_dict(),
                'modelD_YtoX_state_dict': self.G_YtoX.state_dict(),
                'modelD_X_state_dict': self.D_X.state_dict(),
                'modelD_Y_state_dict': self.D_Y.state_dict(),
                'optimizerG_state_dict': self.optimizerG.state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict()
            }, path)

    #FIXME: 这里以后用继承类的方式
    def reverse_norm(self,X:np.ndarray):
        X = ((np.transpose(X,(1,2,0))+ 1) / 2.0) * 255.
        X = X.astype(np.uint8)
        return X

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Cyclegantestac:
    def __init__(self, opt):
        self.opt = opt
        self.G_XtoY = cyclegan.Generateor(opt.nin, opt.nc)
        self.G_YtoX = cyclegan.Generateor(opt.nin, opt.nc)
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
        self.G_XtoY.load_state_dict(checkpoint['modelG_XtoY_state_dict'])
        self.G_YtoX.load_state_dict(checkpoint['modelD_YtoX_state_dict'])

        if torch.cuda.is_available() and len(opt.gpuids) > 0:
            self.G_XtoY = DataParallel(self.G_XtoY, device_ids=opt.gpuids).cuda(opt.gpuids[0])
            self.G_YtoX = DataParallel(self.G_YtoX, device_ids=opt.gpuids).cuda(opt.gpuids[0])
        mkdirs(opt.result_dir,need_remove=True)

    def reverse_norm(self,X:np.ndarray):

        X = ((np.transpose(X, (1, 2, 0)) + 1) / 2.0) * 255.
        X = X.astype(np.uint8)
        return X

    def inference(self,dataloader):
        # self.G_XtoY.eval()
        # self.G_YtoX.eval()

        dataset = dataloader.dataset
        if self.opt.data_mode=="aligned":
            imgAs = dataset.imgs
            imgBs = imgAs
        else:
            # imgAs = [os.path.join(dataset.dirA,ia) for ia in dataset.imgAs]
            imgAs = dataset.imgAs
            imgBs = dataset.imgBs

        result_dir = self.opt.result_dir
        with torch.no_grad():
            for i,(X,Y) in enumerate(dataloader):
                X = X.to(self.device)  # 需要转化的数据
                Y = Y.to(self.device)  # 目标数据
                fake_Y = self.G_XtoY(X)
                fake_recX = self.G_YtoX(fake_Y)
                fake_X = self.G_YtoX(Y)
                fake_recY = self.G_XtoY(fake_X)
                datas=[X,fake_Y,fake_recX,
                       Y,fake_X,fake_recY]
                imgA = imgAs[i] # 针对X
                imgB = imgBs[i]
                name = imgA.rsplit(".")[0]
                realA_name = name+"_realA.png"
                fakeB_name = name+"_fakeB.png"
                rectA_name = name+"_rectA.png"
                name = imgB.rsplit(".")[0]
                realB_name = name + "_realB.png"
                fakeA_name = name + "_fakeA.png"
                rectB_name = name + "_rectB.png"

                with ThreadPoolExecutor() as exe:
                    datas = [x.cpu().float().numpy() for x in datas]
                    datas = list(map(lambda x:np.squeeze(x,0),datas))
                    datas = list(map(lambda x:self.reverse_norm(x),datas))

                    exe.map(cv2.imwrite,[os.path.join(result_dir,realA_name),
                                         os.path.join(result_dir,fakeB_name),
                                         os.path.join(result_dir,rectA_name),
                                         os.path.join(result_dir,realB_name),
                                         os.path.join(result_dir,fakeA_name),
                                         os.path.join(result_dir,rectB_name)],
                                        datas)
            print('inference done!')
    def single_if(self,img):
        im=cv2.imread(img)
        ori=im.copy()
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        ])
        im = trans(im)
        mkdirs("./tempview",)
        # self.G_XtoY.eval()
        name=os.path.splitext(os.path.split(img)[-1])[0]
        with torch.no_grad():
            im=torch.unsqueeze(im,0)
            fake=self.G_XtoY(im)
            fake=fake.cpu().float().numpy()
            fake=np.squeeze(fake,0)
            fake=self.reverse_norm(fake)
            cv2.imwrite(os.path.join("./tempview",name+"_ori.png"),ori)
            cv2.imwrite(os.path.join("./tempview",name+"_fake.png"),
                        fake)