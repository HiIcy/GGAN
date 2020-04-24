# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/1
# __file__ = base_option
# __desc__ =

import argparse

class BaseOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GAN uses manual")
        self.parser.add_argument("-data_root","--data_root",type=str,
                                 default="",help="root dir of image")
        self.parser.add_argument("-direction","--direction",type=str,
                                 choices=['AtoB','BtoA'],default="AtoB",help="from A transform to B or be opposite")
        # self.parser.add_argument("-batch","--batch_size",type=)
        self.parser.add_argument("-crop_size","--crop_size",type=int,
                                 default=256,help="need to be crop as the size")
        self.parser.add_argument("--datasetname",type=str,
                                 default="facade")
        self.parser.add_argument("--netname",type=str,
                                 default="pix2pix")
        self.parser.add_argument("--nin",type=int,default=3)
        self.parser.add_argument("--nc",type=int,default=3)
        self.parser.add_argument("--ngf",type=int,default=64)
        self.parser.add_argument("--pathchshape",type=int,default=60)
        self.parser.add_argument("--data_mode",type=str,default="aligned")
        self.parser.add_argument("--gan_mode",type=str,default="lsgan",choices=['lsgan','vanilla','wgangp'])
        self.parser.add_argument("--gpuids",nargs="*",type=int,default=[0],help="as the name said..")
        self.parser.add_argument("-model_save","--model_save",default="./weight",type=str,help="the dir to save model")
        self.parser.add_argument("--worker",type=int,default=2)

    def print_arg(self,opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def get_arg(self):
        opt = self.parser.parse_args()
        self.print_arg(opt)
        return opt

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        self.parser.add_argument("--nb_epoch",default=3000,type=int)
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--epoch_count',type=int,default=1)
        self.parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')

        self.parser.add_argument("--start_epoch",default=0,type=int)
        self.parser.add_argument("-lr","--lr",default=0.0002,type=float,help="train lr")
        self.parser.add_argument("-weight_decay","--weight_decay",default=0.99,type=float,help="as the name said...")
        self.parser.add_argument("-batch","--batch",default=2,type=int,help="as the name said...")
        self.parser.add_argument("-val_batch","--val_batch",default=4,type=int,help="as the name said...")
        self.parser.add_argument("--show_interval",default=100,type=int,help="train show interval.")
        self.parser.add_argument("--vis_interval",default=50,type=int,help="visualize generator interval.")
        self.parser.add_argument("--alpha",default=100,type=int)
        self.parser.add_argument("--log_dir",default="./log",help="as the name said.")
        self.parser.add_argument("--lambda_X",default=10,type=int,help="as the name said.")
        self.parser.add_argument("--lambda_Y",default=10,type=int,help="as the name said.")
        self.parser.add_argument("--lambda_idt",default=.5,type=int,help="as the name said.")
        self.parser.add_argument("--pool_size",default=50,type=int,help="as the name said.")
        self.parser.add_argument("--preview_dir",default=None,type=str)
        # self.parser.add_argument()

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.parser.add_argument("--test_batch",default=1,type=int,help="as the name said...")
        self.parser.add_argument("--result_dir",default=None,type=str)
