# coding:utf-8
# __author__ = yuan
# __time__ = 2020/4/14
# __file__ = cyclegan
# __desc__ =

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generateor(nn.Module):
    def __init__(self, nic,noc,ngf=64,n_block=6,use_dropout=False,padding_type='reflect'):
        super().__init__()
        use_bias = False
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(nic, ngf, kernel_size=7, padding=0, bias=use_bias),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            #  输出通道比前面通道多2倍
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      nn.BatchNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_block):
            model += [ResnetBlock(ngf * mult,padding_type=padding_type, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling): # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf*mult/2),
                                         kernel_size=(3,3),stride=2,
                                         padding=1,output_padding=1,
                                         bias=use_bias),
                      nn.BatchNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)
                      ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, noc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block=[]
        p=0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PatchGAN(nn.Module):
    def __init__(self,nin,ndf,n_layers=3):
        super(PatchGAN, self).__init__()
        kw = 3
        padw = 1
        sequence = [nn.Conv2d(nin, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2,2),2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        sequence += [
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    mc = Generateor(3,3)
    c = torch.rand(2,3,256,256)
    r=mc(c)
    print(mc)
    print(f"input  shape:{c.shape}")
    print(f"output shape:{r.shape}")