# coding:utf-8
# __author__ = yuan
# __time__ = 2020/3/31
# __file__ = Pix2Pix
# __desc__ =
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as ModelZoo

def resNet34():
    _resnet50 = ModelZoo.resnet34(pretrained=False)
    # for idx,layer in enumerate(list(_resnet50.children())[:-4]):
        # print(idx+1,'->', layer,'\n\n')
    return _resnet50
resnet34 = resNet34()

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # REW: 先求最底层的跨层连接，再逐渐往上层去 大神级代码啊！
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Generator(nn.Module):
    # FIXME:还不够优雅,需要重构
    def __init__(self,nic,ngf=64,noc=3,n_layer=3):
        super(Generator, self).__init__()
        layers = [nn.Conv2d(nic, ngf, kernel_size=4, stride=2, padding=1)]#128
        layer_specs = [
            ngf * 2,#64
            ngf*4,#32
            ngf*8,#16
            ngf*8,#8
            ngf*8,#4
            ngf*8,#2
            ngf*8 #1
        ]
        prenf=ngf
        for out_chan in layer_specs:
            layers.append(nn.Sequential(
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(prenf,out_chan,4,2,1),
                nn.BatchNorm2d(out_chan)
            ))
            prenf=out_chan
        self.down = nn.ModuleList(layers)

        layer_specs = [
            (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
        self.num_down_layer = len(layers) # 8
        uplayers = []
        for index,(out_chan,dropout) in enumerate(layer_specs):
            if index==0:
                uplayers.append(nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(prenf, out_chan, 4, 2, 1, output_padding=0),
                    nn.BatchNorm2d(out_chan),
                    nn.Dropout(dropout,True)
                ))
            else:
                uplayers.append(nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(prenf*2,out_chan,4,2,1,output_padding=0),
                    nn.BatchNorm2d(out_chan),
                    nn.Dropout(dropout,True)
                ))
            prenf=out_chan
        self.up = nn.ModuleList(uplayers)

        self.final = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(prenf*2,noc,4,2,1,output_padding=0),
            nn.Tanh()
        )

    def forward(self,input):
        downsample = []
        x = input
        for i,layer in enumerate(self.down):
            x = layer(x)
            downsample.append(x)
        upsample = []
        for index,layer in enumerate(self.up):
            skip_layer = self.num_down_layer - index -1
            if index==0:
                x=layer(x)
                upsample.append(x)
            else:
                x=torch.cat([x,downsample[skip_layer]],dim=1)
                x=layer(x)
                upsample.append(x)
        x = torch.cat([x,downsample[0]],dim=1)
        output = self.final(x)
        return output


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

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

if __name__ == '__main__':
    ug = Generator(3,3)
    # print(fs)
    # im = torch.rand(2,3,508,508)
    # print(fs)
    # ug = PatchGAN(6,64)
    c = torch.rand(2, 3, 256, 256)
    r = ug(c)
    # print(pc)
    # print(ug)
    print(f"input  shape:{c.shape}")
    print(f"output shape:{r.shape}")
