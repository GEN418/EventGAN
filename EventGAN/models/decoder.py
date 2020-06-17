import torch
import torch.nn as nn

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
        use_bias = norm_layer != nn.BatchNorm2d  #存在batchnorm层即将偏置置为0

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(
            input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]  # negative slope, in_place
        nf_mult = 1      # 每层输出num_channel的乘法因子, 即本层num_filter的乘法因子
        nf_mult_prev = 1 # 每层输入num_channel的乘法因子, 即上一层num_filter的乘法因子
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # 2,4
            sequence += [nn.Conv2d(ndf * nf_mult_prev, #64*1，64*2
                                  ndf * nf_mult, #64*2,64*4
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw,
                                  bias=use_bias)]
            sequence += [ nn.LeakyReLU(0.2, True) ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, #64*4
                               ndf * nf_mult,      #64*8
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=use_bias)]

        sequence += [ nn.LeakyReLU(0.2, True) ]
        sequence += [nn.Conv2d(ndf * nf_mult,  #64*8
                               1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Patch_Discriminator(nn.Module):
    def __init__(self, event_channel=9, image_channel=1,
                 ndf=64, n_layers=3, norm_layer=None):
        super(Patch_Discriminator, self).__init__()
        self.disc = NLayerDiscriminator(event_channel+2*image_channel,
                                        ndf=ndf,
                                        n_layers=n_layers,
                                        norm_layer=norm_layer)

    def forward(self, preds, images):
        return self.disc(torch.cat([preds[0], images], 1))
