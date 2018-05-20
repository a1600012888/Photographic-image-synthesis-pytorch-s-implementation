#!/usr/bin/env mdl
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
def get_output_chn(super_resolution):
    chn = 1024 if super_resolution < 128 else 512

    if super_resolution == 512:
        chn = 128
    if super_resolution == 1024:
        chn = 32

    return chn
class CRN(nn.Module):

    def __init__(self, super_resolution = 256, groups = 6):
        '''

        :param super_resolution: the height(short edge) of output image
        '''
        super(CRN, self).__init__()

        assert super_resolution in [256, 512, 1024], super_resolution
        self.super_resolution = super_resolution
        last_inp_chn = get_output_chn(super_resolution)

        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        return_labels = [True, True, True, True, True, False, False]

        if super_resolution >= 512:
            return_labels.append(False)
        if super_resolution == 1024:
            return_labels.append(False)
        return_labels = return_labels[::-1]

        self.refine_block0 = refine_block(resolutions[0])
        self.refine_block1 = refine_block(resolutions[1])
        self.refine_block2 = refine_block(resolutions[2])
        self.refine_block3 = refine_block(resolutions[3])
        self.refine_block4 = refine_block(resolutions[4])
        self.refine_block5 = refine_block(resolutions[5])
        self.refine_block6 = refine_block(resolutions[6])

        if super_resolution > 256:
            self.refine_block7 = refine_block(resolutions[7])

        if super_resolution > 512:
            self.refine_block8 = refine_block(resolutions[8])

        self.last_conv = nn.Conv2d(last_inp_chn, 3 * groups, kernel_size=1)

        self.kaiming_init_params()

        print(self)
    def forward(self, label):
        x = self.refine_block0(label)

        x = self.refine_block1(label, x)
        x = self.refine_block2(label, x)
        x = self.refine_block3(label, x)
        x = self.refine_block4(label, x)
        x = self.refine_block5(label, x)
        x = self.refine_block6(label, x)

        if self.super_resolution > 256:
            x = self.refine_block7(label, x)

        if self.super_resolution > 512:
            x = self.refine_block8(label, x)

        x = self.last_conv(x)

        x = (x + 1.0) / 2.0 * 255.0

        a = x.split(3, dim = 1)

        x = torch.cat(a, dim = 0)

        return x

    def kaiming_init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class refine_block(nn.Module):

    def __init__(self, super_resolution):
        '''

        :param super_resolution: the height(short edge) of output feature_map
        '''
        super(refine_block, self).__init__()
        self.super_resolution = super_resolution
        x_grid = torch.linspace(-1, 1, 2 * self.super_resolution).repeat(self.super_resolution, 1)
        y_grid = torch.linspace(-1, 1, self.super_resolution).view(-1, 1).repeat(
            1, self.super_resolution * 2)
        self.grid = torch.cat((x_grid.unsqueeze(2), y_grid.unsqueeze(2)), 2)
        self.grid = self.grid.unsqueeze_(0)


        out_chn = get_output_chn(super_resolution)
        in_chn = 20 if super_resolution == 4 else get_output_chn(super_resolution // 2) + 20


        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size = 3, stride = 1, padding = 1)
        self.ln1 = nn.LayerNorm(normalized_shape=[out_chn, super_resolution, super_resolution * 2])

        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride = 1, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[out_chn, super_resolution, super_resolution * 2])

    def forward(self, label, x = None):

        # perform bilinear interpolation to downsample
        #x_grid = torch.linspace(-1, 1, 2 * self.super_resolution).repeat(self.super_resolution, 1)
        #y_grid = torch.linspace(-1, 1, self.super_resolution).view(-1, 1).repeat(
         #   1, self.super_resolution * 2)
        #grid = torch.cat((x_grid.unsqueeze(2), y_grid.unsqueeze(2)), 2)
       # grid = self.grid.repeat(label.size()[0], 1, 1, 1)

        #print('interpolation details: grid:{}, label:{}'.format(grid.size(), label.size()))
        #print('type: grid:{}, label:{}'.format(grid.type(), label.type()))
        grid = self.grid.repeat(label.size(0), 1, 1, 1).cuda()
        #print('Label size {}'.format(label.size()))
        label_downsampled = F.grid_sample(label, grid)

        if self.super_resolution != 4:
            x = F.upsample(x, size=(self.super_resolution, self.super_resolution * 2),
                           mode = 'bilinear', align_corners = True)
            x = torch.cat((x, label_downsampled), 1)
        else:
            x = label_downsampled

        #print('Label size {}'.format(x.size()))
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x, 0.2)

        del label_downsampled
        del grid
        return x


def test(batch_size = 1, resolution = 512):

    net = CRN(resolution)
    net.cuda()

    label = torch.randn([batch_size, 19, 1024, 2048], dtype = torch.float).cuda()

    out = net(label)

    print(out.size(), type(out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--test', action = 'store_true')

    args = parser.parse_args()

    if args.test:
        test()
# vim: ts=4 sw=4 sts=4 expandtab
