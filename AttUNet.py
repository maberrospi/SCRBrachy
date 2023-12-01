#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:18:40 2023

@author: ERASMUSMC+099035
"""
# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% DEFINE ALL UNET PARTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class DoubleConv(nn.Module):
    """ This class makes 3 calculations twice
    
    1. Calculates the convolution of the input channels
    2. Calculates the batch norm of the convolution output
    3. Passes the batch norm output through a ReLU function
    
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            # Padding will ensure that the final segmentation map is the same size as the input image.
            # Can also test with bias parameter as false
            # Convolutions use 3x3 kernels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    """ This class performs the downsampling or contraction of the feature data
        and then a double convolution
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Max pooling uses 2x2 kernel
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSampling(nn.Module):
    """ This class performs the upsampling or expansion of the double convolution"""

    # Technically reduces the number of channels while increasing the spatial resolution of the feature map
    # align_corners preserves the values at input/output border pixels
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsampling = nn.Sequential(
                #nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.upsampling(x)


class OutConv(nn.Module):
    """ This class calculates the output by passing through a final convolution"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


# %% DEFINE UNET MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class AttUNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        self.down4 = DownSampling(512, 1024)
        self.Up4 = UpSampling(1024, 512, bilinear)
        self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv4 = DoubleConv(1024, 512)
        self.Up3 = UpSampling(512, 256, bilinear)
        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv3 = DoubleConv(512, 256)
        self.Up2 = UpSampling(256, 128, bilinear)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv2 = DoubleConv(256, 128)
        self.Up1 = UpSampling(128, 64, bilinear)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv1 = DoubleConv(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d5 = self.Up4(e5)
        x4 = self.Att4(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv4(d5)

        d4 = self.Up3(d5)
        x3 = self.Att3(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv3(d4)

        d3 = self.Up2(d4)
        x2 = self.Att2(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv2(d3)

        d2 = self.Up1(d3)
        x1 = self.Att1(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv1(d2)

        logits = self.outc(d2)
        return logits

    # Use checkpoint on all layers
    # Pytorch recommends using reentrant = False so might use that if there is something wrong
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.Up4 = torch.utils.checkpoint(self.Up4)
        self.Att4 = torch.utils.checkpoint(self.Att4)
        self.Up_conv4 = torch.utils.checkpoint(self.Up_conv4)
        self.Up3 = torch.utils.checkpoint(self.Up3)
        self.Att3 = torch.utils.checkpoint(self.Att3)
        self.Up_conv3 = torch.utils.checkpoint(self.Up_conv3)
        self.Up2 = torch.utils.checkpoint(self.Up2)
        self.Att2 = torch.utils.checkpoint(self.Att2)
        self.Up_conv2 = torch.utils.checkpoint(self.Up_conv2)
        self.Up1 = torch.utils.checkpoint(self.Up1)
        self.Att1 = torch.utils.checkpoint(self.Att1)
        self.Up_conv1 = torch.utils.checkpoint(self.Up_conv1)
        self.outc = torch.utils.checkpoint(self.outc)
