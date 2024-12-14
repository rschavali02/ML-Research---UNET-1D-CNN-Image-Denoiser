import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import skimage as skicd
import torch.nn.functional as F

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        #elif t == '2':
        #    L.append(nn.PixelShuffle(upscale_factor=2))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1

# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        #res = self.res(x)
        return x + self.res(x)
    
class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        if downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
                                  downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
                                  downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
                                  downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), 
                                *[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), 
                                *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), 
                                *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        # Contraction
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        
        x = self.m_body(x4)
        
        # Ensure tensors are 3D before applying interpolation
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Make x 3D if it's 2D
        if x4.dim() == 2:
            x4 = x4.unsqueeze(0)  # Ensure x4 is also 3D
        
        # Expansion (use interpolation with 'linear' for 1D tensors)
        x = F.interpolate(x, size=x4.shape[2:], mode='linear', align_corners=True)  # Match size with x4
        x = self.m_up3(x + x4)
        
        if x3.dim() == 2:
            x3 = x3.unsqueeze(0)  # Ensure x3 is 3D
        x = F.interpolate(x, size=x3.shape[2:], mode='linear', align_corners=True)  # Match size with x3
        x = self.m_up2(x + x3)
        
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)  # Ensure x2 is 3D
        x = F.interpolate(x, size=x2.shape[2:], mode='linear', align_corners=True)  # Match size with x2
        x = self.m_up1(x + x2)
        
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)  # Ensure x1 is 3D
        x = F.interpolate(x, size=x1.shape[2:], mode='linear', align_corners=True)  # Match size with x1
        x = self.m_tail(x + x1)

        return x