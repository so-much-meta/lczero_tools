import torch
from torch import nn
import torch.nn.functional as F
import gzip
import zlib, base64
import numpy as np
import math

from lcztools._weights_file import read_weights_file

# This pytorch implementation slightly optimizes the original by using a simplified "Normalization" layer instead
# of BatchNorm2d, with precalculated normalization/variance divisors: w = 1/torch.sqrt(w + 1e-5).
# Without BatchNorm, this is only useful for eval, never training.

class Normalization(nn.Module):
    r"""Applies per-channel transformation (x - mean)*stddiv
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mean = nn.Parameter(torch.Tensor(channels))        
        self.stddiv = nn.Parameter(torch.Tensor(channels))

    def forward(self, x):
        return (x - self.mean.unsqueeze(1).unsqueeze(2)) * self.stddiv.unsqueeze(1).unsqueeze(2)

    def extra_repr(self):
        return 'channels={}'.format(
            self.channels
        )


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, input_channels, output_channels=None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.conv1_bn = Normalization(output_channels)
    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv1_bn = Normalization(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv2_bn = Normalization(channels)
    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.conv2_bn(self.conv2(out))
        out += x
        out = F.relu(out, inplace=True)
        return out

class LeelaModel(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        # 112 input channels
        self.conv_in = ConvBlock(kernel_size=3,
                               input_channels=112,
                               output_channels=channels)
        self.residual_blocks = []
        for idx in range(blocks):
            block = ResidualBlock(channels)
            self.residual_blocks.append(block)
            self.add_module('residual_block{}'.format(idx+1), block)
        self.conv_pol = ConvBlock(kernel_size=1,
                                   input_channels=channels,
                                   output_channels=32)
        self.affine_pol = nn.Linear(32*8*8, 1858)
        self.conv_val = ConvBlock(kernel_size=1,
                                 input_channels=channels,
                                 output_channels=32)
        self.affine_val_1 = nn.Linear(32*8*8, 128)
        self.affine_val_2 = nn.Linear(128, 1)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.view(-1, 112, 8, 8)
        out = self.conv_in(x)
        for block in self.residual_blocks:
            out = block(out)
        out_pol = self.conv_pol(out).view(-1, 32*8*8)
        out_pol = self.affine_pol(out_pol)
        out_val = self.conv_val(out).view(-1, 32*8*8)
        out_val = F.relu(self.affine_val_1(out_val), inplace=True)
        out_val = F.tanh(self.affine_val_2(out_val))
        return out_pol, out_val
    
class LeelaLoader:
    @staticmethod
    def from_weights_file(filename, train=False):
        filters, blocks, weights = read_weights_file(filename)
        net = LeelaModel(filters, blocks)
        if not train:
            net.eval()
            for p in net.parameters():
                p.requires_grad = False
        parameters = []
        for module_name, module in net.named_modules():
            class_name = module.__class__.__name__
            for typ in ('weight', 'bias', 'mean', 'stddiv'):
                param = getattr(module, typ, None)
                if param is not None:
                    parameters.append((module_name, class_name, typ, param))
        param_idx = 0
        # The unused_bias variable is set each time a convolution is seen; the following bias
        # parameter is not used.
        unused_bias = False
        for i, w in enumerate(weights):
            w = torch.Tensor(w)
            if unused_bias: # ((w**2).mean()==0):
                print(f"{tuple(w.size())} -- Unused bias")
                unused_bias = False
                continue
            module_name, class_name, typ, param = parameters[param_idx]
            print(f"{tuple(w.size())} -- {module_name} - {class_name} - {typ}: {tuple(param.size())}")
            if class_name == 'Normalization' and typ=='mean':
                # print('NMean')
                # w = w
                param.data.copy_(w.view_as(param))
            elif class_name == 'Normalization' and typ=='stddiv':
                # print('NStddiv')
                w = 1/torch.sqrt(w + 1e-5)
                param.data.copy_(w.view_as(param))
            elif len(param.size())==4:
                # Convolutions turn out to be correctly transposed for pytorch
                param.data.copy_(w.view_as(param))
                unused_bias = True
            else:
                param.data.copy_(w.view_as(param))
            param_idx += 1
        return net
