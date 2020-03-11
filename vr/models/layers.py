#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, kaiming_uniform_


class SequentialSaveActivations(nn.Sequential):

    def forward(self, input_):
        self.outputs = [input_]
        for module in self._modules.values():
            input_ = module(input_)
            self.outputs.append(input_)
        return input_


class SimpleVisualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size=3):
        if out_dim is None:
            out_dim = in_dim
        super(SimpleVisualBlock, self).__init__()
        if kernel_size % 2 == 0:
            raise NotImplementedError()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size=3, with_residual=True, with_batchnorm=True,
            shared_block=None, post_linear=False):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        if kernel_size % 2 == 0:
            raise NotImplementedError()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.shared_block=shared_block
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        if post_linear:
            self.post_linear = nn.Conv2d(out_dim, 2 * out_dim, kernel_size=1)
            # UGLY HACK!!!
            self.post_linear.weight.data[:out_dim, :, 0, 0] = torch.eye(out_dim)
        else:
            self.post_linear = None

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        if self.shared_block:
            out = self.shared_block(out)
        if self.post_linear:
            out = self.post_linear(out)
        return out


class SimpleConcatBlock(nn.Module):
    def __init__(self, dim, kernel_size, shared_block=None):
        super().__init__()
        self.proj = nn.Conv2d(3 * dim, dim, kernel_size=1, padding=0)
        self.impl = ResidualBlock(
            dim, dim, kernel_size=kernel_size,
            with_residual=True, with_batchnorm=False, shared_block=shared_block)

    def forward(self, feats, x, y):
        out = torch.cat([feats, x, y], 1) # Concatentate along depth
        return self.impl(F.relu(self.proj(out)))


class ConcatBlock(nn.Module):
    def __init__(self, dim, kernel_size, with_residual=True, with_batchnorm=True,
            shared_block=None, post_linear=False):
        super().__init__()
        self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
        self.vis_block = ResidualBlock(
            dim, kernel_size=kernel_size,
            with_residual=with_residual,with_batchnorm=with_batchnorm,
            shared_block=shared_block, post_linear=post_linear)

    def forward(self, x, y):
        out = torch.cat([x, y], 1) # Concatentate along depth
        out = F.relu(self.proj(out))
        out = self.vis_block(out)
        return out


class GlobalAveragePool(nn.Module):
    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(2).squeeze(2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def build_stem(feature_dim,
               stem_dim,
               module_dim,
               num_layers=2,
               with_batchnorm=True,
               kernel_size=[3],
               stride=[1],
               padding=None,
               subsample_layers=None,
               acceptEvenKernel=False):
    layers = []
    prev_dim = feature_dim

    if len(kernel_size) == 1:
        kernel_size = num_layers * kernel_size
    if len(stride) == 1:
        stride = num_layers * stride
    if padding == None:
        padding = num_layers * [None]
    if len(padding) == 1:
        padding = num_layers * padding
    if subsample_layers is None:
        subsample_layers = []

    for i, cur_kernel_size, cur_stride, cur_padding in zip(range(num_layers), kernel_size, stride, padding):
        curr_out = module_dim if (i == (num_layers-1) ) else stem_dim
        if cur_padding is None:  # Calculate default padding when None provided
            if cur_kernel_size % 2 == 0 and not acceptEvenKernel:
                raise(NotImplementedError)
            cur_padding = cur_kernel_size // 2
        layers.append(nn.Conv2d(prev_dim, curr_out,
                                kernel_size=cur_kernel_size, stride=cur_stride, padding=cur_padding,
                                bias=not with_batchnorm))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(curr_out))
        layers.append(nn.ReLU(inplace=True))
        if i in subsample_layers:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        prev_dim = curr_out
    return SequentialSaveActivations(*layers)


class HybridPool(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=width, stride=width, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=width, stride=width, padding=0)

    def forward(self, x):
        return torch.cat([self.maxpool(x), self.avgpool(x)], 1)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample=None,
                     with_batchnorm=True, dropout=[]):
    layers = []
    prev_dim = module_C * module_H * module_W
    cur_dim = module_C
    if proj_dim is not None and proj_dim > 0:
        layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1, bias=not with_batchnorm))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(proj_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = proj_dim * module_H * module_W
        cur_dim = proj_dim
    if downsample is not None:
        if 'maxpool' in downsample or 'avgpool' in downsample:
            pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
            if 'full' in downsample:
                assert module_H == module_W
                pool_size = module_H
            else:
                pool_size = int(downsample[-1])
            # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
            padding = (0 if ((module_H % pool_size == 0) and (module_W % pool_size == 0)) else 1)
            layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
            prev_dim = cur_dim * math.ceil(module_H / pool_size) * math.ceil(module_W / pool_size)
        if downsample == 'hybrid':
            assert module_H == module_W
            pool = HybridPool(module_H)
            layers.append(pool)
            prev_dim = cur_dim * 2
        if downsample == 'aggressive':
            raise ValueError()
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.AvgPool2d(kernel_size=module_H // 2, stride=module_W // 2))
            prev_dim = proj_dim
            fc_dims = []  # No FC layers here
    layers.append(Flatten())

    if isinstance(dropout, float):
        dropout = [dropout] * len(fc_dims)
    elif not dropout:
        dropout = [0] * len(fc_dims)

    for next_dim, next_dropout in zip(fc_dims, dropout):
        layers.append(nn.Linear(prev_dim, next_dim, bias=not with_batchnorm))
        if with_batchnorm:
            layers.append(nn.BatchNorm1d(next_dim))
        layers.append(nn.ReLU(inplace=True))
        if next_dropout > 0:
            layers.append(nn.Dropout(p=next_dropout))
        prev_dim = next_dim
    layers.append(nn.Linear(prev_dim, num_answers))
    return nn.Sequential(*layers)


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal_
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform_
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)
