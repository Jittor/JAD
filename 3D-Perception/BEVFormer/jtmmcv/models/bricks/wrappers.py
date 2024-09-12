# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/wrappers.py  # noqa: E501

Wrap some nn modules to support empty tensor input. Currently, these wrappers
are mainly used in mask heads like fcn_mask_head and maskiou_heads since mask
heads are trained on only positive RoIs.
"""
import math

from jittor import Function
from jittor import nn as jnn
from jittor.misc import _pair, _triple

from .registry import CONV_LAYERS, UPSAMPLE_LAYERS

#不考虑空var的场景

@CONV_LAYERS.register_module('Conv', force=True)
class Conv2d(jnn.Conv2d):

    def execute(self, x):

        return super().execute(x)


@CONV_LAYERS.register_module('Conv3d', force=True)
class Conv3d(jnn.Conv3d):

    def execute(self, x):

        return super().execute(x)


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv')
@UPSAMPLE_LAYERS.register_module('deconv', force=True)
class ConvTranspose2d(jnn.ConvTranspose2d):

    def execute(self, x):

        return super().execute(x)


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv3d')
@UPSAMPLE_LAYERS.register_module('deconv3d', force=True)
class ConvTranspose3d(jnn.ConvTranspose3d):

    def execute(self, x):

        return super().execute(x)


class MaxPool2d(jnn.MaxPool2d):

    def execute(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet

        return super().execute(x)


class MaxPool3d(jnn.MaxPool3d):

    def execute(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet

        return super().execute(x)


class Linear(jnn.Linear):

    def execute(self, x):
        
        return super().execute(x)
