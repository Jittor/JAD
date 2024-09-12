# Copyright (c) OpenMMLab. All rights reserved.
import jittor
import jittor.nn as jnn


def _fuse_conv_bn(conv, bn):
    """Fuse conv and bn into one module.

    Args:
        conv (Module): Conv to be fused.
        bn (Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else jittor.zeros_like(
        bn.running_mean)

    factor = bn.weight / jittor.sqrt(bn.running_var + bn.eps)
    conv.weight = jnn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]).to(jittor.float32))
    conv.bias = jnn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module):
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (jnn.BatchNorm2d)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = jnn.Identity()
            last_conv = None
        elif isinstance(child, jnn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module
