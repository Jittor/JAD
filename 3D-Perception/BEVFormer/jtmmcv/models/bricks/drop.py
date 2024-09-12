import jittor as jt
from jittor import Module
from jittor import nn as jnn

from jtmmcv import build_from_cfg
from .registry import DROPOUT_LAYERS

#Model 的训练布尔参数名为is_train 该文件需要整体修改为jittor, drop 不会在eval中验证


def drop_path(x, drop_prob=0., is_train=False):
    """jittor版本
    """
    if drop_prob == 0. or not is_train:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)    
    random_var = keep_prob + jt.random(
        shape, dtype=x.dtype)
    output = x/keep_prob * random_var.floor()
    return output

    
@DROPOUT_LAYERS.register_module()
class DropPath(Module):
    """jittor 版本
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_train)


@DROPOUT_LAYERS.register_module()
class Dropout(jnn.Dropout):
    """jittor 版本
    """

    def __init__(self, drop_prob=0.5):
        super().__init__(p=drop_prob)


def build_dropout(cfg, default_args=None):
    """Builder for drop out layers."""
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)
