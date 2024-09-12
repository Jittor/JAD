# Copyright (c) OpenMMLab. All rights reserved.
import inspect

from jittor import nn as jnn

from jtmmcv.utils import is_tuple_of
from .registry import NORM_LAYERS

# norm 不会在eval中验证

NORM_LAYERS.register_module('BN', module=jnn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', module=jnn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', module=jnn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', module=jnn.BatchNorm3d)
# NORM_LAYERS.register_module('SyncBN', module=nn.SyncBatchNorm)  计图的BN会根据分布式情况设定BN类
NORM_LAYERS.register_module('GN', module=jnn.GroupNorm)
NORM_LAYERS.register_module('LN', module=jnn.LayerNorm)
NORM_LAYERS.register_module('IN', module=jnn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', module=jnn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', module=jnn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', module=jnn.InstanceNorm3d)


def infer_abbr(class_type):
    """jittor 版本
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, jnn.InstanceNorm2d):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type,(jnn.BatchNorm1d, jnn.BatchNorm2d, jnn.BatchNorm3d)):
        return 'bn'
    elif issubclass(class_type, jnn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, jnn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'



def build_norm_layer(cfg, num_features, postfix=''):
    """jittor 版本
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'BN' and hasattr(layer, '_specify_ddp_gpu_num'):
        #     layer._specify_ddp_gpu_num(1)                                 分布式训练标签需要整体来看
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer, exclude=None):
    """jittor版本
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (jnn.BatchNorm1d, jnn.BatchNorm2d, jnn.BatchNorm3d, jnn.InstanceNorm2d, jnn.GroupNorm, jnn.LayerNorm)
    return isinstance(layer, all_norm_bases)
