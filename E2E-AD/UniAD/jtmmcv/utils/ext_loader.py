# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

# import jittor


# 随着算子的健全，这些也要删掉

def load_ext(name, funcs):
    ext = importlib.import_module('mmcv.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext

def check_ops_exist():
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
