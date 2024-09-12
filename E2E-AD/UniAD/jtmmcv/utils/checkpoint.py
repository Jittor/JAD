# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
import jittor
from jittor.optim import Optimizer
from .logging import print_log
from .runner_utils import get_dist_info
from ..parallel import is_module_wrapper
from jtmmcv.fileio.file_client import FileClient

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath
        map_location (str): Same as :func:`jittor.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    print_log(
        f'load checkpoint from path: {filename}', logger)
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')

    checkpoint = jittor.load(filename)
    
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_dict = {}
    
        #更改torch版本attention中的jittor不支持的权重
    for key,v in state_dict.items():
        if key.endswith('.in_proj_weight'):
            new_key = key.replace('.in_proj_weight', '.in_proj.weight')
            new_dict[new_key] = v
        elif key.endswith('.in_proj_bias'):
            new_key = key.replace('.in_proj_bias', '.in_proj.bias')
            new_dict[new_key] = v
        else:
            new_dict[key] = v
            
    state_dict = new_dict
            
    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    if is_module_wrapper(model):
        model = model.module
    model.load_state_dict(state_dict)
    # ignore "num_batches_tracked" of BN layers

    return checkpoint

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu

def save_checkpoint(model,
                    filename,
                    optimizer=None,
                    meta=None,
                    file_client_args=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict()),
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    file_client = FileClient.infer_client(file_client_args, filename)
    with io.BytesIO() as f:
        jittor.save(checkpoint, f)
        file_client.put(f.getvalue(), filename)
