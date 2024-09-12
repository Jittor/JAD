# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import sys
import time
import warnings
import functools
import numpy as np
from collections import OrderedDict

import jittor
from getpass import getuser
from socket import gethostname

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def get_host_info():
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but '
                        f'got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `jittor.backends.cudnn.deterministic`
            to True and `jittor.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    jittor.misc.set_global_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def get_dist_info():
    if jittor.in_mpi:
        rank = jittor.rank
        world_size = jittor.world_size
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def allreduce_params(params, coalesce=False, bucket_size_mb=-1):
    """Allreduce parameters.

    Args:
        params (list[jittor.Parameters]): List of parameters or buffers of a
            model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param for param in params]

    for tensor in params:
        jittor.mpi.mpi_reduce(tensor, op='mean')


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[jittor.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad for param in params
        if param.requires_grad and param.grad is not None
    ]
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    
    for tensor in grads:
        jittor.mpi.mpi_reduce(tensor, op='mean')


# def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
#     if bucket_size_mb > 0:
#         bucket_size_bytes = bucket_size_mb * 1024 * 1024
#         buckets = _take_tensors(tensors, bucket_size_bytes)
#     else:
#         buckets = OrderedDict()
#         for tensor in tensors:
#             tp = tensor.type()
#             if tp not in buckets:
#                 buckets[tp] = []
#             buckets[tp].append(tensor)
#         buckets = buckets.values()

#     for bucket in buckets:
#         flat_tensors = _flatten_dense_tensors(bucket)
#         jittor.mpi.mpi_reduce(flat_tensors, op='mean')
#         flat_tensors.div_(world_size)
#         for tensor, synced in zip(
#                 bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
#             tensor.copy_(synced)