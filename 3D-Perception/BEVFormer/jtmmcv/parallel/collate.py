# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import jittor as jt
# from jittor.dataset.dataset import collate_batch

from jtmmcv.parallel.data_container import DataContainer
import numpy as np
from collections.abc import Sequence, Mapping
from PIL import Image
import functools

def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, jt.Var):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper

def collate(batch):
    
    batch = collate_dc(batch)
    data_dict = {}
    for key, value in batch.items():
        if isinstance(value, DataContainer):
            data_dict[key] = value.data[0]
        elif isinstance(value[0], DataContainer):
            data_dict[key] = value[0].data
        else:
            data_dict[key] = value
    return data_dict


def collate_dc(batch):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        samples_per_gpu =len(batch)
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, jt.Var)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            jt.nn.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(collate_batch(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        collate_batch([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_dc(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate_dc([d[key] for d in batch])
            for key in batch[0]
        }
    else:
        return collate_batch(batch)




def collate_batch(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    real_size = len(batch)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, jt.Var):
        temp_data = jt.stack([data for data in batch], 0)
        return temp_data
    if elem_type is np.ndarray:
        temp_data = np.stack([data for data in batch], 0)
        return temp_data
    elif np.issubdtype(elem_type, np.integer):
        return np.int32(batch)
    elif isinstance(elem, int):
        return np.int32(batch)
    elif isinstance(elem, float):
        return np.float32(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Mapping):
        return {key: collate_batch([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return tuple(collate_batch(samples) for samples in transposed)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [collate_batch(samples) for samples in transposed]
    elif isinstance(elem, Image.Image):
        temp_data = np.stack([np.array(data) for data in batch], 0)
        return temp_data
    else:
        raise TypeError(f"Not support type <{elem_type.__name__}>, {batch},{isinstance(batch[0], DataContainer)},{type(batch[0])}")