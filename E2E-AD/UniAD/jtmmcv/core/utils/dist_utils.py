import warnings
from collections import OrderedDict

import jittor
# import jittor.mpi as mpi
from jtmmcv.runner import OptimizerHook



# def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
#     # if bucket_size_mb > 0:
#     #     bucket_size_bytes = bucket_size_mb * 1024 * 1024
#     #     buckets = _take_tensors(tensors, bucket_size_bytes)
#     # else:
#     buckets = OrderedDict()
#     for tensor in tensors:
#         tp = tensor.type()
#         if tp not in buckets:
#             buckets[tp] = []
#         buckets[tp].append(tensor)
#     buckets = buckets.values()

#     for bucket in buckets:
#         flat_tensors = _flatten_dense_tensors(bucket)
#         dist.all_reduce(flat_tensors)
#         flat_tensors.div_(world_size)
#         for tensor, synced in zip(
#                 bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
#             tensor.copy_(synced)


def allreduce_grads(params, coalesce=False, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[jittor.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data for param in params                 # jittor 调用data的方法是否相同
        if param.requires_grad and param.grad is not None
    ]
    world_size = jittor.world_size()
    # if coalesce:
    #     _allreduce_coalesced(grads, world_size, bucket_size_mb)
    # else:
    for var in grads:
        jittor.mpi.mpi_all_reduce(var, op='mean')


class DistOptimizerHook(OptimizerHook):
    """Deprecated optimizer hook for distributed training."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"DistOptimizerHook" is deprecated, please switch to'
                      '"mmcv.runner.OptimizerHook".')
        super().__init__(*args, **kwargs)


def reduce_mean(var):
    """"Obtain the mean of tensor on different GPUs."""
    if not (jittor.mpi == True):
        return var
    var = var.clone()
    jittor.mpi.mpi_all_reduce(var, op='mean')
    return var
