# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from functools import wraps
from typing import Any, Optional, Union

import jittor

log = logging.getLogger(__name__)

class ReduceOp:
    SUM = None

class group:
    WORLD = None


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', int(os.environ.get('LOCAL_RANK', 0)))


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


def _debug(*args, **kwargs):
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.
    It is useful in single-node training when we don't want to connect to a real master node but
    have to set the `MASTER_PORT` environment variable.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def gather_all_tensors(result: Union[jittor.Var], group: Optional[Any] = None):
    """
    jittor 目前不支持all_gather操作.因此仅采用单卡测试，同时该函数输入输出相同
    """

    # convert tensors to contiguous format
    result = result.contiguous()

    # world_size = jittor.world_size

    # gathered_result = [jittor.zeros_like(result) for _ in range(world_size)]

    # # sync and broadcast all
    # torch.distributed.barrier(group=group)
    # torch.distributed.all_gather(gathered_result, result, group)

    return [result]


def sync_ddp_if_available(
    result: Union[jittor.Var],
    group: Optional[Any] = None,
    reduce_op: Optional[Union[ReduceOp, str]] = None
) -> jittor.Var:
    """
    Function to reduce a tensor across worker processes during distributed training
    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value
    """
    if jittor.mpi:
        return sync_ddp(result, group=group, reduce_op=reduce_op)
    return result


def sync_ddp(
    result: Union[jittor.Var],
    group: Optional[Any] = None,
    reduce_op: Optional[Union[ReduceOp, str]] = None
) -> jittor.Var:
    """
    Function to reduce the tensors from several ddp processes to one master process

    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value
    """
    divide_by_world_size = False

    if group is None:
        group = jittor.world_size

    op = reduce_op if isinstance(reduce_op, ReduceOp) else ReduceOp.SUM

    if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
        divide_by_world_size = True

    jittor.mpi.mpi_all_reduce(result, op=op, group=group, async_op=False)

    if divide_by_world_size:
        result = result / jittor.world_size

    return result
