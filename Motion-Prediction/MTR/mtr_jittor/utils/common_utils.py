# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022


import numpy as np
import jittor
import logging
import random
import os
import subprocess
import pickle
import shutil
import multiprocessing as mp

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    is_numpy = False

    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0],dtype=angle.dtype)
    if points.shape[-1] == 2:

        rot_matrix = (
            np.stack((cosa, sina, -sina, cosa), axis=1).reshape(-1, 2, 2).astype(np.float32)
        )
        points_rot = np.matmul(points, rot_matrix)
    else:
        ones = np.ones(points.shape[0],dtype=angle.dtype)
        rot_matrix = (
            np.stack(
                (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1
            )
            .reshape(-1, 3, 3)
        ).astype(np.float32)
        rot_matrix = rot_matrix.astype(np.float32)
        points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot.numpy() if is_numpy else points_rot


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        # tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        tensor_list = [np.expand_dims(x, axis=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = np.zeros(
            (cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2), dtype=cur_tensor.dtype
        )
        new_tensor[:, : cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = np.zeros((cur_tensor.shape[0], maxt_feat0), dtype=cur_tensor.dtype)
        new_mask_tensor[:, : cur_tensor.shape[1]] = 1
        ret_mask_list.append(np.bool8(new_mask_tensor))

    ret_tensor = np.concatenate(ret_tensor_list, axis=0)
    ret_mask = np.concatenate(ret_mask_list, axis=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(axis=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else "ERROR")
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else "ERROR")
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else "ERROR")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_dist_info(return_gpu_per_machine=False):
    pass


def get_batch_offsets(batch_idxs, bs):
    """
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_offsets = jittor.zeros(bs + 1).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jittor.misc.set_global_seed(seed)
    # jittor.cudnn.deterministic = True
    # jittor.cudnn.benchmark = False


def init_dist_slurm(tcp_port, local_rank, backend="nccl"):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    pass


def init_dist_jittor(tcp_port, local_rank, backend="nccl"):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    # num_gpus = jittor.get_device_count()
    # rank = jittor.rank
    num_gpus = 1
    rank = 0


    return num_gpus, rank

def merge_results_dist(result_part, size, tmpdir):
    # rank, world_size = get_dist_info()
    rank = 0
    world_size = 1
    os.makedirs(tmpdir, exist_ok=True)

    # dist.barrier() 暂不支持多卡
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    # dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results
