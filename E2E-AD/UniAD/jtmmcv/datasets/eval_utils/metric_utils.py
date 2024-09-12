import jittor
import math
import numpy as np
from typing import List, Dict, Tuple, Callable, Union

def min_ade(traj: jittor.Var, traj_gt: jittor.Var,
            masks: jittor.Var) -> Tuple[jittor.Var, jittor.Var]:
    """
    Computes average displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape
    [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = err**2
    err = jittor.sum(err, dim=3)
    err = err**0.5
    err = jittor.sum(err * (1 - masks_rpt), dim=2) / \
        jittor.clamp(jittor.sum((1 - masks_rpt), dim=2), min_v=1)
    inds, err = jittor.argmin(err, dim=1)

    return err, inds


def min_fde(traj: jittor.Var, traj_gt: jittor.Var,
            masks: jittor.Var) -> Tuple[jittor.Var, jittor.Var]:
    """
    Computes final displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    lengths = jittor.sum(1 - masks, dim=1).long()
    inds = lengths.unsqueeze(1).unsqueeze(
        2).unsqueeze(3).repeat(1, num_modes, 1, 2) - 1

    traj_last = jittor.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
    traj_gt_last = jittor.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)

    err = traj_gt_last - traj_last[..., 0:2]
    err = err**2
    err = jittor.sum(err, dim=2)
    err = err**0.5
    inds, err = jittor.argmin(err, dim=1)

    return err, inds


def miss_rate(
        traj: jittor.Var,
        traj_gt: jittor.Var,
        masks: jittor.Var,
        dist_thresh: float = 2) -> jittor.Var:
    """
    Computes miss rate for mini batch of trajectories,
    with respect to ground truth and given distance threshold
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :param dist_thresh: distance threshold for computing miss rate.
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = dist**2
    dist = jittor.sum(dist, dim=3)
    dist = dist**0.5
    dist[masks_rpt.bool()] = -math.inf
    _,dist = jittor.argmax(dist, dim=2)
    _,dist = jittor.argmin(dist, dim=1)
    m_r = jittor.sum(jittor.array(dist > dist_thresh)) / len(dist)

    return m_r

def traj_fde(gt_box, pred_box, final_step):
    if gt_box.traj.shape[0] <= 0:
        return np.inf
    final_step = min(gt_box.traj.shape[0], final_step)
    gt_final = gt_box.traj[None, final_step-1]
    pred_final = np.array(pred_box.traj)[:,final_step-1,:]
    err = gt_final - pred_final
    err = np.sqrt(np.sum(np.square(gt_final - pred_final), axis=-1))
    return np.min(err)