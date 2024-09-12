#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import jittor

from jittor import Module
import math
from typing import Tuple

from jtmmcv.models import LOSSES

@LOSSES.register_module()
class TrajLoss(Module):
    """
    MTP loss modified to include variances. Uses MSE for mode selection.
    Can also be used with
    Multipath outputs, with residuals added to anchors.
    """

    def __init__(self, use_variance=False, cls_loss_weight=1., nll_loss_weight=1., loss_weight_minade=0., loss_weight_minfde=1., loss_weight_mr=1.):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing
            regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component,
            compared to regression component
                of loss, default: 1
        """
        super(TrajLoss, self).__init__()
        self.use_variance = use_variance
        self.cls_loss_weight = cls_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.loss_weight_minade = loss_weight_minade
        self.loss_weight_minfde = loss_weight_minfde

    def execute(self,
                traj_prob, 
                traj_preds, 
                gt_future_traj, 
                gt_future_traj_valid_mask):
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories
        and 'probs': mode (log) probabilities
        :param ground_truth: Either a tensor with ground truth trajectories
        or a dictionary
        :return:
        """
        # Unpack arguments
        traj = traj_preds # (b, nmodes, seq, 5)
        log_probs = traj_prob
        traj_gt = gt_future_traj

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # Masks for variable length ground truth trajectories
        masks = 1 - gt_future_traj_valid_mask.to(traj.dtype)

        l_minfde, inds = min_fde(traj, traj_gt, masks)
        try:
            l_mr = miss_rate(traj, traj_gt, masks)
        except:
            l_mr = jittor.zeros_like(l_minfde)
        l_minade, inds = min_ade(traj, traj_gt, masks)
        inds_rep = inds.repeat(
            sequence_length,
            pred_params, 1, 1).permute(3, 2, 0, 1)

        # Calculate MSE or NLL loss for trajectories corresponding to selected
        # outputs:
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1)

        if self.use_variance:
            l_reg = traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = l_minade

        # Compute classification loss
        l_class = - jittor.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        l_reg = jittor.sum(l_reg)/(batch_size + 1e-5) 
        l_class = jittor.sum(l_class)/(batch_size + 1e-5)
        l_minade = jittor.sum(l_minade)/(batch_size + 1e-5) 
        l_minfde = jittor.sum(l_minfde)/(batch_size + 1e-5) 

        loss = l_class * self.cls_loss_weight + l_reg * self.nll_loss_weight + l_minade * self.loss_weight_minade + l_minfde * self.loss_weight_minfde
        return loss, l_class, l_reg, l_minade, l_minfde, l_mr

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

def traj_nll(
        pred_dist: jittor.Var,
        traj_gt: jittor.Var,
        masks: jittor.Var):
    """
    Computes negative log likelihood of ground truth trajectory under a
    predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the
    prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution,
    shape [batch_size, sequence_length, 5]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :return:
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = jittor.pow(1 - jittor.pow(rho, 2), -0.5)

    nll = 0.5 * jittor.pow(ohr, 2) * \
        (jittor.pow(sig_x, 2) * jittor.pow(x - mu_x, 2) + jittor.pow(sig_y, 2) *
         jittor.pow(y - mu_y, 2) - 2 * rho * jittor.pow(sig_x, 1) *
         jittor.pow(sig_y, 1) * (x - mu_x) * (y - mu_y)) - \
        jittor.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = jittor.sum(nll * (1 - masks), dim=1) / (jittor.sum((1 - masks), dim=1) + 1e-5)
    # Note: Normalizing with jittor.sum((1 - masks), dim=1) makes values
    # somewhat comparable for trajectories of
    # different lengths

    return nll

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
    lengths = jittor.sum(1 - masks, dim=1).long()
    valid_mask = lengths > 0
    traj = traj[valid_mask]
    traj_gt = traj_gt[valid_mask]
    masks = masks[valid_mask]
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
    inds,err = jittor.argmin(err, dim=1)

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
    _, dist = jittor.argmax(dist, dim=2)
    _, dist = jittor.argmin(dist, dim=1)
    m_r = jittor.sum(jittor.array(dist > dist_thresh)) / len(dist)

    return m_r
