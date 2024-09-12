#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import jittor
import jittor.nn as jnn
import numpy as np
from skimage.draw import polygon
from jtmmcv.metrics.metric import Metric
from ..occ_head_plugin import calculate_birds_eye_view_parameters, gen_dx_bx


class PlanningMetric(Metric):
    def __init__(
        self,
        n_future=6,
    ):
        super().__init__()
        dx, bx, _ = gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
        dx, bx = dx[:2], bx[:2]
        self.dx = jnn.Parameter(dx).stop_grad()
        self.bx = jnn.Parameter(bx).stop_grad()

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future

        self.add_state("obj_col", default=jittor.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=jittor.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("L2", default=jittor.zeros(self.n_future),dist_reduce_fx="sum")
        self.add_state("total", default=jittor.array(0), dist_reduce_fx="sum")


    def evaluate_single_coll(self, traj, segmentation):
        '''
        gt_segmentation
        traj: jittor.Var (n_future, 2)
        segmentation: jittor.Var (n_future, 200, 200)
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.numpy()) / (self.dx.numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs[:,:,[0,1]] = trajs[:,:,[1,0]] # can also change original tensor
        trajs = trajs / self.dx
        trajs = trajs.numpy() + rc # (n_future, 32, 2)

        r = trajs[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].numpy())

        return jittor.array(collision)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        '''
        trajs: jittor.Var (B, n_future, 2)
        gt_trajs: jittor.Var (B, n_future, 2)
        segmentation: jittor.Var (B, n_future, 200, 200)
        '''
        B, n_future, _ = trajs.shape
        trajs = trajs * jittor.array([-1, 1])
        gt_trajs = gt_trajs * jittor.array([-1, 1])

        obj_coll_sum = jittor.zeros(n_future)
        obj_box_coll_sum = jittor.zeros(n_future)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()

            m1 = jittor.logical_and(
                jittor.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                jittor.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = jittor.logical_and(m1, jittor.logical_not(gt_box_coll))

            ti = jittor.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            m2 = jittor.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        trajs: jittor.Var (B, n_future, 3)
        gt_trajs: jittor.Var (B, n_future, 3)
        '''
        return jittor.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation):
        '''
        trajs: jittor.Var (B, n_future, 3)
        gt_trajs: jittor.Var (B, n_future, 3)
        segmentation: jittor.Var (B, n_future, 200, 200)
        '''
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total +=len(trajs)

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total
        }