#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import jittor
import jittor.nn as jnn

from jittor import Module
from jtmmcv.models.builder import HEADS, build_loss
from jittor.einops import rearrange
from jtmmcv.models.utils.functional import bivariate_gaussian_activation
from jtmmcv.models.bricks.transformer import TransformerDecoderLayer, TransformerDecoder
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np
import copy

@HEADS.register_module()
class PlanningHeadSingleMode(Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = jnn.Embedding(3, embed_dims)
        self.reg_branch = jnn.Sequential(
            jnn.Linear(embed_dims, embed_dims),
            jnn.ReLU(),
            jnn.Linear(embed_dims, planning_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        
        #### planning head
        fuser_dim = 3
        attn_module_layer = TransformerDecoderLayer(embed_dims, 8, dim_feedexecute=embed_dims*2, dropout=0.0, batch_first=False)
        self.attn_module = TransformerDecoder(attn_module_layer, 3)
        
        self.mlp_fuser = jnn.Sequential(
                jnn.Linear(embed_dims*fuser_dim, embed_dims),
                jnn.LayerNorm(embed_dims),
                jnn.ReLU(),
            )
        
        self.pos_embed = jnn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = jnn.ModuleList(self.loss_collision)
        
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            self.bev_adapter = jnn.ModuleList()
            N_Blocks = 3
            for _ in range(N_Blocks):
                self.bev_adapter.append(jnn.Sequential(
                    jnn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                    jnn.ReLU(),
                    jnn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
                ))
                
           
    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (jittor.Var): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (jittor.Var, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (jittor.Var, optional): The mask for the self-driving car's planning.
            command (jittor.Var, optional): The driving command issued to the self-driving car.
            gt_future_boxes (jittor.Var, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        occ_mask = None
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning

    def execute(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (jittor.Var): Bird's eye view feature embedding.
            occ_mask (jittor.Var): Instance mask for occupancy.
            bev_pos (jittor.Var): BEV position.
            sdc_traj_query (jittor.Var): SDC trajectory query.
            sdc_track_query (jittor.Var): SDC track query.
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
        
        
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed.expand(navi_embed.shape[0],P,navi_embed.shape[1])
        plan_query = jittor.concat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).argmax(1, keepdims=True)[1]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = rearrange(plan_query, 'b p c -> p b c')
        
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed +  bev_pos
        
        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')
        ##########################
      
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = jittor.cumsum(sdc_traj_all[...,:2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
        
        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (jittor.Var): SDC trajectory tensor.
            occ_mask (jittor.Var): Occupancy flow instance mask. 
        Returns:
            jittor.Var: Optimized SDC trajectory tensor.
        """
        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = jittor.nonzero(occ_mask[0][cur_t])
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            if pos_xy.shape[0]==0:
                keep_index = jittor.sum((sdc_traj_all[0, t, :2][None, :])**2, dim=-1) < self.occ_filter_range**2
            else:
                keep_index = jittor.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, dim=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].detach().numpy())
            valid_occupancy_num += jittor.sum(keep_index>0)
        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return jittor.array(sdc_traj_optim[None], dtype=sdc_traj_all.dtype)
    
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], jittor.any(sdc_planning_mask[0, :, :self.planning_steps].to(jittor.bool), dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], jittor.any(sdc_planning_mask[0, :, :self.planning_steps].to(jittor.bool), dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict