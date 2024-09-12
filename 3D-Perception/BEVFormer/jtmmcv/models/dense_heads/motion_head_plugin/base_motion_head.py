#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import jittor
from jittor import Module
import copy
import pickle
import jittor.nn as jnn
from jtmmcv.utils.general import Unflatten,LogSoftmax
from jtmmcv.models import  build_loss
from jtmmcv.models.bricks.transformer import build_transformer_layer_sequence

class BaseMotionHead(Module):
    def __init__(self, *args, **kwargs):
        super(BaseMotionHead, self).__init__()
        pass

    def _build_loss(self, loss_traj):
        """
        Build the loss function for the motion prediction task.

        Args:
            loss_traj (dict): A dictionary containing the parameters for the loss function.

        Returns:
            None
        """
        self.loss_traj = build_loss(loss_traj)
        self.unflatten_traj = Unflatten(3, (self.predict_steps, 5))
        self.log_softmax = LogSoftmax(dim=2)

    def _load_anchors(self, anchor_info_path):
        """
        Load the anchor information from a file.

        Args:
            anchor_info_path (str): The path to the file containing the anchor information.

        Returns:
            None
        """
        anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
        self.kmeans_anchors = jittor.stack(
            [jittor.array(a) for a in anchor_infos["anchors_all"]])  # Nc, Pc, steps, 2
        
    def _build_layers(self, transformerlayers, det_layer_num):
        """
        Build the layers of the motion prediction module.

        Args:
            transformerlayers (dict): A dictionary containing the parameters for the transformer layers.
            det_layer_num (int): The number of detection layers.

        Returns:
            None
        """
        self.learnable_motion_query_embedding = jnn.Embedding(
            self.num_anchor * self.num_anchor_group, self.embed_dims)
        self.motionformer = build_transformer_layer_sequence(
            transformerlayers)
        self.layer_track_query_fuser = jnn.Sequential(
            jnn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
            jnn.LayerNorm(self.embed_dims),
            jnn.ReLU()
        )

        self.agent_level_embedding_layer = jnn.Sequential(
            jnn.Linear(self.embed_dims, self.embed_dims*2),
            jnn.ReLU(),
            jnn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.scene_level_ego_embedding_layer = jnn.Sequential(
            jnn.Linear(self.embed_dims, self.embed_dims*2),
            jnn.ReLU(),
            jnn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.scene_level_offset_embedding_layer = jnn.Sequential(
            jnn.Linear(self.embed_dims, self.embed_dims*2),
            jnn.ReLU(),
            jnn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.boxes_query_embedding_layer = jnn.Sequential(
            jnn.Linear(self.embed_dims, self.embed_dims*2),
            jnn.ReLU(),
            jnn.Linear(self.embed_dims*2, self.embed_dims),
        )
    
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        traj_cls_branch = []
        traj_cls_branch.append(jnn.Linear(self.embed_dims, self.embed_dims))
        traj_cls_branch.append(jnn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(jnn.ReLU())
        for _ in range(self.num_reg_fcs-1):
            traj_cls_branch.append(jnn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(jnn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(jnn.ReLU())
        traj_cls_branch.append(jnn.Linear(self.embed_dims, 1))
        traj_cls_branch = jnn.Sequential(*traj_cls_branch)

        traj_reg_branch = []
        traj_reg_branch.append(jnn.Linear(self.embed_dims, self.embed_dims))
        traj_reg_branch.append(jnn.ReLU())
        for _ in range(self.num_reg_fcs-1):
            traj_reg_branch.append(jnn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(jnn.ReLU())
        traj_reg_branch.append(jnn.Linear(self.embed_dims, self.predict_steps * 5))
        traj_reg_branch = jnn.Sequential(*traj_reg_branch)

        num_pred = self.motionformer.num_layers
        self.traj_cls_branches = jnn.ModuleList()
        for _ in range(num_pred):
            layers = []
            layers.append(jnn.Linear(self.embed_dims, self.embed_dims))
            layers.append(jnn.LayerNorm(self.embed_dims))
            layers.append(jnn.ReLU())
            for _ in range(self.num_reg_fcs-1):
                layers.append(jnn.Linear(self.embed_dims, self.embed_dims))
                layers.append(jnn.LayerNorm(self.embed_dims))
                layers.append(jnn.ReLU())
            layers.append(jnn.Linear(self.embed_dims, 1))
            self.traj_cls_branches.append(jnn.Sequential(*layers))

        self.traj_reg_branches = jnn.ModuleList()
        for _ in range(num_pred):
            layers = []
            layers.append(jnn.Linear(self.embed_dims, self.embed_dims))
            layers.append(jnn.ReLU())
            for _ in range(self.num_reg_fcs-1):
                layers.append(jnn.Linear(self.embed_dims, self.embed_dims))
                layers.append(jnn.ReLU())
            layers.append(jnn.Linear(self.embed_dims, self.predict_steps * 5))
            self.traj_reg_branches.append(jnn.Sequential(*layers))
            

    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        extract the bboxes centers and normized according to the bev range
        
        Args:
            bbox_results (List[Tuple[jittor.Var]]): A list of tuples containing the bounding box results for each image in the batch.
            bev_range (List[float]): A list of float values representing the bird's eye view range.

        Returns:
            jittor.Var: A tensor representing normized centers of the detection bounding boxes.
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            xy = bboxes.gravity_center[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / \
                (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / \
                (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                jittor.concat([x_norm[:, None], y_norm[:, None]], dim=-1))
        return jittor.stack(det_bbox_posembed)