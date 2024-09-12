from typing import Tuple

from jittor import Var

from .single_stage import SingleStage3DDetector
from jtmmcv.models import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    """
    Base class for voxelnet.
    """

    def __init__(self,
                voxel_encoder,
                middle_encoder,
                backbone,
                neck,
                bbox_head,
                train_cfg,
                test_cfg,
                data_preprocessor,
                init_cfg):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = build_backbone(voxel_encoder)
        self.middle_encoder = build_backbone(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Var]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x