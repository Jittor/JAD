# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import numpy as np
import jittor
from jittor.dataset import DataLoader
from mtr_jittor.utils import common_utils

from .nuscenes.nuscenes_dataset import NuscenesDataset


__all__ = {
    "NuscenesDataset": NuscenesDataset,
}


def build_dataloader(
    dataset_cfg,
    batch_size,
    dist,
    workers=4,
    logger=None,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=0,
    add_worker_init_fn=False,
):

    def worker_init_fn_(worker_id):
        torch_seed = 1
        np_seed = torch_seed // 2**32 - 1
        np.random.seed(np_seed)

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, "merge_all_iters_to_one_epoch")
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if training:
        sampler = None
    else:
        sampler = jittor.dataset.SequentialSampler(dataset)

    drop_last = dataset_cfg.get("DATALOADER_DROP_LAST", False) and training
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=(sampler is None) and training,
        drop_last=drop_last,
        sampler=sampler,
    )

    return dataset, dataloader, sampler
