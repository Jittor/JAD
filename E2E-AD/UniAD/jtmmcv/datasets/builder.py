import copy
import platform
import random
from functools import partial

import numpy as np
from jittor.dataset import Dataset
# from jtmmcv.parallel.collate import collate
from jtmmcv.utils import Registry, build_from_cfg, get_dist_info



# jittor 的Dataset集成了Dataloader的功能 按需更改

# DATASETS = Registry('dataset')
# PIPELINES = Registry('pipeline')
# OBJECTSAMPLERS = Registry('Object sampler')
 
from .samplers import GroupSampler
# from .dataset_wrappers import CBGSDataset, ClassBalancedDataset, ConcatDataset, RepeatDataset
from .samplers.sampler import build_sampler

if platform.system() != 'Windows':

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
OBJECTSAMPLERS = Registry('Object sampler')



def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)




def build_dataset(cfg, default_args=None):
    # if isinstance(cfg, (list, tuple)):
    #     dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    # elif cfg['type'] == 'ConcatDataset':
    #     dataset = ConcatDataset(
    #         [build_dataset(c, default_args) for c in cfg['datasets']],
    #         cfg.get('separate_eval', True))
    # elif cfg['type'] == 'RepeatDataset':
    #     dataset = RepeatDataset(
    #         build_dataset(cfg['dataset'], default_args), cfg['times'])
    # elif cfg['type'] == 'ClassBalancedDataset':
    #     dataset = ClassBalancedDataset(
    #         build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    # elif cfg['type'] == 'CBGSDataset':
    #     dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    # elif isinstance(cfg.get('ann_file'), (list, tuple)):
    #     dataset = _concat_dataset(cfg, default_args)
    # else:
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build jittor DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A jittor dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A jittor dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:                
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # dataset = dataset(sampler=sampler)

    data_loader = dataset.set_attrs(
        batch_size=batch_size,
        shuffle=shuffle,
        buffer_size=805306368,
        num_workers=num_workers,
        sampler=sampler,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if platform.system() != 'Windows':

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
