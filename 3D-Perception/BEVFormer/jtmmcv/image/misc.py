# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from jtmmcv.image import imdenormalize

try:
    import jittor
except ImportError:
    jittor = None


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (jittor.Var): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if jittor is None:
        raise RuntimeError('jittor is not installed')
    assert jittor.is_var(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].numpy().transpose(1, 2, 0)
        img = imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs
