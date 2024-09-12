import numpy as np
import bisect
import warnings
import jittor
import copy
import math
import functools
import warnings 
from typing import Callable, Optional, Tuple, Union, List, Generic, TypeVar, Iterator, Optional, List, TypeVar, Generic, Sized, Union
from jtmmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

import jittor
import jittor.nn as jnn
from jittor import Var
from jittor import Module
from jittor.dataset import Dataset

T_co = TypeVar('T_co', covariant=True)
DType = int



def ensure_rng(rng=None):
    """Coerces input into a random number generator.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng



def rotate(
    img: Var,
    angle: float,
    expand: bool = False,
    center: Optional[List[int]] = None,
    fill: Optional[List[float]] = None,
) -> Var:
    """jittor版本 目前仅支持Var输入
    """

    if not isinstance(angle, (int, float)):
        if isinstance(angle, jittor.Var) and angle.shape == [1]:
            angle = float(angle)
        else:
            raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    center_f = [0.0, 0.0]
    if center is not None:
        if isinstance(img, jittor.Var):
            _ = 1 if img.ndim == 2 else img.shape[-3]
            height, width = img.shape[-2:]
        else:
            raise TypeError("Argument img should be a jittor.Var")
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return _rotate(img, matrix=matrix, expand=expand, fill=fill)

def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix

def _rotate(
    img: Var,
    matrix: List[float],
    interpolation: str = "nearest",
    expand: bool = False,
    fill: Optional[Union[int, float, List[float]]] = None,
) -> Var:
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_affine_output_size(matrix, w, h) if expand else (w, h)
    dtype = jittor.float32
    theta = jittor.array(matrix, dtype=dtype).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)

    return _apply_grid_transform(img, grid, interpolation, fill=fill)

def _gen_affine_grid(
    theta: Var,
    w: int,
    h: int,
    ow: int,
    oh: int,
) -> Var:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = jittor.empty(1, oh, ow, 3, dtype=theta.dtype)
    x_grid = jittor.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow)
    base_grid[..., 0] = x_grid
    y_grid = jittor.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=ow).unsqueeze(-1)
    base_grid[..., 1] = y_grid
    jittor.init.fill(base_grid[..., 2], 1.0)

    rescaled_theta = theta.transpose(1, 2) / jittor.array([0.5 * w, 0.5 * h], dtype=theta.dtype)
    output_grid = base_grid.view(1, oh * ow, 3)
    output_grid = jittor.nn.bmm(output_grid, rescaled_theta)
    return output_grid.view(1, oh, ow, 2)

def _apply_grid_transform(
    img: Var, grid: Var, mode: str, fill: Optional[Union[int, float, List[float]]]
) -> Var:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        mask = jittor.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype)
        img = jittor.concat((img, mask), dim=1)

    img = jnn.grid_sample(img, grid, mode=mode, padding_mode="zeros")

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        fill_list, len_fill = (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
        fill_img = jittor.array(fill_list, dtype=img.dtype).view(1, len_fill, 1, 1).expand_as(img)
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def _cast_squeeze_in(img: Var, req_dtypes: List[jittor.dtype]) -> Tuple[Var, bool, bool, jittor.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _compute_affine_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:

    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    # Points are shifted due to affine matrix torch convention about
    # the center point. Center is (0, 0) for image center pivot point (w * 0.5, h * 0.5)
    pts = jittor.array(
        [
            [-0.5 * w, -0.5 * h, 1.0],
            [-0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, -0.5 * h, 1.0],
        ]
    )
    theta = jittor.array(matrix, dtype=jittor.float32).view(2, 3)
    new_pts = jittor.matmul(pts, theta.transpose())
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # shift points to [0, w] and [0, h] interval to match PIL results
    min_vals += jittor.array((w * 0.5, h * 0.5))
    max_vals += jittor.array((w * 0.5, h * 0.5))

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = jittor.ceil((max_vals / tol).round() * tol)
    cmin = jittor.floor((min_vals / tol).round() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])  # w, h

def _cast_squeeze_out(img: Var, need_cast: bool, need_squeeze: bool, out_dtype: jittor.dtype) -> Var:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (jittor.uint8, jittor.int8, jittor.int16, jittor.int32, jittor.int64):
            # it is better to round before cast
            img = jittor.round(img)
        img = img.to(out_dtype)

    return img

def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret

def nan_to_num(input):
    mask = jittor.isnan(input)
    output = jittor.masked_fill(input, mask, 0.0)
    return output

def to_jt_var(data):
    """
        convert data to jt_array
    """
    def _to_jt_var(data):
        if isinstance(data,(list,tuple)):
            data =  [_to_jt_var(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_to_jt_var(d) for k,d in data.items()}
        elif isinstance(data,np.ndarray):
            data = jittor.array(data)
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _to_jt_var(data) 

def cdist_p1(u, v):
    # 计算绝对差矩阵
    diff = jittor.abs(u.unsqueeze(1) - v.unsqueeze(0))  # 广播[u.shape[0], v.shape[0], u.shape[1]]
    
    # 计算每个向量对之间的曼哈顿距离
    dist = jittor.sum(diff, dim=2)  # [u.shape[0], v.shape[0]]
    
    return dist

def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
      
class SmoothL1Loss(Module):
    
    __constants__ = ['reduction']
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.beta = beta
        
    def execute(self, input: jittor.Var, target: jittor.Var) -> jittor.Var:
        return jittor.nn.smooth_l1_loss(target, input, self.reduction)
    
def unflatten(input, dim, sizes):
    '''未经过多测试'''
    
    in_shape = list(input.shape)
    insert_len = len(sizes)
    in_shape[dim: dim + insert_len - 1] = sizes
    
    if dim==-1:
        in_shape = in_shape[:-1]
    
    return input.view(in_shape)


class Unflatten(Module):
    r"""
    jittor 用view实现unflatten
    """

    NamedShape = Tuple[Tuple[str, int]]

    __constants__ = ['dim', 'unflattened_size']
    dim: Union[int, str]

    def __init__(self, dim: Union[int, str], unflattened_size) -> None:
        super().__init__()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)
        else:
            raise TypeError("invalid argument type for dim parameter")

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError("unflattened_size must be tuple of tuples, " +
                                    f"but found element of type {type(elem).__name__} at pos {idx}")
            return
        raise TypeError("unflattened_size must be a tuple of tuples, " +
                        f"but found type {type(input).__name__}")

    def _require_tuple_int(self, input):
        if (isinstance(input, (tuple, list))):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " +
                                    f"but found element of type {type(elem).__name__} at pos {idx}")
            return
        raise TypeError(f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}")

    def execute(self, input: Var) -> Var:
        
        return unflatten(input, self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, unflattened_size={self.unflattened_size}'

class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Var. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Var of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def execute(self, input: Var) -> Var:
        return jittor.nn.log_softmax(input, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'
    
class Sampler(Generic[T_co]):
    r"""Base class for mmcv Samplers.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class DistributedSampler(Sampler[T_co]):
    r"""
    Distributed Sampler for distributed data parallel training.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not jittor.mpi:
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = jittor.world_size
        if rank is None:
            if not jittor.mpi:
                raise RuntimeError("Requires distributed package to be available")
            rank = jittor.rank
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = jittor.randperm(len(self.dataset), dtype=int32).tolist  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.
        
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



def sync(data,reduce_mode="mean",to_numpy=True):
    """
        sync data and convert data to numpy
    """
    def _sync(data):
        if isinstance(data, (list,tuple)):
            data =  [_sync(d) for d in data]
        elif isinstance(data, dict):
            data = {k:_sync(d) for k,d in data.items()}
        elif isinstance(data,jittor.Var):
            if data.dtype == bool:
                data = data.to(jittor.int32)
            if jittor.in_mpi:
                data = data.mpi_all_reduce(reduce_mode)
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data,(int,float,str,np.ndarray,LiDARInstance3DBoxes)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _sync(data) 







