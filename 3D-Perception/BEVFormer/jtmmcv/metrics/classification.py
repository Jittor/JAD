# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import wraps
from typing import Callable, Optional, Sequence, Tuple

import jittor
from .utils import get_num_classes as __gnc
from .utils import to_categorical as __tc
from .distributed import rank_zero_warn


def to_categorical(tensor: jittor.Var, argmax_dim: int = 1) -> jittor.Var:
    """
    Converts a tensor of probabilities to a dense label tensor

    """
    rank_zero_warn(
        "This `to_categorical` was deprecated in v1.1.0 in favor of"
        " change with jittor."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __tc(tensor)


def get_num_classes(
    pred: jittor.Var,
    target: jittor.Var,
    num_classes: Optional[int] = None,
) -> int:
    """
    Calculates the number of classes for a given prediction and target tensor.

    .. warning :: Deprecated in favor of :func:`~mmcv.pytorch_lightning.metrics.utils.get_num_classes`

    """
    rank_zero_warn(
        "change by jittor", DeprecationWarning
    )
    return __gnc(pred, target, num_classes)


def stat_scores(
    pred: jittor.Var,
    target: jittor.Var,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[jittor.Var, jittor.Var, jittor.Var, jittor.Var, jittor.Var]:
    """
    Calculates the number of true positive, false positive, true negative
    and false negative for a specific class

    Args:
        pred: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:

        >>> x = jittor.Var([1, 2, 3])
        >>> y = jittor.Var([0, 2, 3])
        >>> tp, fp, tn, fn, sup = stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))

    """
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(jittor.int32).sum()
    fp = ((pred == class_index) * (target != class_index)).to(jittor.int32).sum()
    tn = ((pred != class_index) * (target != class_index)).to(jittor.int32).sum()
    fn = ((pred != class_index) * (target == class_index)).to(jittor.int32).sum()
    sup = (target == class_index).to(jittor.int32).sum()

    return tp, fp, tn, fn, sup


# todo: remove in 1.4
def stat_scores_multiple_classes(
    pred: jittor.Var,
    target: jittor.Var,
    num_classes: Optional[int] = None,
    argmax_dim: int = 1,
    reduction: str = 'none',
) -> Tuple[jittor.Var, jittor.Var, jittor.Var, jittor.Var, jittor.Var]:
    """
    Calculates the number of true positive, false positive, true negative
    and false negative for each class

    .. warning :: Deprecated in favor of :func:`~mmcv.pytorch_lightning.metrics.functional.stat_scores`

    Raises:
        ValueError:
            If ``reduction`` is not one of ``"none"``, ``"sum"`` or ``"elementwise_mean"``.
    """

    rank_zero_warn(
        "This `stat_scores_multiple_classes` was deprecated in v1.2.0 in favor of"
        " `from mmcv.pytorch_lightning.metrics.functional import stat_scores`."
        " It will be removed in v1.4.0", DeprecationWarning
    )
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    num_classes = get_num_classes(pred=pred, target=target, num_classes=num_classes)

    if pred.dtype != bool:
        pred = pred.clamp(max_v=num_classes)
    if target.dtype != bool:
        target = target.clamp(max_v=num_classes)

    possible_reductions = ('none', 'sum', 'elementwise_mean')
    if reduction not in possible_reductions:
        raise ValueError("reduction type %s not supported" % reduction)

    if reduction == 'none':
        pred = pred.view((-1, )).long()
        target = target.view((-1, )).long()

        tps = jittor.zeros((num_classes + 1, ))
        fps = jittor.zeros((num_classes + 1, ))
        fns = jittor.zeros((num_classes + 1, ))
        sups = jittor.zeros((num_classes + 1, ))

        match_true = (pred == target).float()
        match_false = 1 - match_true
        
        tps.scatter(0, pred, match_true,reduce='add')
        fps.scatter(0, pred, match_false,reduce='add')
        fns.scatter(0, target, match_false,reduce='add')
        tns = pred.size(0) - (tps + fps + fns)
        sups.scatter(0, target, jittor.ones_like(match_true),reduce='add')

        tps = tps[:num_classes]
        fps = fps[:num_classes]
        tns = tns[:num_classes]
        fns = fns[:num_classes]
        sups = sups[:num_classes]

    elif reduction == 'sum' or reduction == 'elementwise_mean':
        count_match_true = (pred == target).sum().float()
        oob_tp, oob_fp, oob_tn, oob_fn, oob_sup = stat_scores(pred, target, num_classes, argmax_dim)

        tps = count_match_true - oob_tp
        fps = pred.nelement() - count_match_true - oob_fp
        fns = pred.nelement() - count_match_true - oob_fn
        tns = pred.nelement() * (num_classes + 1) - (tps + fps + fns + oob_tn)
        sups = pred.nelement() - oob_sup.float()

        if reduction == 'elementwise_mean':
            tps /= num_classes
            fps /= num_classes
            fns /= num_classes
            tns /= num_classes
            sups /= num_classes

    return tps.float(), fps.float(), tns.float(), fns.float(), sups.float()

