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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jittor
from .distributed import rank_zero_warn
from .jittormetric import Metric as _Metric
from .jittormetric import MetricCollection as _MetricCollection
# from torchmetrics import MetricCollection as _MetricCollection


def _neg(tensor: jittor.Var):
        return -jittor.abs(tensor)
    
def fmod(x, y):
    b = jittor.divide(x, y).round()
    return (x - y * b)

class Metric(_Metric):
    r"""
    This implementation refers to :class:`~torchmetrics.Metric`.

    .. warning:: This metric is deprecated, use ``torchmetrics.Metric``. Will be removed in v1.5.0.
    """

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def __hash__(self):
        return super().__hash__()

    def __add__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.add, self, other)

    def __and__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_and, self, other)

    def __eq__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.all_equal, self, other)

    def __floordiv__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.floor_divide, self, other)

    def __ge__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.greater_equal, self, other)

    def __gt__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.greater, self, other)

    def __le__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.less_equal, self, other)

    def __lt__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.less, self, other)

    def __matmul__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.matmul, self, other)

    def __mod__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(fmod, self, other)

    def __mul__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.multiply, self, other)

    def __ne__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.ne, self, other)

    def __or__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_or, self, other)

    def __pow__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.pow, self, other)

    def __radd__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.add, other, self)

    def __rand__(self, other: Any):
        from .compositional import CompositionalMetric

        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(jittor.bitwise_and, self, other)

    def __rfloordiv__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.floor_divide, other, self)

    def __rmatmul__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.matmul, other, self)

    def __rmod__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(fmod, other, self)

    def __rmul__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.multiply, other, self)

    def __ror__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_or, other, self)

    def __rpow__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.pow, other, self)

    def __rsub__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.subtract, other, self)

    def __rtruediv__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.divide, other, self)

    def __rxor__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_xor, other, self)

    def __sub__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.subtract, self, other)

    def __truediv__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.divide, self, other)

    def __xor__(self, other: Any):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_xor, self, other)

    def __abs__(self):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.abs, self, None)

    def __inv__(self):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.bitwise_not, self, None)

    def __invert__(self):
        return self.__inv__()

    def __neg__(self):
        from .compositional import CompositionalMetric
        return CompositionalMetric(_neg, self, None)

    def __pos__(self):
        from .compositional import CompositionalMetric
        return CompositionalMetric(jittor.abs, self, None)


    


class MetricCollection(_MetricCollection):
    r"""
    This implementation refers to :class:`~torchmetrics.MetricCollection`.

    .. warning:: This metric is deprecated, use ``torchmetrics.MetricCollection``. Will be removed in v1.5.0.
    """

    def __init__(self, metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]]):
        rank_zero_warn(
            "This `MetricCollection` was deprecated since v1.3.0 in favor of `torchmetrics.MetricCollection`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super().__init__(metrics=metrics)
