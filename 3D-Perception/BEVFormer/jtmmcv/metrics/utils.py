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
from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Mapping, Sequence

import jittor

from .distributed import rank_zero_warn

METRIC_EPS = 1e-6

def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to call of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections is of
            the :attr:`wrong_type` even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to call of ``function``)

    Returns:
        the resulting collection

    Example:
        >>> apply_to_collection(torch.tensor([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
        tensor([64,  0,  4, 36, 49])
        >>> apply_to_collection([8, 0, 2, 6, 7], dtype=int, function=lambda x: x ** 2)
        [64, 0, 4, 36, 49]
        >>> apply_to_collection(dict(abc=123), dtype=int, function=lambda x: x ** 2)
        {'abc': 15129}
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data



def dim_zero_cat(x):
    x = x if isinstance(x, (list, tuple)) else [x]
    return  jittor.concat(x, dim=0)


def dim_zero_sum(x):
    return  jittor.sum(x, dim=0)


def dim_zero_mean(x):
    return  jittor.mean(x, dim=0)

def dim_zero_max(x):
    return  jittor.max(x, dim=0)

def dim_zero_min(x):
    return  jittor.min(x, dim=0)

def _squeeze_scalar_element_tensor(x: jittor.Var) -> jittor.Var:
    return x.squeeze() if x.numel() == 1 else x

# unsure if this is the correct way to implement this
def _squeeze_if_scalar(x):
    return apply_to_collection(x, jittor.Var, _squeeze_scalar_element_tensor)

def _flatten(x):
    return [item for sublist in x for item in sublist]


def _check_same_shape(pred: jittor.Var, target: jittor.Var):
    """ Check that predictions and target have the same shape, else raise error """
    if pred.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")


def _input_format_classification_one_hot(
    num_classes: int,
    preds: jittor.Var,
    target: jittor.Var,
    threshold: float = 0.5,
    multilabel: bool = False
) -> Tuple[jittor.Var, jittor.Var]:
    """Convert preds and target tensors into one hot spare label tensors

    Args:
        num_classes: number of classes
        preds: either tensor with labels, tensor with probabilities/logits or
            multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input
        multilabel: boolean flag indicating if input is multilabel

    Returns:
        preds: one hot tensor of shape [num_classes, -1] with predicted labels
        target: one hot tensors of shape [num_classes, -1] with true labels
    """
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds and target must have same number of dimensions, or one additional dimension for preds")

    if preds.ndim == target.ndim + 1:
        # multi class probabilites
        preds = jittor.argmax(preds, dim=1)[0]

    if preds.ndim == target.ndim and preds.dtype in (jittor.int32, jittor.int16) and num_classes > 1 and not multilabel:
        # multi-class
        preds = to_onehot(preds, num_classes=num_classes)
        target = to_onehot(target, num_classes=num_classes)

    elif preds.ndim == target.ndim and preds.is_floating_point():
        # binary or multilabel probablities
        preds = (preds >= threshold).long()

    # transpose class as first dim and reshape
    if preds.ndim > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)

    return preds.reshape(num_classes, -1), target.reshape(num_classes, -1)


def to_onehot(
    label_tensor: jittor.Var,
    num_classes: Optional[int] = None,
) -> jittor.Var:
    """
    Converts a dense label tensor to one-hot format

    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:

        >>> x = jittor.Var([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = jittor.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def select_topk(prob_tensor: jittor.Var, topk: int = 1, dim: int = 1) -> jittor.Var:
    """
    Convert a probability tensor to binary by selecting top-k highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of highest entries to turn into 1s
        dim: dimension on which to compare entries

    Output:
        A binary tensor of the same shape as the input tensor of type jittor.int32

    Example:
        >>> x = jittor.Var([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=jittor.int32)
    """
    zeros = jittor.zeros_like(prob_tensor)
    topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


def to_categorical(tensor: jittor.Var, argmax_dim: int = 1) -> jittor.Var:
    """
    Converts a tensor of probabilities to a dense label tensor

    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:

        >>> x = jittor.Var([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])

    """
    return jittor.argmax(tensor, dim=argmax_dim)[0]


def get_num_classes(
    pred: jittor.Var,
    target: jittor.Var,
    num_classes: Optional[int] = None,
) -> int:
    """
    Calculates the number of classes for a given prediction and target tensor.

    Args:
        pred: predicted values
        target: true labels
        num_classes: number of classes if known

    Return:
        An integer that represents the number of classes.
    """
    num_target_classes = int(target.max().detach().item() + 1)
    num_pred_classes = int(pred.max().detach().item() + 1)
    num_all_classes = max(num_target_classes, num_pred_classes)

    if num_classes is None:
        num_classes = num_all_classes
    elif num_classes != num_all_classes:
        rank_zero_warn(
            f"You have set {num_classes} number of classes which is"
            f" different from predicted ({num_pred_classes}) and"
            f" target ({num_target_classes}) number of classes",
            RuntimeWarning,
        )
    return num_classes


def reduce(to_reduce: jittor.Var, reduction: str) -> jittor.Var:
    """
    Reduces a given tensor by a given reduction method

    Args:
        to_reduce : the tensor, which shall be reduced
       reduction :  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "elementwise_mean":
        return  jittor.mean(to_reduce)
    if reduction == "none":
        return to_reduce
    if reduction == "sum":
        return  jittor.sum(to_reduce)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(
    num: jittor.Var, denom: jittor.Var, weights: jittor.Var, class_reduction: str = "none"
) -> jittor.Var:
    """
    Function used to reduce classification metrics of the form `num / denom * weights`.
    For example for calculating standard accuracy the num would be number of
    true positives per class, denom would be the support per class, and weights
    would be a tensor of 1s

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'`` or ``None``: returns calculated metric per class

    Raises:
        ValueError:
            If ``class_reduction`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``.
    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    if class_reduction == "micro":
        fraction =  jittor.sum(num) /  jittor.sum(denom)
    else:
        fraction = num / denom

    # We need to take care of instances where the denom can be 0
    # for some (or all) classes which will produce nans
    fraction[fraction != fraction] = 0

    if class_reduction == "micro":
        return fraction
    elif class_reduction == "macro":
        return  jittor.mean(fraction)
    elif class_reduction == "weighted":
        return  jittor.sum(fraction * (weights.float() /  jittor.sum(weights)))
    elif class_reduction == "none" or class_reduction is None:
        return fraction

    raise ValueError(
        f"Reduction parameter {class_reduction} unknown."
        f" Choose between one of these: {valid_reduction}"
    )


def _stable_1d_sort(x: jittor, N: int = 2049):
    """
    Stable sort of 1d tensors. Pytorch defaults to a stable sorting algorithm
    if number of elements are larger than 2048. This function pads the tensors,
    makes the sort and returns the sorted array (with the padding removed)
    See this discussion: https://discuss.pytorch.org/t/is-torch-sort-stable/20714
    """
    if x.ndim > 1:
        raise ValueError('Stable sort only works on 1d tensors')
    n = x.numel()
    if N - n > 0:
        x_max = x.max()
        x =  jittor.concat([x, (x_max + 1) * jittor.ones(N - n, dtype=x.dtype)], 0)
    x_sort = x.sort()
    i = min(N, n)
    return x_sort.values[:i], x_sort.indices[:i]
