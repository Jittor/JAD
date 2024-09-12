import jittor
from jittor import Var
from jittor import Module

import numpy
a=numpy.allclose
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, \
    Tuple, Union, Mapping, OrderedDict, Iterable, Hashable


from jtmmcv.metrics.utils import dim_zero_cat, dim_zero_max, dim_zero_mean, dim_zero_min, \
     dim_zero_sum, _flatten, _squeeze_if_scalar
from jtmmcv.metrics.distributed import rank_zero_warn,gather_all_tensors


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




def allclose(var1: Var, var2: Var, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    """Wrapper of torch.allclose that is robust towards dtype difference."""
    if var1.dtype != var2.dtype:
        var2 = var2.to(dtype=var1.dtype)
    if jittor.abs(var1 - var2) <= (atol + rtol * jittor.abs(var2)):
        return True
    else:
        return False
    


def _flatten_dict(x: Dict) -> Dict:
    """Flatten dict of dicts into single dict."""
    new_dict = {}
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                new_dict[k] = v
        else:
            new_dict[key] = value
    return new_dict


def jit_distributed_available() -> bool:
    return jittor.mpi


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
        >>> apply_to_collection(jittor.Var([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
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


def _neg(tensor: jittor.Var):
    return -jittor.abs(tensor)

def fmod(x, y):
    b = jittor.divide(x, y).round()
    return (x - y * b)


class Metric(Module, ABC):
    """Base class for all metrics present in the Metrics API.

    Implements ``add_state()``, ``forward()``, ``reset()`` and a few other things to
    handle distributed synchronization and per-step metric computation.

    Override ``update()`` and ``compute()`` functions to implement your own metric. Use
    ``add_state()`` to register metric state variables which keep track of state on each
    call of ``update()`` and are synchronized across processes when ``compute()`` is called.

    Note:
        Metric state variables can either be :class:`~jittor.Var` or an empty list which can we used
        to store :class:`~jittor.Var`.

    Note:
        Different metrics only override ``update()`` and not ``forward()``. A call to ``update()``
        is valid, but it won't return the metric value at the current step. A call to ``forward()``
        automatically calls ``update()`` and also returns the metric value at the current step.

    Args:
        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

            - compute_on_cpu: If metric state should be stored on CPU during computations. Only works
                for list states.
            - dist_sync_on_step: If metric state should synchronize on ``forward()``. Default is ``False``
            - process_group: The process group on which the synchronization is called. Default is the world.
            - dist_sync_fn: function that performs the allgather option on the metric state. Default is an
                custom implementation that calls ``jittor.distributed.all_gather`` internally.
            - distributed_available_fn: function that checks if the distributed backend is available.
                Defaults to a check of ``jittor.distributed.is_available()`` and ``jittor.distributed.is_initialized()``.
            - sync_on_compute: If metric state should synchronize when ``compute`` is called. Default is ``True``-
    """

    __jit_ignored_attributes__ = ["device"]
    __jit_unused_properties__ = ["is_differentiable"]
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None
    full_state_update: Optional[bool] = None

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        self.compute_on_cpu = kwargs.pop("compute_on_cpu", False)
        if not isinstance(self.compute_on_cpu, bool):
            raise ValueError(
                f"Expected keyword argument `compute_on_cpu` to be an `bool` but got {self.compute_on_cpu}"
            )

        self.dist_sync_on_step = kwargs.pop("dist_sync_on_step", False)
        if not isinstance(self.dist_sync_on_step, bool):
            raise ValueError(
                f"Expected keyword argument `dist_sync_on_step` to be an `bool` but got {self.dist_sync_on_step}"
            )

        self.process_group = kwargs.pop("process_group", None)

        self.dist_sync_fn = kwargs.pop("dist_sync_fn", None)
        if self.dist_sync_fn is not None and not callable(self.dist_sync_fn):
            raise ValueError(
                f"Expected keyword argument `dist_sync_fn` to be an callable function but got {self.dist_sync_fn}"
            )

        self.distributed_available_fn = kwargs.pop("distributed_available_fn", jit_distributed_available)

        self.sync_on_compute = kwargs.pop("sync_on_compute", True)
        if not isinstance(self.sync_on_compute, bool):
            raise ValueError(
                f"Expected keyword argument `sync_on_compute` to be a `bool` but got {self.sync_on_compute}"
            )

        # initialize
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore
        self._computed = None
        self._forward_cache = None
        self._update_count = 0
        self._to_sync = self.sync_on_compute
        self._should_unsync = True
        self._enable_grad = False
        self._dtype_convert = False

        # initialize state
        self._defaults: Dict[str, Union[List, Var]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        # state management
        self._is_synced = False
        self._cache: Optional[Dict[str, Union[List[Var], Var]]] = None

    @property
    def _update_called(self) -> bool:
        # Needed for lightning integration
        return self._update_count > 0

    def add_state(
        self,
        name: str,
        default: Union[list, Var],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """Adds metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a :class:`~jittor.Var` or an empty list.
                The state will be reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state across multiple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will use `` jittor.sum``,
                `` jittor.mean``, `` jittor.concat``, ``jittor.min`` and ``jittor.max``` respectively, each with argument
                ``dim=0``. Note that the ``"cat"`` reduction only makes sense if the state is a list, and not
                a tensor. The user can also pass a custom function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.
                Default is ``False``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is :class:`~jittor.Var`, the synced value will be a stacked :class:`~jittor.Var`
              across the process dimension if the metric state was a :class:`~jittor.Var`. The original
              :class:`~jittor.Var` metric state retains dimension and hence the synchronized output will be of shape
              ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.
            ValueError:
                If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``None``.
        """
        if not isinstance(default, (Var, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]")

        if isinstance(default, Var):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """``forward`` serves the dual purpose of both computing the metric on the current batch of inputs but also
        add the batch statistics to the overall accumululating metric state.

        Input arguments are the exact same as corresponding ``update`` method. The returned output is the exact same as
        the output of ``compute``.
        """
        # check if states are already synced
        if self._is_synced:
            raise Exception(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_cache = self._forward_full_state_update(*args, **kwargs)
        else:
            self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)

        return self._forward_cache

    def _forward_full_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using two calls to `update` to calculate the metric value on the current batch and
        accumulate global state.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.

        This can be done when the global metric state is a sinple reduction of batch states.
        """
        # store global state and reset to default
        global_state = {attr: getattr(self, attr) for attr in self._defaults.keys()}
        _update_count = self._update_count
        self.reset()

        # local syncronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # reduce batch and global state
        self._update_count = _update_count + 1
        with jittor.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _reduce_states(self, incoming_state: Dict[str, Any]) -> None:
        """Adds an incoming metric state to the current state of the metric.

        Args:
            incoming_state: a dict containing a metric state similar metric itself
        """
        for attr in self._defaults.keys():
            local_state = getattr(self, attr)
            global_state = incoming_state[attr]
            reduce_fn = self._reductions[attr]
            if reduce_fn == dim_zero_sum:
                reduced = global_state + local_state
            elif reduce_fn == dim_zero_mean:
                reduced = ((self._update_count - 1) * global_state + local_state).float() / self._update_count
            elif reduce_fn == dim_zero_max:
                reduced = jittor.max(global_state, local_state)
            elif reduce_fn == dim_zero_min:
                reduced = jittor.min(global_state, local_state)
            elif reduce_fn == dim_zero_cat:
                reduced = global_state + local_state
            elif reduce_fn is None and isinstance(global_state, Var):
                reduced = jittor.stack([global_state, local_state])
            elif reduce_fn is None and isinstance(global_state, list):
                reduced = _flatten([global_state, local_state])
            else:
                reduced = reduce_fn(jittor.stack([global_state, local_state]))  # type: ignore

            setattr(self, attr, reduced)

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        
        # TODO：gather函数缺失导致仅支持单卡推理，
        output_dict = apply_to_collection(
            input_dict,
            Var,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)

            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            if isinstance(output_dict[attr][0], Var):
                output_dict[attr] = jittor.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            if self._enable_grad:
                with jittor.enable_grad():
                    try:
                        update(*args, **kwargs)
                    except RuntimeError as err:
                        if "Expected all tensors to be on" in str(err):
                            raise RuntimeError(
                                "Encountered different devices in metric calculation (see stacktrace for details)."
                                " This could be due to the metric class not being on the same device as input."
                                f" Instead of `metric={self.__class__.__name__}(...)` try to do"
                                f" `metric={self.__class__.__name__}(...).to(device)` where"
                                " device corresponds to the device of the input."
                            ) from err
                        raise err
            else:
                with jittor.no_grad():
                    try:
                        update(*args, **kwargs)
                    except RuntimeError as err:
                        if "Expected all tensors to be on" in str(err):
                            raise RuntimeError(
                                "Encountered different devices in metric calculation (see stacktrace for details)."
                                " This could be due to the metric class not being on the same device as input."
                                f" Instead of `metric={self.__class__.__name__}(...)` try to do"
                                f" `metric={self.__class__.__name__}(...).to(device)` where"
                                " device corresponds to the device of the input."
                            ) from err
                        raise err

            if self.compute_on_cpu:
                self._move_list_states_to_cpu()

        return wrapped_func

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults.keys():
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def sync(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> None:
        """Sync function for manually controlling when metrics states should be synced across processes.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        if self._is_synced and should_sync:
            raise Exception("The Metric has already been synced.")

        if distributed_available is None and self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        is_distributed = distributed_available() if callable(distributed_available) else None

        if not should_sync or not is_distributed:
            return

        if dist_sync_fn is None:
            dist_sync_fn = gather_all_tensors

        # cache prior to syncing
        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

        # sync
        self._sync_dist(dist_sync_fn, process_group=process_group)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        """Unsync function for manually controlling when metrics states should be reverted back to their local
        states.

        Args:
            should_unsync: Whether to perform unsync
        """
        if not should_unsync:
            return

        if not self._is_synced:
            raise Exception("The Metric has already been un-synced.")

        if self._cache is None:
            raise Exception("The internal cache should exist to unsync the Metric.")

        # if we synced, restore to cache so that we can continue to accumulate un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> Generator:
        """Context manager to synchronize the states between processes when running in a distributed setting and
        restore the local cache states after yielding.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            # compute relies on the sync context manager to gather the states across processes and apply reduction
            # if synchronization happened, the current rank accumulated states will be restored to keep
            # accumulation going if ``should_unsync=True``,
            with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
            ):
                value = compute(*args, **kwargs)
                self._computed = _squeeze_if_scalar(value)

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value from state variables synchronized across the
        distributed backend."""

    def reset(self) -> None:
        """This method automatically resets the metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Var):
                setattr(self, attr, default.detach().clone())
            else:
                setattr(self, attr, [])

        # reset internal states
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Make a copy of the metric."""
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute", "_update_signature"]}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("higher_is_better", "is_differentiable", "full_state_update"):
            raise RuntimeError(f"Can't change const `{name}`.")
        super().__setattr__(name, value)


    def type(self, dst_type) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def float(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def double(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def half(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def set_dtype(self, dst_type) -> "Metric":
        """Special version of `type` for transferring all metric states to specific dtype
        Arguments:
            dst_type (type or string): the desired type
        """
        self._dtype_convert = True
        out = super().type(dst_type)
        out._dtype_convert = False
        return out

    def _apply(self, fn: Callable) -> Module:
        """Overwrite _apply function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is garded and will only happen through the special `set_dtype` method.
        """
        this = super()._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if isinstance(value, Var):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Var):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor" f"or a list of Tensor, but encountered {current_val}"
                )

        # make sure to update the device attribute
        # if the dummy tensor moves device by fn function we should also update the attribute

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Var, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Var, fn)

        return this

    def persistent(self, mode: bool = False) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(
        self,
        destination: Dict[str, Any] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Optional[Dict[str, Any]]:
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if not self._persistent[key]:
                continue
            current_val = getattr(self, key)
            if not keep_vars:
                if isinstance(current_val, Var):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [cur_v.detach() if isinstance(cur_v, Var) else cur_v for cur_v in current_val]
            destination[prefix + key] = deepcopy(current_val)  # type: ignore
        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Loads metric states from state_dict."""

        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the update signature of the metric."""

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }

        exists_var_keyword = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in _sign_params.values())
        # if no kwargs filtered, return all kwargs as default
        if not filtered_kwargs and not exists_var_keyword:
            # no kwargs in update signature -> don't return any kwargs
            filtered_kwargs = {}
        elif exists_var_keyword:
            # kwargs found in update signature -> return all kwargs to be sure to not omit any.
            # filtering logic is likely implemented within the update call.
            filtered_kwargs = kwargs
        return filtered_kwargs

    def __hash__(self) -> int:
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Var):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.add, self, other)

    def __and__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.bitwise_and, self, other)

    # Fixme: this shall return bool instead of Metric
    def __eq__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(jittor.all_equal, self, other)

    def __floordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.floor_divide, self, other)

    def __ge__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.greater_equal, self, other)

    def __gt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.greater, self, other)

    def __le__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.less_equal, self, other)

    def __lt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.less, self, other)

    def __matmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.matmul, self, other)

    def __mod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(fmod, self, other)

    def __mul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.multiply, self, other)

    # Fixme: this shall return bool instead of Metric
    def __ne__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(jittor.not_equal, self, other)

    def __or__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.bitwise_or, self, other)

    def __pow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.pow, self, other)

    def __radd__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.add, other, self)

    def __rand__(self, other: "Metric") -> "Metric":
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(jittor.bitwise_and, self, other)

    def __rfloordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.floor_divide, other, self)

    def __rmatmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.matmul, other, self)

    def __rmod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(fmod, other, self)

    def __rmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.mul, other, self)

    def __ror__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.bitwise_or, other, self)

    def __rpow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.pow, other, self)

    def __rsub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.subtract, other, self)

    def __rtruediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.divide, other, self)

    def __rxor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.bitwise_xor, other, self)

    def __sub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.subtract, self, other)

    def __truediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.divide, self, other)

    def __xor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(jittor.bitwise_xor, self, other)

    def __abs__(self) -> "Metric":
        return CompositionalMetric(jittor.abs, self, None)

    def __inv__(self) -> "Metric":
        return CompositionalMetric(jittor.bitwise_not, self, None)

    def __invert__(self) -> "Metric":
        return self.__inv__()

    def __neg__(self) -> "Metric":
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> "Metric":
        return CompositionalMetric(jittor.abs, self, None)

    def __getitem__(self, idx: int) -> "Metric":
        return CompositionalMetric(lambda x: x[idx], self, None)

    def __getnewargs__(self) -> Tuple:
        return (Metric.__str__(self),)

    def __iter__(self):
        raise NotImplementedError("Metrics does not support iteration.")
    

class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute."""

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, Var],
        metric_b: Union[Metric, int, float, Var, None],
    ) -> None:
        """
        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, Var):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Var):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:
        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        val_a = (
            self.metric_a(*args, **self.metric_a._filter_kwargs(**kwargs))
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )
        val_b = (
            self.metric_b(*args, **self.metric_b._filter_kwargs(**kwargs))
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )

        if val_a is None:
            return None

        if val_b is None:
            if isinstance(self.metric_b, Metric):
                return None

            # Unary op
            return self.op(val_a)

        # Binary op
        return self.op(val_a, val_b)

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        repr_str = self.__class__.__name__ + _op_metrics

        return repr_str

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute


class MetricCollection(Module):
    """modified from lightning torchmetrics.MetricCollection
    """

    _groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        super().__init__()

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *additional_metrics)

    def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Iteratively call forward for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        res = {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Iteratively call update for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            for _, cg in self._groups.items():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            if self._state_is_copy:
                # If we have deep copied state inbetween updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False
        else:  # the first update always do per metric to form compute groups
            for _, m in self.items(keep_base=True, copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterates over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one. The complexity of the method is approximately
        ``O(number_of_metrics_in_collection ** 2)``, as all metrics need to be compared to all other metrics.
        """
        n_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups) != n_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups) == n_groups:
                break
            else:
                n_groups = len(self._groups)

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults.keys():
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):
                return False

            if isinstance(state1, Var) and isinstance(state2, Var):
                return state1.shape == state2.shape and allclose(state1, state2)

            if isinstance(state1, list) and isinstance(state2, list):
                return all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2))

        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference
        """
        if not self._state_is_copy:
            for _, cg in self._groups.items():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        # Determine if we just should set a reference or a full copy
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                    setattr(mi, "_update_count", deepcopy(m0._update_count) if copy else m0._update_count)
        self._state_is_copy = copy

    def compute(self) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        for _, m in self.items(keep_base=True, copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            # reset state reference
            self._compute_groups_create_state_ref()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection":
        """Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for _, m in self.items(keep_base=True, copy_state=False):
            m.persistent(mode)

    def add_metrics(
        self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]], *additional_metrics: Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                (metrics if isinstance(m, Metric) else remain).append(m)

            if remain:
                rank_zero_warn(
                    f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored."
                )
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            
            # 后续可以起一个名字
            
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[k] = v
        else:
            raise ValueError("Unknown input to MetricCollection.")

        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group
        """
        if isinstance(self._enable_compute_groups, list):
            self._groups = {i: k for i, k in enumerate(self._enable_compute_groups)}
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                            f" Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}"
                        )
            self._groups_checked = True
        else:
            # Initialize all metrics as their own compute group
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> Dict[int, List[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

    def _to_renamed_ordered_dict(self) -> OrderedDict:
        od = OrderedDict()
        for k, v in self._modules.items():
            od[self._set_name(k)] = v
        return od

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.

        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_ordered_dict().keys()

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.

        Args:
            keep_base: Whether to add prefix/postfix on the collection.
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Module]:
        """Return an iterable of the ModuleDict values.

        Args:
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Module:
        """Retrieve a single metric from the collection.

        Args:
            key: name of metric to retrieve
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules[key]

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"

    def set_dtype(self, dst_type) -> "MetricCollection":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type (type or string): the desired type.
        """
        for _, m in self.items(keep_base=True, copy_state=False):
            m.set_dtype(dst_type)
        return self
