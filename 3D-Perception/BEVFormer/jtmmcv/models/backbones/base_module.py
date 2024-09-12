# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler

from jittor import Module
from jittor import nn as jnn


from jtmmcv.utils import master_only
from jtmmcv.utils.logging import get_logger, logger_initialized, print_log


def hashable_ndarray(arr):
    """
    将任意一个 NumPy ndarray 转换为可哈希的多维列表。
    get_substring_after_first_dot(name)
    参数:
        arr (numpy.ndarray): 输入的 NumPy ndarray。
        
    返回:
        hashable_list (tuple): 可哈希的多维元组形式的数组。
    """
    # 转成数值向量加速计算
    flat_arr = arr.ravel()
    hashable_list = tuple(flat_arr.tolist())
    
    return hashable_list
    


class BaseModule(Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``jittor.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``jittor.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.
        - ``_params_init_info``: Used to track the parameter
            initialization information. This attribute only
            exists during executing the ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `jittor.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        # is_top_level_module = False
        # # check if it is top-level module
        # if not hasattr(self, '_params_init_info'):
        #     # The `_params_init_info` is used to record the initialization
        #     # information of the parameters
        #     # the key should be the obj:`jnn.Parameter` of model and the value
        #     # should be a dict containing
        #     # - init_info (str): The string that describes the initialization.
        #     # - tmp_mean_value (FloatTensor): The mean of the parameter,
        #     #       which indicates whether the parameter has been modified.
        #     # this attribute would be deleted after all parameters
        #     # is initialized.
            
        #     self._params_init_info = defaultdict(dict)
        #     is_top_level_module = True

        #     # Initialize the `_params_init_info`,
        #     # When detecting the `tmp_mean_value` of
        #     # the corresponding parameter is changed, update related
        #     # initialization information
            
        #     for name, param in self.named_parameters():
        #         self._params_init_info[hashable_ndarray(param.data)][
        #             'init_info'] = f'The value is the same before and ' \
        #                            f'after calling `init_weights` ' \
        #                            f'of {self.__class__.__name__} '
        #         self._params_init_info[hashable_ndarray(param.data)][
        #             'tmp_mean_value'] = param.data.mean()


        #     # pass `params_init_info` to all submodules
        #     # All submodules share the same `params_init_info`,
        #     # so it will be updated when parameters are
        #     # modified at any level of the model.
        #     for sub_module in self.modules():
        #         sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'JAD'

        from ..utils import initialize
        from ..utils.weight_init import update_init_info  #需要调试
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    # update_init_info(
                    #     m,
                    #     init_info=f'Initialized by '
                    #     f'user-defined `init_weights`'
                    #     f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        # if is_top_level_module:
        #     self._dump_init_info(logger_name)

        #     for sub_module in self.modules():
        #         del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[name]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f'\n{name} - {param.shape}: '
                    f"\n{self._params_init_info[name]['init_info']} \n ",
                    logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class Sequential(BaseModule, jnn.Sequential):
    """Sequential module in openmmlab format.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        jnn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, jnn.ModuleList):
    """ModuleList in openmmlab format.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if modules is not None:
            jnn.ModuleList.__init__(self, modules)
        else :
            jnn.ModuleList.__init__(self)
