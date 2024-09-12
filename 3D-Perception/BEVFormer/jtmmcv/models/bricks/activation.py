from jittor import Module
from jittor import nn as jnn

from jtmmcv.utils import build_from_cfg, digit_version
from .registry import ACTIVATION_LAYERS

# jt.relu 注册

@ACTIVATION_LAYERS.register_module()
class ReLU(Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png
    """

    def __init__(self):
        super().__init__()

    def execute(self, input):
        return jnn.relu(input)


@ACTIVATION_LAYERS.register_module()
class GELU(Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png
    """

    def __init__(self):
        super().__init__()

    def execute(self, input):
        return jnn.gelu(input)



# for module in [
#         ReLU
# ]:
#     ACTIVATION_LAYERS.register_module(module=module)

# ACTIVATION_LAYERS.register_module(module=GELU)



def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
