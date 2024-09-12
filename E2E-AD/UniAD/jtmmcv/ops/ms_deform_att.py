import jittor as jt
import numpy as np
from jittor import nn
import warnings
import math

from jtmmcv.utils.misc import deprecated_api_warning
from jtmmcv.models.utils.weight_init import constant_init, xavier_init
from jtmmcv.models.bricks.registry import ATTENTION
from jtmmcv.models.backbones.base_module import BaseModule
jt.flags.use_cuda = 1
from .global_header import proj_path


def ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step):
    
    "jittor.float32 是应该使用的数据类型，对于大部分浮点数来说"
    
    
    # value = value.contiguous()
    # spatial_shapes = spatial_shapes.contiguous()
    # level_start_index = level_start_index.contiguous()
    # sampling_loc = sampling_loc.contiguous()
    # attn_weight = attn_weight.contiguous()

    # value = value.to('cuda')
    # spatial_shapes = spatial_shapes.to('cuda').to(jt.int64)
    # level_start_index = level_start_index.to('cuda').to(jt.int64)
    # sampling_loc = sampling_loc.to('cuda')
    # attn_weight = attn_weight.to('cuda')
    
    
        
    value = value.contiguous()
    spatial_shapes = spatial_shapes.contiguous()
    level_start_index = level_start_index.contiguous()
    sampling_loc = sampling_loc.contiguous()
    attn_weight = attn_weight.contiguous()

    value = value.to('cuda')
    spatial_shapes = spatial_shapes.to('cuda').to(jt.int64)
    level_start_index = level_start_index.to('cuda').to(jt.int64)
    sampling_loc = sampling_loc.to('cuda')
    attn_weight = attn_weight.to('cuda')
    
    
    batch = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]
    b = num_heads * channels
    
    num_levels = spatial_shapes.shape[0]
    
    num_query = sampling_loc.shape[1]
    num_q = num_query
    num_point = sampling_loc.shape[4]

    
    im2col_step_ = min(batch, im2col_step)
    
    assert batch % im2col_step_ == 0, f"batch({batch}) must divide im2col_step({im2col_step})"
    
    output = jt.zeros([batch, num_query, num_heads, channels]).to('cuda').to(jt.float32)
    batch_n = im2col_step_

    output_n = output.view(batch // im2col_step_, batch_n, num_query, num_heads, channels)
    
    im2col_step_ = jt.array(im2col_step_)
    per_value_size = spatial_size * num_heads * channels
    per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2
    per_attn_weight_size = num_query * num_heads * num_levels * num_point

    
    value = value.to('cuda').to(jt.float).contiguous()
    spatial_shapes = spatial_shapes.to('cuda').to(jt.int64).contiguous()
    level_start_index = level_start_index.to('cuda').to(jt.int64).contiguous()
    sampling_loc = sampling_loc.to('cuda').to(jt.float).contiguous()
    attn_weight = attn_weight.to('cuda').to(jt.float).contiguous()
    batch_n = jt.array(batch_n).to('cuda').contiguous()
    spatial_size = jt.array(spatial_size).to('cuda').contiguous()
    num_heads = jt.array(num_heads).to('cuda').contiguous()
    channels = jt.array(channels).to('cuda').contiguous()
    num_levels = jt.array(num_levels).to('cuda').contiguous()
    num_query = jt.array(num_query).to('cuda').contiguous()
    num_point = jt.array(num_point).to('cuda').contiguous()
    
    
    for n in range(batch // im2col_step_):
        columns = output_n[n].contiguous()

        n = jt.array(n).contiguous()
        im2col_step_ = jt.array(im2col_step_).contiguous()
        per_value_size = jt.array(per_value_size).contiguous()
        per_sample_loc_size = jt.array(per_sample_loc_size).contiguous()
        per_attn_weight_size = jt.array(per_attn_weight_size).contiguous()
        
        
        got = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[value, 
                        spatial_shapes, 
                        level_start_index, 
                        sampling_loc, n, im2col_step_, per_value_size,
                        per_sample_loc_size,per_attn_weight_size,
                        attn_weight, columns, batch_n, spatial_size, 
                        num_heads, channels, num_levels, num_query, num_point], 
                cuda_header=
                '''
                #include"ms_deform_att.cuh"
                using namespace std;
                ''',
                cuda_src=
                f'''
                @alias(value,in0);
                @alias(spatial_shapes,in1);
                @alias(level_start_index,in2);
                @alias(sampling_loc,in3);
                @alias(n_gpu,in4);
                @alias(im2col_step__gpu,in5);
                @alias(per_value_size_gpu,in6);
                @alias(per_sample_loc_size_gpu,in7);
                @alias(per_attn_weight_size_gpu,in8);
                @alias(attn_weight,in9);
                @alias(columns,in10);
                @alias(batch_n_gpu,in11);
                @alias(spatial_size_gpu,in12);
                @alias(num_heads_gpu,in13);
                @alias(channels_gpu,in14);
                @alias(num_levels_gpu,in15);
                @alias(num_query_gpu,in16);
                @alias(num_point_gpu,in17);
                
                int batch_n;
                int spatial_size;
                int num_heads;
                int channels;
                int num_levels;
                int num_query;
                int num_point;
                cudaMemcpy(&batch_n, batch_n_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&spatial_size, spatial_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&num_heads, num_heads_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&channels, channels_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&num_levels, num_levels_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&num_query, num_query_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&num_point, num_point_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                
                int n;
                int im2col_step_;
                int per_value_size;
                int per_sample_loc_size;
                int per_attn_weight_size;
                cudaMemcpy(&n, n_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&im2col_step_, im2col_step__gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&per_value_size, per_value_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&per_sample_loc_size, per_sample_loc_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&per_attn_weight_size, per_attn_weight_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            
                float *columns_ptr = columns_p;
                
                ms_deformable_im2col_cuda<float>(
                                          (float*)value_p + n * im2col_step_ * per_value_size,
                                          (int64_t*)spatial_shapes_p, 
                                          (int64_t*)level_start_index_p,
                                          (float*)sampling_loc_p + n * im2col_step_ * per_sample_loc_size,
                                          (float*)attn_weight_p + n * im2col_step_ * per_attn_weight_size,
                                          batch_n,
                                          spatial_size, 
                                          num_heads, channels, num_levels, num_query, num_point, 
                                          columns_ptr);
         
                '''
                )
        got.compile_options = {
            f"FLAGS: -I{proj_path}": 1
        }
        if got != 0:
            ...
        else:
            ...        
    

    return output.reshape(batch, num_q, b)


def ms_deform_attn_backward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, grad_value, grad_sampling_loc, grad_attn_weight, im2col_step):
    
    value = value.contiguous()
    spatial_shapes = spatial_shapes.contiguous()
    level_start_index = level_start_index.contiguous()
    sampling_loc = sampling_loc.contiguous()
    attn_weight = attn_weight.contiguous()
    grad_output = grad_output.contiguous()

    value = value.to('cuda')
    spatial_shapes = spatial_shapes.to('cuda').to(jt.int64)
    level_start_index = level_start_index.to('cuda').to(jt.int64)
    sampling_loc = sampling_loc.to('cuda')
    attn_weight = attn_weight.to('cuda')
    grad_output = grad_output.to('cuda')

    batch = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]
    
    num_levels = spatial_shapes.shape[0]
    
    num_query = sampling_loc.shape[1]
    num_point = sampling_loc.shape[4]

    im2col_step_ = min(batch, im2col_step)
    
    assert batch % im2col_step_ == 0, f"batch({batch}) must divide im2col_step({im2col_step})"

    batch_n = im2col_step_

    per_value_size = spatial_size * num_heads * channels
    per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2
    per_attn_weight_size = num_query * num_heads * num_levels * num_point
    grad_output_n = grad_output.view(batch // im2col_step, batch_n, num_query, num_heads, channels)

    
    batch_n = jt.array(batch_n)
    spatial_size = jt.array(spatial_size)
    num_heads = jt.array(num_heads)
    channels = jt.array(channels)
    num_levels = jt.array(num_levels)
    num_query = jt.array(num_query)
    num_point = jt.array(num_point)
    im2col_step_ = jt.array(im2col_step_)
    per_value_size = jt.array(per_value_size)
    per_sample_loc_size = jt.array(per_sample_loc_size)
    per_attn_weight_size = jt.array(per_attn_weight_size)
    spatial_shapes = spatial_shapes.to(jt.int64)
    level_start_index = level_start_index.to(jt.int64)

    for n in range(0, batch // im2col_step_):
        grad_output_g = grad_output_n[n]
        n_gpu = jt.array(n)

        out = jt.code(
            shape=(1, ),
            dtype='int',
            inputs=[grad_output_g,
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_loc,
                    attn_weight,
                    batch_n,
                    spatial_size,
                    num_heads,
                    channels,
                    num_levels,
                    num_query,
                    num_point,
                    grad_value,
                    grad_sampling_loc,
                    grad_attn_weight,
                    n_gpu,
                    im2col_step_,
                    per_value_size,
                    per_sample_loc_size,
                    per_attn_weight_size
                    ],
            cuda_header=
                '''
                #include"ms_deform_att.cuh"
                using namespace std;
                ''',
            cuda_src='''
            @alias(grad_output_g,in0);
            @alias(value,in1);
            @alias(spatial_shapes,in2);
            @alias(level_start_index,in3);
            @alias(sampling_loc,in4);
            @alias(attn_weight,in5);
            @alias(batch_n_gpu,in6);
            @alias(spatial_size_gpu,in7);
            @alias(num_heads_gpu,in8);
            @alias(channels_gpu,in9);
            @alias(num_levels_gpu,in10);
            @alias(num_query_gpu,in11);
            @alias(num_point_gpu,in12);
            @alias(grad_value,in13);
            @alias(grad_sampling_loc,in14);
            @alias(grad_attn_weight,in15);
            @alias(n_gpu,in16);
            @alias(im2col_step__gpu,in17);
            @alias(per_value_size_gpu,in18);
            @alias(per_sample_loc_size_gpu,in19);
            @alias(per_attn_weight_size_gpu,in20);

            int n;
            int im2col_step_;
            int per_value_size;
            int per_sample_loc_size;
            int per_attn_weight_size;
            cudaMemcpy(&n, n_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&im2col_step_, im2col_step__gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&per_value_size, per_value_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&per_sample_loc_size, per_sample_loc_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&per_attn_weight_size, per_attn_weight_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);

            int batch_n;
            int spatial_size;
            int num_heads;
            int channels;
            int num_levels;
            int num_query;
            int num_point;
            cudaMemcpy(&batch_n, batch_n_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&spatial_size, spatial_size_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_heads, num_heads_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&channels, channels_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_levels, num_levels_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_query, num_query_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_point, num_point_gpu_p, sizeof(int), cudaMemcpyDeviceToHost);

            ms_deformable_col2im_cuda<float> (
              (float*)grad_output_g_p,
              (float*)value_p + n * im2col_step_ * per_value_size,
              (int64_t*)spatial_shapes_p,
              (int64_t*)level_start_index_p,
              (float*)sampling_loc_p +
                  n * im2col_step_ * per_sample_loc_size,
              (float*)attn_weight_p +
                  n * im2col_step_ * per_attn_weight_size,
                batch_n,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
              (float*)grad_value_p +
                  n * im2col_step_ * per_value_size,
              (float*)grad_sampling_loc_p +
                  n * im2col_step_ * per_sample_loc_size,
              (float*)grad_attn_weight_p +
                  n * im2col_step_ * per_attn_weight_size
                  );

                int out_cpu = 0;
                cudaMemcpy(out_p, &out_cpu, sizeof(int), cudaMemcpyHostToDevice);
            ''')
        out.compile_options = {
            f"FLAGS: -I{proj_path}": 1
        }
        assert out == 0, 'Error in executing ms_deformable_col2im_cuda'


class MultiScaleDeformableAttnFunction(jt.Function):
        
    def save_for_backward(self, *Vars: jt.Var):
        r"""Save given Vars for a future call to :func:`~Function.grad`.

        ``save_for_backward`` should be called at most once, only from inside the
        :func:`forward` method, and only with tensors.

        All tensors intended to be used in the backward pass should be saved
        with ``save_for_backward`` (as opposed to directly on ``ctx``) to prevent
        incorrect gradients and memory leaks, and enable the application of saved
        tensor hooks. See :class:`jittor.autograd.graph.saved_tensors_hooks`.

        Note that if intermediary tensors, tensors that are neither inputs
        nor outputs of :func:`forward`, are saved for backward, your custom Function
        may not support double backward.
        Custom Functions that do not support double backward should decorate their
        :func:`backward` method with ``@once_differentiable`` so that performing
        double backward raises an error. If you'd like to support double backward,
        you can either recompute intermediaries based on the inputs during backward
        or return the intermediaries as the outputs of the custom Function. See the
        `double backward tutorial <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_
        for more details.

        In :func:`backward`, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``. This is a no-op.

        See :ref:`extending-autograd` for more details on how to use this method.

        Example::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def execute(ctx, x: jittor.Var, y: jittor.Var, z: int):
            >>>         w = x * z
            >>>         out = x * y + y * z + w * y
            >>>         ctx.save_for_backward(x, y, w, out)
            >>>         ctx.z = z  # z is not a tensor
            >>>         return out
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def grad(ctx, grad_out):
            >>>         x, y, w, out = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         gx = grad_out * (y + y * z)
            >>>         gy = grad_out * (x + z + w)
            >>>         gz = None
            >>>         return gx, gy, gz
            >>>
            >>> a = jittor.Var(1., requires_grad=True, dtype=jittor.double)
            >>> b = jittor.Var(2., requires_grad=True, dtype=jittor.double)
            >>> c = 4
            >>> d = Func.apply(a, b, c)

        """
        self.to_save = Vars
    
    def execute(self, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        self.im2col_step = im2col_step
        output = ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=self.im2col_step)
        self.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    def grad(self, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = self.to_save
        grad_value = jt.zeros_like(value)
        grad_sampling_loc = jt.zeros_like(sampling_locations)
        grad_attn_weight = jt.zeros_like(attention_weights)

        ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=self.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_jittor(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        if isinstance(H_, jt.Var):
            H_ = int(H_.data[0])
            W_ = int(W_.data[0])
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = jt.nn.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (jt.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


@ATTENTION.register_module()
class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.0,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        self.msda = MultiScaleDeformableAttnFunction()
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = jt.arange(
            self.num_heads,
            dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().argmax(-1, keepdims=True)[1]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def execute(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().data) == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = jt.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = self.msda(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)
        # else:
        #     output = multi_scale_deformable_attn_jittor(
        #         value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

 

    