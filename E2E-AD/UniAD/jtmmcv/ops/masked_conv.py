import jittor.nn as nn
import jittor as jt
import math
from jittor.misc import _pair
from .global_header import proj_path


def addmm(beta, input, alpha, mat1, mat2):
    return beta * input + alpha * jt.matmul(mat1, mat2)



def masked_im2col_forward_api(features, mask_h_idx, mask_w_idx, data_col, 
                              kernel_h, kernel_w, pad_h, pad_w):
    
    channels, height, width = features.shape
    mask_cnt = mask_h_idx.shape[0]
    output_size = mask_cnt * channels
    
    data_col = jt.array(data_col).to(jt.float32).contiguous().to('cuda')
    features = jt.array(features).to(jt.float32).contiguous().to('cuda')
    mask_h_idx = jt.array(mask_h_idx).contiguous().to('cuda').to(jt.int64)
    mask_w_idx = jt.array(mask_w_idx).contiguous().to('cuda').to(jt.int64)

    
    out = jt.code(
                shape=(1, ), 
                dtype='float32', 
                inputs=[features, mask_h_idx, mask_w_idx, data_col], 
                cuda_header=
                '''
                #include <stdio.h>
                #include"masked_conv.cuh"
                using namespace std;
                ''',
                cuda_src=
                f'''
                @alias(features,in0);
                @alias(mask_h_idx,in1);
                @alias(mask_w_idx,in2);
                @alias(data_col,in3);
                
                const float *features_ptr = features_p;
                const long long *mask_h_idx_ptr = mask_h_idx_p;
                const long long *mask_w_idx_ptr = mask_w_idx_p;
                float *data_col_ptr = data_col_p;
                
                int output_size = {output_size};
                const int height = {height};
                const int width = {width};
                const int kernel_h = {kernel_h};
                const int kernel_w = {kernel_w};
                const int pad_h = {pad_h};
                const int pad_w = {pad_w};
                int mask_cnt = {mask_cnt};
                
                
                MaskedIm2colForward<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0>>>(
                output_size, features_ptr, height, width, kernel_h, kernel_w,
                pad_h, pad_w, mask_h_idx_ptr, mask_w_idx_ptr, mask_cnt, data_col_ptr);
                
                ''',
                )
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    assert out == 0, 'Error in executing MaskedIm2colForward'
    
    return data_col

    
def masked_col2im_forward_api(masked_output, mask_h_idx, mask_w_idx, output,
                              height, width, channels):
    
    mask_cnt = mask_h_idx.shape[0]
    output_size = mask_cnt * channels
    
    masked_output = jt.array(masked_output).to(jt.float32).contiguous().to('cuda')
    mask_h_idx = jt.array(mask_h_idx).contiguous().to('cuda').to(jt.int64)
    mask_w_idx = jt.array(mask_w_idx).contiguous().to('cuda').to(jt.int64)
    output = jt.array(output).to(jt.float32).contiguous().to('cuda')
    
    out = jt.code(
                shape=(1, ), 
                dtype='float32', 
                inputs=[masked_output, mask_h_idx, mask_w_idx, output], 
                cuda_header=
                '''
                #include <stdio.h>
                #include"masked_conv.cuh"
                using namespace std;
                ''',
                cuda_src=
                f'''
                @alias(masked_output,in0);
                @alias(mask_h_idx,in1);
                @alias(mask_w_idx,in2);
                @alias(output,in3);
                
                const float *masked_output_ptr = masked_output_p;
                const long long *mask_h_idx_ptr = mask_h_idx_p;
                const long long *mask_w_idx_ptr = mask_w_idx_p;
                float *output_ptr = output_p;
                
                const int output_size = {output_size};
                const int height = {height};
                const int width = {width};
                const int channels = {channels};
                int mask_cnt = {mask_cnt};
                
                MaskedCol2imForward<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0>>>(
                output_size, masked_output_ptr, height, width, channels, mask_h_idx_ptr,
                mask_w_idx_ptr, mask_cnt, output_ptr);
                '''
        
    )
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    print(out)
    
    pass
    


class MaskedConv2dFunction(jt.Function):

    @staticmethod
    def execute(features, mask, weight, bias, padding=0, stride=1):
        assert mask.dim() == 3 and mask.size(0) == 1
        assert features.dim() == 4 and features.size(0) == 1
        assert features.size()[2:] == mask.size()[1:]
        pad_h, pad_w = _pair(padding)
        stride_h, stride_w = _pair(stride)
        if stride_h != 1 or stride_w != 1:
            raise ValueError(
                'Stride could not only be 1 in masked_conv2d currently.')
        out_channel, in_channel, kernel_h, kernel_w = weight.size()

        batch_size = features.size(0)
        out_h = int(
            math.floor((features.size(2) + 2 * pad_h -
                        (kernel_h - 1) - 1) / stride_h + 1))
        out_w = int(
            math.floor((features.size(3) + 2 * pad_w -
                        (kernel_h - 1) - 1) / stride_w + 1))
        mask_inds = jt.nonzero(mask[0] > 0)
        output = features.new_zeros(batch_size, out_channel, out_h, out_w)
        if mask_inds.numel() > 0:
            mask_h_idx = mask_inds[:, 0].contiguous()
            mask_w_idx = mask_inds[:, 1].contiguous()
            data_col = features.new_zeros(in_channel * kernel_h * kernel_w,
                                          mask_inds.size(0))
            masked_im2col_forward_api(
                features,
                mask_h_idx,
                mask_w_idx,
                data_col,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                pad_h=pad_h,
                pad_w=pad_w)

            masked_output = addmm(1, bias[:, None], 1,
                                        weight.view(out_channel, -1), data_col)
            masked_col2im_forward_api(
                masked_output,
                mask_h_idx,
                mask_w_idx,
                output,
                height=out_h,
                width=out_w,
                channels=out_channel)
        return output

    @staticmethod
    def grad(self, grad_output):
        return (None, ) * 5


masked_conv2d = MaskedConv2dFunction.execute


class MaskedConv2d(nn.Conv):
    """A MaskedConv2d which inherits the official Conv2d.

    The masked forward doesn't implement the backward function and only
    supports the stride parameter to be 1 currently.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(MaskedConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

    def execute(self, input, mask=None):
        if mask is None:  # fallback to the normal Conv2d
            return super(MaskedConv2d, self).execute(input)
        else:
            return masked_conv2d(input, mask, self.weight, self.bias,
                                 self.padding)




def test_masked_forward():
    # Set random seed for reproducibility
    jt.flags.use_cuda = 1
    jt.set_global_seed(42)

    # Define input parameters
    batch_size = 1
    out_channels = 3
    height = 256
    width = 256
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1


    # Create input tensors
    masked_output = jt.randn((out_channels, 256,256))
    out_h = int(math.floor((masked_output.size()[1] + 2 * pad_h -(kernel_h - 1) - 1) / 1 + 1))
    out_w = int(math.floor((masked_output.size()[2] + 2 * pad_w -(kernel_h - 1) - 1) / 1 + 1))
    output = jt.zeros((batch_size, out_channels, out_h, out_w))
    mask = jt.randint(0, 2, [1, height, width], dtype=jt.int16)
    mask_inds = jt.nonzero(mask[0] > 0)
    mask_h_idx = mask_inds[:, 0].stop_grad()
    mask_w_idx = mask_inds[:, 1].stop_grad()
    data_col = jt.zeros((out_channels * 3 * 3, mask_inds.shape[0]))
    
    # Perform masked_col2im_forward
    masked_im2col_forward_api(masked_output, mask_h_idx, mask_w_idx, 
                              data_col, kernel_h, kernel_w, pad_h, pad_w)
    
    weight = jt.randn((3, 3, 3, 3))
    bias = jt.randn((3))
    
    masked_output = addmm(1, bias[:, None], 1, weight.view(out_channels, -1), data_col)
    
    masked_col2im_forward_api(masked_output, mask_h_idx, mask_w_idx, 
                              output, height, width, out_channels)
    
    print(output)
    # Check the output shape
    assert output.shape == (batch_size, out_channels, height, width), "Output shape mismatch"

    print("masked_col2im_forward test passed!")

if __name__ == "__main__":
    test_masked_forward()
