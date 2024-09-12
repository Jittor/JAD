"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention/attention_utils_v2.py
"""

# import torch
# import torch.nn as nn
# from torch.autograd import Function, Variable
import jittor as jt
from jittor import nn
from jittor import Function, Var

# from . import attention_cuda


""" Attention computation code v1."""
class AttentionWeightComputation(Function):
    """
    Generate the attention weight matrix based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)
    Generate the attention weight matrix.
        * (total_query_num, local_size)
    """

    # @staticmethod
    def execute(self,
                query_batch_cnt: Var,
                key_batch_cnt: Var,
                index_pair_batch: Var,
                index_pair: Var,
                query_features: Var,
                key_features: Var):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim]
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, local_size, nhead]
        """

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == query_features.shape[0]

        output = jt.code(
            [total_query_num, local_size, nhead],
            dtype=query_features.dtype,
            inputs=[query_batch_cnt, key_batch_cnt, index_pair_batch,index_pair,
                query_features, key_features],
             cuda_header="""
             #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/attention/src/attention_weight_computation_kernel.cu>
             #define THREADS_PER_BLOCK 256
             #define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
                //@alias(query_batch_cnt,in0)
                //@alias(key_batch_cnt,in1)
                //@alias(index_pair_batch,in2)
                //@alias(index_pair,in3)
                //@alias(query_features,in4)
                //@alias(key_features,in5)
                //@alias(output,out)
                """,
            cuda_src="""
            // params query_batch_cnt: [b]
            // params key_batch_cnt: [b]
            // params index_pair_batch: [total_query_num]
            // params index_pair: [total_query_num, local_size]
            // params query_features: [total_query_num, nhead, hdim]
            // params key_features: [total_key_num, nhead, hdim]
            // params output: [total_query_num, local_size, nhead]
            const int* query_batch_cnt = in0_p;
            const int* key_batch_cnt = in1_p;
            const int* index_pair_batch = in2_p;
            const int* index_pair = in3_p;
            const float* query_features = in4_p;
            const float* key_features = in5_p;
            float* output = out_p;
            
            int b, total_query_num, local_size, total_key_num, nhead, hdim;
            b = in0_shape0;
            total_query_num = in3_shape0;
            local_size = in3_shape1;
            total_key_num = in5_shape0;
            nhead = in5_shape1;
            hdim = in5_shape2;
            
            //dim3 threads(THREADS_PER_BLOCK);
            //dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
            //attention_weight_computation_forward<<<blocks, threads>>>(
            //    b, total_query_num, local_size, total_key_num, nhead, hdim,
            //    query_batch_cnt, key_batch_cnt, index_pair_batch,
            //    index_pair, query_features, key_features,
            //    output);
            cudaMemset(output, 0, total_query_num * local_size * nhead * sizeof(float));
            attention_weight_computation_launcher(
            b, total_query_num, local_size,
            total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair,
            query_features, key_features,
            output);
            """)

        self.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, query_features, key_features
        )
        return output

    # @staticmethod
    def grad(self, grad_out: Var):
        """
        Args:
            ctx:
            grad_out: [total_query_num, local_size, nhead]
        Returns:
            grad_query_features:  [total_query_num, nhead, hdim]
            grad_key_features: [total_key_num, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, query_features, key_features) = self.for_backwards

        grad_query_features , grad_key_features = jt.code(
            [(total_query_num, nhead, hdim), (total_key_num, nhead, hdim)],
            [query_features.dtype, key_features.dtype],
            [query_batch_cnt, key_batch_cnt, index_pair_batch,index_pair,
                query_features, key_features, grad_out],
            cuda_header="""
            #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/attention/src/attention_weight_computation_kernel.cu>
            #define THREADS_PER_BLOCK 256
            #define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
            //@alias(query_batch_cnt,in0)
            //@alias(key_batch_cnt,in1)
            //@alias(index_pair_batch,in2)
            //@alias(index_pair,in3)
            //@alias(query_features,in4)
            //@alias(key_features,in5)
            //@alias(grad_out_data,in6)
            //@alias(grad_query_features,out0)
            //@alias(grad_key_features,out1)
            """,
            cuda_src="""
            // params query_batch_cnt: [b]
            // params key_batch_cnt: [b]
            // params index_pair_batch: [total_query_num]
            // params index_pair: [total_query_num, local_size]
            // params query_features: [total_query_num, nhead, hdim]
            // params key_features: [total_key_num, nhead, hdim]
            // params grad_out: [total_query_num, local_size, nhead]
            // params grad_query_features: [total_query_num, nhead, hdim]
            // params grad_key_features: [total_key_num, nhead, hdim]
            
            const int* query_batch_cnt = in0_p;
            const int* key_batch_cnt = in1_p;
            const int* index_pair_batch = in2_p;
            const int* index_pair = in3_p;
            const float* query_features = in4_p;
            const float* key_features = in5_p;
            float* grad_out = in6_p;
            float* grad_query_features = out0_p;
            float* grad_key_features = out1_p;
            
            int b, total_query_num, local_size, total_key_num, nhead, hdim;
            b = in0_shape0;
            total_query_num = in3_shape0;
            local_size = in3_shape1;
            total_key_num = in5_shape0;
            nhead = in5_shape1;
            hdim = in5_shape2;
            cudaMemset(grad_query_features, 0, total_query_num * nhead * hdim * sizeof(float));
            cudaMemset(grad_key_features, 0, total_key_num * nhead * hdim * sizeof(float));
            dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
            dim3 threads(THREADS_PER_BLOCK);
            attention_weight_computation_backward<<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            """,)



        return None, None, None, None, grad_query_features, grad_key_features


attention_weight_computation = AttentionWeightComputation.apply


class AttentionValueComputation(Function):
    """
    Generate the attention result based on:
        * the generated attention pair index (total_query_num, local_size);
        * value features (total_key_num, nhead, hdim)
        * attn_weight (total_query_num, local_size, nhead)
    Generate the attention result.
        * (total_query_num, nhead, hdim)
    """

    # @staticmethod
    def execute(self,
                query_batch_cnt: Var,
                key_batch_cnt: Var,
                index_pair_batch: Var,
                index_pair: Var,
                attn_weight: Var,
                value_features: Var):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, nhead, hdim]
        """

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == attn_weight.shape[0]

        # output = torch.cuda.FloatTensor(total_query_num, nhead, hdim).zero_()

        output = jt.code(
            (total_query_num, nhead, hdim),
            dtype=value_features.dtype,
            inputs=[query_batch_cnt, key_batch_cnt, index_pair_batch,index_pair,
                attn_weight, value_features],
            cuda_header="""
            #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/attention/src/attention_value_computation_kernel.cu>
            #define THREADS_PER_BLOCK 256
            #define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
            //@alias(query_batch_cnt,in0)
            //@alias(key_batch_cnt,in1)
            //@alias(index_pair_batch,in2)
            //@alias(index_pair,in3)
            //@alias(attn_weight,in4)
            //@alias(value_features,in5)
            //@alias(output,out)
            """,
            cuda_src="""
            // params query_batch_cnt: [b]
            // params key_batch_cnt: [b]
            // params index_pair_batch: [total_query_num]
            // params index_pair: [total_query_num, local_size]
            // params attn_weight: [total_query_num, local_size, nhead]
            // params value_features: [total_key_num, nhead, hdim]
            // params output: [total_query_num, nhead, hdim]
            const int* query_batch_cnt = in0_p;
            const int* key_batch_cnt = in1_p;
            const int* index_pair_batch = in2_p;
            const int* index_pair = in3_p;
            const float* attn_weight = in4_p;
            const float* value_features = in5_p;
            float* output = out_p;
            
            int b, total_query_num, local_size, total_key_num, nhead, hdim;
            b = in0_shape0;
            total_query_num = in3_shape0;
            local_size = in3_shape1;
            total_key_num = in5_shape0;
            nhead = in5_shape1;
            hdim = in5_shape2;
            
            dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
            dim3 threads(THREADS_PER_BLOCK);
            cudaMemset(output, 0, total_query_num * nhead * hdim * sizeof(float));
            attention_value_computation_forward<<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            """,)
            
        self.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features
        )
        return output

    # @staticmethod
    def grad(self, grad_out: Var):
        """
        Args:
            ctx:
            grad_out: [total_query_num, nhead, hdim]
        Returns:
            grad_attn_weight:  [total_query_num, local_size, nhead]
            grad_value_features: [total_key_num, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, attn_weight, value_features) = self.for_backwards

        grad_attn_weight, grad_value_features = jt.code(
            [(total_query_num, local_size, nhead), (total_key_num, nhead, hdim)],
            [attn_weight.dtype, value_features.dtype],
            [query_batch_cnt, key_batch_cnt, index_pair_batch,index_pair,
                attn_weight, value_features, grad_out],
            cuda_header="""
            #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/attention/src/attention_value_computation_kernel.cu>
            #define THREADS_PER_BLOCK 256
            #define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
            //@alias(query_batch_cnt,in0)
            //@alias(key_batch_cnt,in1)
            //@alias(index_pair_batch,in2)
            //@alias(index_pair,in3)
            //@alias(attn_weight,in4)
            //@alias(value_features,in5)
            //@alias(grad_out_data,in6)
            //@alias(grad_attn_weight,out0)
            //@alias(grad_value_features,out1)
            """,
            cuda_src="""
            // params query_batch_cnt: [b]
            // params key_batch_cnt: [b]
            // params index_pair_batch: [total_query_num]
            // params index_pair: [total_query_num, local_size]
            // params attn_weight: [total_query_num, local_size, nhead]
            // params value_features: [total_key_num, nhead, hdim]
            // params grad_out: [total_query_num, nhead, hdim]
            // params grad_attn_weight: [total_query_num, local_size, nhead]
            // params grad_value_features: [total_key_num, nhead, hdim]
            const int* query_batch_cnt = in0_p;
            const int* key_batch_cnt = in1_p;
            const int* index_pair_batch = in2_p;
            const int* index_pair = in3_p;
            const float* attn_weight = in4_p;
            const float* value_features = in5_p;
            float* grad_out = in6_p;
            float* grad_attn_weight = out0_p;
            float* grad_value_features = out1_p;
            
            int b, total_query_num, local_size, total_key_num, nhead, hdim;
            b = in0_shape0;
            total_query_num = in3_shape0;
            local_size = in3_shape1;
            total_key_num = in5_shape0;
            nhead = in5_shape1;
            hdim = in5_shape2;
            cudaMemset(grad_attn_weight, 0, total_query_num * local_size * nhead * sizeof(float));
            cudaMemset(grad_value_features, 0, total_key_num * nhead * hdim * sizeof(float));
            dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
            dim3 threads(THREADS_PER_BLOCK);
            attention_value_computation_backward<<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            """,)

        return None, None, None, None, grad_attn_weight, grad_value_features


attention_value_computation = AttentionValueComputation.apply