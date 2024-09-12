# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi 
# All Rights Reserved


import jittor as jt
import jittor.nn as nn
from jittor import Function

jt.flags.use_cuda = 1


class KNNBatch(Function):
    # @staticmethod
    def execute(self, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        assert k <= m

        idx = jt.code((n, k),
                dtype=jt.int32,
                inputs=[xyz, query_xyz, batch_idxs, query_batch_offsets],
                cuda_header='''
                // input xyz: (n, 3), float
                // input query_xyz: (m, 3), float
                // input batch_idxs: (n), int
                // input query_batch_offsets: (B + 1), int, offsets[-1] = m
                // output idx: (n, k), int
                #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/knn/src/knn_gpu.cu>
                #define THREADS_PER_BLOCK 256
                #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))   
                ''',
                cuda_src='''
                n = in0_shape0;
                m = in1_shape0;
                k = out_shape1;
                
                const float* xyz = in0_p;
                const float* query_xyz = in1_p;
                const int* batch_idxs = in2_p;
                const int* query_batch_offsets = in3_p;
                int* idx = out_p;      
                cudaMemset(idx, 0, n * k * sizeof(int));
                //cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                
                dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                knn_batch_cuda_<<<blocks, threads>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
                '''
                )

        return idx

    #@staticmethod
    def grad(self, a=None):
        return None, None, None, None, None
    

knn_batch = KNNBatch.apply


class KNNBatchMlogK(Function):
    # @staticmethod
    def execute(self, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        assert k <= m
        assert k <= 128

        idx = jt.code((n, k),
                dtype=batch_idxs.dtype,
                inputs=[xyz, query_xyz, batch_idxs, query_batch_offsets],
                cuda_header='''
                // input xyz: (n, 3), float
                // input query_xyz: (m, 3), float
                // input batch_idxs: (n), int
                // input query_batch_offsets: (B + 1), int, offsets[-1] = m
                // output idx: (n, k), int
                #include </data/chenguanyu/MTR-jittor/mtr_jittor/ops/knn/src/knn_gpu.cu>
                #define THREADS_PER_BLOCK 256
                #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))   
                ''',
                cuda_src='''
                int n, m, k;
                n = in0_shape0;
                m = in1_shape0;
                k = out_shape1;
                
                const float* xyz = in0_p;
                const float* query_xyz = in1_p;
                const int* batch_idxs = in2_p;
                const int* query_batch_offsets = in3_p;
                int* idx = out_p;
                cudaMemset(idx, 0, n * k * sizeof(int));
                //cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                
                dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                knn_batch_mlogk_cuda_<<<blocks, threads>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
                '''
                )
        return idx

    # @staticmethod
    def grad(self, a=None):
        return None, None, None, None, None
   
knn_batch_mlogk = KNNBatchMlogK.apply 
