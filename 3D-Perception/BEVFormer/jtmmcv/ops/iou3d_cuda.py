import os
import jittor as jt
import numpy as np
from .global_header import proj_path

def boxes_overlap_bev_gpu(boxes_a, boxes_b, ans_overlap):
#     params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
#     params boxes_b: (M, 5)
#     params ans_overlap: (N, M)

    boxes_a = jt.array(boxes_a).float32().contiguous()
    boxes_b = jt.array(boxes_b).float32().contiguous()
    output = jt.zeros(boxes_a.shape[0], boxes_b.shape[0], dtype='float32')
    
    out = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[boxes_a, boxes_b, output], 
                cuda_header='''
                #include<cstdint>
                #include<vector>
                #include"iou3d_cuda.cuh"''',
                cuda_src='''
                int num_a = in0_shape0;
                int num_b = in1_shape0;
                
                const float *boxes_a_data = in0_p;
                const float *boxes_b_data = in1_p;
                float *ans_overlap_data = in2_p;
                
                boxesoverlapLauncher(num_a,boxes_a_data,num_b,boxes_b_data,ans_overlap_data);
                ''')
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    assert out == 0, 'Error in executing boxesoverlapLauncher'
    ans_overlap = output
    
    return ans_overlap


def nms_gpu(boxes, scores, threshold, pre_max_size=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (jittor.Var): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (jittor.Var): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        jittor.Var: Indexes after nms.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = jt.argsort(scores, descending=True)[0]
    if pre_max_size is not None:
        order = order[:pre_max_size]
        
    boxes = boxes[order].contiguous()
    keep = jt.zeros(boxes.size(0), dtype=jt.int64)
    num_out = jt.zeros((1), dtype=jt.int64)
    threshold = jt.array(threshold)

    out = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[boxes, keep, threshold, num_out], 
                cuda_header='''
                #include<cstdint>
                #include<vector>
                #include"iou3d_cuda.cuh" 
                
                ''',
                cuda_src='''
                int boxes_num = in0_shape0;
                const float *boxes_data = in0_p;
                int64_t *keep_data = new int64_t[boxes_num]; //zhoufangyuan
                
                const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

                float nms_overlap_thresh;
                CHECK_ERROR(cudaMemcpy(&nms_overlap_thresh, in2_p, sizeof(float), cudaMemcpyDeviceToHost));

                size_t mask_size = boxes_num * col_blocks * sizeof(unsigned long long);
                unsigned long long *mask_data = nullptr;
                CHECK_ERROR(cudaMalloc(&mask_data, mask_size));

                nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

                std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
                CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, mask_size, cudaMemcpyDeviceToHost));
                cudaFree(mask_data);

                std::vector<unsigned long long> remv_cpu(col_blocks, 0); 
                int num_to_keep = 0;

                for (int i = 0; i < boxes_num; i++) {
                    int nblock = i / THREADS_PER_BLOCK_NMS;
                    int inblock = i % THREADS_PER_BLOCK_NMS;
                    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
                        keep_data[num_to_keep++] = i;
                        unsigned long long *p = &mask_cpu[0] + i * col_blocks;
                        for (int j = nblock; j < col_blocks; j++) {
                            remv_cpu[j] |= p[j];
                        }
                    }
                }
                cudaMemcpy(in3_p, &num_to_keep, sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(in1_p, keep_data, boxes_num * sizeof(int64_t), cudaMemcpyHostToDevice);
                delete[] keep_data;
                
                ''')
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }

    assert out == 0, 'Error in executing nmsLauncher'
    keep = order[keep[:num_out]].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    
    return keep


def nms_normal_gpu(boxes, scores, threshold, pre_max_size=None, post_max_size=None):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (jittor.Var): Input boxes with shape (N, 5).
        scores (jittor.Var): Scores of predicted boxes with shape (N).
        thresh (jittor.Var): Threshold of non maximum suppression.

    Returns:
        jittor.Var: Remaining indices with scores in descending order.
    """
    order = jt.argsort(scores, descending=True)[0]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    
    boxes = boxes[order].contiguous()
    
    keep = jt.zeros(boxes.size(0), dtype=jt.int64)
    num_out = jt.zeros((1), dtype=jt.int64)
    threshold = jt.array(threshold)

    
    out = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[boxes, keep, threshold, num_out], 
                cuda_header='''
                #include<cstdint>
                #include<vector>
                #include"iou3d_cuda.cuh" 
                
                ''',
                cuda_src='''
                int boxes_num = in0_shape0;
                const float *boxes_data = in0_p;
                int64_t *keep_data = new int64_t[boxes_num]; //zhoufangyuan
                
                const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

                float nms_overlap_thresh;
                CHECK_ERROR(cudaMemcpy(&nms_overlap_thresh, in2_p, sizeof(float), cudaMemcpyDeviceToHost));

                size_t mask_size = boxes_num * col_blocks * sizeof(unsigned long long);
                unsigned long long *mask_data = nullptr;
                CHECK_ERROR(cudaMalloc(&mask_data, mask_size));

                nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

                std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
                CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, mask_size, cudaMemcpyDeviceToHost));
                cudaFree(mask_data);

                std::vector<unsigned long long> remv_cpu(col_blocks, 0); 
                int num_to_keep = 0;

                for (int i = 0; i < boxes_num; i++) {
                    int nblock = i / THREADS_PER_BLOCK_NMS;
                    int inblock = i % THREADS_PER_BLOCK_NMS;
                    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
                        keep_data[num_to_keep++] = i;
                        unsigned long long *p = &mask_cpu[0] + i * col_blocks;
                        for (int j = nblock; j < col_blocks; j++) {
                            remv_cpu[j] |= p[j];
                        }
                    }
                }
                cudaMemcpy(in3_p, &num_to_keep, sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(in1_p, keep_data, boxes_num * sizeof(int64_t), cudaMemcpyHostToDevice);
                delete[] keep_data;
                ''')
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    assert out == 0, 'Error in executing nmsLauncher'
    keep = order[keep[:num_out]].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep




def main():
    jt.flags.use_cuda = 1
    boxes1 = jt.array([
        [10, 10, 20, 20, 0],
        [12, 12, 20, 20, 0],
        [15, 15, 20, 20, 0],
        [30, 30, 20, 20, 0]
    ], dtype='float32')
    boxes2 = jt.array([
        [10, 10, 20, 20, 0],
        [15, 15, 25, 25, 0],
        [12, 14, 21, 24, 0]
    ], dtype='float32')
    
    output = jt.zeros(boxes1.shape[0], boxes2.shape[0], dtype='float32')
    
    
    scores = np.array([0.9, 0.85, 0.7, 0.95])

    thresh = 0.5
    # print(boxes_overlap_bev_gpu(boxes1, boxes2, output))
    print(nms_normal_gpu(boxes1, scores, thresh))
    pass

if __name__ == '__main__':
    main()