import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jittor as jt
import numpy as np
from .global_header import proj_path

def boxes_iou_bev(boxes_a, boxes_b):
    num_a = boxes_a.shape[0]
    num_b = boxes_b.shape[0]
    num_a = jt.array(num_a)
    boxes_a = jt.array(boxes_a).float32()
    num_b = jt.array(num_b)
    boxes_b = jt.array(boxes_b).float32()

    output = jt.zeros(boxes_a.shape[0], boxes_b.shape[0], dtype='float32')
    out = jt.code(
                shape=(1, ), 
                dtype='float32', 
                inputs=[num_a, boxes_a, num_b, boxes_b, output], 
                cuda_header='''
#include<iostream>
#include"iou3d.cuh"
using namespace std;
''',
                cuda_src='''
__global__ void iou3d_boxes_iou_bev_forward_cuda_kernel(@ARGS_DEF) {
    
  @PRECALC
  int num_a = @in0(0);
  int num_b = @in2(0);
  float* boxes_a = &@in1(0, 0);
  float* boxes_b = &@in3(0, 0);
  float* ans_iou = &@in4(0, 0);                                                      
                                                          
  const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
  const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (a_idx >= num_a || b_idx >= num_b) {
    return;
  }

  const float *cur_box_a = boxes_a + a_idx * 5;
  const float *cur_box_b = boxes_b + b_idx * 5;
  float cur_iou_bev = iou_bev(cur_box_a, cur_box_b);
  ans_iou[a_idx * num_b + b_idx] = cur_iou_bev;
}
  int num_a = in1_shape0;
  int num_b = in3_shape0;

dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK_IOU3D),
              DIVUP(num_a, THREADS_PER_BLOCK_IOU3D));
dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

iou3d_boxes_iou_bev_forward_cuda_kernel<<<blocks, threads, 0>>>(@ARGS);
'''
    )
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    assert out == 0, 'Error in executing iou3d_boxes_iou_bev_forward_cuda_kernel'
    return output


def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None):

    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    # order = scores.sort(0, descending=True)[1]
    order = jt.argsort(scores, descending=True)[0]

    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()

    keep = jt.zeros(boxes.size(0), dtype=jt.int64)
    num_out = jt.zeros((1), dtype=jt.int64)

    nms_overlap_thresh = jt.array(thresh)
    iou3d_nms_forward = jt.code(
        shape=(1, ), 
        dtype='int',
        inputs=[boxes, keep, num_out, nms_overlap_thresh], 
        cuda_header='''
#include<iostream>
#include<typeinfo>
#include<vector>
#include"iou3d.cuh"
''',
        cuda_src='''
int boxes_num = in0_shape0;
  //int64_t *keep_data = &@in1(0);
  int64_t *keep_data = new int64_t[boxes_num];
  //jittor::int64 *keep_num_data = &@in2(0);
  //float nms_overlap_thresh = @in3(0);
  float nms_overlap_thresh;
  cudaMemcpy(&nms_overlap_thresh, in3_p, sizeof(float), cudaMemcpyDeviceToHost);

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  // Tensor mask =
  //     at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  // unsigned long long *mask_data =
  //     (unsigned long long *)mask.data_ptr<int64_t>();
  size_t mask_size = boxes_num * col_blocks * sizeof(unsigned long long);
  unsigned long long *mask_data = nullptr;
  cudaMalloc(&mask_data, mask_size);
  if (mask_data == nullptr) {
    printf("iou3d_nms_forward failed to allocate mask_data\\n");
    return;
  }
  IoU3DNMSForwardCUDAKernelLauncher(in0_p, mask_data, boxes_num, nms_overlap_thresh);

  // at::Tensor mask_cpu = mask.to(at::kCPU);
  // unsigned long long *mask_host =
  //     (unsigned long long *)mask_cpu.data_ptr<int64_t>();

  unsigned long long *mask_host = new unsigned long long[mask_size];
  cudaMemcpy(mask_host, mask_data, mask_size, cudaMemcpyDeviceToHost);

  std::vector<unsigned long long> remv_cpu(col_blocks);
  memset(&remv_cpu[0], 0, sizeof(unsigned long long) * col_blocks);

  int64_t num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
    //*keep_num_data = num_to_keep;
  }
  cudaMemcpy(in2_p, &num_to_keep, sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(in1_p, keep_data, boxes_num * sizeof(int64_t), cudaMemcpyHostToDevice);

  delete[] mask_host;
  delete[] keep_data;
  cudaFree(mask_data);
        '''
    )
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    if iou3d_nms_forward != 0:
        print("Error in executing nms_bev")

    num_out = num_out.data[0]
    selected_indices = keep[:num_out].int()
    selected_indices = jt.array(selected_indices.data)
    keep = order[selected_indices]

    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_bev(boxes, scores, thresh):
    assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'
    order = jt.argsort(scores, descending=True)[0]

    boxes = boxes[order].contiguous()

    keep = jt.zeros(boxes.size(0), dtype=jt.int64)
    num_out = jt.zeros((1), dtype=jt.int64)
    nms_overlap_thresh = jt.array(thresh)
    iou3d_nms_forward = jt.code(
        shape=(1, ), 
        dtype='int',
        inputs=[boxes, keep, num_out, nms_overlap_thresh], 
        cuda_header='''
#include<iostream>
#include<typeinfo>
#include<vector>
#include"ops/csrc/iou3d.cuh"
''',
        cuda_src='''
  // int boxes_num = boxes.size(0);
  int boxes_num = in0_shape0;
  // int64_t *keep_data = keep.data_ptr<int64_t>();
  int64_t *keep_data = new int64_t[boxes_num];
//  int64_t *keep_num_data = keep_num.data_ptr<int64_t>();


  float nms_overlap_thresh;
  cudaMemcpy(&nms_overlap_thresh, in3_p, sizeof(float), cudaMemcpyDeviceToHost);

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  // Tensor mask =
  //     at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  // unsigned long long *mask_data =
  //     (unsigned long long *)mask.data_ptr<int64_t>();
  size_t mask_size = boxes_num * col_blocks * sizeof(unsigned long long);
  unsigned long long *mask_data = nullptr;
  cudaMalloc(&mask_data, mask_size);
  if (mask_data == nullptr) {
    printf("iou3d_nms_forward failed to allocate mask_data\\n");
    return;
  }

  IoU3DNMSNormalForwardCUDAKernelLauncher(in0_p, mask_data, boxes_num,
                                nms_overlap_thresh);

  // at::Tensor mask_cpu = mask.to(at::kCPU);
  // unsigned long long *mask_host =
  //     (unsigned long long *)mask_cpu.data_ptr<int64_t>();

  unsigned long long *mask_host = new unsigned long long[mask_size];
  cudaMemcpy(mask_host, mask_data, mask_size, cudaMemcpyDeviceToHost);

  std::vector<unsigned long long> remv_cpu(col_blocks);
  memset(&remv_cpu[0], 0, sizeof(unsigned long long) * col_blocks);
  int64_t num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  // *keep_num_data = num_to_keep;
  cudaMemcpy(in2_p, &num_to_keep, sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(in1_p, keep_data, boxes_num * sizeof(int64_t), cudaMemcpyHostToDevice);

  delete[] mask_host;
  delete[] keep_data;
  cudaFree(mask_data);
        '''
    )
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    print(iou3d_nms_forward)
    if iou3d_nms_forward != 0:
        print("Error")
    return order[keep[:num_out]].contiguous()


def main():
    jt.flags.use_cuda = 1
    boxes1 = jt.array([
        [10, 10, 20, 20, 0],
        [12, 12, 20, 20, 0],
        [15, 15, 20, 20, 0],
        [30, 30, 20, 20, 0]
    ], dtype='float32')
    boxes2 = jt.array([
        [13, 13, 23, 23, 0],
        [15, 15, 25, 25, 0],
        [12, 14, 21, 24, 0]
    ], dtype='float32')

    scores = np.array([0.9, 0.85, 0.7, 0.95])

    thresh = 0.5
    print(nms_bev(boxes1, scores, thresh))
    # a = boxes_iou_bev(boxes1, boxes2)
    # print(a)
    # print(boxes_iou_bev(boxes, scores, thresh))
    pass

if __name__ == '__main__':
    main()