#include <cmath>
// #include <cstdio>
#include <climits>
#include<stdio.h> 
#include<cstdint>
#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

using namespace std;

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}


__device__ float bilinear_interpolate(const float* bottom_data,
    const int height, const int width,
    float y, float x,
    const int index /* index for debug only*/) 
    {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        //empty
        return 0;
    }
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;
    int y_low = (int) y;
    int x_low = (int) x;
    int y_high;
    int x_high;
    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float) y_low;
    } else {
        y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float) x_low;
    } else {
        x_high = x_low + 1;
    }
    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = bottom_data[y_low * width + x_low];
    float v2 = bottom_data[y_low * width + x_high];
    float v3 = bottom_data[y_high * width + x_low];
    float v4 = bottom_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}


__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    float y, float x,
    float & w1, float & w2, float & w3, float & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) 
    {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        //empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;
    y_low = (int) y;
    x_low = (int) x;
    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float) y_low;
    } else {
        y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float) x_low;
    } else {
        x_high = x_low + 1;
    }
    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // reference in forward
    // float v1 = bottom_data[y_low * width + x_low];
    // float v2 = bottom_data[y_low * width + x_high];
    // float v3 = bottom_data[y_high * width + x_low];
    // float v4 = bottom_data[y_high * width + x_high];
    // float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return;
}



__global__ void MaskedIm2colForward(const int n, const float *data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const long long *mask_h_idx,
    const long long *mask_w_idx,
    const int mask_cnt, float *data_col) {
//mask_cnt * channels
CUDA_1D_KERNEL_LOOP(index, n) {
const int m_index = index % mask_cnt;
const int h_col = mask_h_idx[m_index];
const int w_col = mask_w_idx[m_index];
const int c_im = index / mask_cnt;
const int c_col = c_im * kernel_h * kernel_w;
const int h_offset = h_col - pad_h;
const int w_offset = w_col - pad_w;
float *data_col_ptr = data_col + c_col * mask_cnt + m_index;
for (int i = 0; i < kernel_h; ++i) {
int h_im = h_offset + i;
for (int j = 0; j < kernel_w; ++j) {
int w_im = w_offset + j;
if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
*data_col_ptr =
(float)data_im[(c_im * height + h_im) * width + w_im];
} else {
*data_col_ptr = 0.0;
}
data_col_ptr += mask_cnt;
}
}
}
}

__global__ void MaskedCol2imForward(const int n, const float *data_col,
    const int height, const int width,
    const int channels,
    const long long *mask_h_idx,
    const long long *mask_w_idx,
    const int mask_cnt, float *data_im) {
CUDA_1D_KERNEL_LOOP(index, n) {
const int m_index = index % mask_cnt;
const int h_im = mask_h_idx[m_index];
const int w_im = mask_w_idx[m_index];
const int c_im = index / mask_cnt;
// compute the start and end of the output
data_im[(c_im * height + h_im) * width + w_im] = data_col[index];
}
}
