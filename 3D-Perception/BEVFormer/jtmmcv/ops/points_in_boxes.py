import jittor as jt
import numpy as np
from .global_header import proj_path

def points_in_boxes_batch(points, boxes):
    """Find points that are in boxes (CUDA) 该计算将在gpu运行

    Args:
        points (jittor.Var): [B, M, 3], [x, y, z] in LiDAR coordinate
        boxes (jittor.Var): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, w, l, h, ry] in LiDAR coordinate,
            (x, y, z) is the bottom center.

    Returns:
        box_idxs_of_pts (jittor.Var): (B, M, T), default background = 0
    """
    assert boxes.shape[0] == points.shape[0], \
        f'Points and boxes should have the same batch size, ' \
        f'got {boxes.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        f'boxes dimension should be 7, ' \
        f'got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        f'points dimension should be 3, ' \
        f'got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]
    points = jt.array(points).to('cuda').contiguous().float32()
    boxes = jt.array(boxes).to('cuda').contiguous().float32()

    box_idxs_of_pts = points.new_zeros((batch_size, num_points, num_boxes)).fill_(0).int32()
    
    out = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[boxes, points, box_idxs_of_pts], 
                cuda_header='''
                #include"points_in_boxes_cuda.cuh" 
                ''',
                cuda_src='''
                int batch_size = in0_shape0;
                int boxes_num = in0_shape1;
                int pts_num = in1_shape1;
                
                const float *boxes = in0_p;
                const float *pts = in1_p;
                int *box_idx_of_points = in2_p;
                
                points_in_boxes_batch_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                                 box_idx_of_points);
                ''')
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }

    assert out == 0, 'Error in executing points_in_boxes_batch_launcher'

    return box_idxs_of_pts




def points_in_boxes_gpu(points, boxes):
    """
    """
    assert boxes.shape[0] == points.shape[0], \
        f'Points and boxes should have the same batch size, ' \
        f'got {boxes.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        f'boxes dimension should be 7, ' \
        f'got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        f'points dimension should be 3, ' \
        f'got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape

    points = jt.array(points).to('cuda').contiguous().float32()
    boxes = jt.array(boxes).to('cuda').contiguous().float32()

    box_idxs_of_pts = points.new_zeros((batch_size, num_points)).fill_(-1).int32()
    
    out = jt.code(
                shape=(1, ), 
                dtype='int', 
                inputs=[boxes, points, box_idxs_of_pts], 
                cuda_header='''
                #include"points_in_boxes_cuda.cuh" 
                ''',
                cuda_src='''
                int batch_size = in0_shape0;
                int boxes_num = in0_shape1;
                int pts_num = in1_shape1;
                
                const float *boxes = in0_p;
                const float *pts = in1_p;
                int *box_idx_of_points = in2_p;
                
                points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                           box_idx_of_points);
                ''')
    out.compile_options = {
        f"FLAGS: -I{proj_path}": 1
    }
    if out == 0:
        raise RuntimeError('Error in executing points_in_boxes_launcher')
    
    return box_idxs_of_pts





def main():
    # 设置随机种子以便复现结果
    np.random.seed(42)
    B = 3
    # 点的数量
    M = 100
    # 框的数量
    T = 4
    # 生成随机点 [B, M, 3]
    points = np.random.rand(B, M, 3) * 10  # 生成范围在 [0, 10) 的点
    # 生成框 [B, T, 7]
    # 假设框的中心在 (5, 5, 5)，尺寸为 (2, 4, 6)，旋转角度为 0
    boxes = np.zeros((B, T, 7))
    for b in range(B):
        for t in range(T):
            boxes[b, t, :3] = np.random.rand(3) * 10  # 随机中心位置
            boxes[b, t, 3:6] = np.random.rand(3) * 3 + 1  # 随机尺寸，范围在 [1, 4)
            boxes[b, t, 6] = np.random.rand() * 2 * np.pi  # 随机旋转角度，范围在 [0, 2π)

    # 转换为 jittor.Var
    points_var = jt.array(points)
    boxes_var = jt.array(boxes)

    print(points_var.shape, boxes_var.shape)
    print(points_in_boxes_gpu(points_var, boxes_var))
    
    
    
if __name__ == '__main__':
    main()