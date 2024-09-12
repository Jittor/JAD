from jittor import Function
import jittor as jt
import numpy as np

from spconv.utils import Point2VoxelCPU3d as VoxelGenerator


tv = None
try:
    import cumm.tensorview as tv
except:
    pass

''' 本voxelize方法调用了spconv库(hard voxelize),在UniAD未经过实际测试, 请使用参考mmcv源代码进行一些整定
    或依照文档编写对应cuda实现
'''


class Voxelization(Function):

    @staticmethod
    def execute(
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000):
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (jt.Var): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int, optional): maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Default: 35.
            max_voxels (int, optional): maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
                Default: 20000.

        Returns:
            voxels_out (jt.Var): Output voxels with the shape of [M,
                max_points, ndim]. Only contain points and returned when
                max_points != -1.
            coors_out (jt.Var): Output coordinates with the shape of
                [M, 3].

            num_point_features: 点的特征数量
            num_points_per_voxel_out (jt.Var): Num points per voxel with
                the shape of [M]. Only returned when max_points != -1.
        """

        voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=coors_range,
            num_point_features=4,
            max_num_voxels=max_points,
            max_num_points_per_voxel=max_voxels
        )
        points = points.numpy()
        voxel_output = voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        
        voxels = jt.array(voxels).to('cuda')
        coordinates = jt.array(coordinates).to('cuda')
        num_points = jt.array(num_points).to('cuda')
        
        return voxels, coordinates, num_points
       


voxelization = Voxelization.apply





def main():
    # 模拟点云数据
    np.random.seed(42)
    jt.flags.use_cuda = 1
    num_points = 1000
    ndim = 4  # 假设每个点包括3个坐标和1个额外信息
    points = jt.array(np.random.rand(num_points, ndim).astype(np.float32))

    # 体素化参数
    vsize_xyz = [0.05, 0.05, 0.1]  # Assuming a placeholder value for the fourth float
    coors_range_xyz = [0, -40, -3, 70.4, 40, 1]  # Placeholder values for the seventh and eighth floats
    num_point_features = 4  # Example value, adjust based on your actual data
    max_num_voxels = 1000  # Example value, adjust as necessary
    max_num_points_per_voxel = 5  # Example value, adjust as necessary
    # 运行体素化函数
    voxels, coordinates, num_points = Voxelization.execute(
     points, vsize_xyz, coors_range_xyz, max_num_voxels, max_num_points_per_voxel
    )

    # 打印输出结果
    print("Voxels Output:", voxels, coordinates, num_points)
    # print("Coordinates Output:", coors_out)
    # print("Number of Points per Voxel:", num_points_per_voxel_out)

if __name__ == "__main__":
    main()






























