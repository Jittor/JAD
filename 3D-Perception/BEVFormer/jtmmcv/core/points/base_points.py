# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import jittor
import warnings
from abc import abstractmethod


class BasePoints(object):
    """Base class for Points.

    Args:
        tensor (jittor.Var | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (jittor.Var): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        tensor = jittor.array(tensor, dtype=jittor.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, points_dim)).to(
                dtype=jittor.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == \
            points_dim, tensor.size()

        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self):
        """jittor.Var: Coordinates of each point with size (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for jittor.Var and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, jittor.Var):
            tensor = jittor.array(tensor).to(dtype=self.tensor.dtype)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        """jittor.Var: A vector with height of each point."""
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for jittor.Var and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, jittor.Var):
            tensor = jittor.array(tensor).to(dtype=self.tensor.dtype)
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = jittor.concat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        """jittor.Var: A vector with color of each point."""
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for jittor.Var and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, jittor.Var):
            tensor = jittor.array(tensor).to(dtype=self.tensor.dtype)
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = jittor.concat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self):
        """jittor.shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            jittor.Var: The shuffled index.
        """
        idx = jittor.randperm(self.__len__())
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, jittor.Var): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, jittor.Var):
            rotation = jittor.array(rotation).to(dtype=self.tensor.dtype)
        assert rotation.shape == [3, 3] or \
            rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = jittor.sin(rotation)
            rot_cos = jittor.cos(rotation)
            if axis == 1:
                rot_mat_T = jittor.array([[rot_cos, 0, -rot_sin],
                                                 [0, 1, 0],
                                                 [rot_sin, 0, rot_cos]]).to(dtype=rotation.dtype)
            elif axis == 2 or axis == -1:
                rot_mat_T = jittor.array([[rot_cos, -rot_sin, 0],
                                                 [rot_sin, rot_cos, 0],
                                                 [0, 0, 1]]).to(dtype=rotation.dtype)
            elif axis == 0:
                rot_mat_T = jittor.array([[0, rot_cos, -rot_sin],
                                                 [0, rot_sin, rot_cos],
                                                 [1, 0, 0]]).to(dtype=rotation.dtype)
            else:
                raise ValueError('axis should in range')
            rot_mat_T = rot_mat_T.transpose()
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the points in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, jittor.Var): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, jittor.Var):
            trans_vector = jittor.array(trans_vector).to(dtype=self.tensor.dtype)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                f'Unsupported translation vector of shape {trans_vector.shape}'
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | jittor.Var): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            jittor.Var: A binary vector indicating whether each point is \
                inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | jittor.Var): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            jittor.Var: Indicating whether each point is inside \
                the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Box mode.
            rt_mat (np.ndarray | jittor.Var): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a boolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            4. `new_points = points[3:11, vector]`:
                return a slice of points and attribute dims.
            5. `new_points = points[4:12, 2]`:
                return a slice of points with single attribute.
            Note that the returned Points might share storage with this Points,
            subject to jittor's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of  \
                :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if \
                    item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, jittor.Var)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')

        assert p.dim() == 2, \
            f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(
            p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (list[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(jittor.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use jittor.concat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            jittor.concat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].tensor.shape[1],
            attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device (str | :obj:`jittor.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new boxes object on the \
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def clone(self):
        """Clone the Points.

        Returns:
            :obj:`BasePoints`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)


    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            jittor.Var: A point of shape (4,).
        """
        yield from self.tensor

    def new_point(self, data):
        """Create a new point object with data.

        The new point and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (jittor.Var | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        new_tensor = jittor.array(data).to(dtype=self.tensor.dtype) \
            if not isinstance(data, jittor.Var) else data.to(self.tensor.dtype)
        original_type = type(self)
        return original_type(
            new_tensor,
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)
