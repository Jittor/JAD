import math
import jittor
from jittor.einops import rearrange, repeat

def bivariate_gaussian_activation(ip):
    """
    jittor 版本
    Activation function to output parameters of bivariate Gaussian distribution.

    Args:
        ip (jittor.Var): Input tensor.

    Returns:
        jittor.Var: Output tensor containing the parameters of the bivariate Gaussian distribution.
    """
    if ip.shape[-1]>2:
        mu_x = ip[..., 0:1]
        mu_y = ip[..., 1:2]
        sig_x = ip[..., 2:3]
        sig_y = ip[..., 3:4]
        rho = ip[..., 4:5]
        sig_x = jittor.exp(sig_x)
        sig_y = jittor.exp(sig_y)
        rho = jittor.tanh(rho)
        out = jittor.concat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    else:
        mu_x = ip[..., 0:1]
        mu_y = ip[..., 1:2]
        out = jittor.concat([mu_x, mu_y], dim=-1)
    return out

def norm_points(pos, pc_range):
    """
    Normalize the end points of a given position tensor.

    Args:
        pos (jittor.Var): Input position tensor.
        pc_range (List[float]): Point cloud range.

    Returns:
        jittor.Var: Normalized end points tensor.
    """
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) 
    return jittor.stack([x_norm, y_norm], dim=-1)

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (jittor.Var): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        jittor.Var: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = jittor.arange(end=num_pos_feats, dtype=jittor.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = jittor.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = jittor.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    posemb = jittor.concat((pos_y, pos_x), dim=-1)
    return posemb

def rot_2d(yaw):
    """
    Compute 2D rotation matrix for a given yaw angle tensor.

    Args:
        yaw (jittor.Var): Input yaw angle tensor.

    Returns:
        jittor.Var: 2D rotation matrix tensor.
    """
    sy, cy = jittor.sin(yaw), jittor.cos(yaw)
    out = jittor.stack([jittor.stack([cy, -sy]), jittor.stack([sy, cy])]).permute([2,0,1])
    return out

def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    """
    Transform anchor coordinates with respect to detected bounding boxes in the batch.

    Args:
        anchors (jittor.Var): A tensor containing the k-means anchor values.
        bbox_results (List[Tuple[jittor.Var]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        jittor.Var: A tensor containing the transformed anchor coordinates.
    """
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...] # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        if with_rotation_transform: 
            angle = yaw - 3.1415953 # num_agents, 1
            rot_yaw = rot_2d(angle) # num_agents, 2, 2
            rot_yaw = rot_yaw[:, None, None,:, :] # num_agents, 1, 1, 2, 2
            transformed_anchors = rearrange(transformed_anchors, 'b g m t c -> b g m c t')  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12
            rot_yaw = rot_yaw.broadcast([rot_yaw.shape[0], transformed_anchors.shape[1], 
                               transformed_anchors.shape[2], rot_yaw.shape[3], rot_yaw.shape[4]])
            transformed_anchors = transformed_anchors.broadcast([rot_yaw.shape[0], transformed_anchors.shape[1],
                transformed_anchors.shape[2], transformed_anchors.shape[3], transformed_anchors.shape[4]])
            transformed_anchors = jittor.matmul(rot_yaw, transformed_anchors)# -> num_agents, num_groups, num_modes, 12, 2
            transformed_anchors = rearrange(transformed_anchors, 'b g m c t -> b g m t c')
        if with_translation_transform:
            transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
        batched_anchors.append(transformed_anchors)
    return jittor.stack(batched_anchors)


def trajectory_coordinate_transform(trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    """
    Transform trajectory coordinates with respect to detected bounding boxes in the batch.
    Args:
        trajectory (jittor.Var): predicted trajectory.
        bbox_results (List[Tuple[jittor.Var]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        jittor.Var: A tensor containing the transformed trajectory coordinates.
    """
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        transformed_trajectory = trajectory[i,...]
        if with_rotation_transform:
            # we take negtive here, to reverse the trajectory back to ego centric coordinate
            angle = -(yaw - 3.1415953) 
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:,None, None,:, :] # A, 1, 1, 2, 2
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p t c -> a g p c t') # A, G, P, 12 ,2 -> # A, G, P, 2, 12
            rot_yaw = rot_yaw.broadcast([rot_yaw.shape[0], rot_yaw.shape[1], transformed_trajectory.shape[2], 
                               rot_yaw.shape[3], rot_yaw.shape[4]])
            transformed_trajectory = jittor.matmul(rot_yaw, transformed_trajectory)# -> A, G, P, 12, 2
            transformed_trajectory = rearrange(transformed_trajectory, 'a g p c t -> a g p t c')
        if with_translation_transform:
            transformed_trajectory = bbox_centers[:, None, None, None, :2] + transformed_trajectory
        batched_trajectories.append(transformed_trajectory)
    return jittor.stack(batched_trajectories)