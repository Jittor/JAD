# Copyright (c) OpenMMLab. All rights reserved.
import numba
import numpy as np
import jittor

from jtmmcv.ops.iou3d_cuda import nms_gpu, nms_normal_gpu


def box3d_multiclass_nms(mlvl_bboxes,
                         mlvl_bboxes_for_nms,
                         mlvl_scores,
                         score_thr,
                         max_num,
                         cfg,
                         mlvl_dir_scores=None,
                         mlvl_attr_scores=None,
                         mlvl_bboxes2d=None):
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes (jittor.Var): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (jittor.Var): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores (jittor.Var): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score thredhold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (jittor.Var, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (jittor.Var, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (jittor.Var, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[jittor.Var]: Return results after nms, including 3D \
            bounding boxes, scores, labels, direction scores, attribute \
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=jittor.int32)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = jittor.concat(bboxes, dim=0)
        scores = jittor.concat(scores, dim=0)
        labels = jittor.concat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = jittor.concat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = jittor.concat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = jittor.concat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=jittor.int32)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results


def aligned_3d_nms(boxes, scores, classes, thresh):
    """3d nms for aligned boxes.

    Args:
        boxes (jittor.Var): Aligned box with shape [n, 6].
        scores (jittor.Var): Scores of each box.
        classes (jittor.Var): Class of each box.
        thresh (float): Iou threshold for nms.

    Returns:
        jittor.Var: Indices of selected boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    zero = boxes.new_zeros(1, )

    score_sorted = jittor.argsort(scores)
    pick = []
    while (score_sorted.shape[0] != 0):
        last = score_sorted.shape[0]
        i = score_sorted[-1]
        pick.append(i)

        xx1 = jittor.max(x1[i], x1[score_sorted[:last - 1]])
        yy1 = jittor.max(y1[i], y1[score_sorted[:last - 1]])
        zz1 = jittor.max(z1[i], z1[score_sorted[:last - 1]])
        xx2 = jittor.min(x2[i], x2[score_sorted[:last - 1]])
        yy2 = jittor.min(y2[i], y2[score_sorted[:last - 1]])
        zz2 = jittor.min(z2[i], z2[score_sorted[:last - 1]])
        classes1 = classes[i]
        classes2 = classes[score_sorted[:last - 1]]
        inter_l = jittor.max(zero, xx2 - xx1)
        inter_w = jittor.max(zero, yy2 - yy1)
        inter_h = jittor.max(zero, zz2 - zz1)

        inter = inter_l * inter_w * inter_h
        iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
        iou = iou * (classes1 == classes2).float()
        score_sorted = score_sorted[jittor.nonzero(
            iou <= thresh).flatten()]

    indices = jittor.array(pick).to(dtype=boxes.dtype)
    return indices


@numba.jit(nopython=True)
def circle_nms(dets, thresh, post_max_size=83):
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Args:
        dets (jittor.Var): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83

    Returns:
        jittor.Var: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep[:post_max_size]
