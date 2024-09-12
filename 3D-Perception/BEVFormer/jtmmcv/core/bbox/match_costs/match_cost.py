import jittor
from jittor import nn as jnn
from jtmmcv.utils.general import cdist_p1 as cdist
from jtmmcv.core.bbox.iou_calculators import bbox_overlaps
from jtmmcv.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .builder import MATCH_COST


@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from jtmmcv.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import jittor
         >>> self = BBoxL1Cost()
         >>> bbox_pred = jittor.rand(1, 4)
         >>> gt_bboxes= jittor.Var([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = jittor.Var([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            jittor.Var: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = cdist(bbox_pred, gt_bboxes)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from jtmmcv.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import jittor
         >>> self = FocalLossCost()
         >>> cls_pred = jittor.rand(4, 3)
         >>> gt_labels = jittor.Var([0, 1, 2])
         >>> factor = jittor.Var([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            jittor.Var: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from jtmmcv.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import jittor
         >>> self = ClassificationCost()
         >>> cls_pred = jittor.rand(4, 3)
         >>> gt_labels = jittor.Var([0, 1, 2])
         >>> factor = jittor.Var([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            jittor.Var: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from jtmmcv.core.bbox.match_costs.match_cost import IoUCost
         >>> import jittor
         >>> self = IoUCost()
         >>> bboxes = jittor.Var([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = jittor.Var([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            jittor.Var: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight
    
    
@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            jittor.Var: bbox_cost value with weight
        """
        bbox_cost = cdist(bbox_pred, gt_bboxes)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class DiceCost(object):
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from jtmmcv.core.bbox.match_costs.match_cost import IoUCost
         >>> import jittor
         >>> self = IoUCost()
         >>> bboxes = jittor.Var([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = jittor.Var([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, weight=1.):
        self.weight = weight
        self.count = 0

    def __call__(self, input, target):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            jittor.Var: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        # print('INPUT', input.shape)
        # print('target',target.shape)

        N1, H1, W1 = input.shape
        N2, H2, W2 = target.shape

        if H1 != H2 or W1 != W2:
            target = jnn.interpolate(target.unsqueeze(0), size=(H1, W1), mode='bilinear').squeeze(0)

        input = input.contiguous().view(N1, -1)[:, None, :]
        target = target.contiguous().view(N2, -1)[None, :, :]

        a = jittor.sum(input * target, -1)
        b = jittor.sum(input * input, -1) + 0.001
        c = jittor.sum(target * target, -1) + 0.001
        d = (2 * a) / (b + c)
        return (1 - d) * self.weight
