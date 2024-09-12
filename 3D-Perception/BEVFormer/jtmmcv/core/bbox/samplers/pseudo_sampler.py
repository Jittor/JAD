import jittor

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (jittor.Var): Bounding boxes
            gt_bboxes (jittor.Var): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = jittor.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = jittor.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0])
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
