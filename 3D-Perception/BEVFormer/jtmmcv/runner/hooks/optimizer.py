# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

import jittor
from jittor.nn import BatchNorm
from jtmmcv.utils import LossScaler, wrap_fp16_model, digit_version, allreduce_grads
from .hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    # def clip_grads(self, params):
    #     params = list(
    #         filter(lambda p: p.requires_grad and p.grad is not None, params))
    #     if len(params) > 0:
    #         return clip_grad_norm(params)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.grad_clip is not None:
            runner.optimizer.clip_grad_norm(self.grad_clip['max_norm'])
            # if grad_norm is not None:
            #     # Add grad norm to the logger
            #     runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        #  runner.outputs['num_samples'])
        runner.optimizer.step(runner.outputs['loss'])
        jittor.sync_all()
        


@HOOKS.register_module()
class GradientCumulativeOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters=1, **kwargs):
        super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter

        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = runner.outputs['loss']
        loss = loss / loss_factor
        loss.grad()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()



@HOOKS.register_module()
class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook (mmcv's implementation).

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of LossScaler.
            Defaults to 512.
    """

    def __init__(self,
                    grad_clip=None,
                    coalesce=True,
                    bucket_size_mb=-1,
                    loss_scale=512.,
                    distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        if loss_scale == 'dynamic':
            self.loss_scaler = LossScaler(mode='dynamic')
        elif isinstance(loss_scale, float):
            self.loss_scaler = LossScaler(
                init_scale=loss_scale, mode='static')
        elif isinstance(loss_scale, dict):
            self.loss_scaler = LossScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                                f'"dynamic", got {loss_scale}')

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.
        """
        # keep a copy of fp32 weights
        old_groups = runner.optimizer.param_groups
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        state = defaultdict(dict)
        p_map = {
            old_p: p
            for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params']
                        for g in runner.optimizer.param_groups)))
        }
        for k, v in runner.optimizer.state.items():
            state[p_map[k]] = v
        runner.optimizer.state = state
        # convert model to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['loss_scaler']
            self.loss_scaler.load_state_dict(scaler_state_dict)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights,
                                            fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                            fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer `loss_scalar.py`

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        6. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
        scaled_loss.grad()
        # copy fp16 grads in the model to fp32 params in the optimizer

        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce,
                            self.bucket_size_mb)

        has_overflow = self.loss_scaler.has_overflow(fp32_weights)
        # if has overflow, skip this iteration
        if not has_overflow:
            # scale the gradients back
            for param in fp32_weights:
                if param.grad is not None:
                    param.grad.div_(self.loss_scaler.loss_scale)
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(fp32_weights)
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])
            # update fp32 params
            runner.optimizer.step()
            # copy fp32 params to the fp16 model
            self.copy_params_to_fp16(runner.model, fp32_weights)
        self.loss_scaler.update_scale(has_overflow)
        if has_overflow:
            runner.logger.warning('Check overflow, downscale loss scale '
                                    f'to {self.loss_scaler.cur_scale}')

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

@HOOKS.register_module()
class GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook,
                                            Fp16OptimizerHook):
    """Fp16 optimizer Hook (using mmcv implementation) implements multi-
    iters gradient cumulating."""

    def __init__(self, *args, **kwargs):
        super(GradientCumulativeFp16OptimizerHook,
                self).__init__(*args, **kwargs)

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters

        loss = runner.outputs['loss']
        loss = loss / loss_factor

        # scale the loss value
        scaled_loss = loss * self.loss_scaler.loss_scale
        scaled_loss.grad()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            # copy fp16 grads in the model to fp32 params in the optimizer
            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce,
                                self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                # update fp32 params
                runner.optimizer.step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            else:
                runner.logger.warning(
                    'Check overflow, downscale loss scale '
                    f'to {self.loss_scaler.cur_scale}')

            self.loss_scaler.update_scale(has_overflow)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

            # clear grads
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
