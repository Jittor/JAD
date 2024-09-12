"""
Copy-paste from jittor.nn.Transformer, timm, with modifications:
"""
import copy
from typing import Optional, List

import jittor
from jittor import nn as jnn
from jittor import Module


from functools import partial
from jtmmcv.models.utils.builder import TRANSFORMER
from jtmmcv.models.bricks.activation import GELU
from jtmmcv.utils import force_fp32

count = 0


class Mlp(Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=GELU,
                 drop=0.):
        super().__init__()
        self.fp16_enabled = False
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = jnn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = jnn.Linear(hidden_features, out_features)
        self.drop = jnn.Dropout(drop)

    @force_fp32(apply_to=('x', ))
    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(Module):
    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.fp16_enabled = False
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = jnn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = jnn.Dropout(attn_drop)
        self.proj = jnn.Linear(dim, dim)
        self.proj_drop = jnn.Dropout(proj_drop)

    @force_fp32(apply_to=('x', ))
    def execute(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1,
                                                               4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(Module):
    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = jnn.Linear(dim, dim, bias=qkv_bias)
        self.k = jnn.Linear(dim, dim, bias=qkv_bias)
        self.v = jnn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = jnn.Dropout(attn_drop)
        self.proj = jnn.Linear(dim, dim)
        self.proj_drop = jnn.Dropout(proj_drop)
        self.linear_l1 = jnn.Sequential(
            jnn.Linear(self.num_heads, self.num_heads),
            jnn.ReLU(),
        )
        self.linear = jnn.Sequential(
            jnn.Linear(self.num_heads, 1),
            jnn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jittor.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key', 'value'))
    def execute(self, query, key, value, key_padding_mask, hw_lvl):
        B, N, C = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()  #.permute(2, 0, 3, 1, 4)

        v = self.v(value).reshape(B, L,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)

        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        attn = attn.permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, mask

# AttentionTail is a cheap implementation that can make mask decoder 1 layer deeper.
class AttentionTail(Module): 
    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = jnn.Linear(dim, dim, bias=qkv_bias)
        self.k = jnn.Linear(dim, dim, bias=qkv_bias)

        self.linear_l1 = jnn.Sequential(
            jnn.Linear(self.num_heads, self.num_heads),
            jnn.ReLU(),
        )
        
        self.linear = jnn.Sequential(
            jnn.Linear(self.num_heads, 1),
            jnn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jittor.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key'))
    def execute(self, query, key, key_padding_mask, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()  #.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)
        
        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        return mask


class Block(Module):
    def __init__(self,
                 cfg,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=GELU,
                 norm_layer=jnn.LayerNorm,
                 self_attn=False):
        super().__init__()
        self.fp16_enabled = False
        self.head_norm1 = norm_layer(dim)
        self.self_attn = self_attn
        self.attn = Attention(cfg,
                              dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else jnn.Identity()
        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        if self.self_attn:
            self.self_attention = SelfAttention(cfg,
                                                dim,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                attn_drop=attn_drop,
                                                proj_drop=drop)
            self.norm3 = norm_layer(dim)

    @force_fp32(apply_to=('query', 'key', 'value'))
    def execute(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn:
            query = query + self.drop_path(self.self_attention(query))
            query = self.norm3(query)
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        query = query + self.drop_path(x)
        query = self.head_norm1(query)

        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)
        return query, mask


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-53296self.num_heads956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jittor.rand(
        shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @force_fp32(apply_to=('x', ))
    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)


@TRANSFORMER.register_module()
class SegMaskHead(Module):
    def __init__(self,
                 cfg=None,
                 d_model=16,
                 nhead=2,
                 num_encoder_layers=6,
                 num_decoder_layers=1,
                 dim_feedexecute=64,
                 dropout=0,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 self_attn=False):
        super().__init__()

        self.fp16_enabled = False
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        drop_rate = 0
        attn_drop_rate = 0

        norm_layer = None
        norm_layer = norm_layer or partial(jnn.LayerNorm, eps=1e-6)
        act_layer = None
        act_layer = act_layer or GELU
        block = Block(cfg,
                      dim=d_model,
                      num_heads=nhead,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=0,
                      norm_layer=norm_layer,
                      act_layer=act_layer,
                      self_attn=self_attn)
        
        self.blocks = jnn.ModuleList([
            Block(cfg,
                dim=d_model,
                num_heads=nhead,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                self_attn=self_attn)
            for _ in range(num_decoder_layers)     
        ])
            
        self.attnen = AttentionTail(cfg,
                                    d_model,
                                    num_heads=nhead,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop_rate,
                                    proj_drop=0)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jittor.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[jittor.Var]):
        if pos is None:
            return tensor
        else:
            return tensor + pos
        #return tensor if pos is None else tensor + pos
    @force_fp32(apply_to=('memory', 'mask_memory', 'pos_memory', 'query_embed',
                          'mask_query', 'pos_query'))
    def execute(self, memory, mask_memory, pos_memory, query_embed, mask_query,
                pos_query, hw_lvl):
        if mask_memory is not None and isinstance(mask_memory, jittor.Var):
            mask_memory = mask_memory.to(bool)
        masks = []
        inter_query = []
        for i, block in enumerate(self.blocks):
            query_embed, mask = block(self.with_pos_embed(
                query_embed, pos_query),
                                      self.with_pos_embed(memory, pos_memory),
                                      memory,
                                      key_padding_mask=mask_memory,
                                      hw_lvl=hw_lvl)
            masks.append(mask)
            inter_query.append(query_embed)
            #if i == 1:
            #    return mask, masks, inter_query
        attn = self.attnen(self.with_pos_embed(query_embed, pos_query),
                           self.with_pos_embed(memory, pos_memory),
                           key_padding_mask=mask_memory,
                           hw_lvl=hw_lvl)
        return attn, masks, inter_query
