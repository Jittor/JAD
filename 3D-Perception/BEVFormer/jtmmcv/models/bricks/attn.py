import jittor
import warnings
from jittor import init, Module, nn
import numpy as np
import math

def unflatten(input, dim, sizes):
    '''未经过多测试'''
    
    in_shape = list(input.shape)
    insert_len = len(sizes)
    in_shape[dim: dim + insert_len - 1] = sizes
    
    if dim==-1:
        in_shape = in_shape[:-1]
    
    return input.view(in_shape)

jittor.Var.unflatten = unflatten

def _in_projection_packed(
    q,
    k,
    v,
    w,
    b = None,
):
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = nn.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = nn.linear(q, w_q, b_q)
            kv_proj = nn.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.linear(q, w_q, b_q), nn.linear(k, w_k, b_k), nn.linear(v, w_v, b_v)


def _canonical_mask(
    mask,
    mask_name: str,
    other_type,
    other_name: str,
    target_type,
    check_other: bool = True,
):
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = mask.dtype==jittor.float32 or mask.dtype==jittor.float64 or mask.dtype==jittor.float16
        if _mask_dtype != jittor.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = jittor.zeros_like(mask, dtype=target_type).masked_fill(
                mask, float("-inf")
            )
    return mask


def _none_or_dtype(input):
    if input is None:
        return None
    elif isinstance(input, jittor.Var):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or jittor.Var")




class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        batch_first=False,
        qn_block_size=8,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first

        self.num_heads = num_heads
        assert dropout==0, "TODO: dropout>0"

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, ("Self-attention requires query, key and " "value to be of the same size")

        #TODO: quant_noise
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert not add_bias_kv, "TODO: add_bias_kv=True"
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False
        
    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            init.xavier_uniform_(self.in_proj.weight)
        else:
            init.xavier_uniform_(self.k_proj.weight)
            init.xavier_uniform_(self.v_proj.weight)
            init.xavier_uniform_(self.q_proj.weight)

        # init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias, 0.)
            init.constant_(self.in_proj.bias, 0.)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def execute(
        self,
        query,
        key = None,
        value = None,
        key_padding_mask = None,
        incremental_state = None,
        need_weights = True,
        static_kv = False,
        attn_mask = None,
        before_softmax = False,
        need_head_weights = False,
        use_separate_proj_weight=False,
        is_causal = False
    ):
        #TODO:MORE PARAMS IMPLEMENTATION
        is_batched = query.dim() == 3
        
        if need_head_weights:
            need_weights = True

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        assert incremental_state is None, "TODO: incremental_state is not None"
        saved_state = None
        if not use_separate_proj_weight: # ???
            q, k, v = _in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)

        # if self.self_attention:
        #     q = self.q_proj(query)
        #     k = self.k_proj(query)
        #     v = self.v_proj(query)
        # elif self.encoder_decoder_attention:
        #     # encoder-decoder attention
        #     q = self.q_proj(query)
        #     if key is None:
        #         assert value is None
        #         k = v = None
        #     else:
        #         k = self.k_proj(key)
        #         v = self.v_proj(key)
        # else:
        #     assert key is not None and value is not None
        #     q = self.q_proj(query)
        #     k = self.k_proj(key)
        #     v = self.v_proj(value)
            
        q = q*self.scaling

        assert self.bias_k is None, "TODO: self.bias_k is not None:"

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if k is not None:
            k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if v is not None:
            v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        assert saved_state is None, "TODO: saved_state is not None"
        assert k is not None
        src_len = k.shape[1]
        
            
        assert not self.add_zero_attn, "TODO: self.add_zero_attn=True"

        attn_weights = nn.bmm(q, k.transpose(0, 2, 1))

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]
        
        if key_padding_mask is not None:
            key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
            )
            key_padding_mask = nn.pad(key_padding_mask, [0, 1])
        
        
        if attn_mask is not None:
            if key_padding_mask is not None:
                assert attn_mask.shape == attn_weights.shape, "pleas make sure attn_mask.shape == attn_weights.shape, or set mask as None"
                attn_mask = attn_mask + key_padding_mask
            else:
                assert attn_mask.shape == attn_weights.shape, "pleas make sure attn_mask.shape == attn_weights.shape, or set mask as None"
                attn_mask = attn_mask
            attn_weights = attn_weights.masked_fill(attn_mask != 0, -1e9)
            attn_weights = jittor.nn.softmax(attn_weights, dim=-1)
        else:
            attn_weights = nn.softmax(attn_weights, dim=-1)
            
        attn_weights_float = attn_weights.clone()

        assert v is not None
        attn = nn.bmm(attn_weights, v)
        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.shape[1] == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0, 2, 3)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dims=[0])
        
        if self.batch_first and is_batched:
            
            return attn.transpose(1, 0), attn_weights
        
        return attn, attn_weights
