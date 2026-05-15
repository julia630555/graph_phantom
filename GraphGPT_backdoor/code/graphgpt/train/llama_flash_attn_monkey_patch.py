from typing import Optional, Tuple

import torch
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# flash-attn v1 API (legacy)
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except Exception:
    flash_attn_unpadded_qkvpacked_func = None

# flash-attn v2 API (current)
if flash_attn_unpadded_qkvpacked_func is None:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )


def _flash_forward_no_mask(qkv: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
    if flash_attn_unpadded_qkvpacked_func is not None:
        qkv_unpad = rearrange(qkv, "b s ... -> (b s) ...")
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv_unpad, cu_q_lens, q_len, 0.0, softmax_scale=None, causal=True
        )
        return rearrange(output, "(b s) ... -> b s ...", b=bsz)

    return flash_attn_qkvpacked_func(
        qkv, dropout_p=0.0, softmax_scale=None, causal=True
    )


def _flash_forward_with_mask(
    qkv: torch.Tensor, key_padding_mask: torch.Tensor, bsz: int, q_len: int
) -> torch.Tensor:
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    unpad_ret = unpad_input(x, key_padding_mask)
    if len(unpad_ret) == 4:
        x_unpad, indices, cu_q_lens, max_s = unpad_ret
    else:
        x_unpad, indices, cu_q_lens, max_s, _ = unpad_ret
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )

    if flash_attn_unpadded_qkvpacked_func is not None:
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
    else:
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )

    return rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
        "b s (h d) -> b s h d",
        h=nheads,
    )


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We disable _prepare_decoder_attention_mask in LlamaModel,
    # so attention_mask should be key_padding_mask.
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        output = _flash_forward_no_mask(qkv, bsz, q_len)
    else:
        output = _flash_forward_with_mask(qkv, key_padding_mask, bsz, q_len)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


# Disable attention mask transformation in LlamaModel because flash-attn
# expects attention mask to be key_padding_mask.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    return attention_mask


def replace_llama_attn_with_flash_attn():
    llama_mod = transformers.models.llama.modeling_llama
    if hasattr(llama_mod.LlamaModel, "_prepare_decoder_attention_mask"):
        llama_mod.LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask
        )
    llama_mod.LlamaAttention.forward = forward
