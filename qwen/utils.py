import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, List, Callable
from matplotlib import pyplot as plt

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPVisionTransformer, CLIPEncoder
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_flashatt, flash_attn_varlen_func, apply_rotary_pos_emb_vision, Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionBlock, apply_multimodal_rotary_pos_emb, _flash_attention_forward, Qwen2_5_VLCausalLMOutputWithPast, CrossEntropyLoss

logger = logging.get_logger(__name__)



def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Copy from the TOME. 
    https://github.com/facebookresearch/ToMe

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def make_tome_class(visionmodel_class):
    class VisionZipVisionModel(visionmodel_class):
        """
        Useless.
        """
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._info["r"] = self.r
            return super().forward(*args, **kwdargs)

    return VisionZipVisionModel


def apply_info(model: nn.Module, dominant_num: Union[int, torch.Tensor], vision_weight, pooling: bool = False):
    """
    Register reduction info to model.
    """
    if "CLIP" in model.__class__.__name__:
        model.r = [0 for i in range(22)]+ [1]+[0]
    elif "SigLip" in model.__class__.__name__:
        model.r = [0 for i in range(25)]+ [1]
    elif "Qwen2_5" in model.__class__.__name__:
        model.r = [0 for i in range(31)]+ [1]
    else:
        raise ValueError(f"Unknown encoder type: {model.__class__.__name__}")
    
    # VisionZipVisionModel = make_tome_class(model.__class__)  # SigLipVisionModel
    # model.__class__ = VisionZipVisionModel

    model._info = {
        "r": model.r,
        "dominant":dominant_num,
        "vision_weight":vision_weight,
        "pooling":pooling,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer) or isinstance(module, Qwen2_5_VLVisionBlock):
            module._info = model._info


def clear_info(model: nn.Module):
    """
    Clear reduction info.
    """
    model.r = []
    model._info = {}
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer) or isinstance(module, Qwen2_5_VLVisionBlock):
            module._info = model._info


def Qwen2_5_VLVisionBlock_foward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    
    residual = hidden_states

    hidden_states, metric = self.attn(  # ! add metric
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )

    # ! extract attn_weights
    if isinstance(hidden_states, tuple):
        hidden_states, attn_weights = hidden_states
    else:
        attn_weights = None

    # ! store metric if r == 1
    if hasattr(self, "_info") and self._info:  # has attr and not empty
        r = self._info["r"].pop(0)
        if r > 0:
            self.metric = metric

    hidden_states = residual + hidden_states

    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

    return hidden_states, attn_weights  # ! add attn_weights


def Qwen2_5_VLVisionFlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)

    return attn_output, k.mean(1)  # ! add metric [76544, 80]


def Qwen2_5_VisionTransformerPretrainedModel_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    #print(f"start_visionTransformer_hidden_states.shape: {hidden_states.shape}")
    hidden_states = self.patch_embed(hidden_states)
    #print(f"path_embed: hidden_states.shape: {hidden_states.shape}")
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states, attn_weights = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)  # ! return attn_weights
    #print(f"before merge hidden states.shape: {hidden_states.shape}")
    hidden_states = self.merger(hidden_states)
    #print(f"after merger hidden states.shape: {hidden_states.shape}")
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]  # [23040, 3584]
    
    if self._info:  # ! apply phase
        # metric = self.blocks[-1].metric  # [32, 2880, 80]
        # dominant_num = self._info["dominant"]
        top_vision_weight = self._info["vision_weight"]
        frames_num = 1
        #print(f"attn_weights.shape: {attn_weights.shape}")
        attn_rec = attn_weights.mean(dim=1).mean(dim=1)  # [32, 16, 2880, 2880] -> [32, 2880]
        #print(f"attn_rec.shape: {attn_rec.shape}")
        #print(f"hidden_states.shape: {hidden_states.shape}")
        # # Merge and reshuffle
        attn_rec = attn_rec.view(frames_num, -1, self.spatial_merge_unit)
        # metric = metric.view(frames_num, -1, self.spatial_merge_unit, metric.shape[2])
        attn_rec = attn_rec.mean(-1)
        # metric = metric.mean(-2)
        attn_rec = attn_rec.reshape(-1)
        # metric = metric.reshape(-1, metric.shape[2])
        attn_rec = attn_rec[reverse_indices]  
        # metric = metric[reverse_indices,:]
        attn_rec = attn_rec.view(frames_num, -1)  # [32, 720]
       
        # add my code
        batch, q_len = attn_rec.shape
        sorted_values, sorted_indices = torch.sort(attn_rec, dim=1, descending=True)# sorted
        #print(f"sorted_indices: {sorted_indices}")
        #topk_indices = attn_rec.topk(dominant_num, dim=1).indices 

        # DVS
        decay_weights = torch.linspace(top_vision_weight, 1.0, steps=q_len, dtype=attn_rec.dtype ,device=attn_rec.device)  # [576]
        #print(f"decay_weights: {decay_weights}")
        enh_weights = torch.zeros_like(attn_rec)  # [batch, 576]

        for b in range(batch):
            enh_weights[b, sorted_indices[b]] = decay_weights
        #print(f"enh_weights: {enh_weights}, enh_weights[sorted_indices[0]]: {enh_weights[0, sorted_indices[0]]}")
        # metric = metric.view(frames_num, -1, metric.shape[1])  # [32, 720, 80]
    return hidden_states, enh_weights[0]

def last_blk_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    #print(f"start: q.shape: {q.shape}, k.shape: {k.shape}, hidden_states.shape: {hidden_states.shape}")
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)

    # ! return attn weights
    #print(f"after: q.shape: {q.shape}, k.shape: {k.shape}, attn_output.shape:{attn_output.shape}")
    num_frames = cu_seqlens.shape[0] - 1
    #print(f"num_frames: {num_frames}")
    q = q.reshape(num_frames, -1, *q.shape[1:]).permute(0, 2, 1, 3)  # [16, 16, 2880, 80]
    k = k.reshape(num_frames, -1, *k.shape[1:]).permute(0, 2, 1, 3)
    #print(f"attn caculate: q.shape:{q.shape}, k.shape:{k.shape}")
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(80) # ! hyper param
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)  # ! discard float32 [64, 16, 2880, 2880]
    # ! return attn weights

    return (attn_output, attn_weights), k.mean(1)

@torch.no_grad()
def generate_dynamic_qwen25vl(
    self,
    vision_weight = 1,
    enh_para = 1,
    sup_para = 0.9,
    **kwargs
) -> Union[GenerateOutput, torch.LongTensor]:
    # to LLM inference
    self.model.enh_para = enh_para
    self.model.sup_para = sup_para

    # self.visual denotes the vision encoder of Qwen2.5
    clear_info(self.visual)

    apply_info(self.visual, dominant_num=576, vision_weight=vision_weight)
    
    return self.generate_raw(**kwargs)