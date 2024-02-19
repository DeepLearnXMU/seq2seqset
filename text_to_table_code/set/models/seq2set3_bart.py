from typing import Optional, Dict, List, Any
import functools

import logging
import torch
import torch.nn as nn

from torch import Tensor
from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import BARTModel, bart_base_architecture, bart_large_architecture
from fairseq.models.transformer import TransformerEncoderBase, TransformerDecoderBase, Embedding
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from set.modules import set_transformer_layer

TRAINING = 0
VALID_HEADER = 1
VALID_NONHEADER = 2
INFERENCE = 3

logger = logging.getLogger(__name__)
@register_model('seq2set3_bart')
class Seq2Set3BART(BARTModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        # embed_segment: 0: row-header tokens, [1, num_classes]: non-header tokens, num_classes+1: EOS
        embed_segment = nn.Embedding(cfg.max_segment, cfg.decoder_embed_dim, padding_idx=None)
        nn.init.normal_(embed_segment.weight, mean=0, std=cfg.decoder_embed_dim**-0.5)
        return Seq2SetDecoder(cfg, tgt_dict, embed_tokens, embed_segment)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        # if 'decoder.control_code' not in state_dict or state_dict['decoder.control_code'].shape[0] != self.decoder.control_code.shape[0]:
        if 'decoder.control_code' not in state_dict:
            control_code_weight = torch.empty(self.args.max_num, self.cfg.decoder.embed_dim)
            # nn.init.uniform_(control_code_weight, a=-0.1, b=0.1)
            nn.init.normal_(control_code_weight, mean=0, std=self.cfg.decoder.embed_dim**-0.5)
            state_dict['decoder.control_code'] = control_code_weight
            logger.info('Initialized control code from pre-trained BART.')

        if 'decoder.embed_segment.weight' not in state_dict:
            embed_segment_weight = torch.empty(self.cfg.max_segment, self.cfg.decoder.embed_dim)
            nn.init.normal_(embed_segment_weight, mean=0, std=self.cfg.decoder.embed_dim**-0.5)
            state_dict['decoder.embed_segment.weight'] = embed_segment_weight
            logger.info('Initialized segment embedding from pre-trained BART.')

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        segment_ids,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # if classification_head_name is not None:
        #     features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
        )
        if isinstance(prev_output_tokens, list):
            header_prev_output_tokens, non_header_prev_output_tokens = prev_output_tokens
            header_segment_ids, non_header_segment_ids = segment_ids
            x, extra = self.decoder(
                non_header_prev_output_tokens,
                non_header_segment_ids,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
                is_header=False,
                header_prev_output_tokens=header_prev_output_tokens,
                header_segment_ids=header_segment_ids,
            )
            h_len = header_prev_output_tokens.size(2)
            return (x[:, :h_len], extra), (x[:, h_len:], extra)
        else:
            x, extra = self.decoder(
                prev_output_tokens,
                segment_ids,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
                is_header=False,
            )
            return x, extra
        
class Seq2SetDecoder(TransformerDecoderBase):

    def __init__(self, cfg, dictionary, embed_tokens, embed_segment, no_encoder_attn=False, output_projection=None):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        control_code = nn.Parameter(torch.empty(cfg.max_num, cfg.decoder.embed_dim))
        # nn.init.uniform_(control_code, a=-0.1, b=0.1)
        nn.init.normal_(control_code, mean=0, std=cfg.decoder.embed_dim**-0.5)
        self.register_parameter('control_code', control_code)
        self.max_num = cfg.max_num
        self.embed_segment = embed_segment
        self.attn_header = cfg.attn_header
        self.add_row_query = not getattr(cfg, "wo_row_query", False)
        self.add_column_query = not getattr(cfg, "wo_column_query", False)

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = set_transformer_layer.SetTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def forward(
        self,
        prev_output_tokens,
        segment_ids,
        is_header,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        header_prev_output_tokens: Optional[Tensor] = None,
        header_segment_ids: Optional[Tensor] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            segment_ids,
            is_header,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            header_prev_output_tokens=header_prev_output_tokens,
            header_segment_ids=header_segment_ids,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        segment_ids,
        is_header,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        header_prev_output_tokens: Optional[Tensor] = None,
        header_segment_ids: Optional[Tensor] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if full_context_alignment:
            raise NotImplementedError(
                "full_context_alignment is not supported for Seq2SetDecoder"
            )
        if incremental_state is not None and 'control_code_idx' in incremental_state:
            control_code_idx = incremental_state['control_code_idx']
        else:
            control_code_idx = None
        # if control_code_idx is not None:
        #     state = INFERENCE
        # elif incremental_state is None:
        #     state = TRAINING
        # elif is_header:
        #     state = VALID_HEADER
        # else:
        #     state = VALID_NONHEADER
        
        if control_code_idx is None:
            bs, snum, slen = prev_output_tokens.size()
            if is_header:
                assert snum == 1
            elif incremental_state is not None:
                assert snum == self.max_num - 1
        else:
            bs, slen = prev_output_tokens.size()
            snum = 1
        full_slen = slen
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        if control_code_idx is None and is_header: # valid header
            positions = self.embed_positions(prev_output_tokens.reshape(bs*snum, slen))
            positions = positions.reshape(bs, snum, slen, -1)
        else:
            positions = self.embed_positions(prev_output_tokens.reshape(bs*snum, slen), incremental_state=incremental_state)
            if header_prev_output_tokens is not None:
                hlen = header_prev_output_tokens.size(2)
                header_positions = self.embed_positions(header_prev_output_tokens.reshape(bs, hlen), incremental_state=incremental_state)
                header_positions = header_positions.reshape(bs, 1, hlen, -1)
            if incremental_state is None:
                positions = positions.reshape(bs, snum, slen, -1)

        if incremental_state is not None:

            if control_code_idx is not None: # inference
                prev_output_tokens = prev_output_tokens[:, -1:]
                positions = positions[:, -1:]
            elif not is_header: # valid
                prev_output_tokens = prev_output_tokens[:, :, -1:]
                segment_ids = prev_output_tokens.new_zeros(1, 1, 1)
                positions = positions[:, -1:]


        if control_code_idx is None:
            bs, snum, slen = prev_output_tokens.size()
        else:
            bs, slen = prev_output_tokens.size()
            snum = 1
        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if header_prev_output_tokens is not None:
            header_x = self.embed_scale * self.embed_tokens(header_prev_output_tokens)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
            if header_prev_output_tokens is not None:
                header_x = self.quant_noise(header_x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
            if header_prev_output_tokens is not None:
                header_x = self.project_in_dim(header_x)

        if positions is not None:
            x += positions
            if header_prev_output_tokens is not None:
                header_x += header_positions

        # embed control code
        # contorl_code_idx = torch.arange(snum, device=prev_output_tokens.device, dtype=torch.long)
        if self.add_row_query:
            if control_code_idx is None: 
                if is_header: # valid header
                    x += self.control_code[0].reshape(1, 1, 1, -1)
                else: # valid non-header
                    x += self.control_code[1:].reshape(1, snum, 1, -1)
                    if header_prev_output_tokens is not None:
                        header_x += self.control_code[0].reshape(1, 1, 1, -1)
            else: # inference
                # x: bs*snum, slen, dim, control_code_idx: bs*snum
                x += self.control_code[control_code_idx].unsqueeze(1)

        if self.add_column_query:
            x += self.embed_segment(segment_ids)
            if header_prev_output_tokens is not None:
                header_x += self.embed_segment(header_segment_ids)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
            if header_prev_output_tokens is not None:
                header_x = self.layernorm_embedding(header_x)

        x = self.dropout_module(x)
        if header_prev_output_tokens is not None:
            header_x = self.dropout_module(header_x)

        # B x T x C -> T x B x C
        x = x.reshape(bs, snum*slen, -1).transpose(0, 1)
        if header_prev_output_tokens is not None:
            x = torch.cat([header_x.reshape(bs, hlen, -1).transpose(0, 1), x], dim=0)
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx).reshape(bs, snum*slen)
            if header_prev_output_tokens is not None:
                header_attn_padding_mask = header_prev_output_tokens.eq(self.padding_idx).reshape(bs, hlen)
                self_attn_padding_mask = torch.cat([header_attn_padding_mask, self_attn_padding_mask], dim=1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        if control_code_idx is not None:
            # inference
            self_attn_mask = None
        elif incremental_state is None or is_header: 
            # training or valid header
            header_len = header_prev_output_tokens.size(2) if header_prev_output_tokens is not None else 0
            self_attn_mask = self._get_train_self_attn_mask(snum, slen, header_len, self.attn_header).to(x)
        else: 
            # validation
            self_attn_mask = self._get_eval_self_attn_mask(snum, full_slen).to(x)
            if self.attn_header:
                header_mask = self_attn_mask.new_zeros(snum, incremental_state['slen'])
                self_attn_mask = torch.cat((header_mask, self_attn_mask), dim=1)
        for idx, layer in enumerate(self.layers):
            
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                is_header=is_header,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _get_train_self_attn_mask(max_num, max_len, header_len, attn_header=True):
        mask = torch.triu(torch.full([header_len+max_num*max_len, header_len+max_num*max_len], float("-inf")), 1)
        if max_num == 1: # only header
            return mask
        if attn_header:
            for i in range(max_num - 1):
                mask[header_len + i * max_len:header_len + (i + 1) * max_len, header_len:header_len+i * max_len] = float("-inf")
        else:
            for i in range(max_num - 1):
                mask[header_len + i * max_len:header_len + (i + 1) * max_len, :header_len+i * max_len] = float("-inf")
        return mask

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _get_eval_self_attn_mask(max_num, max_len):
        mask = torch.full([max_num, max_num], float("-inf"))
        mask.fill_diagonal_(0)
        mask = mask.repeat(1, max_len)
        # mask = torch.cat([torch.zeros(max_num, max_len), mask], dim=1)
        return mask
    
    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if 'control_code_idx' in incremental_state and incremental_state['control_code_idx'].size() != torch.Size([1]):
            incremental_state['control_code_idx'] = incremental_state['control_code_idx'][new_order]

@register_model_architecture("seq2set3_bart", "seq2set3_bart_base")
def bart_ours_base_architecture(args):
    bart_base_architecture(args)
    args.max_segment = getattr(args, "max_segment", 30)
    args.attn_header = not getattr(args, "not_attn_header", False)



@register_model_architecture("seq2set3_bart", "seq2set3_bart_large")
def bart_ours_large_architecture(args):
    bart_large_architecture(args)
    args.max_segment = getattr(args, "max_segment", 30)
    args.attn_header = not getattr(args, "not_attn_header", False)