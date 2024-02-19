from typing import Optional, Dict, List, Any
import functools

import logging
import torch
import torch.nn as nn

from torch import Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import BARTModel, BARTClassificationHead, bart_base_architecture, bart_large_architecture
from fairseq.models.transformer import TransformerEncoderBase, TransformerDecoderBase, Embedding
from fairseq.modules.grad_multiply import GradMultiply

logger = logging.getLogger(__name__)
@register_model('seq2set2_bart')
class Seq2Set2BART(BARTModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        
        if args.num_classes > 0:
            self.classification_head = BARTClassificationHead(
                input_dim=args.encoder_embed_dim,
                inner_dim=args.encoder_embed_dim,
                num_classes=args.num_classes,
                activation_fn=args.pooler_activation_fn,
                pooler_dropout=args.pooler_dropout,
                do_spectral_norm=getattr(
                    args, "spectral_norm_classification_head", False
                ),
            )
        else:
            self.classification_head = None
        

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        # embed_segment: 0: row-header tokens, [1, num_classes]: non-header tokens, num_classes+1: EOS
        embed_segment = nn.Embedding(cfg.num_classes+2, cfg.decoder_embed_dim, padding_idx=None)
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
            embed_segment_weight = torch.empty(self.cfg.num_classes+2, self.cfg.decoder.embed_dim)
            nn.init.normal_(embed_segment_weight, mean=0, std=self.cfg.decoder.embed_dim**-0.5)
            state_dict['decoder.embed_segment.weight'] = embed_segment_weight
            logger.info('Initialized segment embedding from pre-trained BART.')
        if 'classification_head.dense.weight' not in state_dict:
            state_dict['classification_head.dense.weight'] = self.classification_head.dense.weight
            state_dict['classification_head.dense.bias'] = self.classification_head.dense.bias
            state_dict['classification_head.out_proj.weight'] = self.classification_head.out_proj.weight
            state_dict['classification_head.out_proj.bias'] = self.classification_head.out_proj.bias

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        segment_ids,
        classes_only,
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
        classification_dists = self.forward_classification_head(src_tokens, encoder_out)
        if classes_only:
            return classification_dists

        x, extra = self.decoder(
            prev_output_tokens,
            segment_ids,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        if self.classification_head is not None:
            return x, extra, classification_dists
        else:
            return x, extra
        
    def forward_classification_head(self, src_tokens, encoder_out):
        T, B, C = encoder_out['encoder_out'][0].shape
        sentence_representation = encoder_out['encoder_out'][0].transpose(0,1)[src_tokens.eq(self.eos), :].view(B, C)
        classification_dists = self.classification_head(sentence_representation)
        return classification_dists

class Seq2SetDecoder(TransformerDecoderBase):

    def __init__(self, cfg, dictionary, embed_tokens, embed_segment, no_encoder_attn=False, output_projection=None):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        control_code = nn.Parameter(torch.empty(cfg.max_num, cfg.decoder.embed_dim))
        # nn.init.uniform_(control_code, a=-0.1, b=0.1)
        nn.init.normal_(control_code, mean=0, std=cfg.decoder.embed_dim**-0.5)
        self.register_parameter('control_code', control_code)
        self.max_num = cfg.max_num
        self.grad_mult = getattr(cfg, 'grad_mult', 1.0)
        self.embed_segment = embed_segment
        self.num_classes = cfg.num_classes

    
    def forward(
        self,
        prev_output_tokens,
        segment_ids,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
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
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        segment_ids,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
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
        if control_code_idx is None:
            bs, snum, slen = prev_output_tokens.size()
            assert snum == self.max_num
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
        positions = None
        
        if self.embed_positions is not None:
            positions = self.embed_positions(prev_output_tokens.reshape(bs*snum, slen), incremental_state=incremental_state)
            if incremental_state is None:
                positions = positions.reshape(bs, snum, slen, -1)

        if incremental_state is not None:
            if control_code_idx is None: # valid
                prev_output_tokens = prev_output_tokens[:, :, -1:]
                segment_ids = prev_output_tokens.new_zeros(1, 1, 1)
                # segment_ids = segment_ids[:, :, -1:]
            else: # test
                prev_output_tokens = prev_output_tokens[:, -1:]
                # segment_ids = segment_ids[:, -1:]
            if positions is not None:
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

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        # embed control code
        # contorl_code_idx = torch.arange(snum, device=prev_output_tokens.device, dtype=torch.long)
        if control_code_idx is None:
            code = self.control_code.reshape(1, snum, 1, -1)
            if self.grad_mult != 1.0:
                x += GradMultiply.apply(code, self.grad_mult)
            else:
                x += code
        else:
            # x: bs*snum, slen, dim, control_code_idx: bs*snum
            x += self.control_code[control_code_idx].unsqueeze(1)

        x += self.embed_segment(segment_ids)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.reshape(bs, snum*slen, -1).transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx).reshape(bs, snum*slen)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if control_code_idx is not None:
                # inference
                self_attn_mask = None
            elif incremental_state is None: 
                # training
                self_attn_mask = self._get_train_self_attn_mask(snum, slen).to(x)
            else: 
                # validation
                self_attn_mask = self._get_eval_self_attn_mask(snum, full_slen).to(x)
            
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
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
    def _get_train_self_attn_mask(max_num, max_len):
        mask = torch.triu(torch.full([max_num*max_len, max_num*max_len], float("-inf")), 1)
        for i in range(1, max_num + 1):
            mask[i * max_len:(i + 1) * max_len, :i * max_len] = float("-inf")
        return mask

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _get_eval_self_attn_mask(max_num, max_len):
        mask = torch.full([max_num, max_num], float("-inf"))
        mask.fill_diagonal_(0)
        mask = mask.repeat(1, max_len)
        return mask
    
    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if 'control_code_idx' in incremental_state:
            incremental_state['control_code_idx'] = incremental_state['control_code_idx'][new_order]

@register_model_architecture("seq2set2_bart", "seq2set2_bart_base")
def bart_ours_base_architecture(args):
    bart_base_architecture(args)


@register_model_architecture("seq2set2_bart", "seq2set2_bart_large")
def bart_ours_large_architecture(args):
    bart_large_architecture(args)