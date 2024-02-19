from typing import Optional, Dict, List, Any
import functools

import torch
import torch.nn as nn

from torch import Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import BARTModel, BARTClassificationHead, bart_base_architecture, bart_large_architecture
from fairseq.models.transformer import TransformerEncoderBase, TransformerDecoderBase
from fairseq.modules.grad_multiply import GradMultiply

@register_model('seq2set_bart')
class Seq2SetBART(BARTModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        # add for different learning rate
        for p in self.encoder.parameters():
            p.param_group = 'encoder'
        for p in self.decoder.parameters():
            p.param_group = 'decoder'
        

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return Seq2SetDecoder(cfg, tgt_dict, embed_tokens)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if 'decoder.control_code' not in state_dict:
            control_code_weight = torch.empty(self.args.max_num, self.cfg.decoder.embed_dim)
            # nn.init.uniform_(control_code_weight, a=-0.1, b=0.1)
            nn.init.normal_(control_code_weight, mean=0, std=self.cfg.decoder.embed_dim**-0.5)
            state_dict['decoder.control_code'] = control_code_weight

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        prefix_train: bool = False,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        x_prefix_train = None
        if prefix_train:
            x_prefix_train, _ = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
                prefix_train=True,
            )
        if x_prefix_train is not None:
            return torch.cat([x, x_prefix_train], dim=0), extra
        else:
            return x, extra
class Seq2SetDecoder(TransformerDecoderBase):

    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        control_code = nn.Parameter(torch.empty(cfg.max_num, cfg.decoder.embed_dim))
        # nn.init.uniform_(control_code, a=-0.1, b=0.1)
        nn.init.normal_(control_code, mean=0, std=cfg.decoder.embed_dim**-0.5)
        self.register_parameter('control_code', control_code)
        self.max_num = cfg.max_num
        self.ordered_bos = getattr(cfg, 'ordered_bos', False)
        self.grad_mult = getattr(cfg, 'grad_mult', 1.0)

    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        prefix_train: Optional[int] = False,
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
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            prefix_train=prefix_train,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        prefix_train: Optional[int] = False,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            prefix_train,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        prefix_train: bool = False,
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
            if control_code_idx is None:

                prev_output_tokens = prev_output_tokens[:, :, -1:]
            else:
                prev_output_tokens = prev_output_tokens[:, -1:]
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
        if not self.ordered_bos and not prefix_train:
            if control_code_idx is None:
                code = self.control_code.reshape(1, snum, 1, -1)
                if self.grad_mult != 1.0:
                    x += GradMultiply.apply(code, self.grad_mult)
                else:
                    x += code
            else:
                # x: bs*snum, slen, dim, control_code_idx: bs*snum
                x += self.control_code[control_code_idx].unsqueeze(1)


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

@register_model_architecture("seq2set_bart", "seq2set_bart_base")
def bart_ours_base_architecture(args):
    bart_base_architecture(args)


@register_model_architecture("seq2set_bart", "seq2set_bart_large")
def bart_ours_large_architecture(args):
    bart_large_architecture(args)