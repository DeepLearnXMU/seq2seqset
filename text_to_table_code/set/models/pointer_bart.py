import math
from typing import List, Tuple, Dict, Any, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models.bart import (
    BARTModel,
    bart_base_architecture,
    bart_large_architecture,
)
from fairseq.models import register_model, register_model_architecture

@register_model("pointer_bart")
class PointerBARTModel(BARTModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        BARTModel.add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='N',
                            default=1,
                            help='number of attention heads to be used for '
                                 'pointing')
        parser.add_argument('--alignment-layer', type=int, metavar='I',
                            default=-2,
                            help='layer number to be used for pointing (0 '
                                 'corresponding to the bottommost layer)')
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            default=0,
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
        parser.add_argument('--force-generation', type=float, metavar='P',
                            default=None,
                            help='set the vocabulary distribution weight to P, '
                                 'instead of predicting it from the input (1.0 '
                                 'corresponding to generation, 0.0 to pointing)')
        # fmt: on

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return PointerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return PointerDecoder(args, tgt_dict, embed_tokens)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if "decoder.project_p_gens.weight" not in state_dict:
            p_gen_input_size = self.decoder.p_gen_input_size
            project_p_gens_weight = torch.empty(1, p_gen_input_size)
            nn.init.uniform_(project_p_gens_weight, -math.sqrt(1/p_gen_input_size), math.sqrt(1/p_gen_input_size))
            state_dict["decoder.project_p_gens.weight"] = project_p_gens_weight
            state_dict["decoder.project_p_gens.bias"] = torch.zeros(1)


class PointerEncoder(TransformerEncoder):

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(src_tokens, src_lengths, **kwargs)
        return EncoderOut(
            encoder_out=encoder_out.encoder_out,  # T x B x C
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=None,
        )


class PointerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        self.alignment_heads = args.alignment_heads
        if args.alignment_layer < 0:
            self.alignment_layer = args.alignment_layer + args.decoder_layers
        else:
            self.alignment_layer = args.alignment_layer

        input_embed_dim = embed_tokens.embedding_dim

        # Generation probabilities / interpolation coefficients are predicted
        # from the current decoder input embedding and the decoder output, which
        # is the size of output_embed_dim.
        p_gen_input_size = input_embed_dim + self.output_embed_dim
        self.p_gen_input_size = p_gen_input_size
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        nn.init.zeros_(self.project_p_gens.bias)

        # The dictionary may include a separate entry for an OOV token in each
        # input position, so that their identity can be restored from the
        # original source text.
        self.num_types = len(dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types
        self.force_p_gen = args.force_generation

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (EncoderOut, optional): output from the encoder, used
                for encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False)
            alignment_layer (int, optional): 0-based index of the layer to be
                used for pointing (default: 0)
            alignment_heads (int, optional): number of attention heads to be
                used for pointing (default: 1)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # The normal Transformer model doesn't pass the alignment_layer and
        # alignment_heads parameters correctly. We use our local variables.
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads,
        )
        if not features_only:
            # Embedding the tokens again for generation probability prediction,
            # so that we don't have to reimplement the whole extract_features()
            # method.
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, x), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens)
            x = self.output_layer(x, extra["attn"][0], encoder_out.src_tokens, p_gens)
        return x, extra

    def output_layer(self, features, attn, src_tokens, p_gens, **kwargs):
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """
        if self.force_p_gen is not None:
            p_gens = self.force_p_gen

        # project back to size of vocabulary
        logits = super().output_layer(features, **kwargs)

        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert logits.shape[2] == self.num_embeddings
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        gen_dists = super().get_normalized_probs(
            (logits, None), log_probs=False, sample=None
        )
        gen_dists = torch.mul(gen_dists, p_gens)
        padding_size = (batch_size, output_length, self.num_oov_types)
        padding = gen_dists.new_zeros(padding_size)
        gen_dists = torch.cat((gen_dists, padding), 2)
        assert gen_dists.shape[2] == self.num_types

        # Scatter attention distributions to distributions over the extended
        # vocabulary in a tensor of shape [batch_size, output_length,
        # vocab_size]. Each attention weight will be written into a location
        # that is for other dimensions the same as in the index tensor, but for
        # the third dimension it's the value of the index tensor (the token ID).
        attn = torch.mul(attn, 1 - p_gens)
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn)

        # Final distributions, [batch_size, output_length, num_types].
        # return gen_dists
        return gen_dists + attn_dists

    def get_normalized_probs(self, net_output, log_probs, sample):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs

@register_model_architecture("pointer_bart", "pointer_bart_base")
def bart_ours_base_architecture(args):
    bart_base_architecture(args)


@register_model_architecture("pointer_bart", "pointer_bart_large")
def bart_ours_large_architecture(args):
    bart_large_architecture(args)