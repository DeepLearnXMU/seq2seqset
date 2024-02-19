# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)

import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment


def scale_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, scale=1.0, scale_token=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if scale != 1.0:
        scale_mask = target.eq(scale_token)
        scale_mask = nll_loss.new_ones(nll_loss.size()).detach()
        scale_mask.masked_fill_(target == scale_token, scale)
        nll_loss  = nll_loss * scale_mask
        smooth_loss = smooth_loss * scale_mask

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class Seq2Set3LabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    assign_steps: int = field(
        default=4,
        metadata={"help": "number of steps for assignment"},
    )

    null_scale: float = field(
        default=0.2,
        metadata={"help": "scale for null token"},
    )

    classes_only: bool = field(
        default=False,
        metadata={"help": "only calculate classification loss"}
    )

    generate_only: bool = field(
        default=False,
        metadata={"help": "only calculate generation loss"}
    )

    classes_lambda: float = field(
        default=1.0,
        metadata={"help": "lambda for classification loss"}
    )

@register_criterion(
    "seq2set3_label_smoothed_cross_entropy", dataclass=Seq2Set3LabelSmoothedCrossEntropyCriterionConfig
)
class Seq2Set3LabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        assign_steps,
        null_scale,
        classes_only,
        generate_only,
        classes_lambda,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.assign_steps = assign_steps
        self.null_scale = null_scale
        if classes_only and generate_only:
            raise ValueError("classes_only and generate_only cannot be both True")
        self.classes_only = classes_only
        self.generate_only = generate_only
        self.classes_lambda = classes_lambda
        self.attn_header = not task.cfg.not_attn_header

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.assign_steps > 0 and not self.classes_only:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                is_training = model.training
                model.eval()
                src_tokens, src_lengths = sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]
                encoder_out = model.encoder(src_tokens, src_lengths=src_lengths)
                header_prev_output_tokens, non_header_prev_output_tokens = sample["net_input"]["prev_output_tokens"]
                header_target, non_header_target = sample["target"]
                header_segment_ids, non_header_segment_ids = sample["net_input"]["segment_ids"]
                bs, snum, _ = non_header_target.size()
                slen = header_target.size(2)
                incremental_state = {'slen': slen}
                prev_output_tokens = src_tokens.new_empty(bs, snum, self.assign_steps)
                assert sample["net_input"]["prev_output_tokens"][0][0][0][0] == self.task.target_dictionary.eos()
                prev_output_tokens[:, :, 0] = self.task.target_dictionary.eos()
                decoder_dist = []
                if self.attn_header:
                    model.decoder(header_prev_output_tokens, header_segment_ids, encoder_out=encoder_out, incremental_state=incremental_state, is_header=True, features_only=True)
                for step in range(self.assign_steps):
                    x, _ = model.decoder(
                        prev_output_tokens[:, :, :step + 1],
                        None,
                        encoder_out=encoder_out,
                        incremental_state=incremental_state,
                        is_header=False,
                    )
                    decoder_dist.append(x)
                    if step < self.assign_steps - 1:
                        prev_output_tokens[:, :, step + 1] = x.argmax(dim=-1)
                decoder_dist = torch.stack(decoder_dist, dim=2)
                reorder_index = hungarian_assign(decoder_dist.softmax(-1), non_header_target[:, :, :self.assign_steps], non_header_segment_ids[:, :, :self.assign_steps], [self.task.target_dictionary.null(), self.task.target_dictionary.pad()])

                sample["target"][1] = non_header_target[reorder_index]
                sample["net_input"]["prev_output_tokens"][1] = non_header_prev_output_tokens[reorder_index]
                sample["net_input"]["segment_ids"][1] = non_header_segment_ids[reorder_index]

                if is_training:
                    model.train()

        net_output = model(**sample["net_input"])
        if self.generate_only:
            header_loss, header_nll_loss = 0, 0
        else:
            header_loss, header_nll_loss = self.compute_loss(model, net_output[0], {"target": sample["target"][0]}, reduce=reduce)
        if self.classes_only:
            non_header_loss, non_header_nll_loss = 0, 0
        else:
            non_header_loss, non_header_nll_loss = self.compute_loss(model, net_output[1], {"target": sample["target"][1]}, reduce=reduce)
        loss = header_loss*self.classes_lambda + non_header_loss
        nll_loss = header_nll_loss*self.classes_lambda + non_header_nll_loss
        ntokens = 0
        if not self.generate_only:
            ntokens += sample["ntokens"][0]
        if not self.classes_only:
            ntokens += sample["ntokens"][1]
        sample_size = (
            sample["target"][0].size(0) if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": sample["target"][0].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = scale_label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            scale=self.null_scale,
            scale_token=self.task.target_dictionary.null(),
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

def hungarian_assign(decode_dist, target, segment_ids, ignore_indices, random=False):
    '''

    :param decode_dist: (batch_size, max_kp_num, kp_len, vocab_size)
    :param target: (batch_size, max_kp_num, kp_len)
    :return:
    '''

    batch_size, max_kp_num, kp_len = target.size()
    reorder_rows = torch.arange(batch_size)[..., None]
    if random:
        reorder_cols = np.concatenate([np.random.permutation(max_kp_num).reshape(1, -1) for _ in range(batch_size)], axis=0)
    else:
        score_mask = target.new_zeros(target.size(), dtype=torch.bool)
        for i in ignore_indices:
            score_mask |= (target == i)
        score_mask |= (segment_ids != 0) # mask non-header tokens
        score_mask = score_mask.unsqueeze(1)  # (batch_size, 1, max_kp_num, kp_len)

        score = decode_dist.new_zeros(batch_size, max_kp_num, max_kp_num, kp_len)
        for b in range(batch_size):
            for l in range(kp_len):
                score[b, :, :, l] = decode_dist[b, :, l][:, target[b, :, l]]
        score = score.masked_fill(score_mask, 0)
        score = score.sum(-1)  # batch_size, max_kp_num, max_kp_num

        reorder_cols = []
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(score[b].cpu().numpy(), maximize=True)
            reorder_cols.append(col_ind.reshape(1, -1))
            # total_score += sum(score[b][row_ind, col_ind])
        reorder_cols = np.concatenate(reorder_cols, axis=0)
    return tuple([reorder_rows, reorder_cols])

