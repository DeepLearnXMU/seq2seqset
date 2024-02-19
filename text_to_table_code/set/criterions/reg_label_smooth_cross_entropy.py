import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig, label_smoothed_nll_loss

@dataclass
class RegLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    reg_alpha: float = field(
        default=1.0,
    )

@register_criterion('reg_label_smoothed_cross_entropy', dataclass=RegLabelSmoothedCrossEntropyCriterionConfig)
class RegLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        reg_alpha,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.reg_alpha = reg_alpha
    
    def forward(self, model, sample, reduce=True):
        
        sample_input = sample['net_input']
        sample_concat_input = {
            'src_tokens': torch.cat([sample_input['src_tokens'], sample_input['src_tokens'].clone()], 0),
            'src_lengths': torch.cat([sample_input['src_lengths'], sample_input['src_lengths'].clone()], 0),
            'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
        }
        
        net_output = model(**sample_concat_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        target = torch.cat([target, target.clone()], dim=0)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        
        kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
        loss += self.reg_alpha * kl_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
        
        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss