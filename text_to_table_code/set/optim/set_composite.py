# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer, _build_optimizer
from fairseq.optim.lr_scheduler import FairseqLRScheduler, build_lr_scheduler
from omegaconf import II, open_dict, OmegaConf

from fairseq.optim.composite import FairseqCompositeOptimizer, CompositeLRScheduler, CompositeOptimizer
import json


logger = logging.getLogger(__name__)


@dataclass
class OptimizerAndSchedulerConfig(FairseqDataclass):
    optimizer: Any = None
    lr_scheduler: Optional[Any] = None
    lr: List = II("optimization.lr")
    lr_float: Optional[
        float
    ] = None  # this makes it easier to sweep on learning rate with auto sweepers


@dataclass
class SetCompositeOptimizerConfig(FairseqDataclass):
    groups: str = field(
        default='{}',
        metadata={
            "help": "optimizer name -> optimizer OptimizerAndSchedulerConfig. "
            "Configures a different optimizer and (optionally) lr scheduler for each parameter group"
        },
    )

@register_optimizer("set_composite", dataclass=SetCompositeOptimizerConfig)
class SetFairseqCompositeOptimizer(FairseqCompositeOptimizer):

    optimizers: Dict[str, FairseqOptimizer] = {}
    lr_schedulers: Dict[str, FairseqLRScheduler] = {}
    lr_scheduler: FairseqLRScheduler = None
    _optimizer: torch.optim.Optimizer

    def __init__(self, cfg: SetCompositeOptimizerConfig, params):
        self.cfg = cfg

        assert (
            len(params) > 1
        ), "Composite optimizer only works when there are multiple parameter groups (try fp16_no_flatten_grads: true)"

        groupped_params = defaultdict(list)
        for p in params:
            group = getattr(p, "param_group", "default")
            groupped_params[group].append(p)

        groups = OmegaConf.create(json.loads(cfg.groups))
        assert groupped_params.keys() == groups.keys(), (
            f"Parameter groups {groupped_params.keys()} and optimizer groups {groups.keys()} are not the same! "
            "Try setting 'param_group' on your parameters in the model."
        )

        for group, group_params in groupped_params.items():
            group_cfg = groups[group]
            with open_dict(group_cfg):
                if group_cfg.lr_float is not None:
                    group_cfg.optimizer.lr = [group_cfg.lr_float]
                    group_cfg.lr_scheduler.lr = [group_cfg.lr_float]
                else:
                    group_cfg.optimizer.lr = group_cfg.lr
                    group_cfg.lr_scheduler.lr = group_cfg.lr
            self.optimizers[group] = _build_optimizer(group_cfg.optimizer, group_params)
            if group_cfg.lr_scheduler is not None:
                self.lr_schedulers[group] = build_lr_scheduler(
                    group_cfg.lr_scheduler, self.optimizers[group]
                )

        if len(self.lr_schedulers) > 0:
            assert len(self.lr_schedulers) == len(self.optimizers), (
                f"Please provide an lr scheduler for each optimizer to use pass_through scheduler. "
                f"Optimizers: {self.optimizers}; Lr scheds: {self.lr_schedulers}"
            )
            self.lr_scheduler = CompositeLRScheduler(self.lr_schedulers)

        self._optimizer = CompositeOptimizer(self.optimizers)