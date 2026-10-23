# 论文八：兼容新版 Transformers，保留旧 AdamW 的更新公式。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Transformers 4.5.1 AdamW, retained for the author's correct_bias=False setting.

The optimizer was removed from newer Transformers releases. This preserves its
update order and defaults while leaving the author's training loop untouched.
Reference: transformers v4.5.1, src/transformers/optimization.py, AdamW.
"""

import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        if lr < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or eps < 0:
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      correct_bias=correct_bias))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    step_size *= math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss
