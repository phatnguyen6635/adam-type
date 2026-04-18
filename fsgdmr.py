import torch
from torch import Tensor
from typing import Callable, List, Optional
from torch.optim import Optimizer
import math


def fractional_sgdm(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    prev_param_list: List[Optional[Tensor]],
    *,
    lr: float,
    weight_decay: float,
    fractional_alpha: float,
    delta: float,
    beta: float
):
    eps_power = 1.0 - fractional_alpha
    gamma_val = math.gamma(2 - fractional_alpha)

    for i, param in enumerate(params):
        delta_F = grads[i]

        w_k_pre = prev_param_list[i]

        # snapshot w_t BEFORE update
        w_k = param.detach().clone()
        
        g_k = delta_F + weight_decay * w_k
        
        buf = momentum_buffer_list[i]
        if buf is None:
            buf = (1 - beta) * g_k.detach().clone() 
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(beta).add_((1 - beta) * g_k)

        frac_factor = (torch.abs(w_k - w_k_pre) + delta).pow(eps_power)
        scaled_grad = buf * frac_factor / gamma_val

        param.add_(scaled_grad, alpha=-lr)

        w_k_pre.copy_(w_k)


class FractionalSGDMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        fractional_alpha: float = 0.999,
        delta: float = 1e-8,
        beta: float = 0.99
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (0.0 <= fractional_alpha < 1.0):
            raise ValueError(f"fractional_alpha must be in [0, 1), got {fractional_alpha}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            fractional_alpha=fractional_alpha,
            delta=delta,
            beta=beta
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []
            prev_param_list = []

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            fractional_alpha = group["fractional_alpha"]
            delta = group["delta"]
            beta = group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = None
                    state["prev_param"] = p.detach().clone()

                momentum_buffer_list.append(state["momentum_buffer"])
                prev_param_list.append(state["prev_param"])

            if len(params_with_grad) == 0:
                continue

            fractional_sgdm(
                params_with_grad,
                grads,
                momentum_buffer_list,
                prev_param_list,
                lr=lr,
                weight_decay=weight_decay,
                fractional_alpha=fractional_alpha,
                delta=delta,
                beta=beta
            )

            for p, buf, w_k_pre in zip(params_with_grad, momentum_buffer_list, prev_param_list):
                state = self.state[p]
                state["momentum_buffer"] = buf
                state["prev_param"] = w_k_pre

        return loss