import math
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

def compute_linear_alpha(a: float, b: float, M:int, iter_step: int) -> float:
    alpha = iter_step / M * (b - a) + a
    return alpha

def compute_exponential_alpha(a: float, b: float, M:int, iter_step: int) -> float:
    alpha = a * (b / a) ** (iter_step / M)
    return alpha

def fractional_order_sgdm(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_m_list: List[Optional[Tensor]],
    prev_w_list: List[Optional[Tensor]],
    *,
    lr: float,
    lambda_: float,
    alpha: float,
    delta: float,
    beta: float
):
    eps_power = 1.0 - alpha
    gamma_val = math.gamma(2.0 - alpha)

    for i, w_k in enumerate(params):
        grad_w_k = grads[i]
        w_k_minus_1 = prev_w_list[i]

        # snapshot w_k before update
        w_k_current = w_k.clone()

        # g_k = ∇L(w_k) + λ w_k
        g_k = grad_w_k + lambda_ * w_k_current

        # m_k = β m_{k-1} + (1 - β) g_k
        m_k = momentum_m_list[i]
        if m_k is None:
            m_k = (1.0 - beta) * g_k
            momentum_m_list[i] = m_k
        else:
            m_k.mul_(beta).add_((1.0 - beta) * g_k)

        # (|w_k - w_{k-1}| + δ)^(1-α)
        frac_factor = (torch.abs(w_k_current - w_k_minus_1) + delta).pow(eps_power)

        # w_{k+1} = w_k - μ * m_k * frac_factor / Γ(2-α)
        scaled_grad = m_k * frac_factor / gamma_val
        w_k.add_(scaled_grad, alpha=-lr)

        # update w_{k-1} <- w_k (for next step)
        w_k_minus_1.copy_(w_k_current)


class FractionalOrderSGDMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.1,
        lambda_: float = 5e-4,
        fractional_alpha: float = 0.999,
        delta: float = 1e-8,
        beta: float = 0.99
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate mu: {lr}")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid weight_decay lambda_: {lambda_}")
        if not (0.0 < fractional_alpha <= 1.001):
            raise ValueError(f"alpha must be in (0, 1.001], got {fractional_alpha}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")

        defaults = dict(
            lr=lr,
            lambda_=lambda_,
            alpha=fractional_alpha,
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
            momentum_m_list = []
            prev_w_list = []

            lr = group["lr"]
            lambda_ = group["lambda_"]
            alpha = group["alpha"]
            delta = group["delta"]
            beta = group["beta"]

            for w in group["params"]:
                if w.grad is None:
                    continue

                params_with_grad.append(w)
                grads.append(w.grad)

                state = self.state[w]
                if len(state) == 0:
                    state["momentum_m"] = None
                    state["prev_w"] = w.clone()

                momentum_m_list.append(state["momentum_m"])
                prev_w_list.append(state["prev_w"])

            if len(params_with_grad) == 0:
                continue

            fractional_order_sgdm(
                params_with_grad,
                grads,
                momentum_m_list,
                prev_w_list,
                lr=lr,
                lambda_=lambda_,
                alpha=alpha,
                delta=delta,
                beta=beta
            )

            for w, m_k in zip(params_with_grad, momentum_m_list):
                state = self.state[w]
                state["momentum_m"] = m_k

        return loss