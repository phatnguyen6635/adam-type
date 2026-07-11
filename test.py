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
        lr=0.1,
        lambda_=5e-4,

        alpha_start=0.6,
        alpha_end=0.99,

        max_iters=100000,
        alpha_schedule="linear",

        delta=1e-8,
        beta=0.99,
    ):

        defaults = dict(
            lr=lr,
            lambda_=lambda_,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            max_iters=max_iters,
            alpha_schedule=alpha_schedule,
            delta=delta,
            beta=beta,
        )

        super().__init__(params, defaults)

        self.iter_step = 0

    def _compute_alpha(self, group):

        a = group["alpha_start"]
        b = group["alpha_end"]
        M = group["max_iters"]

        t = min(self.iter_step, M)

        schedule = group["alpha_schedule"].lower()

        if schedule == "linear":
            alpha = compute_linear_alpha(a, b, M, t)

        elif schedule in ["exp", "exponential"]:
            alpha = compute_exponential_alpha(a, b, M, t)

        else:
            raise ValueError(f"Unknown alpha schedule: {schedule}")

        return alpha

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
            delta = group["delta"]
            beta = group["beta"]

            alpha = self._compute_alpha(group)

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
                params=params_with_grad,
                grads=grads,
                momentum_m_list=momentum_m_list,
                prev_w_list=prev_w_list,
                lr=lr,
                lambda_=lambda_,
                alpha=alpha,
                delta=delta,
                beta=beta,
            )

            for w, m_k in zip(params_with_grad, momentum_m_list):
                self.state[w]["momentum_m"] = m_k

        self.iter_step += 1

        return loss

if __name__ == "__main__":
    print("OK")