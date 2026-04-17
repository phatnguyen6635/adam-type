import torch
from torch import Tensor
from typing import Callable, List, Optional
from torch.optim import Optimizer
import math


def compute_momentum(iter_step: int) -> float:
    b = 0.9
    a = 1000
    if iter_step <= 10000:
        return b * iter_step / 10000
    elif iter_step <= 15000:
        return b - 2 * (0.9 * 0.999999 ** (15 * a) - b) + iter_step / (5 * a) * (0.9 * 0.999999 ** (15 * a) - b)
    else:
        return 0.9 * 0.999999 ** iter_step


def fractional_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    prev_param_list: List[Optional[Tensor]],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    fractional_alpha: float,
    delta: float,
):
    eps_power = 1.0 - fractional_alpha
    gamma_val = math.gamma(2 - fractional_alpha)

    for i, param in enumerate(params):
        grad = grads[i]

        prev_param = prev_param_list[i]

        # snapshot w_t BEFORE update
        current_param = param.detach().clone()

        # fractional factor uses w_t and w_{t-1}
        frac_factor = (torch.abs(current_param - prev_param) + delta).pow(eps_power)
        scaled_grad = grad * frac_factor / gamma_val

        buf = momentum_buffer_list[i]
        if buf is None:
            buf = scaled_grad.detach().clone()
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(momentum).add_(scaled_grad)

        # paper-style update: w_{t+1} = w_t - lr * (v_{t+1} + wd * w_t)
        update = buf
        if weight_decay != 0:
            update = update.add(current_param, alpha=weight_decay)

        param.add_(update, alpha=-lr)

        # store w_t for next step
        prev_param.copy_(current_param)


class FractionalSGDAdaptiveMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        fractional_alpha: float = 0.999,
        delta: float = 1e-8,
        momentum_schedule: Optional[Callable[[int], float]] = compute_momentum,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (0.0 <= fractional_alpha < 1.0):
            raise ValueError(f"fractional_alpha must be in [0, 1), got {fractional_alpha}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            fractional_alpha=fractional_alpha,
            delta=delta,
            momentum_schedule=momentum_schedule,
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

            # one step per param group, not max over params
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            current_step = group["step"]

            if group["momentum_schedule"] is not None:
                momentum_t = float(group["momentum_schedule"](current_step))
            else:
                momentum_t = float(group["momentum"])

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

            fractional_sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                prev_param_list,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum_t,
                fractional_alpha=fractional_alpha,
                delta=delta,
            )

            for p, buf, prev_p in zip(params_with_grad, momentum_buffer_list, prev_param_list):
                state = self.state[p]
                state["momentum_buffer"] = buf
                state["prev_param"] = prev_p

        return loss