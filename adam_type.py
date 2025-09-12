from typing import List

import torch
from torch.optim import Optimizer


class RMSpropAdaptiveMomentum(Optimizer):
    def __init__(self, params, lr=1e-1, alpha=0.99, eps=1, weight_decay=0, momentum=0, centered=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super(RMSpropAdaptiveMomentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is None: continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(self.state[p]["square_avg"])
                # group['momentum'] *= 0.999999
                if group['momentum'] > 0:
                    momentum_buffer_list.append(self.state[p]['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(self.state[p]['grad_avg'])

                state["step"] += 1
            rmsprop(params_with_grad,
                    grads,
                    square_avgs,
                    grad_avgs,
                    momentum_buffer_list,
                    lr=group["lr"],
                    alpha=group["alpha"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    momentum=compute_momentum(state["step"]),
                    centered=group['centered'])

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


def rmsprop(params: List[torch.Tensor],
            grads: List[torch.Tensor],
            square_avgs: List[torch.Tensor],
            grad_avgs: List[torch.Tensor],
            momentum_buffer_list: List[torch.Tensor],
            *,
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool):

    for i, param in enumerate(params):
        grad = grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        grad_norm = grad.norm(2)
        square_avg = square_avgs[i]
        square_avg.mul_(alpha).addcmul_(grad_norm, grad_norm, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)


def compute_momentum(iter):
    b = 0.9
    a = 1000
    if iter <= 10000:
        return b * iter/ 10000
    elif iter > 10000 and iter <= 15000:
        return b - 2*(0.9 * 0.999999**(15*a) - b) + iter/(5*a)*(0.9*0.999999**(15*a) - b)
    else:
        return 0.9 * 0.999999**iter