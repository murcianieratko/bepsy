from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OscillatorParameters:
    mu: Tensor
    omega: Tensor
    lyapunov: Tensor
    coupling: Tensor


def stuart_landau_rhs(state: Tensor, parameters: OscillatorParameters) -> Tensor:
    if state.shape[-1] != 2:
        raise ValueError('state must end in real and imaginary coordinates')
    n = state.shape[-2]
    if parameters.coupling.shape[-2:] != (n, n):
        raise ValueError('coupling must have shape (..., oscillators, oscillators)')
    x, y = state.unbind(dim=-1)
    radius_sq = x.square() + y.square()
    growth = parameters.mu - parameters.lyapunov * radius_sq
    dx = growth * x - parameters.omega * y
    dy = growth * y + parameters.omega * x
    differences = state.unsqueeze(-3) - state.unsqueeze(-2)
    coupling = (parameters.coupling.unsqueeze(-1) * differences).sum(dim=-2)
    return torch.stack((dx, dy), dim=-1) + coupling


def coupling_rhs(phase: Tensor, epsilon_1: float, epsilon_2: float) -> Tensor:
    return epsilon_1 * torch.sin(phase.unsqueeze(-1) - phase.unsqueeze(-2) + epsilon_2)


def hopf_radial_rhs(radius: Tensor, alpha: Tensor, beta: Tensor, beta_star: Tensor, lyapunov: Tensor) -> Tensor:
    return alpha * (beta - beta_star) * radius - lyapunov * radius.pow(3)


def rk4_step(state: Tensor, dt: Tensor, parameters: OscillatorParameters) -> Tensor:
    scale = dt
    while scale.ndim < state.ndim:
        scale = scale.unsqueeze(-1)
    k1 = stuart_landau_rhs(state, parameters)
    k2 = stuart_landau_rhs(state + scale * k1 / 2, parameters)
    k3 = stuart_landau_rhs(state + scale * k2 / 2, parameters)
    k4 = stuart_landau_rhs(state + scale * k3, parameters)
    return state + scale * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class TrainableOscillators(nn.Module):
    off_diagonal: Tensor

    def __init__(self, compartments: int) -> None:
        super().__init__()
        if compartments < 1:
            raise ValueError('at least one compartment is required')
        self.mu = nn.Parameter(torch.zeros(compartments))
        self.omega = nn.Parameter(torch.ones(compartments))
        self.raw_lyapunov = nn.Parameter(torch.zeros(compartments))
        self.raw_coupling = nn.Parameter(torch.zeros(compartments, compartments))
        self.register_buffer('off_diagonal', 1 - torch.eye(compartments))

    def parameters_for_flow(self) -> OscillatorParameters:
        return OscillatorParameters(
            self.mu,
            self.omega,
            torch.nn.functional.softplus(self.raw_lyapunov),
            self.raw_coupling * self.off_diagonal,
        )

    def forward(self, state: Tensor, dt: Tensor) -> Tensor:
        return rk4_step(state, dt, self.parameters_for_flow())
