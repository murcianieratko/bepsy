from typing import cast

import torch
from torch import Tensor, nn


def canonical_symplectic_form(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype, device=device)


def anchored_energy(state: Tensor) -> Tensor:
    if state.shape[-1] != 4:
        raise ValueError('expected radial-phase state with four components')
    immune_radius, _, stromal_radius, _ = state.unbind(dim=-1)
    return 0.5 * (immune_radius.square() + stromal_radius.square()) - immune_radius.square() * stromal_radius.square()


def dissipative_field(gradient: Tensor) -> Tensor:
    return cast(Tensor, -gradient * torch.linalg.vector_norm(gradient, dim=-1, keepdim=True))


def hamiltonian_field(gradient: Tensor) -> Tensor:
    matrix = canonical_symplectic_form(gradient.dtype, gradient.device)
    return torch.einsum('ij,...j->...i', matrix, gradient)


class PseudoHamiltonian(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.energy_correction = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, 1))
        self.dissipation = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, 1))
        self.energy_scale = nn.Parameter(torch.tensor(0.0))

    def energy(self, state: Tensor) -> Tensor:
        return cast(Tensor, anchored_energy(state) + self.energy_scale * self.energy_correction(state).squeeze(-1))

    def forward(self, state: Tensor, coupling_term: Tensor) -> Tensor:
        if state.shape != coupling_term.shape:
            raise ValueError('coupling term and state shapes must match')
        with torch.enable_grad():
            state_for_gradient = state if state.requires_grad else state.detach().requires_grad_(True)
            energy = self.energy(state_for_gradient)
            potential = self.dissipation(state_for_gradient).squeeze(-1)
            grad_energy = torch.autograd.grad(energy.sum(), state_for_gradient, create_graph=True)[0]
            grad_potential = torch.autograd.grad(potential.sum(), state_for_gradient, create_graph=True)[0]
            return hamiltonian_field(grad_energy) + dissipative_field(grad_potential) + coupling_term
