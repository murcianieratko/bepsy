import torch
from torch import Tensor, nn

from .oscillators import TrainableOscillators


class OscillatorForecast(nn.Module):
    def __init__(self, modalities: int = 9, compartments: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.compartments = compartments
        self.encoder = nn.Sequential(nn.Linear(2 * modalities, hidden), nn.Tanh(), nn.Linear(hidden, 2 * compartments))
        self.flow = TrainableOscillators(compartments)
        self.decoder = nn.Sequential(nn.Linear(2 * compartments, hidden), nn.Tanh(), nn.Linear(hidden, modalities))

    def forward(self, values: Tensor, mask: Tensor, delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        observed = torch.where(mask, values, torch.zeros_like(values))
        features = torch.cat((observed, mask.to(values.dtype)), dim=-1)
        state = self.encoder(features).reshape(*values.shape[:-1], self.compartments, 2)
        predicted_state = self.flow(state, delta)
        return self.decoder(predicted_state.flatten(start_dim=-2)), state, predicted_state


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    if count.item() == 0:
        raise ValueError('no observed target values')
    return ((prediction - target).square() * mask.to(prediction.dtype)).sum() / count
