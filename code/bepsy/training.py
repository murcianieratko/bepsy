import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import OscillatorForecast, masked_mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model: OscillatorForecast, loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total = 0.0
    count = 0
    for values, mask, target, target_mask, delta in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = model(values, mask, delta)
        loss = masked_mse(prediction, target, target_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        torch.autograd.backward(loss)
        optimizer.step()
        total += loss.item() * values.shape[0]
        count += values.shape[0]
    if count == 0:
        raise ValueError('empty dataloader')
    return total / count


def save_state(path: Path, model: OscillatorForecast, optimizer: torch.optim.Optimizer, seed: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.state-', suffix='.tmp')
    os.close(descriptor)
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'seed': seed, 'epoch': epoch,
                    'torch_rng': torch.get_rng_state()}, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_state(path: Path, model: OscillatorForecast, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    return int(state['seed']), int(state['epoch'])
