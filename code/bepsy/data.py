import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Trajectory:
    subject: str
    times: Tensor
    values: Tensor
    mask: Tensor


def read_longitudinal_csv(path: Path, modalities: int = 9) -> list[Trajectory]:
    columns = [f'modality_{index}' for index in range(modalities)]
    groups: dict[str, list[tuple[float, list[float], list[bool]]]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'subject', 'time', *columns}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('missing subject, time, or modality columns')
        for row in reader:
            subject = row['subject'].strip()
            if not subject:
                raise ValueError('empty subject identifier')
            values = []
            mask = []
            for name in columns:
                raw = row[name].strip()
                present = raw != ''
                mask.append(present)
                values.append(float(raw) if present else 0.0)
            groups.setdefault(subject, []).append((float(row['time']), values, mask))
    trajectories = []
    for subject, records in groups.items():
        records.sort(key=lambda record: record[0])
        times = torch.tensor([record[0] for record in records], dtype=torch.float32)
        value_tensor = torch.tensor([record[1] for record in records], dtype=torch.float32)
        mask_tensor = torch.tensor([record[2] for record in records], dtype=torch.bool)
        if len(records) < 2 or not torch.all(times[1:] > times[:-1]):
            raise ValueError('each subject needs at least two distinct observation times')
        if not torch.isfinite(times).all() or not torch.isfinite(value_tensor).all():
            raise ValueError('observations must be finite')
        trajectories.append(Trajectory(subject, times, value_tensor, mask_tensor))
    if not trajectories:
        raise ValueError('no observations')
    return trajectories


class ConsecutivePairs(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, trajectories: list[Trajectory]) -> None:
        self.pairs = [(trajectory, index) for trajectory in trajectories for index in range(len(trajectory.times) - 1)]
        if not self.pairs:
            raise ValueError('no observation pairs')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        trajectory, position = self.pairs[index]
        return (
            trajectory.values[position], trajectory.mask[position],
            trajectory.values[position + 1], trajectory.mask[position + 1],
            trajectory.times[position + 1] - trajectory.times[position],
        )
