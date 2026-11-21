import math

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .hmp2 import ModalityTable, centered_log_ratio
from .model import OscillatorForecast


class HMP2MetagenomicPairs(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, table: ModalityTable, allowed_subjects: set[str], pseudocount: float,
                 days_per_model_unit: float, max_gap_days: float) -> None:
        if table.modality != 'metagenomics':
            raise ValueError('expected HMP2 metagenomics')
        if not allowed_subjects:
            raise ValueError('an explicit subject set is required')
        if not math.isfinite(days_per_model_unit) or days_per_model_unit <= 0:
            raise ValueError('days per model unit must be positive')
        if not math.isfinite(max_gap_days) or max_gap_days <= 0:
            raise ValueError('maximum gap must be positive')
        self.features = table.features
        self.values = torch.as_tensor(centered_log_ratio(table.values, pseudocount), dtype=torch.float32)
        self.days_per_model_unit = days_per_model_unit
        by_subject: dict[str, list[int]] = {}
        for index, sample in enumerate(table.samples):
            if sample.subject in allowed_subjects:
                by_subject.setdefault(sample.subject, []).append(index)
        self.pairs: list[tuple[int, int, float]] = []
        for indices in by_subject.values():
            indices.sort(key=lambda index: table.samples[index].week)
            for left, right in zip(indices, indices[1:]):
                days = 7 * (table.samples[right].week - table.samples[left].week)
                if 0 < days <= max_gap_days:
                    self.pairs.append((left, right, days))
        if not self.pairs:
            raise ValueError('no eligible consecutive sample pairs')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        left, right, days = self.pairs[index]
        first = self.values[left]
        second = self.values[right]
        mask = torch.ones_like(first, dtype=torch.bool)
        return first, mask, second, mask.clone(), torch.tensor(days / self.days_per_model_unit, dtype=torch.float32)


def model_for_metagenomic_prototype(table: ModalityTable, hidden: int = 64, compartments: int = 12) -> OscillatorForecast:
    if table.modality != 'metagenomics' or not table.features:
        raise ValueError('a nonempty metagenomics table is required')
    return OscillatorForecast(modalities=len(table.features), hidden=hidden, compartments=compartments)


def assert_subject_disjoint(*groups: set[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        if not group:
            raise ValueError('subject sets must be nonempty')
        if seen.intersection(group):
            raise ValueError('a subject occurs in more than one partition')
        seen.update(group)
