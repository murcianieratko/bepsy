from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .hmp2 import open_text


@dataclass(frozen=True)
class GeoSeries:
    accession: str
    title: str
    samples: tuple[str, ...]


def read_geo_soft(path: Path, expected: str) -> GeoSeries:
    accession = ''
    title = ''
    samples = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith('!Series_geo_accession = '):
                accession = line.split(' = ', 1)[1].strip()
            elif line.startswith('!Series_title = '):
                title = line.split(' = ', 1)[1].strip()
            elif line.startswith('^SAMPLE = '):
                samples.append(line.split(' = ', 1)[1].strip())
    if accession != expected or not title or not samples or len(samples) != len(set(samples)):
        raise ValueError('GEO SOFT series does not match the expected accession or contains invalid metadata')
    return GeoSeries(accession, title, tuple(samples))


def filter_and_normalize_counts(counts: NDArray[np.float64], mitochondrial: NDArray[np.bool_],
                                doublet: NDArray[np.bool_], target_sum: float) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if counts.ndim != 2 or mitochondrial.shape != (counts.shape[1],) or doublet.shape != (counts.shape[0],):
        raise ValueError('expected cells by genes, gene mitochondrial flags and cell doublet flags')
    if not mitochondrial.any():
        raise ValueError('mitochondrial genes were not identified')
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError('counts and target sum must be finite and nonnegative')
    library = counts.sum(axis=1)
    fraction = np.divide(counts[:, mitochondrial].sum(axis=1), library,
                         out=np.ones_like(library), where=library > 0)
    keep = (library > 0) & (fraction < 0.20) & ~doublet
    filtered = counts[keep]
    return np.log1p(filtered * (target_sum / library[keep])[:, None]), keep


def compartment_frequencies(sample_ids: list[str], cell_types: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], NDArray[np.float64]]:
    if len(sample_ids) != len(cell_types) or not sample_ids:
        raise ValueError('sample identifiers and cell labels must have equal nonzero length')
    if any(not sample or not cell_type for sample, cell_type in zip(sample_ids, cell_types)):
        raise ValueError('sample identifiers and cell labels cannot be empty')
    samples = tuple(sorted(set(sample_ids)))
    types = tuple(sorted(set(cell_types)))
    sample_index = {value: index for index, value in enumerate(samples)}
    type_index = {value: index for index, value in enumerate(types)}
    counts = np.zeros((len(samples), len(types)), dtype=np.float64)
    for sample, cell_type in zip(sample_ids, cell_types):
        counts[sample_index[sample], type_index[cell_type]] += 1
    return samples, types, counts / counts.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class ProcessedExpression:
    cells: tuple[str, ...]
    genes: tuple[str, ...]
    expression: NDArray[np.float64]


def read_geo_processed_csv(path: Path, selected_genes: set[str]) -> ProcessedExpression:
    import csv

    if not selected_genes:
        raise ValueError('select at least one gene')
    with open_text(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cells = tuple(header[1:])
        if not cells or not all(cells) or len(set(cells)) != len(cells):
            raise ValueError('cell identifiers must be unique and nonempty')
        features: dict[str, list[float]] = {}
        for row in reader:
            if not row or row[0] not in selected_genes:
                continue
            gene = row[0]
            if gene in features or len(row) != len(header):
                raise ValueError('duplicate gene or inconsistent matrix row')
            features[gene] = [float(value) for value in row[1:]]
        if set(features) != selected_genes:
            raise ValueError('one or more selected genes are missing')
    genes = tuple(sorted(features))
    expression = np.asarray([features[gene] for gene in genes], dtype=np.float64).T
    if not np.isfinite(expression).all():
        raise ValueError('processed expression contains nonfinite values')
    return ProcessedExpression(cells, genes, expression)
