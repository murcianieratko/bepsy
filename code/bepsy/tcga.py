import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .hmp2 import open_text


@dataclass(frozen=True)
class GeneCounts:
    gene_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    counts: NDArray[np.int64]
    strand: str


def read_star_counts(path: Path, strand: str) -> GeneCounts:
    allowed = {'unstranded', 'stranded_first', 'stranded_second'}
    if strand not in allowed:
        raise ValueError('choose an available GDC STAR count column explicitly')
    with open_text(path) as handle:
        lines = (line for line in handle if not line.startswith('#'))
        reader = csv.DictReader(lines, delimiter='\t')
        if not {'gene_id', 'gene_name', strand}.issubset(reader.fieldnames or []):
            raise ValueError('not a GDC STAR gene count table')
        ids = []
        seen: set[str] = set()
        names = []
        values = []
        for row in reader:
            gene = row['gene_id']
            if gene.startswith('N_'):
                continue
            if not gene or gene in seen:
                raise ValueError('empty or duplicate gene identifier')
            value = int(row[strand])
            if value < 0:
                raise ValueError('negative gene count')
            seen.add(gene)
            ids.append(gene)
            names.append(row['gene_name'])
            values.append(value)
    if not ids:
        raise ValueError('no gene counts')
    return GeneCounts(tuple(ids), tuple(names), np.asarray(values, dtype=np.int64), strand)
