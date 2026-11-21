import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleMetadata:
    sample: str
    subject: str
    week: float
    site: str
    diagnosis: str
    data_type: str


@dataclass(frozen=True)
class ModalityTable:
    modality: str
    features: tuple[str, ...]
    samples: tuple[SampleMetadata, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.samples), len(self.features)):
            raise ValueError('sample and feature dimensions do not match the values')
        if not np.isfinite(self.values).all():
            raise ValueError('feature matrix contains nonfinite values')


def open_text(path: Path) -> TextIO:
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8', newline='')
    return path.open('r', encoding='utf-8', newline='')


def read_metadata(path: Path, data_type: str) -> dict[str, SampleMetadata]:
    result: dict[str, SampleMetadata] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {'External ID', 'Participant ID', 'week_num', 'site_name', 'diagnosis', 'data_type'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('not an HMP2 metadata table')
        for row in reader:
            if row['data_type'] != data_type or not row['week_num'].strip():
                continue
            sample = row['External ID'].strip()
            subject = row['Participant ID'].strip()
            if not sample or not subject:
                raise ValueError('missing sample or subject identifier')
            week = float(row['week_num'])
            if not math.isfinite(week):
                raise ValueError('nonfinite sampling week')
            item = SampleMetadata(sample, subject, week, row['site_name'].strip(), row['diagnosis'].strip(), data_type)
            if sample in result and result[sample] != item:
                raise ValueError('conflicting metadata for a sample')
            result[sample] = item
    if not result:
        raise ValueError('no eligible samples for this modality')
    return result


def read_feature_table(path: Path, metadata: dict[str, SampleMetadata], modality: str,
                       suffix: str = '', feature_marker: str | None = None,
                       exclude_marker: str | None = None) -> ModalityTable:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        if len(header) < 2:
            raise ValueError('feature table has no samples')
        sample_ids = [name.removesuffix(suffix) if suffix else name for name in header[1:]]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError('duplicate sample identifiers in feature table')
        indexes = [i for i, sample in enumerate(sample_ids) if sample in metadata]
        if len(indexes) != len(sample_ids):
            raise ValueError('feature table has samples without usable metadata')
        features: list[str] = []
        seen: set[str] = set()
        feature_rows: list[list[float]] = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError('feature row has an inconsistent number of columns')
            feature = row[0]
            if feature_marker is not None and feature_marker not in feature:
                continue
            if exclude_marker is not None and exclude_marker in feature:
                continue
            if feature in seen:
                raise ValueError('duplicate feature identifier')
            seen.add(feature)
            features.append(feature)
            feature_rows.append([float(row[i + 1]) for i in indexes])
        if not features:
            raise ValueError('no features selected')
    samples = tuple(metadata[sample_ids[i]] for i in indexes)
    values = np.asarray(feature_rows, dtype=np.float64).T
    return ModalityTable(modality, tuple(features), samples, values)


def read_hmp2_metagenomics(metadata_path: Path, profile_path: Path) -> ModalityTable:
    metadata = read_metadata(metadata_path, 'metagenomics')
    return read_feature_table(profile_path, metadata, 'metagenomics', suffix='_profile',
                              feature_marker='|s__', exclude_marker='|t__')


def centered_log_ratio(values: NDArray[np.float64], pseudocount: float) -> NDArray[np.float64]:
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('expected a sample-by-feature matrix')
    if not np.isfinite(values).all() or (values < 0).any() or not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('composition and pseudocount must be finite and nonnegative')
    logged = np.log(values + pseudocount)
    return cast(NDArray[np.float64], logged - logged.mean(axis=1, keepdims=True))


def by_subject(table: ModalityTable) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    indices: dict[str, list[int]] = {}
    for index, sample in enumerate(table.samples):
        indices.setdefault(sample.subject, []).append(index)
    result = {}
    for subject, sample_indices in indices.items():
        ordered = sorted(sample_indices, key=lambda index: table.samples[index].week)
        times_days = np.asarray([table.samples[index].week * 7 for index in ordered], dtype=np.float64)
        result[subject] = (times_days, table.values[ordered])
    return result
