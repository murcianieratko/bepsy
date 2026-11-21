from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Archive:
    accession: str
    actual_tissue: str
    actual_disease: str
    paper_description_matches: bool
    longitudinal: bool


ARCHIVES = {
    'HMP2': Archive('HMP2', 'intestinal and stool samples', 'inflammatory bowel disease and controls', True, True),
    'TCGA-COAD': Archive('TCGA-COAD', 'colon', 'colon adenocarcinoma', True, False),
    'TCGA-READ': Archive('TCGA-READ', 'rectum', 'rectal adenocarcinoma', True, False),
    'GSE116222': Archive('GSE116222', 'colon epithelium', 'ulcerative colitis and controls', True, False),
    'GSE150115': Archive('GSE150115', 'colon lamina propria', 'ulcerative colitis', False, False),
    'GSE181919': Archive('GSE181919', 'head and neck', 'head and neck squamous cell carcinoma', False, False),
}


def require_paper_compatible(accession: str) -> Archive:
    archive = ARCHIVES[accession]
    if not archive.paper_description_matches:
        raise ValueError('archive content does not match the tissue or disease assigned in the paper')
    return archive


def methylation_m_values(beta: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    if not 0 < epsilon < 0.5:
        raise ValueError('epsilon must be strictly between zero and one half')
    if not np.isfinite(beta).all() or (beta < 0).any() or (beta > 1).any():
        raise ValueError('methylation beta values must lie in the unit interval')
    clipped = np.clip(beta, epsilon, 1 - epsilon)
    return np.log2(clipped / (1 - clipped))


def copy_number_log_ratio(copy_number: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    if not np.isfinite(copy_number).all() or not np.isfinite(reference).all():
        raise ValueError('copy-number values must be finite')
    if (copy_number <= 0).any() or (reference <= 0).any():
        raise ValueError('copy-number and reference must be strictly positive')
    return np.log2(copy_number / reference)


def read_gdc_project(path: Path, expected: str) -> Archive:
    import json

    with path.open(encoding='utf-8') as handle:
        response = json.load(handle)
    project = response['data']['project_id']
    if project != expected or project not in ('TCGA-COAD', 'TCGA-READ'):
        raise ValueError('GDC project identifier does not match the selected TCGA cohort')
    return ARCHIVES[project]
