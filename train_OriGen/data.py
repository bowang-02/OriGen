"""Dynamic single-plane augmentation for multi-sequence training cases."""

from __future__ import annotations

import random
from typing import Dict

from custom_model.multi_sequence_fusion import AXIAL_SEQUENCE_IDS, SAGITTAL_SEQUENCE_IDS
from train_fusion_only.data import MultiSequenceMRIDataset, MultiSequenceMRITrainingCollator


def sample_plane_view(record: Dict[str, object], axial_only_prob: float, sagittal_only_prob: float) -> Dict[str, object]:
    """Sample a complete plane view using the requested probabilities."""
    if not (record["has_axial"] and record["has_sagittal"]):
        return record
    draw = random.random()
    if draw < axial_only_prob:
        keep_ids = AXIAL_SEQUENCE_IDS
        has_axial, has_sagittal = True, False
    elif draw < axial_only_prob + sagittal_only_prob:
        keep_ids = SAGITTAL_SEQUENCE_IDS
        has_axial, has_sagittal = False, True
    else:
        return record
    selected = [
        (path, sequence_id)
        for path, sequence_id in zip(record["nifti"], record["sequence_type_ids"])
        if sequence_id in keep_ids
    ]
    augmented = dict(record)
    augmented["nifti"] = [path for path, _ in selected]
    augmented["sequence_type_ids"] = [sequence_id for _, sequence_id in selected]
    augmented["has_axial"] = has_axial
    augmented["has_sagittal"] = has_sagittal
    return augmented


class OrientationSamplingDataset(MultiSequenceMRIDataset):
    def __init__(self, jsonl_path: str, axial_only_prob: float = 0.1, sagittal_only_prob: float = 0.1):
        if axial_only_prob < 0 or sagittal_only_prob < 0 or axial_only_prob + sagittal_only_prob > 1:
            raise ValueError("Single-plane probabilities must be nonnegative and sum to at most one.")
        super().__init__(jsonl_path)
        self.axial_only_prob = axial_only_prob
        self.sagittal_only_prob = sagittal_only_prob

    def __getitem__(self, index: int) -> Dict[str, object]:
        return sample_plane_view(super().__getitem__(index), self.axial_only_prob, self.sagittal_only_prob)


