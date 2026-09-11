"""Subject-level train/test split (all ROI rows of a subject stay together)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_indices_by_subject(subject_ids, y, test_size, seed):
    subj_max: dict = {}
    for s, yi in zip(subject_ids, y):
        yi = int(yi)
        if s not in subj_max or yi > subj_max[s]:
            subj_max[s] = yi
    uniq = np.array(sorted(subj_max.keys(), key=str), dtype=object)
    labels = np.array([subj_max[s] for s in uniq], dtype=np.int64)
    n_subj = len(uniq)
    c0, c1 = int((labels == 0).sum()), int((labels == 1).sum())
    if n_subj < 4 or c0 < 2 or c1 < 2:
        raise ValueError(
            "Held-out split needs ≥4 subjects and ≥2 pass and ≥2 fail subjects "
            "(have n={0}, pass={1}, fail={2}).".format(n_subj, c0, c1)
        )
    idx = np.arange(n_subj)
    strat = labels
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strat
    )
    subj_train = {str(uniq[i]) for i in idx_train}
    subj_test = {str(uniq[i]) for i in idx_test}
    subj_arr = np.asarray(subject_ids, dtype=object)
    train_mask = np.array([str(s) in subj_train for s in subj_arr], dtype=bool)
    test_mask = np.array([str(s) in subj_test for s in subj_arr], dtype=bool)
    return train_mask, test_mask, subj_train, subj_test
