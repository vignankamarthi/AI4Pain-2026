"""Subject-disjoint splits + K-subject subset sampler.

ANTIPATTERNS rule 3: per-subject splits only. No subject in multiple splits.
ANTIPATTERNS rule 19: subset-transfer requires HIP-B before becoming the
                     loop's evaluation set.
"""


def subject_disjoint_split(all_subjects: list[str], n_val: int,
                           seed: int = 0) -> tuple[list[str], list[str]]:
    """Split subjects into (train, val) with no overlap. Stratification by
    label is left to the caller since labels live alongside signals.
    """
    raise NotImplementedError


def k_subject_subset(train_subjects: list[str], k: int,
                     seed: int = 0) -> list[str]:
    """Sample K subjects from train for the inner-loop fitness evaluation.
    Used by framework.eval after HIP-B selects K.
    """
    raise NotImplementedError
