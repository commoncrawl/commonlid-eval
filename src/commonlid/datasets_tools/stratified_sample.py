"""Class-balanced stratified sampler.

Byte-equivalent port of the research repo's
``generate_nano_datasets.generate_small_version``: pick at least ``min_size``
samples per class (dropping classes below that), then stratified-sample the
remaining pool so the non-minimum slice is proportional to each class's
remaining size, rounded to integers. The total output size is approximately
``max_size + min_size * n_classes``.

Reproducibility detail — the original used the *global* ``random`` module
throughout, with ``pop_n`` re-seeding the global RNG on each call so the
per-class min-pop is deterministic per call, while the outer stratified
sample then continues from whatever residual state the last ``pop_n`` left
behind. To stay byte-equivalent without polluting the global module, we
allocate a single ``random.Random`` instance and re-seed it inside ``_pop_n``
before each per-class min selection. ``None`` labels are kept as a distinct
class (matching the original); upstream callers may filter them beforehand
if undesired.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampledIndices:
    """Result of :func:`stratified_sample_with_minimum_per_class`."""

    selected: list[int]
    dropped_classes: list[str | None]


def _pop_n(
    items: list[int],
    n: int,
    *,
    seed: int,
    rng: random.Random,
) -> list[int]:
    """Reseed ``rng`` and pop ``n`` random indices off ``items`` (in-place).

    Mirrors the original ``generate_nano_datasets.pop_n``: each call resets
    the shared RNG to ``seed`` so the per-class min selection is independent
    of iteration order. Residual entropy (after ``rng.sample``) carries over
    into the outer stratified-sample step.
    """
    rng.seed(seed)
    if n > len(items):
        msg = "n larger than list length"
        raise ValueError(msg)
    idxs = rng.sample(range(len(items)), n)
    idxs.sort(reverse=True)
    picked: list[int] = []
    for i in idxs:
        picked.append(items[i])
        del items[i]
    return picked


def stratified_sample_with_minimum_per_class(
    labels: list[str | None],
    *,
    max_size: int = 1000,
    min_size: int = 5,
    seed: int = 42,
) -> SampledIndices:
    """Stratified sample of ``labels`` with a minimum quota per class.

    Returns the selected indices (shuffled) and the classes dropped because
    they had fewer than ``min_size`` members. ``None`` is treated as a
    distinct class — pre-filter the input if you want to drop it.
    """
    rng = random.Random()
    rng.seed(seed)

    label_to_idxs: dict[str | None, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_idxs[label].append(idx)

    dropped = sorted(
        (lbl for lbl, idxs in label_to_idxs.items() if len(idxs) < min_size),
        key=lambda lbl: (lbl is None, lbl or ""),
    )
    label_to_idxs = {lbl: idxs for lbl, idxs in label_to_idxs.items() if len(idxs) >= min_size}

    min_label_to_idxs: dict[str | None, list[int]] = {}
    remaining_label_to_idxs: dict[str | None, list[int]] = {}
    for label, idxs in label_to_idxs.items():
        idxs = copy.deepcopy(idxs)
        min_label_to_idxs[label] = _pop_n(idxs, min_size, seed=seed, rng=rng)
        remaining_label_to_idxs[label] = idxs

    total_remaining = sum(len(idxs) for idxs in remaining_label_to_idxs.values())

    selected_remaining: list[int] = []
    selected_min: list[int] = []
    for label, idxs in remaining_label_to_idxs.items():
        target = round(max_size * (len(idxs) / total_remaining)) if total_remaining > 0 else 0
        selected_remaining.extend(rng.sample(idxs, target))
        selected_min.extend(min_label_to_idxs[label])

    combined = selected_remaining + selected_min
    rng.shuffle(combined)
    return SampledIndices(selected=combined, dropped_classes=dropped)
