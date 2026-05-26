"""Dataset loading and splitting helpers used by HGBO-DSE training scripts.

``generate_dataset`` walks a directory of per-benchmark ``*.pt`` PyG sample
shards and concatenates them. ``split_dataset`` produces a deterministic
random 80/20 split; ``leave_one_board_out_split`` produces a board-stratified
split for ATAPP-style cross-FPGA generalisation experiments.

Loss helpers (:func:`msle_loss`, :func:`mape_loss`, :func:`mae_loss`) are kept
here for backward compatibility with the original HGBO-DSE single-target
training scripts.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import numpy as np
import random
import torch
from torch.utils.data import random_split


log = logging.getLogger(__name__)


def msle_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared log error. Both tensors should be non-negative."""
    output = torch.log(output + 1)
    target = torch.log(target + 1)
    return torch.mean(torch.square(output - target))


def mape_loss(output: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Mean absolute percentage error.

    ``eps`` is added to ``|target|`` in the denominator so that targets of zero
    do not produce NaN. The original HGBO-DSE implementation used a bare
    division; this guarded form is backwards-compatible for non-zero targets
    while keeping training stable when an occasional label is zero.
    """
    return torch.mean(torch.abs((target - output) / (target.abs() + eps)))


def mae_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(target - output))


def generate_dataset(
    dataset_dir: str | os.PathLike[str],
    dataset_name_list: Iterable[str],
    print_info: bool = False,
) -> list:
    """Load and concatenate ``*.pt`` PyG sample shards in ``dataset_dir``.

    Non-``.pt`` names in ``dataset_name_list`` are skipped; remaining names are
    sorted for deterministic ordering before loading.
    """
    dataset_list: list = []
    for ds in sorted(name for name in dataset_name_list if name.endswith(".pt")):
        ds_path = os.path.join(dataset_dir, ds)
        if os.path.isfile(ds_path):
            tem_data = torch.load(ds_path, weights_only=False)
            dataset_list = dataset_list + tem_data
            if print_info:
                log.info("loaded %s", ds_path)
    return dataset_list


def generate_multi_board_dataset(
    dataset_root: str | os.PathLike[str],
    boards: Iterable[str],
    split_subdir: str,
    *,
    print_info: bool = False,
) -> list:
    """Load and concatenate per-board PyG sample shards.

    Layout expected on disk (matches :func:`hgp.data_process.gen_dataset_board.augment_dataset_dirs`):

        ``<dataset_root>/<board>/<split_subdir>/*.pt``

    where ``board`` is the canonical device key (as registered in
    :data:`hgp.board_utils._BOARD_PROFILES`) and ``split_subdir`` is e.g.
    ``"std_arch"`` or ``"rdc_arch"``. Returns the flat list of samples; each
    sample carries the ``board_device`` field added during augmentation.

    Skips boards whose directory does not exist, logging a warning. Raises
    ``RuntimeError`` if no samples were loaded.
    """
    root = os.fspath(dataset_root)
    combined: list = []
    boards = list(boards)
    for board in boards:
        board_dir = os.path.join(root, board, split_subdir)
        if not os.path.isdir(board_dir):
            log.warning("board dataset directory missing: %s", board_dir)
            continue
        shard_names = sorted(name for name in os.listdir(board_dir) if name.endswith(".pt"))
        for shard in shard_names:
            shard_path = os.path.join(board_dir, shard)
            samples = torch.load(shard_path, weights_only=False)
            combined.extend(samples)
            if print_info:
                log.info("loaded %s (%d samples)", shard_path, len(samples))
    if not combined:
        raise RuntimeError(
            "No samples loaded under {!r} for boards={} split={!r}".format(
                root, boards, split_subdir
            )
        )
    return combined


def _log_first_ten(all_list: Sequence, label: str) -> None:
    sample_ys = [getattr(item, "y", None) for item in all_list[:10]]
    log.debug("first ten %s graph Y: %s", label, sample_ys)


def split_dataset(all_list, shuffle: bool = True, seed: int | None = 6666):
    """Shuffle ``all_list`` and produce an 80/20 train/test split.

    The shuffle (when enabled) and the subsequent :func:`torch.utils.data.random_split`
    both use ``seed`` for reproducibility — previously the inner split was
    hard-coded to ``42``, which caused the same train/test partition regardless
    of the user-supplied ``--seed``.
    """
    if not isinstance(all_list, list):
        all_list = list(all_list)

    _log_first_ten(all_list, "train (before shuffle)")

    if shuffle:
        if seed is not None:
            np.random.RandomState(seed=seed).shuffle(all_list)
        else:
            random.shuffle(all_list)
        log.debug("split_dataset seed = %s", seed)

    _log_first_ten(all_list, "train (after shuffle)")

    split_seed = 42 if seed is None else int(seed)
    generator = torch.Generator().manual_seed(split_seed)
    n_total = len(all_list)
    n_train = round(0.8 * n_total)
    n_test = n_total - n_train
    train_ds, test_ds = random_split(all_list, [n_train, n_test], generator=generator)
    return train_ds, test_ds


def _sample_board_device(sample) -> str | None:
    """Best-effort extractor for the ``board_device`` attribute on a sample."""
    if hasattr(sample, "board_device"):
        value = sample.board_device
        if value is not None:
            return str(value)
    return None


def leave_one_board_out_split(
    all_list,
    *,
    train_boards: Iterable[str] | None,
    test_board: str,
    val_fraction: float = 0.1,
    seed: int = 128,
):
    """Split samples by ``board_device`` for cross-FPGA generalisation.

    Samples tagged with a device in ``train_boards`` form the training set; a
    ``val_fraction`` slice of them is reserved as the in-distribution validation
    set. All samples tagged with ``test_board`` form the out-of-distribution
    test set. If ``train_boards`` is ``None``, every device other than
    ``test_board`` is used for training.

    The shuffle used to carve out the validation set is seeded with ``seed`` so
    the partition is reproducible.

    Returns ``(train, val, test)`` lists. Raises ``ValueError`` if the test set
    is empty (no samples for that board) or the training set is empty.
    """
    if not isinstance(all_list, list):
        all_list = list(all_list)

    train_set = None if train_boards is None else {str(board) for board in train_boards}
    train_samples: list = []
    test_samples: list = []
    unknown_samples = 0

    for sample in all_list:
        board = _sample_board_device(sample)
        if board is None:
            unknown_samples += 1
            continue
        if board == test_board:
            test_samples.append(sample)
        elif train_set is None or board in train_set:
            train_samples.append(sample)

    if unknown_samples:
        log.warning(
            "leave_one_board_out_split: dropped %d sample(s) with no board_device tag",
            unknown_samples,
        )

    if not test_samples:
        raise ValueError("No samples tagged with board_device={!r}".format(test_board))
    if not train_samples:
        raise ValueError(
            "No samples remain for training after holding out board_device={!r}".format(test_board)
        )

    rng = np.random.RandomState(seed=seed)
    rng.shuffle(train_samples)
    n_val = int(round(max(0.0, min(val_fraction, 1.0)) * len(train_samples)))
    val_samples = train_samples[:n_val]
    train_samples = train_samples[n_val:]
    return train_samples, val_samples, test_samples
