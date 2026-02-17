from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def iter_dt_chunks(
    df: pd.DataFrame,
    col: str = "dt",
    chunk_size: int = 1000,
    *,
    assume_sorted: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield chunks split by runs of equal values in `col` (e.g. dt).

    Guarantees:
    - Never splits the same datetime value across chunks.
    - Each yielded chunk has len >= chunk_size, except possibly the last chunk.
    - Memory-friendly: no groupby, no per-row python loops; slices by iloc.

    Notes:
    - If `assume_sorted=True`, requires `df[col]` to be monotonic increasing.
    - If `assume_sorted=False`, will stably sort by `col` (extra memory).
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    n = len(df)
    if n == 0:
        return
        yield  # pragma: no cover

    s = df[col]

    if assume_sorted:
        if not s.is_monotonic_increasing:
            raise ValueError(f"Column {col!r} must be sorted (monotonic increasing)")
    else:
        if not s.is_monotonic_increasing:
            df = df.sort_values(col, kind="mergesort")
            s = df[col]

    v = s.to_numpy()
    if n == 1:
        yield df
        return

    change_idx = np.flatnonzero(v[1:] != v[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [n]))
    run_sizes = np.diff(boundaries)

    start = 0
    acc = 0
    for i, run_len in enumerate(run_sizes, start=1):
        acc += int(run_len)
        if acc >= chunk_size:
            end = int(boundaries[i])
            yield df.iloc[start:end]
            start = end
            acc = 0

    if start < n:
        yield df.iloc[start:n]
