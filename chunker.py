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
    """Возвращает чанки DataFrame по сериям одинаковых значений в `col` (например dt)

    Гарантии
    - Одинаковое значение даты никогда не окажется в разных чанках
    - Каждый чанк имеет размер >= chunk_size кроме возможно последнего
    - Память расходуется экономно без groupby и без построчных python циклов, используются срезы iloc

    Примечания
    - Если assume_sorted=True то требуется чтобы df[col] был монотонно неубывающим
    - Если assume_sorted=False то при необходимости будет выполнена стабильная сортировка по col что потребует дополнительной памяти
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
