from __future__ import annotations

from typing import Iterator

import pandas as pd


def iter_dt_chunks(
    df: pd.DataFrame,
    col: str = "dt",
    chunk_size: int = 1000,
    *,
    assume_sorted: bool = True,
) -> Iterator[pd.DataFrame]:
    """Возвращает чанки DataFrame по сериям одинаковых значений в `col` например dt

    Гарантии
    - Одинаковое значение даты никогда не окажется в разных чанках
    - Каждый чанк имеет размер >= chunk_size кроме возможно последнего
    - Память расходуется оптимально по памяти O(1) доп памяти, обход делается одним проходом по значениям

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

    # Берем numpy представление без копирования когда это возможно
    v = s.to_numpy(copy=False)

    if n == 1:
        yield df
        return

    start = 0
    run_start = 0
    acc = 0

    prev = v[0]

    for i in range(1, n):
        cur = v[i]
        if cur != prev:
            run_end = i
            run_len = run_end - run_start
            acc += run_len

            if acc >= chunk_size:
                yield df.iloc[start:run_end]
                start = run_end
                acc = 0

            run_start = run_end
            prev = cur

    # Закрываем последнюю серию и отдаем остаток
    if start < n:
        yield df.iloc[start:n]
