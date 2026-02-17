import pandas as pd
import pytest

from chunker import iter_dt_chunks


def make_df(vals):
    return pd.DataFrame({"dt": pd.to_datetime(vals)})


def test_example_like_splitting():
    df = make_df(
        [
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:03",
        ]
    )

    chunks = list(iter_dt_chunks(df, "dt", chunk_size=2))

    # последний чанк может быть меньше chunk_size
    assert [len(c) for c in chunks] == [2, 3, 1]
    assert pd.concat(chunks, ignore_index=True).equals(df.reset_index(drop=True))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
def test_no_dt_overlap_between_chunks(chunk_size):
    df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=5, freq="s").repeat(3)})
    chunks = list(iter_dt_chunks(df, "dt", chunk_size=chunk_size))

    seen = set()
    for ch in chunks:
        dts = set(ch["dt"].unique())
        assert seen.isdisjoint(dts)
        seen |= dts


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
def test_all_except_last_are_at_least_chunk_size(chunk_size):
    df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=10, freq="s").repeat(3)})
    chunks = list(iter_dt_chunks(df, "dt", chunk_size=chunk_size))

    for ch in chunks[:-1]:
        assert len(ch) >= chunk_size

    assert pd.concat(chunks, ignore_index=True).equals(df.reset_index(drop=True))


def test_unsorted_raises_by_default():
    df = make_df(["2023-01-01 00:00:02", "2023-01-01 00:00:01"])
    with pytest.raises(ValueError):
        list(iter_dt_chunks(df, "dt", chunk_size=1, assume_sorted=True))


def test_empty_df():
    df = pd.DataFrame({"dt": pd.to_datetime([])})
    assert list(iter_dt_chunks(df, "dt", chunk_size=3)) == []
