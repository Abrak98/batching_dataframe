import pandas as pd
import pytest

from batching_module import get_batch_generator_from_df


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 25, 500])
@pytest.mark.parametrize("data_multiplier", [1, 2, 25, 1500])
def test_data_equality(batch_size, data_multiplier):
    input_df = pd.DataFrame(
        {
            "dt": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"]
            * data_multiplier,
            "value": [1, 2, 3, 4, 5] * data_multiplier,
        }
    )

    df_generator = get_batch_generator_from_df(
        input_df.copy(), batch_size=batch_size, batch_column="dt"
    )
    result_batches = list(df_generator)
    result_df = pd.concat(result_batches)

    assert len(input_df) == len(result_df)
    assert input_df["value"].sum() == result_df["value"].sum()
    assert input_df.columns.tolist() == result_df.columns.tolist()


@pytest.mark.parametrize("batch_size", [2, 3, 4, 5, 10])
def test_one_batch(batch_size):
    input_df = pd.DataFrame(
        {
            "dt": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
                "2023-01-02",
                "2023-01-02",
                "2023-01-02",
            ],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    df_generator = get_batch_generator_from_df(
        input_df.copy(), batch_size=batch_size, batch_column="dt"
    )
    result_batches = list(df_generator)

    assert len(result_batches) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
def test_two_batches(batch_size):
    input_df = pd.DataFrame(
        {
            "dt": [
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
            ],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    df_generator = get_batch_generator_from_df(
        input_df.copy(), batch_size=batch_size, batch_column="dt"
    )
    result_batches = list(df_generator)

    assert len(result_batches) == 2


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 10, 100])
@pytest.mark.parametrize("data_multiplier", [1, 2, 5, 25, 500])
def test_batch_size_bigger_or_equal(batch_size, data_multiplier):
    input_df = pd.DataFrame(
        {
            "dt": [
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
                "2023-01-03",
            ]
            * data_multiplier,
            "value": [1, 2, 3, 4, 5, 6] * data_multiplier,
        }
    )

    df_generator = get_batch_generator_from_df(
        input_df.copy(), batch_size=batch_size, batch_column="dt"
    )
    result_batches = list(df_generator)
    bad_batches = []

    for batch_df in result_batches[:-1]:
        if len(batch_df) < batch_size:
            bad_batches.append(len(batch_df))

    assert len(bad_batches) == 0


@pytest.mark.parametrize("batch_size", [1, 2, 3, 10])
def test_sorting(batch_size):
    input_df = pd.DataFrame(
        {"date": ["2023-01-03", "2023-01-01", "2023-01-02"], "value": [3, 1, 2]}
    )
    df_generator = get_batch_generator_from_df(
        input_df.copy(), batch_size=batch_size, batch_column="date"
    )
    result_batches = list(df_generator)
    assert result_batches[0]["date"].iloc[0] == "2023-01-01"
    assert result_batches[-1]["date"].iloc[-1] == "2023-01-03"


@pytest.mark.parametrize("batch_size", [1, 10])
def test_empty_df(batch_size):
    empty_df = pd.DataFrame(columns=["dt", "value"])
    df_generator = get_batch_generator_from_df(
        empty_df, batch_size=batch_size, batch_column="dt"
    )
    result_batches = list(df_generator)

    assert result_batches[0].empty is True
    assert len(result_batches) == 1
