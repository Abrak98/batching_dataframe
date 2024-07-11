from typing import Generator
import pandas as pd


def get_batch_generator_from_df(
    df: pd.DataFrame, batch_size: int, batch_column: str
) -> Generator[pd.DataFrame, None, None]:
    """Splits df into batches of size batch_size or larger.
    Returns batches sorted by batch_column.
    The first one to return is the batch with the lowest value of batch_column.

    :param df: df for splitting into batches.
    :param batch_size: The desired size of the batch.
    :param batch_column: Which column should be divided into batches.
        Rows with the same value in this column will always be in the same batch.
    :return: Batch generator
    """

    df = df.sort_values(batch_column)

    cum_sum, batch_start = 0, 0
    batch_starts = [batch_start]

    for group_size in df.groupby(batch_column).size():
        if cum_sum >= batch_size:
            batch_starts.append(batch_start)
            cum_sum = 0
        cum_sum += group_size
        batch_start += group_size

    batch_starts.append(len(df))

    for start, end in zip(batch_starts, batch_starts[1:]):
        yield df.iloc[start:end]
