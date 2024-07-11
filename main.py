import pandas as pd

from batching_module import get_batch_generator_from_df


def main():
    dfs_1 = pd.date_range("2023-01-01 00:00:01", "2023-01-01 00:00:15", freq="s")
    dfs_2 = pd.date_range("2023-01-01 00:00:10", "2023-01-01 00:00:25", freq="s")
    dfs_3 = pd.date_range("2023-01-01 00:00:15", "2023-01-01 00:00:30", freq="s")
    data = pd.concat(
        [
            pd.DataFrame({"dt": dfs_1.repeat(1)}),
            pd.DataFrame({"dt": dfs_2.repeat(1)}),
            pd.DataFrame({"dt": dfs_3.repeat(3)}),
        ]
    )

    batch_size = 5
    df_generator = get_batch_generator_from_df(data, batch_size, batch_column="dt")

    for batch_number, batch_df in enumerate(df_generator):
        print(
            f"\nBatch number {batch_number}. batch_df size = {len(batch_df)}:\n{batch_df}\n"
        )


if __name__ == "__main__":
    main()
