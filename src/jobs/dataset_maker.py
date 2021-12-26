import argparse
import os
import pandas as pd

from src.utils.import_downcasting import import_downcasting


def make_dataset(
    raw_data_dir: str, export_dir: str, main_file: str, store_id: str = None
):
    os.makedirs(export_dir, exist_ok=True)

    sales_df = import_downcasting(os.path.join(raw_data_dir, main_file))
    sales_df = filter_store(sales_df, store_id)

    calendar_df = import_downcasting(os.path.join(raw_data_dir, "calendar.csv"))
    price_df = import_downcasting(os.path.join(raw_data_dir, "sell_prices.csv"))

    merged_df = merge_dataset(sales_df, calendar_df, price_df)
    merged_df.to_csv(os.path.join(export_dir, "dataset.csv"), index=False)


def filter_store(sales_df: pd.DataFrame, store_id: str):
    if store_id == None:
        return sales_df
    else:
        return sales_df[sales_df.store_id == store_id]


def merge_dataset(
    sales_df: pd.DataFrame, calendar_df: pd.DataFrame, price_df: pd.DataFrame
):
    sales_df = pd.melt(
        sales_df,
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="date_code",
        value_name="sales",
    )
    sales_df = pd.merge(
        left=sales_df, right=calendar_df, left_on="date_code", right_on="d", how="left"
    )
    sales_df = pd.merge(
        left=sales_df,
        right=price_df,
        left_on=["store_id", "item_id", "wm_yr_wk"],
        right_on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    return sales_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="assets/data/raw")
    parser.add_argument("--export-dir", default="assets/data/data_CA_1")
    parser.add_argument("--main-file", default="sales_train_evaluation.csv")
    parser.add_argument("--store-id", default="CA_1")
    args = parser.parse_args()

    make_dataset(args.raw_data_dir, args.export_dir, args.main_file, args.store_id)
