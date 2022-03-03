import sys

sys.path.append(f"/home/npanj/personal_works/m5-forecasting-accuracy")


import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import copy
from src.utils.plotting import PdfFile


def get_wmape(data: pd.DataFrame, resid_column: str) -> pd.DataFrame:
    """
    Function to get WMAPE
    """

    data["abs_residual"] = data[resid_column].abs()
    sum_abs_residual = (
        data[["store_id", "item_id", "abs_residual"]]
        .groupby(["store_id", "item_id"])
        .abs_residual.sum()
        .reset_index()
    )
    sum_actual = (
        data[["store_id", "item_id", "sales"]]
        .groupby(["store_id", "item_id"])
        .sales.sum()
        .reset_index()
    )
    sum_abs_residual.rename(columns={"abs_residual": "sum_abs_residual"}, inplace=True)
    wmape_df = pd.merge(
        left=sum_abs_residual, right=sum_actual, on=["store_id", "item_id"], how="left"
    )
    wmape_df["wmape"] = wmape_df["sum_abs_residual"] / wmape_df["sales"]

    return wmape_df


def plotting_sales_forecast(
    data: pd.DataFrame, score: pd.DataFrame, file_name: str, product_cat: str = None
):
    """
    Function to plot prediction and actual
    """

    pd.options.mode.chained_assignment = None
    data_df = data.copy()
    score_df = score.copy()

    if os.path.exists(file_name):
        os.remove(file_name)

    if product_cat is not None:
        data_df = data_df.loc[data_df.cat_id == product_cat]
        score_df = score_df[score_df.item_id.isin(data_df.item_id.tolist())]
        score_df.sort_values(by="wmape", ascending=False, inplace=True)

    item_ids = list(score_df.item_id.unique())

    pdf = PdfFile(file_name)
    for item_id in item_ids:
        df = data_df.loc[data_df.item_id == item_id]
        eval_df = score_df.loc[score_df.item_id == item_id]

        store_id = str(df.store_id.unique()[0])
        item_id = str(df.item_id.unique()[0])
        wmape = str(eval_df.wmape.unique()[0])

        title = f"graph of store_id:{store_id} on item_id:{item_id} with WMAPE:{wmape}"
        pdf.save_fig(df, title)

    pdf.close()


def get_wmape_custom_groupby(data: pd.DataFrame, resid_column: str, groupby_keys: list):
    """
    Function to get WMAPE wigh customize in groupby key
    """

    data["abs_residual"] = data[resid_column].abs()
    residual_columns = copy.deepcopy(groupby_keys)
    residual_columns.append("abs_residual")

    actual_columns = copy.deepcopy(groupby_keys)
    actual_columns.append("sales")

    sum_abs_residual = data[residual_columns].groupby(groupby_keys).agg({"abs_residual": ["sum", "mean", "count"]})
    sum_abs_residual.columns = ["sum_abs_residual", "mean_abs_residual", "num_obs"]
    sum_abs_residual = sum_abs_residual.reset_index()

    sum_actual = data[actual_columns].groupby(groupby_keys).sales.sum().reset_index()
    wmape_df = pd.merge(
        left=sum_abs_residual, right=sum_actual, on=groupby_keys, how="left"
    )
    wmape_df["wmape"] = wmape_df["sum_abs_residual"] / wmape_df["sales"]

    return wmape_df
