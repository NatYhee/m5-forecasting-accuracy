import pandas as pd
import os
from tqdm import tqdm
import ast

from src.models.timeseries.unitrootTest import searchStationarySeriesADF
from src.utils.utils import (
    load_json,
    save_json,
    convert_tuple_to_str,
    convert_str_to_tuple,
)
from sktime.forecasting.arima import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path


class autoARIMA:
    def __init__(self, asset_dir: str, **params) -> None:
        self._config = load_json(asset_dir, "auto-arima-config.json")

    def __call__(self):
        """Using optimal ARIMA order from training to create data file with residual from
        model generated from optimal ARIMA order for studying qualitative factors for LGBM.
        """

        data = pd.read_csv(self._config["data_dir"])
        data.set_index("date", inplace=True)
        data.index = pd.DatetimeIndex(data.index).to_period("D")

        arima_score = {"store_id": [], "item_id": [], "mse": [], "mae": [], "sse": []}

        store_ids = list(self._config["ARIMA_orders"].keys())

        for store_id in tqdm(store_ids):
            data_store = data[data.store_id == store_id]
            item_ids = list(self._config["ARIMA_orders"][store_id].keys())

            for item_id in tqdm(item_ids, leave=False):
                data_store_id = data_store[data_store.item_id == item_id]

                fitted_params = ast.literal_eval(
                    self._config["ARIMA_orders"][store_id][item_id]
                )
                arima_order = fitted_params["order"]

                if "intercept" in fitted_params.keys():
                    model = ARIMA(
                        endog=data_store_id["sales"], order=arima_order, trend="c"
                    )
                else:
                    model = ARIMA(endog=data_store_id["sales"], order=arima_order)

                model_result = model.fit()

                data_store_id = data_store_id.assign(prediction=model_result.predict())
                data_store_id = data_store_id.assign(arima_residual=model_result.resid)
                data = data.loc[
                    ~((data.store_id == store_id) & (data.item_id == item_id))
                ]
                data = data.append(data_store_id)

                arima_score["store_id"].append(store_id)
                arima_score["item_id"].append(item_id)
                arima_score["mse"].append(model_result.mse)
                arima_score["mae"].append(model_result.mae)
                arima_score["sse"].append(model_result.sse)

        data["abs_residual"] = data.arima_residual.abs()
        path = Path(self._config["asset_dir"])
        data.to_csv(os.path.join(path, "data_with_arima_resid.csv"))

        wampe_df = self._gat_wmape(data, "abs_residual")
        arima_score_df = pd.DataFrame(arima_score)
        arima_score_df = pd.merge(
            left=arima_score_df, right=wampe_df, on=["store_id, sku_id"], how="left"
        )
        arima_score_df.to_csv(os.path.join(path, "arima_model_score.csv"), index=False)

    @staticmethod
    def train(data_dir: str, asset_dir: str, **params):
        """Creates asset_dir and saves config.json with optimal ARIMA orders.

        Args:
            asset_dir (str): Directory to be created to save model assets.
            data_dir (str): Directory with file name of raw data.
            **params: The training keyword arguments parameters.
        """

        data = pd.read_csv(data_dir)
        data.set_index("date")
        data["revenue"] = data["revenue"] = data["sales"] * data["sell_price"]

        store_ids = autoARIMA._get_store_ids(data)
        results = {}

        for store_id in tqdm(store_ids):
            data_store = data[data.store_id == store_id]
            item_ids = autoARIMA._get_item_ids(data_store)
            results[str(store_id)] = {}

            for item_id in tqdm(item_ids, leave=False):

                print(f"training store:{str(store_id)}, item:{str(item_id)}")

                # For handling multiple trainings
                if os.path.isfile(os.path.join(asset_dir, "auto-arima-config.json")):
                    temp_config = load_json(asset_dir, "auto-arima-config.json")
                    results = temp_config["ARIMA_orders"]

                    if str(item_id) in temp_config["ARIMA_orders"][store_id].keys():
                        print(
                            f"stroe: {str(store_id)}, item:{str(item_id)} already trained"
                        )
                        continue

                data_store_item = data_store[data_store.item_id == item_id]
                adf_test = searchStationarySeriesADF(data_store_item["sales"])
                integrated_order = adf_test.get_diff_order_stationary()

                fitted_params = autoARIMA._perform_auto_arima(
                    ts=data_store_item["sales"], diff_order=integrated_order
                )
                results[str(store_id)].update({str(item_id): str(fitted_params)})

                # For handling multiple trainings
                os.makedirs(asset_dir, exist_ok=True)
                config = {
                    "classname": "autoARUNA",
                    "asset_dir": asset_dir,
                    "data_dir": data_dir,
                    "ARIMA_orders": results,
                }
                save_json(config, asset_dir, "auto-arima-config.json")

    @staticmethod
    def _get_store_ids(data: pd.DataFrame) -> list:

        # sort store_ids from largest to smallest revenue
        total_store_revenue = data.groupby("store_id")["revenue"].sum()
        store_ids = total_store_revenue.sort_values(ascending=False).index

        return store_ids

    @staticmethod
    def _get_item_ids(data: pd.DataFrame) -> list:

        total_item_revenue = data.groupby("item_id")["revenue"].sum()
        item_ids = total_item_revenue.sort_values(ascending=False).index

        return item_ids

    @staticmethod
    def _perform_auto_arima(ts: pd.Series, diff_order: int) -> dict:

        model = AutoARIMA(
            start_p=1,
            start_q=1,
            max_p=15,
            max_q=15,
            d=diff_order,
            max_d=diff_order,
            seasonal=False,
            start_P=0,
            D=0,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            alpha=0.05,
            with_intercept=False,
        )

        model.fit(ts)
        fitted_params = model.get_fitted_params()

        return fitted_params

    @staticmethod
    def _gat_wmape(data: pd.DataFrame, abs_resid_column: str) -> pd.DataFrame:
        """
        Method to calculate WMAPE, Weighted Mean Average Percentage of Error.

        Args:
            data (pd.DataFrame): Dataframe contain information of prediction and actual sales.
            abs_resid_column (str): Column name of absolute value of prediciton residual.

        Returns:
            pd.DataFrame: Dataframe contain information of WMAPE of each store and sku_id
        """

        sum_abs_residual = (
            data[["store_id", "item_id", abs_resid_column]]
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

        sum_abs_residual.rename(
            columns={"abs_residual": "sum_abs_residual"}, inplace=True
        )

        wmape_df = pd.merge(
            left=sum_abs_residual,
            right=sum_actual,
            on=["store_id", "item_id"],
            how="left",
        )
        wmape_df["wmape"] = wmape_df["sum_abs_residual"] / wmape_df["sales"]

        return wmape_df
