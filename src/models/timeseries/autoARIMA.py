import pandas as pd
import os
from tqdm import tqdm

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
        model generated from optimal ARIMA order for studying qualitative factors for LGBM
        """

        data = pd.read_csv(self._config["data_dir"])
        data.set_index("date", inplace=True)
        data.index = pd.DatetimeIndex(data.index).to_period("D")

        store_ids = list(self._config["ARIMA_orders"].keys())

        for store_id in tqdm(store_ids):
            data_store = data[data.store_id == store_id]
            item_ids = list(self._config["ARIMA_orders"][store_id].keys())

            for item_id in tqdm(item_ids, leave=False):
                data_store_id = data_store[data_store.item_id == item_id]
                arima_order = convert_str_to_tuple(
                    self._config["ARIMA_orders"][store_id][item_id]
                )

                model = ARIMA(data_store_id["sales"], order=arima_order)
                data_store_id = data_store_id.assgin(prediction=model.fit().predict)
                data_store_id = data_store_id.assign(arima_residual=model.fit().resid)
                data_store_id = data_store_id.assign(arima_mse=model.fit().mse)
                data_store_id = data_store_id.assign(arima_sse=model.fit().sse)
                data = data.loc[
                    ~((data.store_id == store_id) & (data.item_id == item_id))
                ]
                data = data.append(data_store_id)

        path = Path(self._config["asset_dir"])
        data.to_csv(os.path.join(path, "data_with_arima_resid.csv"))

    @staticmethod
    def train(data_dir: str, asset_dir: str, **params):
        """Creates asset_dir and saves config.json with optimal ARIMA orders.

        Args:
            asset_dir (str): Directory to be created to save model assets
            data_dir (str): Directory with file name of raw data
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

                #For handling multiple trainings
                if os.path.isfile(os.path.join(asset_dir, "auto-arima-config.json")):
                    temp_config = load_json(asset_dir, "auto-arima-config.json")
                    if str(item_id) in temp_config["ARIMA_orders"][store_id].keys():
                        print(
                            f"stroe: {str(store_id)}, item:{str(item_id)} already trained"
                        )
                        continue

                data_store_item = data_store[data_store.item_id == item_id]
                adf_test = searchStationarySeriesADF(data_store_item["sales"])
                integrated_order = adf_test.get_diff_order_stationary()

                arima_order = autoARIMA._perform_auto_arima(
                    ts=data_store_item["sales"], diff_order=integrated_order
                )
                results[str(store_id)].update(
                    {str(item_id): convert_tuple_to_str(arima_order)}
                )

                #For handling multiple trainings
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
    def _perform_auto_arima(ts: pd.Series, diff_order: int) -> tuple:

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
            with_intercept=True
        )

        model.fit(ts)
        arima_order = model.get_fitted_params()["order"]

        return arima_order
