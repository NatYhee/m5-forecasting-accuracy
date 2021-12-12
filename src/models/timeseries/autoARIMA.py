import pandas as pd
import os
import tqdm

from unitrootTest import searchStationarySeriesADF
from src.utils.utils import load_json, save_json
from sktime.forecasting.arima import AutoARIMA


class autoARIMA:
    def __init__(self, asset_dir: str, **params) -> None:
        self._config = load_json(asset_dir, "config.json")

    def __call__(self):
        pass

    @staticmethod
    def train(asset_dir: str, data_dir: str, **params):
        """Creates asset_dir and saves config.json with optimal ARIMA orders.

        Args:
            asset_dir (str): Directory to be created to save model assets
            data_dir (str): Directory with file name of raw data
            **params: The training keyword arguments parameters.
        """

        config = {
            "classname": "autoARUNA",
            "asset_dir": asset_dir,
            "data_dir": data_dir,
            "ARIMA_orders": {},
        }

        os.makedirs(asset_dir, exist_ok=True)
        save_json(asset_dir)

        data = pd.read_csv(data_dir)
        data["revenue"] = data["revenue"] = data["sales"] * data["sell_price"]

        store_ids = autoARIMA._get_store_ids(data)

        for store_id in tqdm(store_ids):
            data_store = data[data.store_id == store_id]
            item_ids = autoARIMA._get_store_ids(data_store)

            for item_id in tqdm(item_ids, leave=False):
                data_store_item = data_store_item[data_store_item.item_id == item_id]
                integrated_order = searchStationarySeriesADF(data_store_item['sales'])
                
                model = AutoARIMA(
                        start_p=1,
                        start_q=1,
                        max_p=30,
                        max_q=30
                        d=integrated_order,
                        max_d=integrated_order,
                        seasonal=False,
                        start_P=0,
                        D=0,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        alpha=0.05
                      )

    @staticmethod
    def _get_store_ids(data: pd.DataFrame):

        # sort store_ids from largest to smallest revenue
        total_store_revenue = data.groupby("store_id")["revenue"].sum()
        store_ids = total_store_revenue.sort_values(ascending=False).index

        return store_ids

    @staticmethod
    def _get_item_ids(data: pd.DataFrame):

        total_item_revenue = data.groupby("item_id")["revenue"].sum()
        item_ids = total_item_revenue.sort_values(ascending=False).index

        return item_ids
