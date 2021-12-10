import pandas as pd

from statsmodels.tsa.stattools import adfuller


class searchStationarySeriesADF:
    """
    Searching for Integrated Order, different order, the transform time series data from non stationary
    to stationay if the current state of time series data is non stationary by using Augmented Dickey Fuller ("ADF") test
    """

    def __init__(
        self,
        timeseries: pd.Series,
        max_diff_order: int = 10,
        adf_test_lag: int = None,
        adf_alpha: float = 0.05,
        adf_regression: str = "ct",
    ) -> None:
        """
        Args:
            timeseries (pd.Series): time series data
            max_diff_order (int): maximum number of diff order
            adf_test_lag (int): maximum lag for using to test ADF (Default: Use Auto Lag)
            adf_alpha (float): alpha threshold for testing statistical significant
            adf_refression (str): how we contruct regression model to perform ADF (Default: With Trend and Constant)
        """
        self._timeseries = timeseries
        self._max_diff_order = max_diff_order
        self._adf_test_lag = adf_test_lag
        self._adf_alpha = adf_alpha
        self._adf_regression = adf_regression

    def get_diff_order_stationary(self) -> int:
        """
        Searching for integrated orders by checking which order transform time series data from non stationary
        to stationary

        Returns:
            int: required diff order to create stationary series
        """
        stationary = False

        for diff_order in range(self._max_diff_order + 1):

            if diff_order == 0:
                stationary = self._is_stationary(self._timeseries)
            else:
                stationary = self._is_stationary(
                    self._timeseries.diff(diff_order).dropna()
                )

            if stationary:
                return diff_order

        if not stationary:
            return -999

    def _is_stationary(self, timeseries: pd.Series) -> bool:
        """
        Referring from Augmented Dicky Fuller test to determine whether the series is stationary ot not
        Args:
            timeseries (pd.Series): time series data

        Returns:
            bool: boolean value represent stationary or not
        """
        adfuller_result = self.test_adfuller(timeseries)

        if adfuller_result["p-value"] < self._adf_alpha:
            return True
        else:
            return False

    def test_adfuller(self, timeseries: pd.Series) -> dict:
        """
        Referring from Augmented Dicky Fuller test to determine whether the series is stationary ot not
        Args:
            timeseries (pd.Series): time series data

        Returns:
            dict: dictionary contain information from Augmented Dicky Fuller test
        """
        if self._adf_test_lag is None:
            dftest = adfuller(
                timeseries, autolag="AIC", regression=self._adf_regression
            )
        else:
            dftest = adfuller(
                timeseries,
                maxlag=self._adf_test_lag,
                regression=self._adf_regression,
            )

        adfuller_properties = [
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
            "critical vakyes",
            "icbest",
            "resstore",
        ]

        adfuller_result = dict(zip(adfuller_properties, list(dftest)))

        return adfuller_result
