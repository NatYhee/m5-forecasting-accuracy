import pandas as pd

from statsmodels.tsa.stattools import adfuller


class searchStationarySeriesADF:

    def __init__(self, timeseries:pd.series, max_diff_order:int = 10, adf_test_lag:int = None, adf_alpha:float = 0.05, adf_regression:str = "ct" ) -> None:
        self._timeseries = timeseries
        self._max_diff_order = max_diff_order
        self._adf_test_lag = adf_test_lag
        self._adf_alpha = adf_alpha
        self._adf_regression = adf_regression


    def get_diff_order_stationary(self) -> int:
        
        stationary = False

        for diff_order in range(self._max_diff_order + 1):
            if self._is_stationary:
                stationary = True
                return diff_order
        
        if not stationary:
            return -999

    def test_adfuller(self) -> dict:

        if self._adf_lag_test is None:
            dftest = adfuller(self._timeseries, autolag="AIC", regression=self._adf_regression)
        else:
            dftest = adfuller(self._timeseries, maxlag=self._adf_test_lag, regression=self._adf_regression)
        
        adfuller_properties = [
                        "Test Statistic", 
                        "p-value", 
                        "#Lags Used", 
                        "Number of Observations Used",
                        "critical vakyes",
                        "icbest",
                        "resstore"
                    ]
        
        adfuller_result = dict(zip(adfuller_properties, list(dftest)))

        return adfuller_result


    def _is_stationary(self) -> bool:
        
        adfuller_result = self.test_adfuller()

        if adfuller_result["p-value"] < self._adf_alpha:
            return True
        else:
            return False


