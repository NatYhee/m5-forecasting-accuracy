import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller


def adf_test_autolag(timeseries:pd.Series, lag:int=None) -> None:
    print("Results of Dickey-Fuller Test:")

    if lag is None:
        dftest = adfuller(timeseries, autolag="AIC")
    else:
        dftest = adfuller(timeseries, lag=lag)

    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = f"{value:.6f}"
    print(dfoutput)
    


def gen_weekday_num(day:int) -> int:
    if day == 'Monday':
        return 1
    elif day == 'Tuesday':
        return 2
    elif day == 'Wednesday':
        return 3
    elif day == 'Thursday':
        return 4
    elif day == 'Friday':
        return 5
    elif day == 'Saturday':
        return 6
    elif day == 'Sunday':
        return 7


def ts_sales_plot(df:pd.DataFrame, col_y_name:str) -> None:
    sns.lineplot(x=df.index, y=col_y_name, data=df)
    plt.xlabel("date")
    plt.ylabel(col_y_name)
    item_id = str(df['item_id'].unique()[0])
    plt.title(item_id)
    plt.show()