import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def gen_weekday_num(day: int) -> int:
    if day == "Monday":
        return 1
    elif day == "Tuesday":
        return 2
    elif day == "Wednesday":
        return 3
    elif day == "Thursday":
        return 4
    elif day == "Friday":
        return 5
    elif day == "Saturday":
        return 6
    elif day == "Sunday":
        return 7


def ts_sales_plot(df: pd.DataFrame, col_y_name: str) -> None:
    sns.lineplot(x=df.index, y=col_y_name, data=df)
    plt.xlabel("date")
    plt.ylabel(col_y_name)
    item_id = str(df["item_id"].unique()[0])
    plt.title(item_id)
    plt.show()
