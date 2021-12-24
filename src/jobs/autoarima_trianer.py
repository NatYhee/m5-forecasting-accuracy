"""Script to train auto arima
"""
import os

from src.models.timeseries.autoARIMA import autoARIMA

if __name__ == "__main__":
    root = "assets/data"
    path = os.path.join(root, "sales_ca1_melted.csv")
    model = autoARIMA.train(root, path)
