"""Script to execute ARIMA from trained with AutoARIMA
"""
import os
import argparse
import pandas as pd

from src.models.timeseries.autoARIMA import autoARIMA

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default="assets/data/data_CA_1")
    args = parser.parse_args()

    ts_model = autoARIMA(args.assets_dir)
    ts_model()
