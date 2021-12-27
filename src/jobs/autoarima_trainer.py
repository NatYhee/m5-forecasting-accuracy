"""Script to train auto arima
"""
import os
import argparse
import pandas as pd

from src.models.timeseries.autoARIMA import autoARIMA

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="assets/data/data_CA_1/dataset.csv")
    parser.add_argument("--assets-dir", default="assets/data/data_CA_1")
    args = parser.parse_args()

    agent = autoARIMA()
    agent.train(args.data_dir, args.assets_dir)
