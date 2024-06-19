import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.WarningMessage =False


def main():
    data_set = load_diabetes()
    X = data_set['data']
    y = data_set['target']
    feature_names = data_set['feature_names']
    rf = RandomForestRegressor(max_depth=10  rf.fit(X, y)
    pred = rf.predict(X)
    mse = mean_squared_error(pred, y)
    print(f"mse: {mse:0.3f}")

if __name__ == "__main__":
    """
    Description: explain the module
    """

    main()
    











