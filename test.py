import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from src.utils import test_train_split
import tqdm
import tabulate

import os
os.getcwd()


def test_labels(features, m='RF'):
        data = pl.scan_ipc("./data/Stocks/*.arrow").collect()
        if m == 'RF':
                model = RandomForestRegressor(max_depth=4, random_state=42, n_jobs=20)
        elif m == 'ridge':
                model = Ridge(alpha=0.5)
        else:
                model = LinearRegression()

        targets = ['log_vwap', 'log_close', 'log_open', 'log_mid', 'log_twap_mid']

        # Loop through each target label
        res = [['Label', 'mse', 'rmse', 'r2', 'in-corr', 'out-corr']]
        for label in tqdm.tqdm(targets):

                X_train, y_train, X_test, y_test, train_info, test_info = test_train_split(data, label, features)

                # Model training
                model.fit(X_train, y_train.reshape(-1))

                # Predictions and evaluation
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test.reshape(-1), predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)

                train_info = train_info.insert_at_idx(train_info.shape[1], pl.Series(label+'_hat', model.predict(X_train)))
                train_info = train_info.with_columns(pl.corr(label, label+'_hat').over('Stock', 'Date').alias('corr'))

                test_info = test_info.insert_at_idx(test_info.shape[1], pl.Series(label+'_hat', model.predict(X_test)))
                test_info = test_info.with_columns(pl.corr(label, label+'_hat').over('Stock', 'Date').alias('corr'))

                res.append([label, round(mse, 4), round(rmse, 4), round(r2, 4),
                            round(train_info['corr'].mean(), 4),
                            round(test_info['corr'].mean(), 4)])

        # Store results
        print(tabulate.tabulate(res))
        return res

