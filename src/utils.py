import polars as pl
import numpy as np
import os
import tqdm

class CustomSet:
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = set()

    def add(self, item):
        self.data.add(item)
        if len(self.data) >= self.max_len:
            smallest = min(self.data)
            self.data.remove(smallest)

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return str(self.data)

    def to_list(self):
        return list(self.data)


def test_train_split(df: pl.DataFrame, label, features=None, r: float = 0.8, skip: int = 1, folds=5):
    days = df['Date'].unique().sort().to_list()
    fold_size = len(days) // folds
    features = df.columns if features is None else features
    for i in range(folds):
        this_fold = days[i * fold_size: (i + 1) * fold_size]
        train_ = this_fold[:int(len(this_fold)*r)-skip]
        test_ = this_fold[int(len(this_fold)*r):]
        train_data = df.filter(pl.col('Date').is_in(train_))
        test_data = df.filter(pl.col('Date').is_in(test_))

        drops = ['Stock', 'Date', 'Minutes', 'Time', 'log_open', 'log_close',  'trade_mask',
                 'lift_mask', 'hit_mask', 'log_vwap', 'log_mid', 'log_twap_mid']
        # print(train_data.schema)

        train_X = train_data.select(features).cast(pl.Float32).drop(drops).to_numpy()

        train_y = train_data[label].to_numpy()

        test_X = test_data.select(features).cast(pl.Float32).drop(drops).to_numpy()
        test_y = test_data[label].to_numpy()

        train_info = train_data[['Stock', 'Date', 'Minutes', label]]
        test_info = test_data[['Stock', 'Date', 'Minutes', label]]

        return train_X, train_y, test_X, test_y, train_info, test_info



def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)

def get_data(path='sample_data'):
    all_files = os.listdir(path)

    all_data = []
    all_trading_days = [file.split('_')[-1].split('.')[0] for file in all_files if file.endswith('.csv')]
    # get the unique trading days
    all_trading_days = np.unique(all_trading_days)

    columns = ['Date', 'Minutes', 'Stock', 'open', 'high', 'low', 'close',
               'volume', 'num_trade', 'last_bid', 'last_ask', 'bid_twap',
               'num_lift', 'lift_vwap', 'lift_volume', 'num_hit', 'vwap',
               'ask_twap', 'hit_vwap', 'hit_volume']

    for day in tqdm.tqdm(all_trading_days):
        # get all files for that day
        files = [file for file in all_files if file.endswith('.csv') and file.split('_')[-1].split('.')[0] == day]
        # read all files
        dfs = pl.DataFrame()

        for file in files:
            fields = file.split('_')
            item = '_'.join(fields[:-1])
            schema = {'Minutes': str}
            for i in range(1, 101):
                schema['Stock' + str(i)] = float
            df = pl.read_csv(os.path.join(path, file), dtypes=schema)
            df = df.melt(id_vars='Minutes', variable_name='Stock', value_name=item).sort('Minutes')

            if len(dfs) == 0:
                dfs = df
            else:
                dfs = dfs.join(df, on=['Minutes', 'Stock'], how='inner')
        dfs = dfs.insert_at_idx(0, pl.Series('Date', [day] * len(dfs)))
        all_data.append(dfs.select(columns))

    all_data = pl.concat(all_data)

    # we add three masks here denoting that the conditions we could buy or sell
    # 0: no trade
    # 1: 1-3 trades
    # 2: 4+ trades
    all_data = all_data.with_columns(pl.when(pl.col('num_trade') == 0).
                                         then(pl.lit(0)).
                                         when(pl.col('num_trade').is_between(1, 3)).
                                         then(pl.lit(1)).
                                         otherwise(pl.lit(2)).alias('trade_mask'))
    all_data = all_data.with_columns(pl.when(pl.col('num_hit') == 0).
                                        then(pl.lit(0)).
                                        when(pl.col('num_hit').is_between(1, 3)).
                                        then(pl.lit(1)).
                                        otherwise(pl.lit(2)).alias('lift_mask'))
    all_data = all_data.with_columns(pl.when(pl.col('num_lift') == 0).
                                        then(pl.lit(0)).
                                        when(pl.col('num_lift').is_between(1, 3)).
                                        then(pl.lit(1)).
                                        otherwise(pl.lit(2)).alias('hit_mask'))

    return all_data


class preprocessor():
    def __init__(self, feature_data:pl.DataFrame):
        self.feature_data = feature_data

    def fillnull(self, feature_lists=None, method='forward'):
        if not feature_lists:
            feature_lists = self.feature_data.columns

        assert method in ['forward', '0'], "Invalid method specified"

        self.feature_data = self._fillnull(feature_lists, method)
        return self.feature_data

    def _fillnull(self, feature_lists, method):
        if method == 'forward':

            self.feature_data = self.feature_data.fill_nan(pl.lit(None)).with_columns(pl.col(feature_lists).forward_fill().over('Date')).drop_nulls()

        elif method == '0':
            self.feature_data = self.feature_data.with_columns(pl.all().fill_null(0))
        else:
            raise ValueError("Invalid method specified")
        return self.feature_data

    def winsorize(self,feature_lists=None, method='quantile', lower=0.01, upper=0.99, n=3):
        if not feature_lists:
            feature_lists = self.feature_data.columns

        assert method in ['quantile', 'gaussian', 'iqr', 'mad'], "Invalid method specified"

        self.feature_data = self._winsorize(feature_lists, method, lower, upper, n)
        return self.feature_data

    def _winsorize(self, feature_lists, method, lower, upper, n=3):
        data = self.feature_data[feature_lists]
        if method == 'quantile':
            lower_limit = data.quantile(lower, interpolation='lower')
            upper_limit = data.quantile(upper, interpolation='higher')
        elif method == 'gaussian':
            mean = data.mean()
            std = data.std(ddof=0)
            upper_limit = mean + n * std
            lower_limit = mean - n * std
        elif method == 'iqr':
            quantiler_ = []
            for i in [0.25, 0.75]:
                quantiler_.append(data.quantile(i, interpolation='linear'))
            iqr = quantiler_[1] - quantiler_[0]
            upper_limit = quantiler_[1] + (iqr * n)
            lower_limit = quantiler_[0] - (iqr * n)
        elif method == 'mad':
            median = data.median()
            mad = data.select((pl.all() .sub(pl.all().median())).abs().median())
            upper_limit = median + n * mad
            lower_limit = median - n * mad
        else:
            raise ValueError("Invalid method specified")
        self.feature_data = self.feature_data.with_columns(*[pl.col(i).clip(lower_bound=lower_limit[0, i],
                                                                            upper_bound=upper_limit[0, i])
                                                             for i in feature_lists])
        # self.feature_data[feature_lists] = data.with_columns(pl.all().clip(lower_limit, upper_limit))
        return self.feature_data
