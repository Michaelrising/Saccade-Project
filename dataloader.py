import polars as pl
import os
import tqdm
import numpy as np
from src.utils import preprocessor

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


class DataLoader():
    def __init__(self, path='./data', fields=None, derivatives=None):
        self.data_path = path
        self.features = fields
        self.derivatives = derivatives
        self.datamap = {}
        self.market_datamap ={}
        self.processor = preprocessor
        self.stocks = None
        self.load_data(path)

    def load_raw_data(self):
        return pl.read_ipc(os.path.join(self.data_path, 'raw_data.arrow'))

    def get_features(self):
        return pl.read_ipc(os.path.join(self.data_path, 'factors.arrow'))

    def get_derivatives(self):
        return pl.read_ipc(os.path.join(self.data_path, 'derivatives.arrow'))

    def load_data(self, path='./data'):

        self.data_path = path
        self.datamap['raw_data'] = self.load_raw_data()
        self.datamap['features'] = self.get_features()
        self.datamap['derivatives'] = self.get_derivatives()
        self.stocks = sorted(self.datamap['raw_data']['Stock'].unique().to_list())

        return self.datamap

    def load_dataset_to_map(self, horizon: int = 5):

        for stock in tqdm.tqdm(self.stocks):

            print(f"Loading {stock}...")
            # in_raw_data = self.datamap['raw_data'].filter(pl.col('Stock') == stock)

            in_derivatives = self.datamap['derivatives'].filter(pl.col('Stock') == stock)

            in_features = self.datamap['features'].filter(pl.col('Stock') == stock)
            # join all the data
            data = in_derivatives.join(in_features, on=set(in_derivatives.columns).intersection(set(in_features.columns)), how='inner')
            # preprocess data

            # # get the target
            target = self.get_target(data[['Date', 'Minutes', 'open', 'close', 'vwap', 'last_mid', 'twap_mid']],
                                     groupby_date=True, horizon=horizon)

            data = data.join(target, on=set(data.columns).intersection(set(target.columns)), how='inner')

            feature_lists = data.drop('Date', 'Minutes', 'Stock').columns

            processor = self.processor(data)
            processor.fillnull(feature_lists=feature_lists)
            data = processor.winsorize(feature_lists=feature_lists)

            data = data.with_columns((pl.col('Date') + pl.lit(' ') + pl.col('Minutes')).alias('Time'))

            self.market_datamap[stock] = data

        return self.market_datamap

    def get_target(self, price_data, groupby_date=True, horizon:int=5):
        # get the target
        horizon += 1
        # we cant trade when we observe the close price or the vwap or any price that we already observed
        # so we use the next open price as the base price
        if groupby_date:
            price_data = price_data.with_columns([(pl.col('close').log().shift(-horizon) - pl.col('open').log().shift(-1)).over('Date').alias('log_close'),
                                                  (pl.col('open').log().shift(-horizon) - pl.col('open').log().shift(-1)).over('Date').alias('log_open'),
                                                  (pl.col('vwap').log().shift(-horizon) - pl.col('vwap').log().shift(-1)).over('Date').alias('log_vwap'),
                                                  (pl.col('last_mid').log().shift(-horizon) - pl.col('open').log().shift(-1)).over('Date').alias('log_mid'),
                                                  (pl.col('twap_mid').log().shift(-horizon) - pl.col('twap_mid').log().shift(-1)).over('Date').alias('log_twap_mid')])
        else:
            pass
        return price_data.drop(['open', 'close', 'vwap', 'last_mid', 'twap_mid'])

    def save_dataset(self, path='./data/Stocks'):
        os.makedirs(path, exist_ok=True)
        for stock in self.stocks:
            self.market_datamap[stock].write_ipc(os.path.join(path, f"{stock}.arrow"))
        # return self.market_datamap


loader = DataLoader()
loader.load_dataset_to_map()
loader.save_dataset()
