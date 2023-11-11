import polars as pl
import polars.selectors as cs
import numpy as np
import os
import tqdm
from src.utils import preprocessor


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
        return pl.read_ipc(os.path.join(self.data_path, 'features.arrow'))

    def get_derivatives(self):
        return pl.read_ipc(os.path.join(self.data_path, 'derivatives.arrow'))

    def load_data(self, path='./data'):

        self.data_path = path
        self.datamap['raw_data'] = self.load_raw_data()
        self.datamap['features'] = self.get_features()
        self.datamap['derivatives'] = self.get_derivatives()
        self.stocks = self.datamap['raw_data'].select('Stock').unique().to_list()

        return self.datamap

    def load_dataset_to_map(self, horizon: int = 5):

        for stock in self.stocks:

            print(f"Loading {stock}...")
            in_raw_data = self.datamap['raw_data'].filter(pl.col('Stock') == stock)

            in_derivatives = self.datamap['derivatives'].filter(pl.col('Stock') == stock)
            in_data = in_raw_data.join(in_derivatives, on=['Date', 'Minutes', 'Stock'], how='inner')

            in_features = self.datamap['features'].filter(pl.col('Stock') == stock)

            # preprocess data
            processor = self.processor(in_features)
            feature_lists = in_features.drop('Date', 'Minutes', 'Stock').columns
            in_features = processor.winsorize(feature_lists=feature_lists)

            assert len(in_data) == len(in_raw_data) == len(in_features) == len(in_derivatives), "Data length mismatch"

            # join all the data
            data = in_data.join(in_features, on=['Date', 'Minutes', 'Stock'], how='inner')

            # get the target
            target = self.get_target(in_data[['Date', 'Minutes', 'open', 'close', 'vwap', 'last_mid']],
                                     groupby_date=True, horizon=horizon)
            data = data.join(target, on=['Date', 'Minutes'], how='inner')
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
        return price_data

    def save_dataset(self, path='./data/processed_data'):
        os.makedirs(path, exist_ok=True)
        for stock in self.stocks:
            self.market_datamap[stock].write_ipc(os.path.join(path, f"{stock}.arrow"))
        # return self.market_datamap


