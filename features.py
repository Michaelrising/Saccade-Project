import polars as pl
import numpy as np
import talib
from src.utils import get_data, preprocessor
from talib import abstract
import tqdm

# fields = ['BBANDS', 'MIDPOINT', 'SAR', 'T3', 'TRIMA', 'AROONOSC', 'CMO', 'MFI', 'MOM', 'STOCH', 'RSI', 'WILLR', 'ULTOSC',
#           'ADOSC', 'AD', 'OBV', 'TRANGE', 'NATR', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
#           'CDL3BLACKCROWS', '']

class PriceVolumeIndicators:
    def __init__(self, data: pl.DataFrame, fields=None, derivatives=None):
        self.fields = talib.get_functions()  if fields is None else fields
        self.derivatives = ['spread', 'imbalance', 'lift', 'hit', 'mid'] if derivatives is None else derivatives
        self.price_volume = data[['Date', 'Minutes', 'Stock', 'open', 'high', 'low', 'close', 'volume']]
        self.data = data
        self.tab_factors = talib.get_functions()
        self.factors = pl.DataFrame()
        self.extra_factors = pl.DataFrame()

    # get the factors by using tab lib
    def get_technical_factors(self):
        # groupby stock
        for stock, data in tqdm.tqdm(self.price_volume.group_by('Stock')):
            # sort by date and minutes
            data = data.sort(['Date', 'Minutes'])
            processor = preprocessor(data)
            # preprocess data
            price_volume = processor.fillnull(feature_lists=['open', 'high', 'low', 'close', 'volume'])
            # get the date and minutes
            this_stock = price_volume[['Date', 'Minutes', 'Stock']]
            # convert to numpy array for tab lib
            inputs = {'open': price_volume['open'].to_numpy(),
                      'high': price_volume['high'].to_numpy(),
                      'low': price_volume['low'].to_numpy(),
                      'close': price_volume['close'].to_numpy(),
                      'volume': price_volume['volume'].to_numpy()}

            # cal the factors
            for factor in self.fields:
                if factor.upper() in self.tab_factors:
                    if factor.upper() in ['MAVP', 'ASIN', 'ACOS', 'EXP', 'COSH', 'SINH']:
                        continue
                    try:
                        func = getattr(abstract, factor.upper())

                        outputs = func(inputs)

                        # Check if the output is 2D (list of lists)
                        if len(outputs) and isinstance(outputs[0], np.ndarray):
                            # Handle 2D list
                            for i, sublist in enumerate(outputs):
                                series_name = f"{factor}_{i}"
                                if np.isnan(sublist).sum() > 0.3 * len(sublist):
                                    print(f"Error in {factor} all NAN")
                                    continue
                                this_stock = this_stock.insert_at_idx(this_stock.shape[1],
                                                                      pl.Series(series_name, sublist))
                        else:
                            # Handle 1D list
                            series_name = f"{factor}"
                            this_stock = this_stock.insert_at_idx(this_stock.shape[1],
                                                                  pl.Series(series_name, outputs))
                    except Exception:
                        print(f"Error in {factor}")
                        continue
            if len(self.factors) == 0:
                self.factors = this_stock
            else:
                self.factors = pl.concat([self.factors, this_stock])

        return self.factors

    def get_derivatives(self):
        # add vanilla factors
        for derivative in self.derivatives:
            func = getattr(self, 'add_' + derivative)
            func()

        return self.data

    def add_spread(self):
        self.data = self.data.with_columns((pl.col('last_ask') - pl.col('last_bid')).alias('spread'),
                                           ((pl.col('last_ask') - pl.col('last_bid'))/2).alias('half_spread'))

    def add_imbalance(self):
        self.data = self.data.with_columns(
                        (pl.col('hit_volume') / (pl.col('volume')+0.01)).alias('hit_pressure'),
                                (pl.col('lift_volume') / (pl.col('volume')+0.01)).alias('lift_pressure'),
                                (pl.col('hit_volume') / (pl.col('lift_volume')+0.1)).alias('imb_ratio'),
                               )

    def add_lift(self):
        self.data = self.data.with_columns((pl.col('lift_vwap') - pl.col('vwap')).alias('lift_pre'),
                                           (pl.col('num_lift') - (pl.col('num_lift').shift(1))).over('Stock').alias('lift_change'))

    def add_hit(self):
        self.data = self.data.with_columns((pl.col('hit_vwap') - pl.col('vwap')).alias('hit_pre'),
                                           (pl.col('num_hit') - (pl.col('num_hit').shift(1))).over('Stock').alias('hit_change'))

    def add_mid(self):
        self.data = self.data.with_columns(((pl.col('last_ask') + pl.col('last_bid'))/2.).alias('last_mid'),
                               ((pl.col('ask_twap') + pl.col('bid_twap'))/2.).alias('twap_mid'))

    def write_ipc(self, path):
        self.factors.write_ipc(path)
        return self.factors



try:
    data = pl.read_ipc('data/raw_data.arrow')
except FileNotFoundError:
    data = get_data()
    data.write_ipc('data/raw_data.arrow')
indicators = PriceVolumeIndicators(data)
factors = indicators.get_technical_factors()
factors.write_ipc('data/factors.arrow')

derivatives = indicators.get_derivatives()

derivatives.write_ipc('data/derivatives.arrow')