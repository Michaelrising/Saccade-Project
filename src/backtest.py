import os
import numpy as np
import polars as pl
from tqdm import tqdm
from multiprocessing import get_context


class Backtest:
    """
    Backtest class is used for reading historical market data,
    executing strategies, simulating trades, and estimating returns.
    Call Backtest.run during initialization to initiate the backtesting process.
    """

    def __init__(self,
                 data: np.array,
                 strategy,
                 broker):
        """
        Construct backtesting object. Required parameters include: historical data,
        strategy object, initial capital, commission rate, etc.
        The initialization process includes checking input types, filling data null values, etc.
        Parameters:
        :param data:           pl.LazyFrame        Historical data in pandas DataFrame format
        :param broker:         type(Broker)        Broker type responsible for executing buy and sell operations
                                                   as well as maintaining account status.
        :param strategy:       type(Strategy)      Strategy Type
        :param cash:           float               Initial funding amount
        :param commission:     float               Transaction fee rate for each transaction. For example, if the fee is
                                                   .2%, then it should be entered as 0.002 here.
        """

        # Sort the market data by time if it is not already sorted.

        # Initialize exchange and strategy objects using data.
        self._data = data
        self._calenders = list(sorted(set(data[:, 2].reshape(-1))))
        self._broker = broker
        self._strategy = strategy
        self._results = None

    def run(self, day:str=None):
        """
        Run backtesting, iterate through historical data, execute simulated trades, and return backtesting results.
        Run the backtest. Returns `pd.Series` with results and statistics.
        Keyword arguments are interpreted as strategy parameters.
        """
        # Set the start and end positions for back testing
        # Back testing main loop, update market status, and execute strategy

        if day:
            self.run_one_day(day)
            return
        else:
            num_workers = 15
            with get_context('spawn').Pool(num_workers) as pool:

                results = list(tqdm(pool.imap(self.run_one_day, self._calenders[3:])))

            self.agg_res(results)
        # return results

    def run_one_day(self, day):

        # Strategy Initialization
        idx = self._calenders.index(day)

        focus_days = self._calenders[max(0, idx - 3):idx + 1]
        market_data = self._data[(self._data[:, 2]<=day) & (self._data[:, 2]>=focus_days[0])]

        broker = self._broker(market_data, cash=10000000, commission=0.0002)
        strategy = self._strategy(broker, risk_manage=False, model='linear')

        strategy.init(day)

        data = self._data[self._data[:, 2] == day]

        ticks = np.unique(data[:, 1].reshape(-1))
        ticks.sort()
        for tick in tqdm(ticks):
            broker.next(tick)
            strategy.next(tick)
            broker.write_ratio(tick)

        broker.close_all_positions()

        result = broker.get_result()

        tbt_values, positions, transaction_his, returns = result.values()

        tbt_values = pl.from_records(tbt_values, orient='row', schema={'Time': str, 'tbt_value': float})
        transaction_history = pl.from_records(transaction_his, orient='row',
                                              schema={'Time': pl.Utf8, 'Stock': pl.Utf8, 'Price': pl.Float32, 'Quantity': pl.Int64, # signed int
                                                      'OrderType': pl.Utf8, 'OrderSide': pl.Utf8, 'Status': pl.Int32})
        returns = pl.from_dict({'Date': day, 'DailyReturn': returns})
        returns = returns.with_columns(pl.col('DailyReturn').cumprod().alias('CumReturn'))
        os.makedirs(f'./results/{day}', exist_ok=True)
        transaction_history.write_ipc(f'./results/{day}/transaction_history.arrow')
        tbt_values.write_ipc(f'./results/{day}/tbt_values.arrow')
        returns.write_ipc(f'./results/{day}/returns.arrow')
        records = []
        for key, value in positions.items():
            if len(value):
                for stock, (quantity, average_price) in value.items():
                    records.append({'Time': key, 'Stock': stock, 'Quantity': quantity, 'AveragePrice': average_price})
        positions = pl.from_records(records, orient='row', schema={'Time': str, 'Stock': pl.Utf8,
                                                                   'Quantity': pl.Int64, 'AveragePrice': pl.Float32})

        positions.write_ipc(f'./results/{day}/positions.arrow')

        return result

    def agg_res(self, results, save=True):
        tbt_values = []
        transaction_history = []
        returns = []
        positions = []
        for res in results:
            tbt_value, position, transaction_his, _return = res.values()
            tbt_values += tbt_value
            transaction_history += transaction_his
            returns.append(_return)
            records = []
            for key, value in positions.items():
                if len(value):
                    for stock, (quantity, average_price) in value.items():
                        records.append({'Time': key, 'Stock': stock, 'Quantity': quantity, 'AveragePrice': average_price})
            positions += records

        tbt_values = pl.from_records(tbt_values, orient='row', schema={'Time': str, 'tbt_value': float})

        transaction_history = pl.from_records(transaction_history, orient='row',
                                              schema={'Time': pl.Utf8, 'Stock': pl.Utf8, 'Price': pl.Float32,
                                                      'Quantity': pl.Int32,
                                                      'OrderType': pl.Utf8, 'OrderSide': pl.Utf8, 'Status': pl.Int32})
        returns = pl.from_dict({'Date': self._calenders[3:], 'DailyReturn': returns})
        returns = returns.with_columns((pl.col('DailyReturn') + pl.lit(1)).cumprod().alias('CumReturn'))
        positions = pl.from_records(positions, orient='row', schema={'Time': str, 'Stock': pl.Utf8,
                                                                   'Quantity': pl.Int32, 'AveragePrice': pl.Float32})

        if save:
            os.makedirs('./results', exist_ok=True)
            transaction_history.write_ipc('./results/transaction_history.arrow')
            tbt_values.write_ipc('./results/tbt_values.arrow')
            returns.write_ipc('./results/returns.arrow')
            positions.write_ipc('./results/positions.arrow')
        # return tbt_values, transaction_history, returns






