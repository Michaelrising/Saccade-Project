import copy
import os
import numpy as np
import pandas as pd
import polars as pl
from .strategy import Strategy
from .broker import Broker
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import get_context


class Backtest:
    """
    Backtest class is used for reading historical market data,
    executing strategies, simulating trades, and estimating returns.
    Call Backtest.run during initialization to initiate the backtesting process.
    """

    def __init__(self,

                 data: np.array,
                 strategy: Strategy,
                 broker: Broker):
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
        # if not data.index.is_monotonic_increasing:
        #     data = data.sort_index()
        # Initialize exchange and strategy objects using data.
        self._data = data[:, :3] # keep stock, time, date
        self._calenders = broker._calenders
        self._broker = broker
        self._strategy = strategy
        self._results = None

    def run(self, groupby_date=True):
        """
        Run backtesting, iterate through historical data, execute simulated trades, and return backtesting results.
        Run the backtest. Returns `pd.Series` with results and statistics.
        Keyword arguments are interpreted as strategy parameters.
        """
        # Set the start and end positions for back testing
        # Back testing main loop, update market status, and execute strategy
        num_workers = 20
        # self.run_one_day((self._calenders[3], self._broker, self._strategy))

        with get_context('spawn').Pool(num_workers) as pool:

            results = list(tqdm(pool.imap(self.run_one_day, [(d, self._broker, self._strategy) for d in self._calenders[3:]])))

        self.agg_res(results)
        # return results

    def run_one_day(self, args):
        day, broker, strategy = args
        # Strategy Initialization
        strategy.init(day)

        data = self._data[self._data[:, 2] == day]

        ticks = np.unique(data[:, 1].reshape(-1))
        ticks.sort()
        for tick in tqdm(ticks):

            broker.next(tick)
            strategy.next(tick)
            broker.write_ratio(tick)

        broker.close_all_positions()

        return broker.get_result()

    def agg_res(self, results, save=True):
        tbt_values = []
        transaction_history = []
        returns = []
        for res in results:
            tbt_value, transaction_his, _return = res.values()
            tbt_values.append(tbt_value)
            transaction_history += transaction_his
            returns.append(_return)
        tbt_values = pl.from_records(tbt_values, orient='row', schema={'Time': str, 'tbt_value': float})

        transaction_history = pl.from_records(transaction_history, orient='row',
                                              schema={'Time': str, 'Stock': str, 'Price': float, 'Quantity': int,
                                                      'OrderType': str, 'OrderSide': str, 'Status': int})
        returns = pl.from_dict({'Date': self._calenders[3:], 'DailyReturn': returns})
        returns = returns.with_columns(pl.col('DailyReturn').cumprod().alias('CumReturn'))
        if save:
            os.makedirs('./results', exist_ok=True)
            transaction_history.write_ipc('./results/transaction_history.arrow')
            tbt_values.write_ipc('./results/tbt_values.arrow')
            returns.write_ipc('./results/returns.arrow')
        return tbt_values, transaction_history, returns






