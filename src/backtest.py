import pandas as pd
import polars as pl
from .strategy import Strategy
from .broker import Broker
from tqdm import tqdm


class Backtest:
    """
    Backtest class is used for reading historical market data,
    executing strategies, simulating trades, and estimating returns.
    Call Backtest.run during initialization to initiate the backtesting process.
    """

    def __init__(self,

                 data: pl.LazyFrame,
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
        self._data = data
        self._calenders = self._data.select('Date').unique().collect()['Date'].sort().to_list()
        self._broker = broker
        self._strategy = strategy
        self._results = None

    def run(self, groupby_date=True):
        """
        Run backtesting, iterate through historical data, execute simulated trades, and return backtesting results.
        Run the backtest. Returns `pd.Series` with results and statistics.
        Keyword arguments are interpreted as strategy parameters.
        """
        strategy = self._strategy
        broker = self._broker
        # Strategy Initialization
        strategy.init()
        # Set the start and end positions for back testing
        # Back testing main loop, update market status, and execute strategy
        if groupby_date:
            for i, day in tqdm(enumerate(self._calenders)):
                data = self._data.filter(pl.col('Date') == day).sort('Time').collect()
                for tick in tqdm(data['Time'].unique().sort().to_list()):
                    # tick_data = self._data.loc[self._data['date'] == tick]
                    broker.next(tick)
                    strategy.next(tick)
                    broker.write_ratio(tick)
                # TODO close all the positions
                broker.close_all_positions()
        else:
            if not self._data.select('Time').collect()['Time'].is_sorted():
                self._data = self._data.sort('Time')
            for tick in tqdm(self._data.select('Time').unique().collect()['Time'].sort().to_list()):
                broker.next(tick)
                strategy.next(tick)
                broker.write_ratio(tick)

        # After completing the strategy execution, calculate the results and return them.
        res = broker.get_result()
        broker.plot_ratio()
        return res

