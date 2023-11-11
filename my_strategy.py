import numpy as np

from src.strategy import Strategy
import pandas as pd
# from data_process import DataLoader
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import polars as pl

class MyStrategy(Strategy):
    def __init__(self, broker, risk_manage=True, label='log_mid', model='random_forest'):
        super().__init__(broker, risk_manage, model)
        self.y = None
        self.x = None
        self._data = None
        self.data = None
        self.market = None
        self.label = label
        self.all_labels = ['log_close', 'log_open', 'log_mid', 'log_vwap', 'log_twap_mid']
        self.max_drawdown = 0.15
        self.take_profit_percent = 0.2
        self.stop_loss_percent = 0.05
        self.risk_manage = risk_manage
        self.warm_up = False
        self.train_x = None
        self.train_y = None
        self.warm_up_date = '0501'

        if model == 'ridge':
            self.model = Ridge(alpha=0.5)
        elif model == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=20, max_depth=3)
        else:
            self.model = LinearRegression()

    def init(self, data_path="./data/Stocks/*.arrow"):
        self.data = pl.scan_ipc(data_path)
        self.data = self.data.with_columns((pl.col('Date', 'Minutes').str.concat(' ')).alias('Time'))
        self.market = self.data.select(['Date', 'Minutes', 'Time', 'Stock', 'trade_mask',
                                        'lift_mask', 'hit_mask', 'close'])

    def _warm_up(self, length=20):
        """
        apply the earliest-month data for training and warming up the model
        :param length:
        :return:
        """
        warmup_data = self.data.filter(pl.col('Date') < self.warm_up_date).drop(['Date', 'Minutes', 'Time'])
        warmup_x = warmup_data.drop(self.all_labels + ['trade_mask', 'lift_mask', 'hit_mask']).collect().to_numpy()
        warmup_y = warmup_data.select(self.label).collect().to_numpy()
        self.model.fit(warmup_x, warmup_y)

        self.train_x = warmup_x
        self.train_y = warmup_y
        self.warm_up = True

    def next(self, tick):
        if tick < self.warm_up_date:
            return
        elif not self.warm_up:
            self._warm_up()
        else:
            self.train_x = np.concatenate([self.train_x, self.tick_x])
            self.train_y = np.concatenate([self.train_y, self.tick_y])
            self.model.fit(self.train_x, self.train_y)

        tick_data = self.market.filter((pl.col('Time') == tick)).collect()
        if not len(tick_data):
            return
        # if self.risk_manage and self.risk_manager():
        #     return
        self.tick_x = (self.data.filter((pl.col('Time') == tick)).drop(
                        self.all_labels + ['trade_mask', 'lift_mask', 'hit_mask']).collect().to_numpy())

        self.tick_y = (self.data.filter((pl.col('Time') == tick)).select(self.label).collect().to_numpy())

        signal = self.model.predict(self.tick_x)

        # split into 5 groups based on signal
        signal = pl.Series('signal', signal)
        signal = signal.qcut(quantiles=5, labels=[str(i) for i in range(1, 6)]).to_pandas()
        # buy the top 20% stocks
        buy_list = signal == '5'
        buy_stocks = set(tick_data.loc[buy_list]['Stock'].tolist())
        # sell the bottom 20% stocks
        sell_list = signal == '1'
        sell_stocks = set(tick_data.loc[sell_list]['Stock'].tolist())
        # sell first and then buy

        price = 0.0
        #TODO conditionning on the market info, the price could be bid or ask or mid
        amount = int(self._broker.cash / len(buy_stocks))
        for stock in buy_stocks:
            self.execute(stock, amount, price)

        for stock in sell_stocks:
            self.execute(stock, amount, price)

        for stock in buy_stocks:
            amount = int(self._broker.cash / len(buy_stocks))
            self.buy(stock, amount)

    def risk_manager(self):
        # Calculate the current portfolio value based on today's prices
        zero_count = 0
        for symbol, position in self._broker.position.items():
            if position == [0, 0]:
                zero_count += 1
            else:
                break
        if zero_count == len(self._broker.position):
            return False

        current_portfolio_value = self._broker.market_value
        # Calculate the drawdown
        drawdown = (self._broker.initial_cash - current_portfolio_value) / self._broker.initial_cash

        if drawdown > self.max_drawdown:
            # If drawdown exceeds the acceptable level, liquidate the portfolio
            for stock, pos in self._broker.position.items():
                if pos[0] > 0:
                    self.sell(stock)
            return True

        return False

