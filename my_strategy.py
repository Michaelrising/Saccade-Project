import numpy as np
import numba as nb
from src.strategy import Strategy
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import polars as pl


class MyStrategy(Strategy):
    def __init__(self, broker, risk_manage=True, label='log_mid', model='linear'):
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
        self.date = '0'
        self.warm_up = False
        self.train_x = None
        self.train_y = None
        self.features = pl.read_csv('./selected_features.csv')['Features'].to_list()
        self.market_cols = ['Date', 'Minutes', 'Time', 'Stock', 'trade_mask', 'lift_mask', 'hit_mask']
        self.market_price_cols = ['close', 'last_bid', 'last_ask', 'last_mid']

        if model == 'ridge':
            self.model = Ridge(alpha=0.5)
        elif model == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=20, max_depth=3, n_jobs=10)
        else:
            self.model = LinearRegression(n_jobs=-1)

    def init(self, date: str, data_path="./data/Stocks/*.arrow"):
        data = pl.scan_ipc(data_path).select(self.features+self.market_cols + self.market_price_cols + [self.label])
        self.date = date
        idx = self._broker._calenders.index(date)
        self.days_focus = self._broker._calenders[min(0, idx-3):idx+1]
        self.train_data = data.filter(pl.col('Date').is_in(self.days_focus)).select(
            ['Time', self.label] + self.features)#.collect().to_numpy()

        self.market = data.select(['Time', 'Stock'])#.collect().to_numpy()


    def _warm_up(self, length=20):
        """
        apply the earliest-month data for training and warming up the model
        :param length:
        :return:
        """
        # warmup_data = self.data.filter(pl.col('Date') < self.date)
        warmup_data = self.train_data.filter(pl.col('Time') < self.date + ' 09:30:00')#[self.train_data[:, 0] < self.date + ' 09:30:00']
        warmup_x = warmup_data.select(self.features).collect().to_numpy()#[:, 2:].astype(np.float32)
        warmup_y = warmup_data.select(self.label).collect().to_numpy().reshape(-1) #[:, 1].reshape(-1).astype(np.float32)

        self.mean_x = warmup_x.mean(axis=0)
        self.std_x = warmup_x.std(axis=0) + 10e-6
        self.model.fit((warmup_x - self.mean_x)/self.std_x, warmup_y)

        self.train_x = warmup_x
        self.train_y = warmup_y
        self.warm_up = True

    # @nb.jit(parallel=True)
    def next(self, tick):

        if tick[:4] < self.date:
            return
        elif not self.warm_up:
            self._warm_up()
        else:
            train_data = self.train_data.filter((pl.col('Time') < tick))
            self.train_x = train_data.select(self.features).collect().to_numpy()#np.concatenate((self.train_x, self.tick_x), axis=0)
            self.mean_x = self.train_x.mean(axis=0)
            self.std_x = self.train_x.std(axis=0) + 10e-6
            train_x = (self.train_x - self.mean_x) / self.std_x

            self.train_y = train_data.select(self.label).collect().to_numpy().reshape(-1) #np.concatenate((self.train_y, self.tick_y), axis=0)

            self.model.fit(train_x, self.train_y)

        tick_data = self.market.filter(pl.col('Time') == tick).collect()#[self.market[:, 0] == tick]
        if not len(tick_data):
            return
        # if self.risk_manage and self.risk_manager():
        #     return
        self.tick_x = self.train_data.filter(pl.col('Time') == tick).select(self.features).collect().to_numpy() #[self.train_data[:, 0] == tick][:, 2:].astype(np.float32)
        # self.tick_y = self.train_data.filter(pl.col('Time') == tick).select(self.label).collect().to_numpy().reshape(-1) #[self.train_data[:, 0] == tick][:, 1].reshape(-1).astype(np.float32)
        tick_x = (self.tick_x - self.mean_x) / self.std_x

        signal = self.model.predict(tick_x)

        # split into 5 groups based on signal
        signal = pl.Series('signal', signal)
        signal = signal.qcut(quantiles=5, labels=[str(i) for i in range(1, 6)])
        # buy the top 20% stocks
        buy_list = signal == '5'
        # buy_stocks = tick_data.filter
        buy_stocks = set(tick_data.filter(buy_list)['Stock'].to_list())
        # sell the bottom 20% stocks
        sell_list = signal == '1'
        sell_stocks = set(tick_data.filter(sell_list)['Stock'].to_list())

        # sell first and then buy
        ava_cash = self._broker.cash
        amount = int(ava_cash / len(buy_stocks))
        self._execute(buy_stocks, amount, 1)
        self._execute(sell_stocks, amount, 2)

    # @nb.jit()
    def _execute(self, stocks, amount, order_side):
        order_type = 1  # set always market order for instant trading
        price = 0.0
        amount = -amount if order_side == 2 else amount
        for stock in stocks:
            order = (stock, -amount, price, order_type, order_side)
            self.execute(order)

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

