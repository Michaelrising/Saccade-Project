from .utils import assert_msg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Broker:
    def __init__(self, data, cash, commission):
        assert_msg(0 < cash, "Enter the initial cash quantity.：{}".format(cash))
        assert_msg(0 <= commission <= 0.05, "Please input the commission fee rate.：{}".format(commission))
        self._initial_cash = cash
        self._data = data
        self._stocks = data['Stock'].unique()
        self._commission = commission
        self._position = {i: [0, 0.0]for i in self._stocks}
        self.tick_data = {i: 0.0 for i in self._stocks}
        self._cash = cash
        self._i = 0
        self._last_i = None
        self.day_value = []
        self.maxdown_point = []
        self.transaction_history = []

    @property
    def cash(self):
        """
        :return: Return the current amount of cash in the account
        """
        return self._cash

    @property
    def position(self):
        """
        :return: Retrieve current account position
        """
        return self._position

    @property
    def initial_cash(self):
        """
        :return: Return initial cash amount
        """
        return self._initial_cash

    @property
    def market_value(self):
        """
        :return: Return current market value
        """
        val = self._cash
        for s in self._stocks:
            pos = self._position.get(s)
            if pos is not None:
                val += pos[0] * pos[1]
        return val

    @property
    def current_price(self):
        """
        :return: Return current market price
        """
        new_tick_data = self._data.loc[self._data['Date'] == self._i][['Stock', 'close']].to_dict(orient='records')
        for i in new_tick_data:
            self.tick_data[i['Stock']] = i['close']
        return self.tick_data

    def execute(self, stock, amount, price, order_type='limit'):
        """
        Buy all at market price using the remaining funds in the current account.
        """
        assert_msg(stock in self._stocks, "Please enter the stock code.：{}".format(stock))
        quantity = int(amount / price)
        if quantity < 1:
            return
        self._cash -= float(quantity * price * (1 + self._commission))

        # Update position
        if stock in self._position:
            existing_quantity, average_price = self._position[stock]
            new_quantity = existing_quantity + quantity

            # Adjust average price if not closing the position
            if new_quantity != 0:
                new_average_price = (existing_quantity * average_price + price) / new_quantity
            else:
                new_average_price = 0

            self._position[stock] = (new_quantity, new_average_price)
        else:
            self._position[stock] = (quantity, price)
        # Record the transaction
        self.transaction_history.append(
            {'Stock': stock, 'Quantity': quantity, 'Price': price, 'OrderType': order_type})
        return True

    def next(self, tick):
        self._i = tick

    # record daily value
    def write_ratio(self, tick):
        the_value = self.market_value
        self.day_value.append({'Date': tick, 'Value': the_value})

    def calculate_pnl(self):
        """
        Calculate the unrealized P&L of the portfolio.
        """
        total_pnl = 0
        for stock, (quantity, average_price) in self._position.items():
            # Assume we have a function to get the current market price
            current_market_price = self.tick_data[stock]
            total_pnl += (current_market_price - average_price) * quantity

        return total_pnl

    def get_absolute_return(self):
        _cash = self.market_value
        return (_cash - self._initial_cash) / self._initial_cash

    # annual return rate
    def get_annualized_return(self):
        _cash = self.market_value
        ratio = (_cash - self._initial_cash) / self._initial_cash
        return_value = (1 + ratio) ** (252 / len(self.day_value)) - 1
        return return_value

    # sharpe ratio
    def get_sharpe_ratio(self):
        _cash = pd.DataFrame(self.day_value)['Value']
        ratio = ((_cash - self._initial_cash) / self._initial_cash).dropna()
        return_value = (self.get_annualized_return() - 0.02) / ratio.std() * (252 ** 0.5)
        return return_value

        # 最大回撤

    def get_maxdown(self):
        _df = pd.DataFrame(self.day_value)
        return_list = _df['Value'].dropna()
        i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)) + 1
        if i == 0:
            return 0
        j = np.argmax(return_list[:i]) + 1

        # record the maxmum drawdown points
        self.maxdown_point.append(_df.iloc[j])
        self.maxdown_point.append(_df.iloc[i])

        return (return_list[j] - return_list[i]) / (return_list[j])

    def get_result(self):
        _dic = {'return': round(self.get_absolute_return(), 4),
                'annual_rate': round(self.get_annualized_return(), 4),
                'max_drawdown': round(self.get_maxdown(), 4),
                'sharp_ratio': round(self.get_sharpe_ratio(), 4)}
        self.result = _dic
        return _dic

    def plot_ratio(self, w=20, h=7):
        sns.set()

        _day_price = pd.DataFrame(self.day_value).dropna()
        _day_price['trade_ratio'] = (_day_price['Value'] - self._initial_cash) / self._initial_cash

        plt.figure(figsize=(w, h))
        # return ratio
        plt.plot(_day_price['Date'], _day_price['trade_ratio'], linewidth='2', color='#1E90FF')
        # set x ticks to be readable
        plt.xticks(_day_price['Date'][::int(len(_day_price) / 10)], rotation=45)

        # max drawdown points
        x_list = [date['Date'] for date in self.maxdown_point]
        y_list = [(date['Value'] - self._initial_cash) / self._initial_cash for date in self.maxdown_point]
        plt.scatter(x_list, y_list, c='g', linewidths=7, marker='o')

        # benchmark
        plt.title('Total Returns {0}|Annualized Returns {1}|Max Drawdown {2}'.format(
            self.result['return'],
            self.result['annual_rate'],
            self.result['max_drawdown']), fontsize=16)
        plt.grid(True)
        # plt.legend(['Benchmark Returns', 'Total Returns'], loc=2, fontsize=14)
        plt.show()










