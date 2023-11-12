from my_strategy import MyStrategy
from src.broker import Broker
from src.backtest import Backtest
import pandas as pd
import polars as pl
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    data = pl.scan_ipc("./data/Stocks/*.arrow")
    market_data = data.select(pl.col(['Date', 'Minutes', 'Time', 'Stock', 'trade_mask', 'lift_mask',
                               'hit_mask', 'close', 'last_bid', 'last_ask', 'last_mid']))

    broker = Broker(market_data, cash=10000000, commission=0.0002)
    strategy = MyStrategy(broker, risk_manage=False, model='linear')
    backtest_engine = Backtest(market_data, strategy, broker)
    ret = backtest_engine.run()