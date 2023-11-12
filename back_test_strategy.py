from my_strategy import MyStrategy
from src.broker import Broker
from src.backtest import Backtest
import polars as pl
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    day = '1112'
    data = pl.scan_ipc("./data/Stocks/*.arrow")
    market_data = data.select(pl.col(['Stock', 'Time', 'Date', 'Minutes',  'trade_mask', 'lift_mask',
                                'hit_mask', 'close', 'last_bid', 'last_ask', 'last_mid'])).collect().to_numpy()

    backtest_engine = Backtest(market_data, MyStrategy, Broker)
    ret = backtest_engine.run(day)