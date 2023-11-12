from my_strategy import MyStrategy
from src.broker import Broker
from src.backtest import Backtest
import polars as pl
import warnings
warnings.filterwarnings('ignore')

@profile
def main_function():
    data = pl.scan_ipc("./data/Stocks/*.arrow")
    market_data = data.select(pl.col(['Stock', 'Time', 'Date', 'Minutes',  'trade_mask', 'lift_mask',
                                'hit_mask', 'close', 'last_bid', 'last_ask', 'last_mid'])).collect().to_numpy()

    broker = Broker(market_data, cash=10000000, commission=0.0002)
    strategy = MyStrategy(broker, risk_manage=False, model='linear')
    backtest_engine = Backtest(market_data, strategy, broker)
    ret = backtest_engine.run()


if __name__ == "__main__":
    main_function()