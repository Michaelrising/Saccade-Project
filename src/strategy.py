import abc


class Strategy(metaclass=abc.ABCMeta):
    """
    Abstract strategy class used to define trading strategies.
    If you want to define your own strategy class, you need to inherit from this base class and implement two abstract methods:
    Strategy.init
    Strategy.next
    """

    def __init__(self, broker, risk_manager=True, model='random_forest'):
        """
        Construct strategy object.
        @params broker:  Broker        Trading API interface for simulating trades
        @params data:    list           Market data
        """
        self._indicators = []
        self._broker = broker
        self._tick = 0
        self._data = None

    @property
    def tick(self):
        return self._tick

    @abc.abstractmethod
    def init(self):
        """
         Initialize strategy. Called once during the strategy backtesting/execution process
         to initialize the internal state of the strategy. Here, auxiliary parameters of
         the strategy can also be pre-calculated. For example, based on historical market data:
         Calculate the buy/sell indicator vector; Train/initialize model parameters.
        """
        pass

    @abc.abstractmethod
    def next(self, tick):
        """
        Step function that executes the strategy for the current tick, which represents the
        current "time". For example, data[tick] is used to access the current market price.
        """
        pass

    def execute(self, order):
        self._broker.execute(order)


