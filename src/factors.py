import pandas as pd
import numpy as np


def MOM(close: pd.Series, timeperiod=10):
    """
    Calculate Momentum (MOM) of a stock's close prices over a specified time period.
    """
    # Calculate the difference in close prices between current time and 'timeperiod' periods ago
    mom_values = close.apply(lambda x: x - x.shift(timeperiod))
    return mom_values


def SMA(values, n):
    """
    Return Simple Moving Average
    """
    return pd.Series(values).rolling(n).mean()


def RSI(close, timeperiod=14):
    """
    Calculate the Relative Strength Index (RSI) of a stock's close prices over a specified time period.

    """
    # Calculate the price changes (daily returns)
    delta = close.diff()

    # Calculate the gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gains and losses over the specified time period
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()

    # Calculate the relative strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def MACD(close, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence/Divergence (MACD) of a stock's close prices.
    """
    # Calculate the short-term and long-term EMAs
    short_ema = close.ewm(span=short_period, adjust=False).mean()
    long_ema = close.ewm(span=long_period, adjust=False).mean()

    # Calculate the MACD line (difference between short and long EMAs)
    macd_line = short_ema - long_ema

    # Calculate the signal line (EMA of the MACD line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Create a DataFrame to store MACD and Signal line values
    macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})

    return macd_df


def OBV(close, volume):
    """
    Calculate the On-Balance Volume (OBV) of a stock.
    """
    # Calculate the price changes (daily returns)
    price_changes = close.diff()
    volume = np.log(volume + 10e-7)
    # Initialize the OBV with the first value
    obv = pd.Series(0.0, index=close.index)

    for loc, i in enumerate(price_changes.index[1:]):
        pre_i = price_changes.index[loc-1]
        if price_changes[i] > 0:
            obv[i] = obv[pre_i] + volume[i]
        elif price_changes[i] < 0:
            obv[i] = obv[pre_i] - volume[i]
        else:
            obv[i] = obv[pre_i]

    return obv

def pe_ratio(close, eraning):
    """
    Calculate the Earning Per Share (EPS) of a stock.
    """
    # Calculate the price changes (daily returns)
    eps = close / (eraning+ 10e-7)
    return eps

def reg_factors(close, timeperiod=14, nbdev=1):
    """
    Calculate regression factors indicators of a stock's close prices.
    """
    # Calculate Linear Regression
    linearreg = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # Calculate Linear Regression Angle
    linearreg_angle = np.arctan(linearreg) * (180 / np.pi)

    # Calculate Linear Regression Intercept
    linearreg_intercept = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[1], raw=True)

    # Calculate Linear Regression Slope
    linearreg_slope = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # Calculate STDDEV (Standard Deviation)
    stddev = close.rolling(window=timeperiod).std() * nbdev

    # Calculate TSF (Time Series Forecast)
    tsf = close.rolling(window=timeperiod).mean()

    # Calculate VAR (Variance)
    var = close.rolling(window=timeperiod).var() * nbdev

    # Create a DataFrame with the calculated indicators
    indicators = pd.DataFrame({
        'linear_reg': linearreg,
        'linear_reg_angle': linearreg_angle,
        'intercept': linearreg_intercept,
        'slope': linearreg_slope,
        'stddev': stddev,
        'tsf': tsf,
        'var': var
    })

    return indicators

