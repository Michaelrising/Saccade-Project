import polars as pl
import numpy as np

def test_tran_split(df:pl.DataFrame, r:float=0.2, skip:int=2):
    days = df['Date'].unique().sort()
    train_days = days[:int(len(days) * 0.8 - 2)]
    test_days = days[int(len(days) * 0.8):]
    t1, t2 = train_days.max(), test_days.min()

    train_data = df.filter(pl.col('Date') <= t1)
    test_data = df.filter(pl.col('Date') >= t2)



