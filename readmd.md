This is the project description
```
In this project, there are several steps to be followed:
    1) Data Loader & Data Preprocessing
    2) Future Engineering & Feature Selection
    3) Model Building & Model Evaluation
    4) Strategy Implementation & Backtesting
    5) Performance Evaluation & Conclusion
```
## 1-2 Data Processing &  Future Engineering
```
1. We are given a dataaset of 181-day one-minute bar data for 100 stocks. We first load 
    the data day-by-day, item-by-item, then melt the data into a long format. And then 
    concatenate the data into a single dataframe, which is convenitent for further data processing. 

2. technical indicators and derivative prices are calculated and added to the
    dataframe. The technical indicators are calculated by using the TA-Lib package.
    And the derivative prices are spread, imbalance and some other hit lift-related
    properties. And mid price is also calculated by the average of last bid and last 
    ask price. This gives us the factors and derivative prices for the model building, 
    which is saved as arrows format.
    
3. I have to stress one of the derivative features are trading/hit/lift masks. These masks
    are derived as: trade_mask = 0 if trade_num = 0 elif trade_num in [1, 2, 3] trade_mask = 1 
    else trade_mask = 2. This is the indicator that whether we could trade or trade with which side's
    price in the backtesing. For instance, the model or strategy generates one signal at time t to
    long 1000 volumes of stock A, but in history at time t, there is no trade happened, then we could
    not trade at time t at all or we could set the order price as the other side's best price to lift. 
    This mask denotes the trading activities that we could perform or not in backtesing, and at what price.
    In this way to make the backtesting more realistic. 

4. With the price columns, we generate several labels for the model building, through 
    cooperating open, close, vwap, last mid and mid twap prices. The horizon of the labels 
    is set as 5 minutes for simplicity, we could imagine that further horizon could leads to 
    low correlations between the labels and the features and leads to lower predicton performance.
    For avoiding any future information leakage, we only use the information before the current 
    time point t to generate the labels for t. For instance, when generating the log return 
    close label for 10:00:00, we have seen the prices at 10:00:00, then we could not use 
    this close price to generate the label for 10:00:00, because this price happened already 
    and we can't trade in this price anymore. Instead of that, we use the open price for the 
    next bar as the base price and calculate the log return close price. 

        log_return_close = log(close_price.shift(-horizon-1) / open_price.shift(-1)).


5. The above priocess gave us around 200 features. The data is saved in an arrow format for each one stock. 
    This is convenient for further model building and evaluation. 

```
## 2-3 Feature Selection & Model Building
```
1) Feature Selection
    Due to the large number of features, we need to select the features for the model building.
    Since multicollinearity could lead to the overfitting of the model, We first apply VIF to select 
    the features. Without lossing too much information, we get rid of the features with VIF larger 
    than 100. And then we find the correlation between the features and the targets. One step, 
    we get rid of the features with correlation smaller than 10% quantile for both possitive and 
    negative correlation. Through this two processes, we shortlist to 73 features for the model building.

2) Model Building & Label Choosing
    We choose the basic linear modeling to predict the log mid price return. We have tried and tested
    several models from ordinary linear regression, to ridge model and random forest model. 
    Due to the large data set and the insufficient machine memory, I split the data set into five 
    folds without overlapping in time scale (purge and time series split), but I only apply the models 
    on the first fold to evaluate the model. The results shows that last mid log return with linear 
    regression model has the best performance interms of the in-sample and out-sample daily correlation. 
    And the random forest model has the worst performance, this may need much more fine tuning of the hyper-parameters.
    Due to time limitation, I decide to use linear model. We then choose the last mid log return as the label 
    for the model building.

```

## 4 Strategy & Backtesting
```
4.1 Strategy

The strategy is implemented in the Strategy class, which is to predict the mid price log return and 
generate trading signals. We applied the last three-days data to train the model and predict the log return
of last mid price for the next 5 minutes. And then we generate the trading signals based on the prediction.
The signal generation is simple, with the prediction available, we split the stocks into 5 groups and 
long the stocks in the top 20% quantile and short the stocks in the bottom 20% quantile.  And with the
requirment that 'Zero net exposure is preferred', we long and short with the same amount of
money, with the cash as the 60-80% of the total porfolio. And if the long or short positions are larger than
the 50% initial cash, we adjust the positions and close all the corresponding positions for risk control.

Recall the trading masks that I derived before, I always set a market order to trade the stocks in no matter
long or short positions. But the trading status and trading prices are determined by the masks. For instance,
if there is no trading activities at this tick, I will give up the trade, and if there is trade, I will trade
at best our side or best opposite side price based on the lift/hit masks. This part is implemented in the
broker class.

We set no cross-night positions, which means we will close all the positions at the end of the day.

4.2 Backtesting

Considering no cross-night positions, I implemented the backtesting with the multiprocessing for each 
day independently (though this may leads to some sort of bias of the strategy performance, since we assume 
the same amount of money is invested each day). But the multiprocessing could lead to a hugh improvement 
of the backtesting speed. After retrieving the backtesing results for each day, we aggregate the results 
and calculate the performance of the strategy.

```
### 4.1 Assumptions
```
In this project. I tried to make the backtesting as realistic as possible. But there are still some assumptions:

1) The trading masks are derived from the history data, which means the trading masks are not available in real time.

2) We assume all the orders are market orders, which means we could trade at the best price at our side or at 
the other side. This is unrealistic. Since the limit orders are the main orders in the market. But we could 
assume that if we wanna make the trade happen, then we could set a limit order at the best price at our side
or the other side.

3) We assume the trading commission and rebate are fixed, which is not true in real market.

4) As we discussed before, we assume the same amount of money is invested each day for backtesing the
strategy. This is not true in real market.
 
5) As we can short the positions, but we does not consider the cost when we have to borrow the stocks. 
The cost for borrowing is significant in reality. 

```
## 5 Future Work
```
1) The model building could be improved by using the more advanced models, such as the neural network or we could do more
work to the fine-tuning od the models like the random forest.

2) The labels could coorperate with the commission and rebate to make more realistic labels to reflect
the stock price. For instance, I was thinking use a transformation of weighted-mid price named as poly-weighted mid
price, which is the weighted average of the last bid and ask price with the combination of commission and rebate.
The form is as follows:
    poly_weighted_mid_price = P_mid + theta * I(I**2 + 1) / 2
where P_mid is the mid price, I is the imbalance, theta is defined as
    theta = A_t * s + c_t, here A_t and s_t could be learnable or set as 0.6, 0.4, which are experimentally workable.
and s = (S+ F_t)/2, here S = (P_a - P_b)/(P_a + P_b) is the spread, F_t is the commission and rebate. 
    
3)Optimize the codes efficiency, especially for model building and backtesting. Due to the large tck data.
but my algorithm for backtesing is event-driven, which is impossible to run this large dataset.

```
## 6 How to run the code
```
1. python3 features.py
2. python3 dataloader.py
3. select features and models and labels in notebook of feature_selections 
4. python3 back_test_strategy.py

```

##  Performance Evaluation & Conclusion
```
We could output tick-by-tick returns and over-all returns and sharp-ratios etc. But due to time limitation,
I could not perform analysis in terms of there metrics. But based on what I got now, I have to say a toy modeling
is not enough to predict the stock movement, more sophisticated real live conditions are needed to be considered. 
Hope I cold have more time to further discuss my ideas and improve the performance of the strategy with your team. 
```