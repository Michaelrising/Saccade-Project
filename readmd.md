This is the project description
```
In this project, there are several steps to be followed:
    1) Data Loader & Data Preprocessing
    2) Future Engineering & Feature Selection
    3) Model Building & Model Evaluation
    4) Strategy Implementation & Backtesting
    5) Performance Evaluation & Conclusion
```
## Step 1-2 Data Processing &  Future Engineering
```
1. get_data() function is used to load the data from the local directory and 
save as arrow format. We are given a dataaset of 181-day one-minute bar data 
for 100 stocks. We first load the data day-by-day, item-by-item, then melt 
the data into a long format. And then concatenate the data into a single 
dataframe, which is convenitent for further data processing. 

2. technical indicators and derivative prices are calculated and added to the
dataframe. The technical indicators are calculated by using the TA-Lib package.
And the derivative prices are spread, imbalance and some other hit lift-related
properties. And mid price is also calculated by the average of last bid and last 
ask price. This gives us the factors and derivative prices for the model building, 
which is saved as arrows format.

3. With the price columns, we generate several labels for the model building. By 
cooperating open, close, last mid and mid twap prices. The horizon of the labels 
if 5 minutes for simplicity, we could imagine that further horizon could leads to 
low correlations between the labels and the features. For avoiding any future information
leakage, we only use the information before the current time point to generate the labels.
For instance, when generating the log return close price for 10:00:00, we only have seen
the prices at 10:00:00, then we could not use this close price to generate the label for
10:00:00. Instead of that, we use the open price for the next bar as the base price and
calculate the log return close price. This is the same for the other labels.

4. The data is then split into one stock an arrow file. This is convenient for
the model building and evaluation. 

```
## Step 2-3 Feature Selection & Model Building
```
1) Feature Selection
    We first apply VIF to select the features. Without lossing too much information, we get rid
    of the features with VIF larger than 100. And then we find the correlation between the features
    and the targets. One step, we get rid of the features with correlation smaller than 10% quantile 
    for both possitive and negative correlation. Through this two processes, we shortlist to 73 features
    for the model building.

2) Model Building & Label Choosing
    We use the basic linear modeling to predict the mid price change movement. The model we 
    evaluate is the ordinary linear regression, ridge model and random forest model. Due to the
    large data set, we split the data set into five folds without overlapping in time scale and 
    only evaluate the model on the first fold. The results shows that last mid log return with linear 
    regression model has the best performance. And the random forest model has the worst performance. 
    We then choose the last mid log return as the label for the model building. And we use the 
    linear regression model to predict the last mid log return. 

```

## Step 4 Strategy Implementation & Backtesting
```
I independently implement the Strategy and Backtesinf kits with numpy and polars ad the main backbone.
The strategy is implemented in the Strategy class. The strategy is to predict the mid price log return.
The strategy is simple, with the prediction available, we long the stocks in the top 20% quantile and
short the stocks in the bottom 20% quantile. 

The backtesting is implemented in the Backtesting class. Considering no cross-night positions, I implemented 
the backtesting with the multiprocessing for each day independently (though this may leads to some sort of bias
of the strategy and model, since we assume the same amount of money is invested each day). But the multiprocessing
could lead to a hugh improvement of the backtesting speed. After retrieving the backtesing results for each day,
 we aggregate the results and calculate the performance of the strategy.

```
## Step 5 Performance Evaluation & Conclusion
```


```
## Step 6 Future Work
```
1) The model building could be improved by using the more advanced models, such as the neural network.

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