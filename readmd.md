This is the project description
```
In this project, there are several steps to be followed:
    1) Data Loader & Data Preprocessing
    2) Future Engineering & Feature Selection
    3) Model Building & Model Evaluation
    4) Strategy Implementation & Backtesting
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

3. 

```

```