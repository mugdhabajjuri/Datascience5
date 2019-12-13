world wide products
==============================

Time series forecasting

# World Wide Products Inc.

### product demand forecasting - Project 4

#### Submitted by: Mugdha Bajjuri

The dataset contains historical product demand for a manufacturing company with footprints globally. The company provides thousands of products within dozens of product categories. There are four central warehouses to ship products within the region it is responsible for.

This dataset contains 1 CSV file.

Product_demand.csv - CSV data file containing product demand for encoded product id's

## Exploratory Data Analysis


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
```


```python
df = pd.read_csv('../data/raw/product_demand.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Warehouse</th>
      <th>Product_Category</th>
      <th>Date</th>
      <th>Order_Demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Product_0993</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012/7/27</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012/1/19</td>
      <td>500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012/2/3</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012/2/9</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012/3/2</td>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Removing Extra characters and converting order demand to float datatype
```


```python
df['Order_Demand'] = df['Order_Demand'].replace( '[()]','', regex=True ).astype(float)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1048575 entries, 0 to 1048574
    Data columns (total 5 columns):
    Product_Code        1048575 non-null object
    Warehouse           1048575 non-null object
    Product_Category    1048575 non-null object
    Date                1037336 non-null object
    Order_Demand        1048575 non-null float64
    dtypes: float64(1), object(4)
    memory usage: 40.0+ MB



```python
#checking for any null values
print(pd.isnull(df).sum())
```

    Product_Code            0
    Warehouse               0
    Product_Category        0
    Date                11239
    Order_Demand            0
    dtype: int64



```python
#Dropping the rows without date values, as we cant forecast with the null date values
df = df.dropna()
print(pd.isnull(df).sum())
```

    Product_Code        0
    Warehouse           0
    Product_Category    0
    Date                0
    Order_Demand        0
    dtype: int64



```python
df.shape
```




    (1037336, 5)




```python
#Lets find the products with highest demand
```


```python
prod = df.groupby('Product_Code')['Order_Demand'].sum().reset_index(name='Order_Demand')
```


```python
prod.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Order_Demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Product_0001</td>
      <td>460000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product_0002</td>
      <td>8836000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Product_0003</td>
      <td>118300.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Product_0004</td>
      <td>124600.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Product_0005</td>
      <td>22300.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod = prod.sort_values('Order_Demand', ascending = False)
```


```python
prod = prod.head(15)
```

### Plot showing products with highest demand


```python
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(12,9)})
ax = sns.barplot(x="Order_Demand", y="Product_Code", data=prod, clip_on=True)
```


![png](output_17_0.png)


Product 1359 is with highest demand 


```python
print(df['Date'].min())
print(df['Date'].max())
```

    2011/1/8
    2017/1/9


Products data is from 2011 till 2017


```python
#Changing datatype for date column
import dateutil
df['Date'] = df['Date'].apply(dateutil.parser.parse)
```


```python
df.dtypes
```




    Product_Code                object
    Warehouse                   object
    Product_Category            object
    Date                datetime64[ns]
    Order_Demand               float64
    dtype: object




```python
# Extracting date features
df['date'] = df.Date.dt.day
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Warehouse</th>
      <th>Product_Category</th>
      <th>Date</th>
      <th>Order_Demand</th>
      <th>date</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Product_0993</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-07-27</td>
      <td>100.0</td>
      <td>27</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-01-19</td>
      <td>500.0</td>
      <td>19</td>
      <td>1</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-03</td>
      <td>500.0</td>
      <td>3</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-09</td>
      <td>500.0</td>
      <td>9</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-03-02</td>
      <td>500.0</td>
      <td>2</td>
      <td>3</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Product_Code                object
    Warehouse                   object
    Product_Category            object
    Date                datetime64[ns]
    Order_Demand               float64
    date                         int64
    month                        int64
    year                         int64
    dtype: object




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Warehouse</th>
      <th>Product_Category</th>
      <th>Date</th>
      <th>Order_Demand</th>
      <th>date</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Product_0993</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-07-27</td>
      <td>100.0</td>
      <td>27</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-01-19</td>
      <td>500.0</td>
      <td>19</td>
      <td>1</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-03</td>
      <td>500.0</td>
      <td>3</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-09</td>
      <td>500.0</td>
      <td>9</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-03-02</td>
      <td>500.0</td>
      <td>2</td>
      <td>3</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = df.groupby('year')['Order_Demand'].mean().reset_index(name='avg_order_demand')
```


```python
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>avg_order_demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011</td>
      <td>13068.584375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>4661.575815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>4645.429285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>4949.900958</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>5243.695256</td>
    </tr>
  </tbody>
</table>
</div>



### Plot of order demand over the years


```python
sns.set(style="whitegrid")
ax = sns.barplot(x="year", y="avg_order_demand", data=df3)
```


![png](output_29_0.png)



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Warehouse</th>
      <th>Product_Category</th>
      <th>Date</th>
      <th>Order_Demand</th>
      <th>date</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Product_0993</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-07-27</td>
      <td>100.0</td>
      <td>27</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-01-19</td>
      <td>500.0</td>
      <td>19</td>
      <td>1</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-03</td>
      <td>500.0</td>
      <td>3</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-02-09</td>
      <td>500.0</td>
      <td>9</td>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Product_0979</td>
      <td>Whse_J</td>
      <td>Category_028</td>
      <td>2012-03-02</td>
      <td>500.0</td>
      <td>2</td>
      <td>3</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfprod1359 = df.loc[df['Product_Code'] == 'Product_1359'].sort_values(['Date'],ascending=False)
```


```python
dfprod1359.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Code</th>
      <th>Warehouse</th>
      <th>Product_Category</th>
      <th>Date</th>
      <th>Order_Demand</th>
      <th>date</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>921328</th>
      <td>Product_1359</td>
      <td>Whse_J</td>
      <td>Category_019</td>
      <td>2017-01-06</td>
      <td>100000.0</td>
      <td>6</td>
      <td>1</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>943424</th>
      <td>Product_1359</td>
      <td>Whse_J</td>
      <td>Category_019</td>
      <td>2016-12-28</td>
      <td>50000.0</td>
      <td>28</td>
      <td>12</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>870423</th>
      <td>Product_1359</td>
      <td>Whse_J</td>
      <td>Category_019</td>
      <td>2016-12-28</td>
      <td>3000.0</td>
      <td>28</td>
      <td>12</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>877287</th>
      <td>Product_1359</td>
      <td>Whse_J</td>
      <td>Category_019</td>
      <td>2016-12-28</td>
      <td>3000.0</td>
      <td>28</td>
      <td>12</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>871104</th>
      <td>Product_1359</td>
      <td>Whse_J</td>
      <td>Category_019</td>
      <td>2016-12-28</td>
      <td>10000.0</td>
      <td>28</td>
      <td>12</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfprod1359 = dfprod1359.drop(columns=['Product_Code','Product_Category','date','month','year'])
dfprod1359.index=pd.to_datetime(dfprod1359.Date,format='%Y/%m/%d')
dfprod1359.drop(columns=['Date'],inplace=True)
```


```python
dfprod1359.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Warehouse</th>
      <th>Order_Demand</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-01-05</th>
      <td>Whse_J</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>2012-01-05</th>
      <td>Whse_J</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>2012-01-05</th>
      <td>Whse_J</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>2012-01-05</th>
      <td>Whse_J</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>2012-01-05</th>
      <td>Whse_J</td>
      <td>1000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#resampling dfprod1359 with Month

prod1359DmndMnth = dfprod1359.resample('M').sum()
prod1359DmndMnth.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order_Demand</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-04-30</th>
      <td>6341000.0</td>
    </tr>
    <tr>
      <th>2016-05-31</th>
      <td>6854000.0</td>
    </tr>
    <tr>
      <th>2016-06-30</th>
      <td>7870000.0</td>
    </tr>
    <tr>
      <th>2016-07-31</th>
      <td>7108000.0</td>
    </tr>
    <tr>
      <th>2016-08-31</th>
      <td>6934000.0</td>
    </tr>
    <tr>
      <th>2016-09-30</th>
      <td>6960000.0</td>
    </tr>
    <tr>
      <th>2016-10-31</th>
      <td>7727000.0</td>
    </tr>
    <tr>
      <th>2016-11-30</th>
      <td>8814000.0</td>
    </tr>
    <tr>
      <th>2016-12-31</th>
      <td>5653000.0</td>
    </tr>
    <tr>
      <th>2017-01-31</th>
      <td>100000.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Product 1359 demand over the years


```python
prod1359DmndMnth.Order_Demand.plot(figsize=(15,8), title= 'Product 1359 Demand')
plt.show()
```


![png](output_37_0.png)


## Forecasting Models

### FbProphet Model


```python
#Using FbProphet model for forecasting demand
from fbprophet import Prophet
```


```python
df_group2 = df[df['Product_Code'] == 'Product_1359']
df_group2 = df_group2.drop(['Warehouse','Product_Category','date','year','Product_Code','month'],1)
```


```python
df_group2 = df_group2.rename(columns = {'Date': 'ds', 'Order_Demand': 'y' })
```


```python
obj = Prophet(daily_seasonality=True)
obj.fit(df_group2)
```




    <fbprophet.forecaster.Prophet at 0x1a1fe709e8>




```python
future = obj.make_future_dataframe(periods=365)
```


```python
forecast = obj.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17296</th>
      <td>2018-01-02</td>
      <td>34329.331077</td>
      <td>-61794.799028</td>
      <td>122157.062064</td>
    </tr>
    <tr>
      <th>17297</th>
      <td>2018-01-03</td>
      <td>30927.000131</td>
      <td>-63116.269280</td>
      <td>118976.464877</td>
    </tr>
    <tr>
      <th>17298</th>
      <td>2018-01-04</td>
      <td>27621.133290</td>
      <td>-51593.634050</td>
      <td>113113.823413</td>
    </tr>
    <tr>
      <th>17299</th>
      <td>2018-01-05</td>
      <td>28651.694584</td>
      <td>-56649.060327</td>
      <td>123186.586049</td>
    </tr>
    <tr>
      <th>17300</th>
      <td>2018-01-06</td>
      <td>5381.198094</td>
      <td>-76411.589302</td>
      <td>99183.048864</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig1 = obj.plot(forecast)
```


![png](output_46_0.png)



```python
fig2 = obj.plot_components(forecast)
```


![png](output_47_0.png)


### Observations :

Looking at the various forecasting plots derived from fbProphet model, for Product_1359 
following observations can be made:

Demand for the product_1359 is increasing from 2012 to 2018 with a small dip from 2014 to 2015.

Considering yearly trend, there is a sharp increase in demand between January to March

Considering weekly trend, demand is high on sundays and then goes to negative on saturdays
    

# ARIMA Model

SARIMAX,  is used to model and predict future points of a time series.

ARIMA component is used to fit time-series data to better understand and forecast future points in the time series.


```python
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(prod1359DmndMnth,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.2859      0.389     -0.736      0.462      -1.048       0.476
    ma.L1         -0.7501      0.166     -4.524      0.000      -1.075      -0.425
    ar.S.L12      -0.2269      0.235     -0.967      0.333      -0.687       0.233
    sigma2      3.771e+12   4.25e-14   8.86e+25      0.000    3.77e+12    3.77e+12
    ==============================================================================



```python
results.plot_diagnostics(figsize=(16, 8))
plt.show()
```


![png](output_52_0.png)



```python
pd.plotting.register_matplotlib_converters()
```


```python
pred = results.get_prediction(start=pd.to_datetime('2016-08-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = prod1359DmndMnth['2012':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()
```


![png](output_54_0.png)


