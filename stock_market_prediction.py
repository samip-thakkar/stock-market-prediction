# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 02:23:52 2019

@author: Samip
"""

#Import libraries
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import math
import numpy as np


#Load dataset
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('aaba.us.txt',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

#Visualizing per day close price of the stock
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.plot(data['Close'])
plt.title('Altaba Inc. closing Price')
plt.show()

#Plotting a scatterplot
df_close = data['Close']
df_close.plot(style = 'k.')
plt.title('Scatter plot of closing price')
plt.show()

#Testing if time series data is stationary or not
def test_stationarity(timeseries):
    
    #Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    #Plot rolling statistics
    plt.plot(timeseries, color = 'red')
    plt.plot(rolmean, color = 'blue')
    plt.plot(rolstd, color = 'black')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
    
    #Perform ADFT 
    print("Result of Auto Dickery Fuller Test")
    adft = adfuller(timeseries, autolag = 'AIC')
    
    #Output will not give information of what values it define, hence we need to manually set it up
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df_close)

#We can see that mean and standard deviation are increasing and thus, data is not stationary
"""p-value > 0.05 hennce we can't reject Null hypothesis and test-statistics value is greater than critical values.
Thus, we need to separate seasonality and trend from our series."""

result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)

"""We will take log of series to reduce the magnitude of values and reduce the rising trends in series. Then we will find
rolling statistics and apply ADTF."""

from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()

#Now we will create an ARIMA model and train with closing price. So, split data into train and test and visualize it.
#ARIMA = Auto Regressive Integrated Moving Average
train_data, test_data = df_log[3:int(len(df_log) * 0.9)], df_log[int(len(df_log) * 0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
 
#Now we will choose p,q and d value withput using ACF and PACF, we will use Auto ARIMA.
""" p: The number of lag observations included in the model, also called the lag order.
    d: The number of times that the raw observations are differenced, also called the degree of differencing.
    q: The size of the moving average window, also called the order of moving average."""

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())
#We got values of p,d,q as 3,2,1


#PLot residual plot of ARIMA
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()
"""Findings:  The residual errors seem to fluctuate around a mean of zero and have a uniform variance.
-->The density plot suggest normal distribution with mean zero.
-->All the dots should fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed.
--> The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated."""


#Create ARIMA model with parameters calculated
model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

#Forecast stock prices on test set keeping 95% confidence level
fc, se, conf = fitted.forecast(544, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=50)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.3)
plt.title('Altaba Inc. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#Checking accuracy
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))