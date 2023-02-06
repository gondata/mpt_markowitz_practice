# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as optimize

yf.pdr_override()

# Then we have to download the data

assets = ['AAPL', 'TSLA', 'KO']
startdate = '2012-01-01'
enddate = '2023-02-02'
data = pd.DataFrame()

for t in assets:
    data[t] = pdr.get_data_yahoo(t, start=startdate, end=enddate)['Adj Close']

# Returns

log_returns = np.log(1+data.pct_change())

# Portfolio Variables

port_log_returns = []
port_vols = []

for i in range (10000):
    num_assets = len(assets)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights) # The sum must be 1
    port_ret = np.sum(log_returns.mean() * weights) * 252
    port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))       
    port_log_returns.append(port_ret)
    port_vols.append(port_var)

def portfolio_stats(weights, log_returns):
    port_ret = np.sum(log_returns.mean() * weights) * 252
    port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    sharpe = port_ret/port_var    #rf = 0
    return {'Return': port_ret, 'Volatility': port_var, 'Sharpe': sharpe}

# We maximize the Sharpe ratio

def minimize_sharpe(weights, log_returns): 
    return -portfolio_stats(weights, log_returns)['Sharpe'] 

port_log_returns = np.array(port_log_returns)
port_vols = np.array(port_vols)
sharpe = port_log_returns/port_vols

max_sr_vol = port_vols[sharpe.argmax()]
max_sr_ret = port_log_returns[sharpe.argmax()]

constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
bounds = tuple((0,1) for x in range(num_assets))
initializer = num_assets * [1./num_assets,]

optimal_sharpe = optimize.minimize(minimize_sharpe, initializer, method = 'SLSQP', args = (log_returns,) ,bounds = bounds, constraints = constraints)
optimal_sharpe_weights = optimal_sharpe['x'].round(4)
optimal_stats = portfolio_stats(optimal_sharpe_weights, log_returns)

# Prints

print("Optimal Portfolio Weights: ", list(zip(assets, list(optimal_sharpe_weights))))
print("Optimal Portfolio log_returns: ", np.round(optimal_stats['Return'], 3))
print("Optimal Portfolio Volatility: ", np.round(optimal_stats['Volatility'], 3))
print("Optimal Portfolio Sharpe: ", np.round(optimal_stats['Sharpe'], 3))

# Graphs

plt.figure(figsize=(12, 6))
plt.scatter(port_vols, port_log_returns, c=(port_log_returns/port_vols))
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=30)
plt.colorbar(label = 'Sharpe Ratio, rf=0')
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.show()