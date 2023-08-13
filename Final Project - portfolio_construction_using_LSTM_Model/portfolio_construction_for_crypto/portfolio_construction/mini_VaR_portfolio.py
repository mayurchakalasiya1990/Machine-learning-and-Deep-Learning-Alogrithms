# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:57:12 2022

@author: ChakalasiyaMayurVash
"""

import pandas as pd  
import numpy as np
from pandas_datareader import data, wb
import datetime
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

def calc_portfolio_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_var

def min_VaR(mean_returns, cov, alpha, days):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, alpha, days)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_VaR, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def scatter_plot_for_min_VaR_portfoilio(data):
    min_VaR = data.iloc[data['VaR'].idxmin()]
    #create scatter plot coloured by VaR
    plt.subplots(figsize=(15,10))
    plt.scatter(data.stdev,data.ret,c=data.VaR,cmap='RdYlBu')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Returns')
    plt.colorbar()
    #plot red star to highlight position of minimum VaR portfolio
    plt.scatter(min_VaR[1],min_VaR[0],marker=(5,1,0),color='r',s=500)
    plt.show()

min_port = min_VaR(mean_returns, cov, alpha, days)

pd.DataFrame([round(x,2) for x in min_port_VaR['x']],index=tickers).T


tickers = ['AAPL', 'MSFT', 'NFLX', 'AMZN', 'GOOG']
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2018, 12, 31)
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in tickers]).T
df.columns = tickers