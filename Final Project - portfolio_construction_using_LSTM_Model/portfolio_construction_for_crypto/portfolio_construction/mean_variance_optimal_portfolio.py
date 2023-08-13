# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:34:03 2022

@author: ChakalasiyaMayurVash
"""

import pandas as pd  
import numpy as np
from pandas_datareader import data, wb
import datetime
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt

class mean_variance_optimal_portfolio:
    

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def calc_portfolio_perf(self,weights, mean_returns, cov, rf):
    
        annual_return = np.sum(mean_returns * weights) * 252
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        sharpe_ratio = (annual_return - rf) / std_dev
    
        return annual_return, std_dev, sharpe_ratio
    
    def monte_carlo_simulate_portfolios(self,num_portfolios, mean_returns, cov, rf, tickers):
        
        output = np.zeros((len(mean_returns)+3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            annual_return, std_dev, sharpe_ratio = self.calc_portfolio_perf(weights, mean_returns, cov, rf)
            output[0,i] = annual_return
            output[1,i] = std_dev
            output[2,i] = sharpe_ratio
            #iterate through the weight vector and add data to results array
            for j in range(len(weights)):
                output[j+3,i] = weights[j]
                
        results_df = pd.DataFrame(output.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
        
        return results_df
    
    def scatter_plot_for_portfolio(data):
        max_sharpe_port = data.iloc[data['sharpe'].idxmax()]
        #locate positon of portfolio with minimum standard deviation
        min_vol_port = data.iloc[data['stdev'].idxmin()]
        #create scatter plot coloured by Sharpe Ratio
        plt.subplots(figsize=(15,10))
        plt.scatter(data.stdev,data.ret,c=data.sharpe,cmap='RdYlBu')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Returns')
        plt.colorbar()
        #plot red star to highlight position of portfolio with highest Sharpe Ratio
        plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
        #plot green star to highlight position of minimum variance portfolio
        plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
        plt.show()
        
    
    def calc_neg_sharpe(self,weights, mean_returns, cov, rf):
  
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - rf) / portfolio_std
        return -sharpe_ratio    

    def max_sharpe_ratio(self,mean_returns, cov, rf):
        num_assets = len(mean_returns)
        args = (mean_returns, cov, rf)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(self.calc_neg_sharpe, num_assets*[1./num_assets,], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        return result