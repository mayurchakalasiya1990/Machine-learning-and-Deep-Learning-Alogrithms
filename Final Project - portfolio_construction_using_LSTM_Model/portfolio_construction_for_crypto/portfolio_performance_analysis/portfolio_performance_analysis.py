# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:01:06 2022

@author: ChakalasiyaMayurVash
"""

from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

#252 trading days in a year
N = 252

#2% risk free rate
rf =0.02


def portfolio_performance_analysis(df,portfolios, N, rf, assets):
        
    #df = df.copy()
    num_runs = len(portfolios)
    
    result = np.zeros((num_runs,(len(assets)+9)))    
    
    for i in range(num_runs):
        
        # randomized weights
        weights = np.array(portfolios[i]) 
        #Rebalance w/ constraints (SUM of all weights CANNOT BE > 1)
        #weights = weights/np.sum(weights)
    
        # daily return of the portfolio based on a given set of weights
        df['portfolio_ret'] = df.iloc[:,0]*weights[0]+df.iloc[:,1]*weights[1]+df.iloc[:,2]*weights[2]+df.iloc[:,3]*weights[3]+df.iloc[:,4]*weights[4]
            
        # Calculating Mean
        E = df['portfolio_ret'].mean()
    
        # drawdown    
        comp_ret = (df['portfolio_ret']+1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret/peak)-1
        max_drawdown = dd.min()    
    
        # Annualizing Mean
        E_AN = E * N    
    
        n_days = df['portfolio_ret'].shape[0]
    
        # Annulized return
        annualized_return = (df['portfolio_ret']+1).prod()**(N/n_days) - 1
    
        # Volatility ratio
        annualized_volatility = df['portfolio_ret'].std()*N**0.5
        
        # Sharpe ratio
        annualized_sharpe = (annualized_return - rf) / annualized_volatility
         
        # Calculating Downside Standard Deviation
        mean = E * N -rf
        std_neg = df['portfolio_ret'][df['portfolio_ret']<0].std()*np.sqrt(N)
          
        # Calculating Upside Standard Deviation
        std_pos = df['portfolio_ret'][df['portfolio_ret']>=0].std()*np.sqrt(N)
        
        # Calculating Volatility Skewness
        VS = std_pos/std_neg
        
        # Sortino
        Sortino = mean/std_neg          
       
        # Populating the 'result' array with the required values: Mean, SD, Sharpe followed by the weights                   
        result[i,0] = E_AN
        result[i,1] = std_neg
        result[i,2] = std_pos
        result[i,3] = VS
        result[i,4] = Sortino
        result[i,5] = annualized_return
        result[i,6] = annualized_sharpe
        result[i,7] = annualized_volatility
        result[i,8] = max_drawdown
        
        for j in range(len(assets)):
            result[i,j+9]= weights[j]
            
    columns = ['Mean','Downside SD', 'Upside SD', 'Volatility Skewness', 'Sortino', 'Annualized Return', 'Annualized Sharpe Ratio','Annualized Volatility', 'Draw Down'] + assets

    result = pd.DataFrame(result,columns=columns)
    
    return result        
        
def plot_bar_chart(df, chart_label, xlabel, ylabel):
    ts = str(time.time())    
    fig_name="./"+chart_label+"_" + ts +".png"
    df[chart_label].plot.bar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_name)
    plt.show()
    


# Establishing Assets
assets = ['USD','ethereum','binance','bitcoin','tether']

# Creating an empty dataframe
df = pd.DataFrame()

# Running the function that we just made and saving the results to the DataFrame
df = pd.read_csv('./output/crypto_predicted_close_price.csv')
df.set_index('Date',inplace=True)
df.sort_index(inplace=True)


# Calculating Log Return
df = np.log(df/df.shift(1))

# Dropping the first row because it's N/A
df = df.dropna()

# Creating 10000 random simulations of each portfolio weight configuration
num_runs = 5 # number of rows/iterations

# Creating a Matrix with 10000 rows, with each row representing a random portfolio:
    #first 3 columns are Mean Returns, Standard Deviation, and Sortino Ratio
    # remaining columns are each assets random weight within that random portfolio
result = np.zeros((num_runs,(len(assets)+9)))

equal_weight_portfolio = [0.20,0.20,0.20,0.20,0.20]
max_sharpe_ration_portfolio = [0.636536,0.010561,0.004236,0.006141,0.342525]
min_variance_portfolio = [0.235863,0.00018,0.338035,0.00121,0.424711]
Hierarchical_Risk_Parity_portfolio = [0.55726,0.0,0.0,0.0,0.44273]
kellys_criteria_portfolio = [0.08,0.0,0.0,0.0,0.07]
portfolios = [equal_weight_portfolio,max_sharpe_ration_portfolio,min_variance_portfolio,Hierarchical_Risk_Parity_portfolio,kellys_criteria_portfolio]


porf_analysis_output = portfolio_performance_analysis(df,portfolios, N, rf, assets)

print(porf_analysis_output)

plot_bar_chart(porf_analysis_output, "Sortino", "Porfolios", "Sortino")
plot_bar_chart(porf_analysis_output, "Annualized Sharpe Ratio", "Porfolios", "Annualized Sharpe Ratio")
plot_bar_chart(porf_analysis_output, "Annualized Volatility", "Porfolios", "Annualized Volatility")
plot_bar_chart(porf_analysis_output, "Draw Down", "Porfolios", "Draw Down")