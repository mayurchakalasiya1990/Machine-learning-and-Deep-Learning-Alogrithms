'''
https://github.com/thk3421-models/KellyPortfolio/blob/main/config.json
'''
import argparse
import datetime
import json
import sys
import numpy as np
import pandas as pd
import yfinance
from cvxopt import matrix
from cvxopt.solvers import qp
#from sklearn.covariance import LedoitWolf
from typing import Dict


estimation_mode='custom'
implied=''
price_data= None
config = './config.json'

def load_config(path:str)->Dict:
    "load required config file"
    with open(path) as config_file:
        data = json.load(config_file)
    return data

def load_prices()->pd.DataFrame:    
    data = pd.read_csv('./crypto_historical_close_price_2022.csv')
    data.set_index('Date',inplace=True)
    #data.drop('Unnamed: 0', axis=1, inplace=True)  
    data.sort_index(inplace=True)
    return data


def annual_excess_returns(prices:pd.DataFrame, config:Dict)->pd.DataFrame:
    '''Stock data only changes on weekdays. Crypto data is available all days.
       Compute daily returns using Friday to Monday returns for all data'''
    returns = prices.pct_change(1)
    excess_returns = returns - config['annual_risk_free_rate'] / 252
    return excess_returns

def annual_covar(excess_returns:pd.DataFrame, config:Dict)->pd.DataFrame:
    "annualized covariance of excess returns"
    if config['use_Ledoit_Wolf'] == True:
        lw = LedoitWolf().fit(excess_returns.dropna()).covariance_
        ann_covar = pd.DataFrame(lw, columns=excess_returns.columns) * 252
    else:
        ann_covar = excess_returns.cov() * 252
    print('Condition number of annualized covariance matrix is:', np.linalg.cond(ann_covar))
    try:
        eigvals, __ = np.linalg.eig(ann_covar)
    except:
        print('Error in Eigen decomposition of covariance matrix')
        eigvals = []
        sys.exit(-1)
    if min(eigvals) <= 0:
        print('Error!  Negative eigenvalues in covariance matrix detected!')
        sys.exit(-1)
    return ann_covar

def kelly_optimize_unconstrained(M:pd.DataFrame, C:pd.DataFrame)->pd.DataFrame:
    "calc unconstrained kelly weights"
    results = np.linalg.inv(C) @ M
    kelly = pd.DataFrame(results.values, index=C.columns, columns=['Weights'])
    return kelly

def kelly_optimize(M_df:pd.DataFrame, C_df:pd.DataFrame, config:Dict)->pd.DataFrame:
    "objective function to maximize is: g(F) = r + F^T(M-R) - F^TCF/2"
    r = config['annual_risk_free_rate']
    M = M_df.to_numpy()
    C = C_df.to_numpy()

    n = M.shape[0]
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    G = matrix(0.0, (n, n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n, 1))
    try:
        max_pos_size = float(config['max_position_size'])
    except KeyError:
        max_pos_size = None
    try:
        min_pos_size = float(config['min_position_size'])
    except KeyError:
        min_pos_size = None
    if min_pos_size is not None:
        h = matrix(min_pos_size, (n, 1))

    if max_pos_size is not None:
       h_max = matrix(max_pos_size, (n,1))
       G_max = matrix(0.0, (n, n))
       G_max[::n+1] = 1.0
       G = matrix(np.vstack((G, G_max)))
       h = matrix(np.vstack((h, h_max)))

    S = matrix((1.0 / ((1 + r) ** 2)) * C)
    q = matrix((1.0 / (1 + r)) * (M - r))
    sol = qp(S, -q, G, h, A, b)
    kelly = np.array([sol['x'][i] for i in range(n)])
    kelly = pd.DataFrame(kelly, index=C_df.columns, columns=['Weights'])
    return kelly

def display_results(df:pd.DataFrame, config:Dict, msg:str)->None:
    "display asset allocations"
    df['Capital_Allocation'] = df['Weights'] * config['capital']
    print(msg)
    print(df.round(2))
    cash = config['capital'] - df['Capital_Allocation'].sum()
    print('Cash:', np.round(cash))
    print('*'*100)

def kelly_implied(covar:pd.DataFrame, config:Dict)->pd.DataFrame:
    "caculate return rates implied from allocation weights: mu = C*F"
    F = pd.DataFrame.from_dict(config['position_sizes'], orient='index').transpose()
    F = F[covar.columns]
    implied_mu = covar @ F.transpose()
    implied_mu.columns = ['implied_return_rate']
    return implied_mu

def correlation_from_covariance(covariance:pd.DataFrame)->pd.DataFrame:
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def main():
    "load data and begin primary calculation"
    config = load_config('./config.json')
    prices = load_prices()
    excess_returns = annual_excess_returns(prices, config)
    covar = annual_covar(excess_returns, config)
    mu = pd.DataFrame(columns=covar.columns)
    if estimation_mode == 'identical':
        rate = config['identical_annual_excess_return_rate']
        mu.loc[0] = rate
    elif estimation_mode == 'historical':
        mu.loc[0] = excess_returns.mean()*252
    elif estimation_mode == 'custom':
        rates = config['expected_annual_excess_return_rates']
        mu = pd.DataFrame.from_dict(rates, orient='index').transpose()
    else:
        print('unexpected estimation mode for annual excess return rates')
        sys.exit(-1)
    mu = mu[covar.columns].transpose()

    if implied is not None and implied.upper() == 'TRUE':
        implied_returns = kelly_implied(covar, config)
        print('*'*100)
        print(implied_returns.round(2))
        return 0
    print('*'*100)
    ann_excess_returns = mu
    ann_excess_returns.columns = ['Annualized Excess Returns']
    print(ann_excess_returns)
    print('*'*100)
    print('Estimated Correlation Matrix of Annualized Excess Returns (rounded to 2 decimal places)')
    print(correlation_from_covariance(covar).round(2))
    print('*'*100)
    unc_kelly_weights = kelly_optimize_unconstrained(mu, covar)
    display_results(unc_kelly_weights, config, 'Unconstrained Kelly Weights (no constraints on shorting or leverage')
    print('Begin optimization')
    kelly_weights = kelly_optimize(mu, covar, config)
    print('*'*100)
    display_results(kelly_weights, config, 'Allocation With Full Kelly Weights')
    kelly_fraction = float(config['kelly_fraction'])
    partial_kelly = kelly_fraction*kelly_weights
    display_results(partial_kelly, config,
                    'Allocation With Partial Kelly Fraction:'+str(kelly_fraction))
    return 0


price_data= None
main()
