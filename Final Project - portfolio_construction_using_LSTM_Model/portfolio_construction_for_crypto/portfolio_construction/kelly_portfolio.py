# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:08:17 2022

@author: ChakalasiyaMayurVash
"""

import argparse
import datetime
import json
import sys
import numpy as np
import pandas as pd
import yfinance
from cvxopt import matrix
from cvxopt.solvers import qp
from sklearn.covariance import LedoitWolf
from typing import Dict

class kelly_portfolio:
    
    def load_prices(config:Dict)->pd.DataFrame:
        "load prices from web or from local file"
        if OPTIONS.price_data is not None:
            try:
                #Expects a CSV with Date, Symbol header for the prices, i.e. Date, AAPL, GOOGL
                price_data = pd.read_csv(OPTIONS.price_data, parse_dates=['Date'])
                price_data.set_index(['Date'], inplace=True)
            except (OSError, KeyError):
                print('Error loading local price data from:', OPTIONS.price_data)
                sys.exit(-1)
        else:
            stock_symbols, crypto_symbols = [], []
            start_date = (datetime.datetime.today()
                          - datetime.timedelta(days=365*config['max_lookback_years'])).date()
            end_date = datetime.datetime.today().date() - datetime.timedelta(days=1)
            try:
                if 'stock_symbols' in config['assets'].keys():
                    stock_symbols = config['assets']['stock_symbols']
                if 'crypto_symbols' in config['assets'].keys():
                    crypto_symbols = config['assets']['crypto_symbols']
                symbols = sorted(stock_symbols + crypto_symbols)
            except KeyError:
                print('Error retrieving symbols from config file. Config file should be \
                       formatted in JSON such that config[\'assets\'][\'stock_symbols\'] \
                       is valid. See example config file from GitHub')
                sys.exit(-1)
            if len(symbols) > 0:
                print('Downloading adjusted daily close data from Yahoo! Finance')
                try:
                    price_data = yfinance.download(symbols, start=str(start_date), end=str(end_date),
                                                   interval='1d', auto_adjust=True, threads=True)
                except:
                    print('Error downloading data from Yahoo! Finance')
                    sys.exit(-1)
                cols = [('Close', x) for x in symbols]
                price_data = price_data[cols]
                price_data.columns = price_data.columns.get_level_values(1)
                price_data.to_csv('sample_data.csv', header=True)
        price_data = price_data.sort_index()
        return price_data
    
    def annual_excess_returns(prices:pd.DataFrame, config:Dict)->pd.DataFrame:
        '''Stock data only changes on weekdays. Crypto data is available all days.
       Compute daily returns using Friday to Monday returns for all data'''
        returns = prices[prices.index.dayofweek < 5].pct_change(1)
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
    
    def display_results(df:pd.DataFrame, config:Dict, msg:str)->None:
        "display asset allocations"
        df['Capital_Allocation'] = df['Weights'] * config['capital']
        print(msg)
        print(df.round(2))
        cash = config['capital'] - df['Capital_Allocation'].sum()
        print('Cash:', np.round(cash))
        print('*'*100)       
        
    def correlation_from_covariance(covariance:pd.DataFrame)->pd.DataFrame:
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation
    
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

    def kelly_implied(covar:pd.DataFrame, config:Dict)->pd.DataFrame:
        "caculate return rates implied from allocation weights: mu = C*F"
        F = pd.DataFrame.from_dict(config['position_sizes'], orient='index').transpose()
        F = F[covar.columns]
        implied_mu = covar @ F.transpose()
        implied_mu.columns = ['implied_return_rate']
        return implied_mu