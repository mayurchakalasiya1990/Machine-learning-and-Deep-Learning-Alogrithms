# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import Libraries
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas_datareader as webreader


# The List of 20 type of cryptocurrency
coin_list = ['BUSD-USD','BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'USDT-USD', 'DOGE-USD', 'XLM-USD', 'DOT-USD', 'UNI7083-USD', 'LINK-USD', 'USDC-USD', 'BCH-USD', 'LTC-USD', 'GRT6719-USD', 'ETC-USD', 'FIL-USD', 'AAVE-USD', 'ALGO-USD', 'EOS-USD','BNB-USD','TRX-USD','UNI7083-USD','SHIB-USD','SOL-USD','DAI-USD','MATIC-USD']

#defining the dataframe
main_df = pd.DataFrame()

for coin in coin_list:
    coin_df = pd.DataFrame()
    df = pd.DataFrame(index=[0])
    
    # Defining the Start Date and End Date
    datetime_check = datetime(2018, 1, 1, 0, 0)
    datetime_end = datetime(2022, 10, 31, 0, 0)
    
    while len(df) > 0:
        if datetime_end == datetime_check:
            break
        
        #we are using the request to fetch the data from the api in the json format and then storing it into the dataframe.
        temp_data = webreader.DataReader(coin, start=datetime_check, end=datetime_end, data_source="yahoo")
        temp_data.reset_index(inplace=True)
        df = pd.DataFrame(temp_data)
        df.columns = ['Date', 'High', 'Low', 'Open', 'Close','Volume','Adj Close']
        
        #adding datetime and symbol to dataframe
        #df = df.drop(['Timestamp'], axis=1)
        #df['Datetime'] = [datetime_end - relativedelta(minutes=len(df)-i) for i in range(0, len(df))]
        coin_df = df.append(coin_df)
        datetime_end = datetime_check
        
    print('Coin:',coin)
    print('coin.shape:',coin_df.shape)
    coin_df['Symbol'] = coin
    main_df = main_df.append(coin_df)

main_df = main_df[['Date','Symbol', 'High', 'Low', 'Open', 'Close','Volume','Adj Close']].reset_index(drop=True)
main_df

main_df.to_csv('../data/final_df.csv', index=False)


