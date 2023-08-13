# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:55:16 2022

@author: ChakalasiyaMayurVash
"""

import json
import numpy as np
import os
import pandas as pd
import urllib.request
import h5py

# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1356998100&end=9999999999&period=300'

# parse json returned from the API to Pandas DF
"""
openUrl = urllib2.urlopen(url)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())

res = urllib.request.urlopen(url)
r = res.read()
d = json.loads(r.decode())

df = pd.DataFrame(d)
"""

coins = ['BTC']
df_list=[]
for coin in coins:
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_'+coin+'&start=1439014500&end=9999999999&period=300'
    res = urllib.request.urlopen(url)
    r = res.read()
    d = json.loads(r.decode())
    print(pd.DataFrame(d).shape)

df = pd.DataFrame(d)
original_columns=[u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close','Timestamp','High','Low','Open']
df = df.loc[:,original_columns]
df.columns = new_columns
df.to_csv('../data/bitcoin2015to2017.csv',index=None)



with h5py.File(''.join(['../data/bitcoin2015to2017_wf.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value
    input_times = hf['input_times'].value
    output_times = hf['output_times'].value
    original_inputs = hf['original_inputs'].value
    original_outputs = hf['original_outputs'].value
    original_datas = hf['original_datas'].value
    
print(original_inputs[0].shape)
df.to_csv('data/bitcoin2015to2017.csv',index=None)
