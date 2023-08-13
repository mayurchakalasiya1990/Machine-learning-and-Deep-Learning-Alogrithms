# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:17:40 2022

@author: ChakalasiyaMayurVash
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PastSampler import PastSampler

#data file path
dfp = '../data/bitcoin2017to2021.csv'
file_name='../data/bitcoin2017to2021_close.h5'

#Columns of price data to use
columns = ['close']
df = pd.read_csv(dfp)
time_stamps = df['unix']
df = df.loc[:,columns]
original_df = pd.read_csv(dfp).loc[:,columns]


scaler = MinMaxScaler()
# normalization
for c in columns:
    df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))
    
#Features are input sample dimensions(channels)
A = np.array(df)[:,None,:]
print(A)
original_A = np.array(original_df)[:,None,:]
time_stamps = np.array(time_stamps)[:,None,None]

#Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 256, 16         #Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
input_times, output_times = ps.transform(time_stamps)
original_B, original_Y = ps.transform(original_A)

import h5py
with h5py.File(file_name, 'w') as f:
    f.create_dataset("inputs", data = B)
    f.create_dataset('outputs', data = Y)
    f.create_dataset("input_times", data = input_times)
    f.create_dataset('output_times', data = output_times)
    f.create_dataset("original_datas", data=np.array(original_df))
    f.create_dataset('original_inputs',data=original_B)
    f.create_dataset('original_outputs',data=original_Y)