# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:16:53 2022

@author: ChakalasiyaMayurVash
"""

import pandas as pd

df = pd.read_csv('./Bitcoin_tweets.csv')

start_date = '2020-07-02'
end_date = '2021-08-03'



# Select DataFrame rows between two dates
mask = (df['date'] > start_date) & (df['date'] <= end_date)
df_out =  df.loc[mask]

df_out.to_csv('./Bitcoin_tweets_2020_2021.csv')
