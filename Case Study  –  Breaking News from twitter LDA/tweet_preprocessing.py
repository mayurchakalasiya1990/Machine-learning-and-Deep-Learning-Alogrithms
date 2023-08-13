#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:35:00 2022

@author: geekarea
"""

import preprocessor as p
import pandas as pd

def clean_tweet_data(filename,field_list):    
    df = pd.read_csv(filename)
    for col in df.columns:        
        if col in field_list:            
            print("Column name===>",col)
            print("before cleaning:",df[col])
            df[col] = df[col].apply(lambda x: p.clean(str(x)))
            print("After cleaning:",df[col])
    df.to_csv(filename)


clean_tweet_data('scraped_tweets.csv', ['text'])

