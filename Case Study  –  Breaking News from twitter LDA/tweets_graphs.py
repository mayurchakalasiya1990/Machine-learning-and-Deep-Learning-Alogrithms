#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:18:50 2022

@author: geekarea
"""

import pandas as pd
from datetime import datetime

#The top 10 most active users by the number of tweets posted
def top_10_users_by_no_of_tweets(filename):
    df = pd.read_csv(filename)    
    #print(datetime.fromisoformat(df['created_at'][0]).hour)
    top_10_users = df.groupby("username")['username'].value_counts().nlargest(10).reset_index(level=1, drop=True)
    top_10_users.plot(kind = 'bar')



def tweets_per_hours(filename):
    df = pd.read_csv(filename)        
    tweet_per_hr = df.groupby(['date','hour']).size()
    print(tweet_per_hr)
    tweet_per_hr.plot(kind = 'bar')

#top_10_users_by_no_of_tweets("scraped_tweets.csv")
tweets_per_hours("scraped_tweets.csv")
