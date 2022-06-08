#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:13:46 2022

@author: geekarea
"""

import datetime
import tweepy
import pandas as pd
import requests
import re
import preprocessor as p
import sys

consumer_key = "RaS51chGBc6FhFGfwz8hF78bL"
consumer_secret = "7RlZnRB5hxDpYQUYMdO2UsqtbUaBAYsJ8vvEFZ5bFELqr7hler"
access_token = "1515612339739009024-Y5lk1rfAnXZaJFIAcgG8T6T6xRV0xj"
access_token_secret = "89LK9ZG5mnolzsEGeuyyNS16S2HGkidrTYx3PdSlEJvTr"
bearer_token ="AAAAAAAAAAAAAAAAAAAAAH5XbgEAAAAAPKdN0ASoOQ4A0mlbZzPUlyoe%2BeU%3DF4akxLj5YhNRFSeC630hNA1nqJT7fCmxV24ZKCd28nGi6poFbH"
tweet_per_day = 100

def validate_date_range(from_date,to_date):    
    try:        
        d1=datetime.datetime.strptime(from_date,'%Y-%m-%d')
        d2=datetime.datetime.strptime(to_date,'%Y-%m-%d')
        print((d2-d1).days)
        if (((d2-d1).days + 1) == 3):
            return True
    except ValueError:
        return False
    return False

def validate_date_input(date):
    if len(date) == 10:
         x = date.split("-", 2)
         if len(x[0]) == 4 and len(x[1])==2 and len(x[2]) == 2:
             try:        
                 datetime.datetime.strptime(date, '%Y-%m-%d')        
                 return True
             except ValueError:
                 return False
    return False

def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    txt = re.sub(r'#', '', txt)
    txt = re.sub(r'RT : ', '', txt)
    txt = re.sub(r'https?:\|\|[A-Za-z0-9\.\|]', '', txt)
    return txt


    
def fetch_tweet(from_date,to_date,query):
    client = tweepy.Client( bearer_token=bearer_token, 
                            consumer_key=consumer_key, 
                            consumer_secret=consumer_secret, 
                            access_token=access_token, 
                            access_token_secret=access_token_secret, 
                            return_type = requests.Response,
                            wait_on_rate_limit=True)
   
    for x in range(int(tweet_per_day/100)):
        tweets = client.search_recent_tweets(query=query,                                  
                                      start_time=from_date,
                                      end_time=to_date, 
                                      max_results=100)
        
        
        # Save data as dictionary
        tweets_dict = tweets.json() 
        #print(tweets_dict)
        # Extract "data" value from dictionary
        tweets_data = tweets_dict['data'] 
        
        # Transform to pandas Dataframe
        df = pd.json_normalize(tweets_data) 
        df['Tweet'] = df['text'].apply(cleanUpTweet)
        df['Tweet'] = df['Tweet'].apply(lambda x: p.clean(str(x)))    
        with open('tweets.csv', 'a') as f:
            df.to_csv(f, header=False)
        #print(df)


query = input("Enter Query to Search Tweets for Celebraty profile: ")
if(len(query) > 512):    
    sys.exit("Query Length Should be not greater than 512 characters.") 
    
from_date = input("Enter From date(yyyy-mm-dd): ")
if(not validate_date_input(from_date)):
    sys.exit("Please Enter Validate Date Formate (YYYY-MM-DD)")
   
    
to_date = input("Enter To date(yyyy-mm-dd): ")
if(not validate_date_input(to_date)):
    sys.exit("Please Enter Validate Date Formate (YYYY-MM-DD)")

if(not validate_date_range(from_date,to_date)):
    sys.exit(" Please enter Date Range for exact 3 days.")
   

d1=datetime.datetime.strptime(from_date,'%Y-%m-%d')
d2=datetime.datetime.strptime(to_date,'%Y-%m-%d')

fetch_tweet(d1,d2,query)



"""
query: #AskSRK OR #RedChilliesentertainment OR (srk mannat) OR SRKians OR #ShahRukhKhan    


from_date = '2022-05-01'
to_date = '2022-05-03'
d1=datetime.datetime.strptime(from_date,'%Y-%m-%d')
d2=datetime.datetime.strptime(to_date,'%Y-%m-%d')
query = '(#AskSRK OR #RedChilliesentertainment OR (srk mannat) OR SRKians OR #ShahRukhKhan) lang:en -is:retweet'
fetch_tweet(d1,d2,query)
"""
