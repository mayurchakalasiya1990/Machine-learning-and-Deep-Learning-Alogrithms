# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tweepy
import sys
import os 
import pandas as pd
import datetime
import requests
import random


consumer_key = "RaS51chGBc6FhFGfwz8hF78bL"
consumer_secret = "7RlZnRB5hxDpYQUYMdO2UsqtbUaBAYsJ8vvEFZ5bFELqr7hler"
access_token = "1515612339739009024-Y5lk1rfAnXZaJFIAcgG8T6T6xRV0xj"
access_token_secret = "89LK9ZG5mnolzsEGeuyyNS16S2HGkidrTYx3PdSlEJvTr"
bearer_token ="AAAAAAAAAAAAAAAAAAAAAH5XbgEAAAAAPKdN0ASoOQ4A0mlbZzPUlyoe%2BeU%3DF4akxLj5YhNRFSeC630hNA1nqJT7fCmxV24ZKCd28nGi6poFbH"
tweet_per_day = 300

client = tweepy.Client( bearer_token=bearer_token, 
                 consumer_key=consumer_key, 
                 consumer_secret=consumer_secret, 
                 access_token=access_token, 
                 access_token_secret=access_token_secret, 
                 return_type = requests.Response,
                 wait_on_rate_limit=True)       

def save_to_csv(tweets,db):  
    
    #print("before ======>",db)
    # Save data as dictionary
    tweets_x = tweets.json() 
    print('tweets_x',tweets_x)
    #print(tweets['data'][0]['created_at'],'|',tweets['data'][0]['author_id'],'|',tweets['data'][0]['text'])
    i=0
    for tweet in tweets_x['data']:
        # now song is a dictionary
        #print('========================',i+1)          
        i=i+1
        #print(tweet)
        #print(attribute,":", value) # example usage
        hashtag_list=[]
        username=tweet['author_id']
        text=tweet['text']
        created_at=tweet['created_at']
        date=datetime.datetime.strptime(created_at,'%Y-%m-%dT%H:%M:%S.%f%z').date().strftime("%Y-%m-%d")
        hour=datetime.datetime.strptime(created_at,'%Y-%m-%dT%H:%M:%S.%f%z').hour        
        if 'entities' in tweet:
            for key, value in tweet['entities'].items():
                if key == 'hashtags':
                    for hashtag in value:
                        hashtag_list.append(hashtag['tag'])                               
                        
        
        ith_tweet = [username,text,hashtag_list,created_at,date,hour]
        #print("ith_tweet:",ith_tweet)        
        db.loc[len(db)] = ith_tweet
    
    #print("after ======>",db)
    fileName = 'scraped_tweets.csv'
    if not os.path.exists(fileName):
        header=True
    else:
        header=False
    # we will save our database as a CSV file.
    with open(fileName, 'a') as f:
        db.to_csv(f, header=header)

def scrape(query, start_time,end_time):        

    db = pd.DataFrame(columns=['username','text','hashtags','created_at','date','hour'])   
    rand_limit=random.randint(10, 100)
    tweets = client.search_recent_tweets(query=query,                                  
                          start_time=start_time,
                          tweet_fields=['author_id', 'created_at','entities'],
                          end_time=end_time, 
                          max_results=100)    
        
    #print("after ======>",db)
    save_to_csv(tweets,db)

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

# Enter Hashtag and initial date

query = input("Enter Query to Search tranding events or news : ")
if(len(query) > 512):    
    sys.exit("Query Length Should be not greater than 512 characters.") 

date_since = input("Enter Date since The Tweets are required in yyyy-mm-dd: ")
if(not validate_date_input(date_since)):
    sys.exit("Please Enter Validate Date Formate (YYYY-MM-DD)")

start_time=datetime.datetime.strptime(date_since+'T09:00:00Z','%Y-%m-%dT%H:%M:%SZ')        
end_time=datetime.datetime.strptime(date_since+'T09:59:00Z','%Y-%m-%dT%H:%M:%SZ')                

if start_time.weekday()==6 or start_time.weekday()==5:
    sys.exit("Please enter date for Weekday.")

scrape(query, start_time,end_time)
print('Scraping has completed!')

"""
query = "((breaking news) OR #Ukraine OR #TATAIPL OR covid OR News) lang:en -is:retweet"
date_since = "2022-05-06"
start_time=datetime.datetime.strptime(date_since+'T17:56:00Z','%Y-%m-%dT%H:%M:%SZ')        
end_time=datetime.datetime.strptime(date_since+'T17:59:00Z','%Y-%m-%dT%H:%M:%SZ')                
scrape(query, start_time,end_time)
print('Scraping has completed!')
"""