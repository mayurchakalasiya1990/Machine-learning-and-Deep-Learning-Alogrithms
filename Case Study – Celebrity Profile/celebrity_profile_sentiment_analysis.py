#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:23:22 2022

@author: geekarea
"""

import tweepy

import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
import codecs
import nltk
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

CONSUMER_KEY = "RaS51chGBc6FhFGfwz8hF78bL"
CONSUMER_SECRET = "7RlZnRB5hxDpYQUYMdO2UsqtbUaBAYsJ8vvEFZ5bFELqr7hler"
ACCESS_TOKEN = "1515612339739009024-Y5lk1rfAnXZaJFIAcgG8T6T6xRV0xj"
ACCESS_TOKEN_SECRET = "89LK9ZG5mnolzsEGeuyyNS16S2HGkidrTYx3PdSlEJvTr"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
prebuild_classifier = NaiveBayesAnalyzer()

def getTweets(twitterAccount):
    tweets = tweepy.Cursor(api.user_timeline, 
                       screen_name=twitterAccount, 
                       count=None,
                       since_id=None,
                       max_id=None,
                       trim_user=True,
                       exclude_replies=True,
                       contribubtor_details=False,
                       include_entities=False).items(50)

    df = pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])
    print(df.head())
    return df;

def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    txt = re.sub(r'#', '', txt)
    txt = re.sub(r'RT : ', '', txt)
    txt = re.sub(r'https?:\|\|[A-Za-z0-9\.\|]', '', txt)
    return txt

def getTxtPositivity(txt, prebuild_classifier):
    #print(TextBlob(txt,analyzer=prebuild_classifier).sentiment.p_pos)
    return TextBlob(txt,analyzer=prebuild_classifier).sentiment.p_pos

def getTxtNegativity(txt, prebuild_classifier):
    #print(TextBlob(txt,analyzer=prebuild_classifier).sentiment.p_neg)
    return TextBlob(txt,analyzer=prebuild_classifier).sentiment.p_neg

def getTextAnalysis(p_pos,p_neg):
    if p_neg >= 0.7:
        return "Negative"
    elif p_pos >= 0.7:
        return "Positive"
    else:
        return "Neutral"

#print('df.shape[0]',df.shape[0],' df[df[Score]==Positive].shape[0]',df[df['Score']=='Positive'].shape[0],'%',df[df['Score']=='Positive'].shape[0]/df.shape[0]*100)
#print('df.shape[0]',df.shape[0],' df[df[Score]==Negative].shape[0]',df[df['Score']=='Negative'].shape[0],'%',df[df['Score']=='Negative'].shape[0]/df.shape[0]*100)
#print('df.shape[0]',df.shape[0],' df[df[Score]==Neutral].shape[0]',df[df['Score']=='Neutral'].shape[0],'%',df[df['Score']=='Neutral'].shape[0]/df.shape[0]*100)

def getPositiveTweetPercetage(df):        
        return df[df['Score']=='Positive'].shape[0]/df.shape[0]*100

def getNegativeTweetPercetage(df):        
        return df[df['Score']=='Negative'].shape[0]/df.shape[0]*100
    
def getNeutralTweetPercetage(df):        
        return df[df['Score']=='Neutral'].shape[0]/df.shape[0]*100

def plotPieChart(pos,neg,neutral):
    labels=['Positive','Negative','Neutral']
    sizes = [pos,neg,neutral]
    colors = ['yellowgreen','lightcoral','gold']
    explode=(0,0.1,0)
    plt.pie(sizes,explode=explode,colors=colors,autopct='%1.1f%%',startangle=120)
    plt.legend(labels,loc=[-0.05,0.05],shadow=True)
    plt.axis('equal')
    plt.savefig('Sentiment_Analysis.png')

def plotPosVsNeg(df):
    labels = df.groupby('Score').count().index.values
    values = df.groupby('Score').size().values
    plt.bar(labels, values)

def plotPositivityNegativity(df):
    for index,row in df.iterrows():
        print(index)
        if row['Score'] == 'Positive':
            print(row['Score'],' Positive')
            plt.scatter(row['p_pos'], row['p_neg'], color='green')
        elif row['Score'] == 'Negative':
            print(row['Score'],' Negative')
            plt.scatter(row['p_pos'], row['p_neg'], color='red')
        elif row['Score'] == 'Neutral':
            print(row['Score'],' Neutral')
            plt.scatter(row['p_pos'], row['p_neg'], color='blue')
    plt.title("")
    plt.xlabel('p_pos')
    plt.ylabel('p_neg')
    plt.show()


df = pd.read_csv('tweets.csv')

df['Tweet'] = df['Tweet'].apply(cleanUpTweet)

df['p_pos']=df['Tweet'].apply(lambda x: getTxtPositivity(x,prebuild_classifier))
df['p_neg']=df['Tweet'].apply(lambda x: getTxtNegativity(x,prebuild_classifier))
df = df.drop(df[df['Tweet']==''].index)
df['Score']=df.apply(lambda x : getTextAnalysis(x['p_pos'],x['p_neg']), axis=1)
df.to_csv('tweets_output.csv')


pos = getPositiveTweetPercetage(df)
neg = getNegativeTweetPercetage(df)
neutral = getNeutralTweetPercetage(df)
#plotPieChart(pos, neg, neutral)
#plotPolarityVsSubjectivity(df)
#plotPoloritySubjectivity(df)