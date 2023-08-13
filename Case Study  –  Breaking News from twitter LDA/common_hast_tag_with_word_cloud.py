#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:37:57 2022

@author: geekarea
"""
import codecs
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import ast
import json

word_counter = Counter()
en_stopwords = stopwords.words("english")

def make_wordcloud(counts):
 cloud = WordCloud(width=800, height=400)
 cloud.generate_from_frequencies(dict(counts.most_common(200)))
 image = cloud.to_image()
 image.save("wordcloud.png")


def find_common_tweets_from_file(filename):
    df = pd.read_csv(filename)
    
    for hashtags in df['hashtags']:    
        try:
            hastags_x = list(eval(hashtags))    
            word_counter.update(hastags_x)               
        except:
            print("An exception occurred")
    
    print(word_counter)
    make_wordcloud(word_counter)

find_common_tweets_from_file("scraped_tweets.csv")
