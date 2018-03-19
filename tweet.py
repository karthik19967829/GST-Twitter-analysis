# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:41:20 2017

@author: Akhilesh
"""

import tweepy
import re
#import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
consumer_key=''#type in your consumer key
consumer_secret=''#type in your customer secret key
access_token=''#type in your access token
access_token_secret=''#type in your secret access token
#nltk.download('stopwords')
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
public_tweets=api.search('GST')
fh=open("gstlstm.txt","a+")
#lines=[]
for tweet in public_tweets:
    review=tweet.text
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    #review=review.split()
    #ps=PorterStemmer()
    #review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review=' '.join(review)
    #lines.append("kar is a don")
    
    fh.write(review+"\n")
    analysis=TextBlob(review)
    print(review)
    print(analysis.sentiment)
#fh.writelines(lines)
fh.close()    
#print(lines)    
    