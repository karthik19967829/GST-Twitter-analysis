# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:25:43 2017

@author: Akhilesh
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
#dataset=pd.read_csv('gstanalysis.tsv',delimiter='\t',quoting=3)
fh=open("gstanalysis.txt","r")
content=fh.read().split("\t")
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
y=[]
for i in content:
    analysis=TextBlob(i)
    if analysis.polarity==0:
        y.append(0)
    elif analysis.polarity>0:
        y.append(1)
    elif analysis.polarity<0:
        y.append(-1)
y=pd.Series(y)
u=pd.Series(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)     
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(content).toarray()
#y=dataset.iloc[:,1]    


# Feature Scaling
# Fitting Naive Bayes to the Training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)