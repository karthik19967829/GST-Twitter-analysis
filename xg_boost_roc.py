# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:47:05 2017

@author: Akhilesh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
