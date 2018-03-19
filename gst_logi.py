# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:41:15 2017

@author: Akhilesh P Patil
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

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
