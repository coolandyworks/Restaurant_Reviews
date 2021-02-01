# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:34:45 2020

@author: tripa
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# cleaning the data
corpus = []
import re
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1]

# classify function
from sklearn.model_selection import cross_val_score, train_test_split
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)
    
    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)
    
import xgboost as xgb
model = xgb.XGBClassifier()
classify(model, X, y)

# using Random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model, X, y)

y_pred = model.pre

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_train)




