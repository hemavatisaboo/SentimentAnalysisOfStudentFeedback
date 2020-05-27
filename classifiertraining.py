# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:34:34 2017

@author: kamini
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lexicon_calculate
#import pickle
from sklearn.externals import joblib

listofapostrophe=['\'s','\'m','\'d','n\'t','\'ve','\'re','\'\'']
STOPWORDS = stopwords.words('english') + list(punctuation) + listofapostrophe

def tokenize(text):
    words = word_tokenize(text)
    #print(words)
    words = [w.lower() for w in words]
    return [w for w in words if w not in STOPWORDS]
    
#corpus = ['This is the first document.',\
#'This isn\'t the, second - second document.',\
#'And the third one.',\
#'Is this the first document?']
corpus=['positivefeedback.txt','negativefeedback.txt','neutralfeedback.txt']
vectorizer = TfidfVectorizer(input='filename',stop_words=STOPWORDS)
tfidfmatrix=vectorizer.fit_transform(corpus)    
a=vectorizer.vocabulary_
idf=vectorizer.idf_
words=vectorizer.get_feature_names()
tfidfPos=tfidfmatrix.toarray()[0].tolist()
tfidfNeg=tfidfmatrix.toarray()[1].tolist()
tfidfNeu=tfidfmatrix.toarray()[2].tolist()


outcsvName=open('output.csv','w')
fid=open('database3.csv','r')
count=0
names=[]
X=[]
Y=[]
Z=[]
for line in fid:
    [text,label]=line.strip().split(',')
    words = tokenize(text)
    count+=1
    tfidfpos=0
    tfidfneg=0
    tfidfneu=0
    for word in words:
        ind=words.index(word)
        tfidfpos+=tfidfPos[ind]
        tfidfneg+=tfidfNeg[ind]
        tfidfneu+=tfidfNeu[ind]
    lexicon=lexicon_calculate.lexiconcal(line)
    Z.append(lexicon)
    X.append([tfidfpos,tfidfneg,tfidfneu,lexicon])
    names.append('Comment'+str(count))
    Y.append(label)
   

model=RandomForestClassifier(random_state=4, max_features=3)
X1=np.array(X)
Y=np.array(Y).astype(int)
model.fit(X1,Y)

#s=pickle.dumps(model)
joblib.dump(model,'tfidflexiconClassifier.pkl')