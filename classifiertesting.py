# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:49:15 2017

@author: kamini
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lexicon_calcuate
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


line=raw_input('Enter the Feedback:')
text=line.strip()
words = tokenize(text)
tfidfpos=0
tfidfneg=0
tfidfneu=0
for word in words:
    ind=words.index(word)
    tfidfpos+=tfidfPos[ind]
    tfidfneg+=tfidfNeg[ind]
    tfidfneu+=tfidfNeu[ind]
lexicon=lexicon_calculate.lexiconcal(line)
Z=lexicon
X=[tfidfpos,tfidfneg,tfidfneu,lexicon]
X1=np.array(X)

model=joblib.load('tfidflexiconClassifier.pkl')
Ytest_predprob1=model.predict_proba(X1)    
Ytest_predclasspos=Ytest_predprob1[:,2]
Ytest_predclassneg=Ytest_predprob1[:,0]
Ytest_predclassneu=Ytest_predprob1[:,1]
Ytest_predpos=(Ytest_predclasspos>0.75).astype(np.int)
Ytest_predneg=(Ytest_predclassneg>0.25).astype(np.int)*-1
Ytest_pred=Ytest_predpos+Ytest_predneg
if Ytest_pred==1:
    print ('Positive')
elif Ytest_pred==-1:
    print ('Negative')
