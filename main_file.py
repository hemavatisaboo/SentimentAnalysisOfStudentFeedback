# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:01:55 2017

@author: hemavati
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
#import sklearn.metrics as metric

import lexicon_calculate

listofapostrophe=['\'s','\'m','\'d','n\'t','\'ve','\'re','\'\'']
STOPWORDS = stopwords.words('english') + list(punctuation) + listofapostrophe

def tokenize(text):
    words = word_tokenize(text)
    #print(words)
    words = [w.lower() for w in words]
    return [w for w in words if w not in STOPWORDS]
    
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
#    fidfeature.write('Comment'+str(count)+','+str(tfidfpos)+','+str(tfidfneg)+','+str(tfidfneu)+','+label+'\n')
    names.append('Comment'+str(count))
#    X.append([tfidfpos,tfidfneg,tfidfneu])
    Y.append(label)
   
 
#modelsvm=svm.SVC()
model=RandomForestClassifier(random_state=4, max_features=3)
#scores=cross_val_score(model,np.array(X),np.array(Y).astype(np.int))
X1=np.array(X)
Y=np.array(Y).astype(int)
Y_test=[]
Ytest_predclass=[]
Ytest_predclasspos=[]
Ytest_predclassneg=[]
Ytest_predclassneu=[]
audioNames=np.array([])
skf = StratifiedKFold(n_splits=10, shuffle=False)
for train_index, test_index in skf.split(X1, Y):
    X_train1, X_test1 = X1[train_index], X1[test_index]
    Y_train1, Y_test1 = Y[train_index], Y[test_index]
    
    model.fit(X_train1,Y_train1)
    Ytest_predprob1=model.predict_proba(X_test1)    
    Ytest_predclass1=np.array(Ytest_predprob1[:,0])

#    modelsvm.fit(X_train1,Y_train1)
#    Ytest_predclass1=modelsvm.predict(X_test1)

    Ytest_predclass.extend(Ytest_predclass1)
    Y_test.extend(Y_test1)
    nameslist=np.array(names)[test_index]
    audioNames=np.append(audioNames,nameslist)
    
    Ytest_predclasspos.extend(Ytest_predprob1[:,2])
    Ytest_predclassneg.extend(Ytest_predprob1[:,0])
    Ytest_predclassneu.extend(Ytest_predprob1[:,1])
#    Ytest_predclass1pos=(Ytest_predclasspos>=0.75).astype(np.int)
#    Ytest_predclass.extend(Ytest_predclass1pos)

Ytest_predclasspos=np.array(Ytest_predclasspos)
Ytest_predclassneu=np.array(Ytest_predclassneu)
Ytest_predclassneg=np.array(Ytest_predclassneg)
Ytest_predpos=(Ytest_predclasspos>0.75).astype(np.int)
Ytest_predneg=(Ytest_predclassneg>0.25).astype(np.int)*-1
Ytest_pred=Ytest_predpos+Ytest_predneg
#Ytest_predclass=np.array(Ytest_predclass)
Y_test=np.array(Y_test)
counts=0
pos=0
neg=0
neu=0
#for thresh in np.arange(0.4,0.95,0.1):
#for thresh in [0.1,0.2,0.3]:
#    Ytest_pred=(Ytest_predclass>thresh).astype(np.int)
for i in range(0,len(Y_test)):
    if Y_test[i]==Ytest_pred[i]:
        counts+=1
        if Y_test[i]==1:
            pos+=1
        elif Y_test[i]==-1:
            neg+=1
        else:
            neu+=1
#Ytest_pred=np.array(Ytest_predclass)
error=Y_test-Ytest_pred 
    
#print(len(error))    
acc=100.0*(len(error)-np.count_nonzero(error))/len(error)
print ("Model Accuracy (%) ",acc)
#print(counts)
#print(pos)
#print(neg)
#print(neu)
outcsv=np.transpose(np.vstack((audioNames,Y_test,Ytest_pred)))
csv_writer=csv.writer(outcsvName)
csv_writer.writerows(outcsv)
fid.close()