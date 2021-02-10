# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:46:38 2021

@author: charlescolgan
"""

#https://iq.opengenus.org/text-classification-using-k-nearest-neighbors/

#Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from datetime import datetime

start = datetime.now()

#Load Data
data_train = open("data_train.txt", "r")

data_valid = open("data_valid.txt", "r")

labels_train_original = open("labels_train_original.txt", "r")

labels_valid_original = open("labels_valid_original.txt", "r")

#Listize
dat_train = [line.strip() for line in data_train]

dat_val = [line.strip() for line in data_valid]

train_lab = [line.strip() for line in labels_train_original]

val_lab = [line.strip() for line in labels_valid_original]

#close files
data_train.close()

data_valid.close()

labels_train_original.close()

labels_valid_original.close()

#Endocde labels
for i in range(len(train_lab)):
    if train_lab[i] == "News":       
        train_lab[i] = 0        
    if train_lab[i] == "Opinion":
        train_lab[i] = 1        
    if train_lab[i] == "Classifieds":        
        train_lab[i] = 2        
    if train_lab[i] == "Features":        
        train_lab[i] = 3

for i in range(len(val_lab)):
    if val_lab[i] == "News":       
        val_lab[i] = 0        
    if val_lab[i] == "Opinion":
        val_lab[i] = 1        
    if val_lab[i] == "Classifieds":        
        val_lab[i] = 2        
    if val_lab[i] == "Features":        
        val_lab[i] = 3

#Form Algo
def knn_text(train, labels, valid, valid_labels, n = 5):                       
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=n)),
    ])
    text_clf.fit(train, labels)
    predicted = text_clf.predict(valid)
    acc = np.mean(predicted == valid_labels)*100
    return [predicted, acc]

def svm_text(train, labels, valid, valid_labels, n = 5):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.SVC(C = n, kernel = "linear")),
    ])
    text_clf.fit(train, labels)
    predicted = text_clf.predict(valid)
    acc = np.mean(predicted == valid_labels)*100
    return [predicted, acc] 


#Process data       
N = list(range(1,26))

Res1 = []

Res2 = []

for i in N:
    Res1.append(knn_text(dat_train, train_lab, dat_val, val_lab, i)[1])
    
    Res2.append(svm_text(dat_train, train_lab, dat_val, val_lab, i)[1])
 
Res = pd.DataFrame({"N":N, "KNN":Res1, "SVM":Res2})

#Graph Resultts
Res.plot(x = "N", y = ["KNN", "SVM"])

#Get process time
stop = datetime.now()

print("Total time elapsed: ", stop - start)