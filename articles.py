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
def Proc_KNN(train, labels, valid, valid_labels, dist = 10):
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
    ACC = []   
    for n in range(1, dist):
        jim = knn_text(train, labels, valid, valid_labels, n)[1]
        ACC.append(jim)    
    kbest_pred = knn_text(train, labels, valid, valid_labels, ACC.index(max(ACC)) + 1)
    return kbest_pred

def Proc_SVM(train, labels, valid, valid_labels, cost = 10):
    def svm_text(train, labels, valid, valid_labels, n = 5):
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', svm.SVC(C = n)),
        ])
        text_clf.fit(train, labels)
        predicted = text_clf.predict(valid)
        acc = np.mean(predicted == valid_labels)*100
        return [predicted, acc]
    ACC = []
    for n in range(1, cost):
        jam = svm_text(train, labels, valid, valid_labels, n)[1]
        ACC.append(jam)    
    cbest_pred = svm_text(train, labels, valid, valid_labels, ACC.index(max(ACC)) + 1)
    return cbest_pred   

#Process data       
J1 = Proc_KNN(dat_train, train_lab, dat_val, val_lab, 10)
        
J2 = Proc_SVM(dat_train, train_lab, dat_val, val_lab, 10)   
        
        