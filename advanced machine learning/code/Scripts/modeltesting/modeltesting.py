#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:54:34 2018

@author: diego
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
import random

def test(classifier,param,df, cv = 3, iterations= 3, test_size = 0.2, random_state = 0):
    X = df.drop(['Class'],1)
    y = df['Class']
    
    list_of_parameters = ['F1 Score','Precision','Recall']
    for key in param:
        list_of_parameters.append(key)
        
   
    #DC dataframe to store results    
    df_results = pd.DataFrame(columns = list_of_parameters)
    
    for j in range(0,iterations):
        
        #choose parameters
        sel_par = {}
        for key, value in param.items():
            if value[0] == 0:
                min_v = value[1]
                max_v = value[2]
                p = 10**random.uniform(min_v,max_v)
            elif value[1]:
                p = random.choice(value[1:])
            sel_par[key] = p
            
        total_precision = 0
        total_recall = 0        
        for i in range(1,cv+1):
            #DC define splits
            X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state = random_state+i)
            clf = classifier(**sel_par)
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
                        
            prec,rec = precisionRecallScore(y_test.values,y_pred)
            total_precision +=prec
            total_recall += rec
            
        precision = total_precision/cv
        recall = total_recall/cv
        f1score = f1Score(precision,recall)
        
        print("Iteration %i: (%.3f) %s" % (j,f1score,sel_par))
        
        sel_par['F1 Score'] = f1score
        sel_par['Precision'] = precision
        sel_par['Recall'] = recall
        
        df_results = df_results.append(sel_par,ignore_index=True)
        df_results = df_results.sort_values(by='F1 Score',ascending = False)
  
    return df_results

def precisionRecallScore(y_true,y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0,len(y_true)):
        if y_pred[i] ==1:
            if y_pred[i] == y_true[i]:
                tp +=1
            else:
                fp +=1
        else:
            if y_pred[i] == y_true[i]:
                tn +=1
            else:
                fn +=1
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    return precision, recall

def f1Score(precision,recall):
    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)


