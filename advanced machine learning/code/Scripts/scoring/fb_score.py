#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    return [precision, recall],[tp,fp,tn,fn]

def fbScore(precision,recall,beta):
    if precision + recall == 0:
        return 0
    else:
        return (1+beta**2)*precision*recall/((beta**2)*precision+recall)