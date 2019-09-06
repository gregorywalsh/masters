#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')

import pandas as pd
import datasplit.preprocessing as pp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def loadData():
    #Reading and preprocessing data
    df = pp.get_processed_data('../../Dataset/creditcard.csv')      
    #Reading splits
    splits = pd.read_pickle("../cvsplits/cvsplits.pkl")
    return df,splits

if __name__=='__main__':
    df,splits = loadData()
    df_pos = df[df['Class']==1]
    df_pos = df_pos.drop(['Class'],1)

    clf = KMeans(n_clusters = 5)
    clf.fit(df_pos)
    
    lbl = clf.labels_
    
    df_pos['Class'] = lbl
    
    class0 = df_pos[df_pos['Class']==0]