#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:43:10 2018

@author: diego
"""

import pandas as pd
#from datasplit.datasplit import *
#from modeltesting.modeltesting import *
from sklearn import svm
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing




#df = df - df.mean()

def fisherDiscriminant(df):
    my_set = df
    
    my_arr = my_set.as_matrix()
    
    x = range(0,my_arr.shape[1]-3) 
    
    
    my_set = (my_set - my_set.mean()) / my_set.std()
    
    neg_set = my_set.loc[my_set['Class'] < 0]
    pos_set = my_set.loc[my_set['Class'] > 0]
    
    neg_set = neg_set.drop(['Class'],1)
    pos_set = pos_set.drop(['Class'],1)
    
    neg_set_mean = neg_set.mean()
    pos_set_mean = pos_set.mean()
    
    neg_set_std = neg_set.std()
    pos_set_std = pos_set.std()
    
    set_mean_diff = abs(pos_set_mean - neg_set_mean)
    
    
    a= pd.DataFrame({'A':neg_set_std,'B':pos_set_std})
    
    #df_analysis = pd.DataFrame({'|m1-m2|':set_mean_diff,'|s1/s2|':set_std_diff})
    
    
    set_std_add = a['A']**2 + a['B']**2
    
    df_analysis = pd.DataFrame({'|m1-m2|':set_mean_diff,'|s1^2+s2^2|':set_std_add})
    
    #df_analysis = abs(df_analysis - df_analysis.mean())/df_analysis.std()
    
    df_analysis ['Fisher']=(df_analysis['|m1-m2|']/df_analysis['|s1^2+s2^2|'])
    
    df_analysis.sort_values(by='Fisher',ascending=True,inplace=True)
    
    order = df_analysis.index.values
    
    
    x = list(range(0,neg_set.shape[1]))
    #for i in range(0,len(neg_set)):
    #    plt.boxplot(neg_set.values[:,i])
    
    """
    fig, ax = plt.subplots()
    neg = neg_set.boxplot(ax = ax,positions =[x-.2 for x in list(x)], widths = 0.2,patch_artist = True, notch = True,showfliers=False)
    #plt.xticks([x+.2 for x in list(x)])
    
    pos = pos_set.boxplot(ax=ax,positions =[x+.2 for x in list(x)],widths = 0.2,patch_artist = True, notch = True,showfliers=False)
    plt.xticks([x for x in list(x)])
    
    for patch in neg['boxes']:
        patch.set_facecolor('lightblue')
    for patch in pos['boxes']:
        patch.set_facecolor('lightcoral')
    
    #ax.set_ylim(-1,1)
    #ax.set_xlim(-1,29)
    plt.show()
    """
    return order

if __name__ == '__main__':
    
    #read data
    df = pd.read_csv('../../Dataset/creditcard.csv')
    df = df / df.std()
    mylist = fisherDiscriminant(df)