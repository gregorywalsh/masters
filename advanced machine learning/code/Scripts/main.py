#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:03:48 2018

@author: diego
"""
import pandas as pd
import scripts.datasplit.datasplit as ds
import scripts.datasplit.preprocessing as pre
import scripts.modeltesting.modeltesting as mt
from sklearn import svm

df = pd.read_csv("../Dataset/creditcard.csv")

kfolds = ds.kFold(df, n_splits=10, test_size=0.01)

for indexes in kfolds:

    train_set = df.iloc[indexes[0], :]
    val_set = df.iloc[indexes[1], :]

    # Run preprocessing on the data
    # df = pre.preprocess(train_set)
    # df = pre.get_processed_data(path='../dataset/creditcard.csv')

    # Reducing size of dataframe by removing non_fraud data
    my_set = ds.reduceSize(df, total_data=10000, proportion=-2)

    # Define classifier
    my_clf = svm.SVC

    # Define parameters for classifier
    # The first number in the array defines the type of parameter:
    # Use 0 for continuous parameter, 1 for discrete
    c_param = [0, -10, 12]
    g_param = [0, -7, 5]
    d_param = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    k_param = [1, 'rbf']

    # Generate dictionary with parameters
    my_param = {'C': c_param, 'gamma': g_param, 'kernel': k_param, 'class_weight': [1, 'balanced', None],
                'degree': d_param}

    # test the model
    a = mt.test(my_clf, my_param, my_set, random_state=3, cv=3, iterations=1000)
