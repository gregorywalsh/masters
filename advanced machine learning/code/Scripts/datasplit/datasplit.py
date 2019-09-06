#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:05:07 2018

@author: diego
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from math import ceil

def just_msg_format(msg, *a, **b):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = just_msg_format

def reduce_training_splits(splits, class_labels, observation_reqs):
    """reduce_training_splits

    Reduce the number of samples in a training split, whilst trying to include a certain number of the target class.
    Useful for doing multi-class splits.

    :param splits:                      a generator which produces splits
    :param class_labels:                a series containing all classes labels in the same order as the dataframe
    :param target_classes:              a list containing at least one target class
    :param proportion:                  a float in the range 0-1 specifying the proportion of
    :param observation_reqs:            a dict with class labels as keys and number of required observations as values

    :return:    Yields training and test data splits with a reduced number of training_ids
    """
    need_warning = False
    for training_ids, test_ids in splits:

        reduced_training_ids = np.empty(shape=(0,), dtype=np.int64)
        Y_train = class_labels.iloc[training_ids]

        for target_class, required_number in observation_reqs.items():
            class_ids = Y_train.index[Y_train == target_class]
            available_num = class_ids.shape[0]

            if available_num < required_number:
                need_warning = True

            selected_ids = np.random.choice(a=class_ids,
                                            size=min(required_number, available_num),
                                            replace=False)

            reduced_training_ids = np.concatenate((reduced_training_ids, selected_ids), axis=0)

        yield reduced_training_ids, test_ids

    if need_warning:
        warnings.warn(message="Not all observation requirements could be met for all classes")


def dataSplit(df,n_samples = 500, test_size = 0.2, random_state = 0):
    
    """dataSplit
    
    Split the data into training and test set.
    
    Arguments:
    -----------
    df: Pandas dataframe
        dataframe with the raw data
    n_samples: int
        number of false samples to be included in the set
    test_size: float
        fraction of data to use in the test set
    random_state: int
        random seed
    """
    
    #DC split dataset into two classes
    df_fraud = df.loc[df['Class']==1]
    df_non_fraud = df.loc[df['Class']==0]
    
    #DC select a size N of samples of class 0
    df_non_fraud_selection = df_non_fraud.sample(n_samples,random_state=random_state)
    
    #DC generate a shuffled dataset with the two classes
    frames = [df_fraud, df_non_fraud_selection]
    dataset = pd.concat(frames)
    dataset = dataset.sample(frac=1,random_state = random_state)
    #print(dataset.head())
    
    #DC define training and test set
    X = np.array(dataset.drop(['Class'],1))
    y = np.array(dataset['Class'])
    
    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state = random_state)
    
    n_ones_training = sum(y_train)
    n_ones_test = sum(y_test)
    
    r_train = n_ones_training / len(y_train)
    r_test = n_ones_test / len(y_test)
    
    return X_train, X_test, y_train, y_test, r_train, r_test

def reduceSize(df,total_data = 3000, proportion = -1,random_state=0):
    """reduceSize
    
    Reduce the size of the dataset by removing negative features.
    
    Arguments:
    ---------
    df: Pandas dataframe
        dataframe with the raw data
    ntotal_data: int
        size of the set to be generated
    proportion: float (0-1)
        Fraction of positives over total. Set to -1 to keep original proportions
    random_state: int
        Random state seed
        
    """
    #Count values of different classes
    val_neg = df[df['Class']==0].count()[0]
    val_pos = df[df['Class']==1].count()[0]
    
    #Proportion
    if proportion == -1:
        proportion = val_pos / (val_neg +val_pos)
    elif proportion == -2:
        proportion = total_data - val_neg / total_data
    
    #Define number of negatives and positives in the split
    num_pos = ceil(total_data * proportion)
    if num_pos > val_pos:
        num_pos = val_pos
        warnings.warn("\nWarning: the proportion %.3f can't me met since there are only %i positives\n" %(proportion,val_pos))
        
    num_neg = total_data - num_pos
    
    #print(num_pos)
    #print(num_neg)
      
    #Split set into negatives and positives
    data_neg = df[df['Class']==0]
    data_pos = df[df['Class']==1]
    
    df_neg = data_neg.sample(num_neg)
    df_pos = data_pos.sample(num_pos)
    
    df_reduced = [df_neg,df_pos]
    df_reduced = pd.concat(df_reduced)
    df_reduced = df_reduced.sample(frac=1)
    
    df_reduced.reset_index(inplace=True)
    df_reduced.drop(['index'],1,inplace=True)
    
    return df_reduced


def train_test_Prop_Split(df,total_data = 3000, proportion = -1, test_size = 0.2):
    """train_test_Prop_Split
    
    Generates a Train-Test set with a specific proportion positive/negatives
    
    Arguments:
    ---------
    df: Pandas dataframe
        dataframe with the raw data
    total_data: int
        size of the set to be generated
    proportion: float (0-1)
        Fraction of positives over total. Set to -1 to keep original proportions
    test_size: float
        Fraction for the test size
        
    """
    
    #Count values of different classes
    val_neg = df[df['Class']==0].count()[0]
    val_pos = df[df['Class']==1].count()[0]
    
    #Proportion
    if proportion == -1:
        proportion = val_pos / (val_neg +val_pos)
    
    #Define number of negatives and positives in the split
    num_pos = ceil(total_data * proportion)
    num_neg = total_data - num_pos
    
    #Define size of test set
    num_test = ceil(test_size*total_data)
    num_test_pos = ceil(proportion * num_test)
    num_test_neg = num_test - num_test_pos
    
    num_train_pos = num_pos - num_test_pos
    num_train_neg = num_neg - num_test_neg
    
    
    #Split set into negatives and positives
    data_neg = df[df['Class']==0]
    data_pos = df[df['Class']==1]
    
    df_neg_train = data_neg.sample(num_train_neg)
    df_neg_test = data_neg.sample(num_test_neg)
    
    df_pos_train = data_pos.sample(num_train_pos)
    df_pos_test = data_pos.sample(num_test_pos)
    
    #define training set
    train = [df_neg_train,df_pos_train]
    train = pd.concat(train)
    train = train.sample(frac=1)
    
    #define test set
    test = [df_neg_test,df_pos_test]
    test = pd.concat(test)
    test = test.sample(frac=1)
    
    X_train = train.drop(['Class'],1)
    y_train = train['Class']
    
    X_test = test.drop(['Class'],1)
    y_test = test['Class']
    
    return X_train, X_test, y_train, y_test

"""
if __name__ == '__main__':
    df = pd.read_csv('../../../Dataset/creditcard.csv')
    #Xtr, Xts, ytr, yts, rtr, rts = dataSplit(df)

    df = reduceSize(df,total_data = 10000, proportion=-2)
    print("Reduced set: "+str(len(df.index))+", with " + str(df['Class'].mean() * 100) + "% fraud cases.")
    print("-------------------------------")

    for i, fold in enumerate(kFold(df, n_splits=1, test_size=0.01)):
        print("Fold "+str(i))

        train_set = df.iloc[fold[0]]
        print("Train set: "+str(len(fold[0]))+", with "+str(train_set['Class'].mean()*100)+"% fraud cases.")
        print("First indexes: "+str(fold[0][1:5]))

        test_set = df.iloc[fold[1]]
        print("Test set: "+str(len(fold[1]))+", with "+str(test_set['Class'].mean()*100)+"% fraud cases.")
        print("First indexes: " + str(fold[1][1:5]))



        print("-------------------------------")
"""