#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:46:32 2018

@author: diego
"""

import numpy as np
from sklearn import svm, preprocessing, cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd

#DC importing Dataset
df = pd.read_csv('./Dataset/creditcard.csv')

#DC split dataset into two classes
df_fraud = df.loc[df['Class']==1]
df_non_fraud = df.loc[df['Class']==0]

#DC select a size N of samples of class 0
N = 10000
df_non_fraud_selection = df_non_fraud.sample(N,random_state=0)

#DC generate a shuffled dataset with the two classes
frames = [df_fraud, df_non_fraud_selection]
dataset = pd.concat(frames)
dataset = dataset.sample(frac=1,random_state = 0)
print(dataset.head())

#DC define training and test set
X = np.array(dataset.drop(['Class','Time'],1))
y = np.array(dataset['Class'])

X_train, X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state = 0)

#DC tuned parameters
no_C = 0
no_G = 0
C= []
n= 2e-3
while n < 2e9:
    C.append(n)
    n *=2
    no_C += 1
#print(C)

gamma= []
n= 2e-8
while n < 2e3:
    gamma.append(n)
    n *=2
    no_G += 1
#print(gamma)

kernels = ['linear','poly','rbf']

#DC simple classifier
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

results = np.zeros((no_G,no_C+1))
col_name = np.copy(C)
col_name = np.insert(col_name,0,0)
file_name = 'results_%i.csv' %N
for i in range(0,no_G):
    results[i][0] = gamma[i]
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
for i in range(0,no_G):
    g = gamma[i]
    for j in range(0,no_C):
        c = C[j]
        print('Gamma: %.6f, C: %.2f' %(g,c))
        classifier = svm.SVC(kernel=kernels[2], gamma= g, C=c,random_state=0)
        classifier.fit(X_train,y_train)
        y_score = classifier.decision_function(X_test)
        
        #DC average precision
        average_precision = average_precision_score(y_test, y_score)
        #print('Average precision-recall score: {0:0.4f}'.format(average_precision))
        
        #DC plot the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        results[i][j+1] = average_precision
    
    df = pd.DataFrame(data = results)
    #print(col_name)
    df.columns = col_name
    df.to_csv(file_name, index=False)
    
        #plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        



""" THIS IS FOR GRID SEARCH USING SKLEARN
tuned_parameters = [{'kernel': ['rbf'], 'gamma':gamma,'C':C},{'kernel':['linear'],'C':[1, 10, 100, 1000]}]
scores = ['precision','recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

"""
