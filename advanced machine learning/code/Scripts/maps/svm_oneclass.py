#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../')

import pandas as pd
import numpy as np
import datasplit.preprocessing as pp
import datasplit.datasplit as ds
import visualisation.data_visualisation as dv
import random
from sklearn import svm
import scoring.fb_score as fb
import datetime



load_from_file = False
file_location = "./AE output/TOP1247-20180312T1405_epochs-20_Lin(31,27)-Lin(27,16)-Lin(16,31).pkl"

#parameters
no_random_search = 300

#set output folder
output_folder = "./SVM output/"

#File name
name = 'SVM'
now = datetime.datetime.now()
time = now.strftime("%Y%m%dT%H%M")
file_name = output_folder + time + "_combinations_" + str(no_random_search)

#Total score
total_scores = pd.DataFrame(columns=['Name','Reference','F_score_name','Precision','Recall','TP','FP','TN','FN','Score'])


#Reading and preprocessing data
df = pp.get_processed_data('../../Dataset/creditcard.csv')    
#Reading splits
splits = pd.read_pickle("../cvsplits/cvsplits.pkl")

#Statistics container
total_statistics = []

#Define parameters
# Use 0 for continuous parameter, 1 for discrete
c_param = [0, -10, 12]
g_param = [0, -7, 5]
n_param = [1, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]
d_param = [1, 2, 3, 4, 5]
k_param = [1, 'rbf']
w_param = [1, 'balanced', None]
train_size_list = [5000, 10000, 50000, 100000]

check_specific_model = False
if check_specific_model == True:
    #good classifier parameters (from tests)
    c_param = [1, 1.388371]
    g_param = [1, 1.82383e-06]
    d_param = [1, 2]
    k_param = [1, 'rbf']
    w_param = [1, 'balanced']
    train_size_list = [10000]
    features = dv.fisherDiscriminant(df)
    no_random_search = len(features)



# Generate dictionary with parameters
my_param = {'gamma': g_param, 'kernel': k_param, 'degree': d_param, 'nu':n_param}

total_statistics = []
stat_param = []

#Save info for positive classes
df_pos = df[df['Class']==1].copy()
df_pos['N_classified'] = 0
df_pos['Correct']=0

for k in range(no_random_search):
    if check_specific_model == True:
        if k !=0:
            print("Dropping feature [%s]" % features[k-1])
            df = df.drop(features[k-1],1)
        
    #choose parameters
    sel_par = {}
    par_string = ""
    for key, value in my_param.items():
       
        if value[0] == 0:
            min_v = value[1]
            max_v = value[2]
            p = 10**random.uniform(min_v,max_v)
        elif value[1]:
            p = random.choice(value[1:])
        sel_par[key] = p
        par_string = par_string + key + "(" + str(p) + ")"
    train_size = random.choice(train_size_list)
    par_string = par_string + "samples(%i)" % train_size
    
    print ("\n********* Iteration %i/%i *********" % (k+1,no_random_search) )
    print("Classifier architecture:")
    print(sel_par)
    print("Data size: %i" % train_size)
    
        
    #Iterate through the CV sets
    statistics = []
    

    
    for i,dataset in enumerate(splits):
        print("------ Cross validation step %i ------" % (i+1))
        #indices for train and test set
        train_id = dataset[0]
        test_id = dataset[1]
        #generate the splits
        train = df.loc[train_id]
        
        ################################ ESTOY AQUI
        #positive indices
        indices = []
        indices_ref = []
        for jj in range(len(test_id)):
            if test_id[jj] in df_pos.index:
                indices.append(test_id[jj])
                indices_ref.append(jj)
                
        #Updating number of times classified
        df_pos['N_classified'].loc[indices]+=1
        
        #Reduce size of train set
        train_reduced = ds.reduceSize(train,train_size,0.5)
        test = df.loc[test_id]
    
       
        #Divide set into training and test set
        X_train = train_reduced[train_reduced.Class == 0].drop(['Class'], axis=1)
        #X_train = train_reduced.drop(['Class'], axis=1)
        #y_train = train_reduced['Class']
        
        #Test set
        X_test = test.drop(['Class'], axis=1)
        y_test = test['Class']
        
        #To array
        X_train = X_train.values
        #y_train = y_train.values
        X_test = X_test.values
        y_test = y_test.values
        
        clf = svm.OneClassSVM(**sel_par)
        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        
        [precision,recall],[tp,fp,tn,fn] = fb.precisionRecallScore(y_test,y_pred)
        fbscore = fb.fbScore(precision,recall,1)
        f1 = fb.fbScore(precision,recall,1)
        f10 = fb.fbScore(precision,recall,10)
        f50 = fb.fbScore(precision,recall,50)
        st = [precision,recall,tp,fp,tn,fn,f1,f10,f50]
        #st = [fbscore,precision,recall,tp,fp,tn,fn]
        statistics.append(st)
        
        #updating list of correct classifications
        for jj in range(len(indices_ref)):
            if y_test[indices_ref[jj]]==y_pred[indices_ref[jj]]:
                df_pos['Correct'].loc[indices[jj]]+=1
            
        
        
    stat_param.append(par_string)
    total_statistics.append(np.mean(statistics,0))
    
    
    total = np.array(total_statistics)
        
    th_df = pd.DataFrame({
            'Parameters': stat_param,
            #'FB_mean':total[:,0],
            'Precision':total[:,0],
            'Recall':total[:,1],
            'TP':total[:,2],
            'FP':total[:,3],
            'TN':total[:,4],
            'FN':total[:,5],
            'F1':total[:,6],
            'F10':total[:,7],
            'F50':total[:,8],
            })

    th_df = th_df[['Parameters','Precision','Recall','TP','FP','TN','FN','F1','F10','F50']]   
    #th_df = th_df.sort_values(by=['Recall'],ascending=False) 
    th_df.to_csv(file_name+".csv")

    #Generate results
    f1_res = pd.DataFrame(th_df.iloc[th_df['F1'].idxmax()]).T
    f1_res = f1_res.drop(['F10','F50'],1)
    f1_res.rename(columns={'F1':'Score'},inplace=True)
    f1_res['F_score_name'] = 'F1'
    
    f10_res = pd.DataFrame(th_df.iloc[th_df['F10'].idxmax()]).T
    f10_res = f10_res.drop(['F1','F50'],1)
    f10_res.rename(columns={'F10':'Score'},inplace=True)
    f10_res['F_score_name'] = 'F10'
    
    f50_res = pd.DataFrame(th_df.iloc[th_df['F50'].idxmax()]).T
    f50_res = f50_res.drop(['F1','F10'],1)
    f50_res.rename(columns={'F50':'Score'},inplace=True)
    f50_res['F_score_name'] = 'F50'
    
    scores = [f1_res,f10_res,f50_res]
    scores = pd.concat(scores,axis=0)
    scores['Name'] = name
    scores['Reference'] = par_string
    scores = scores[['Name','Reference','F_score_name','Precision','Recall','TP','FP','TN','FN','Score']] 
    #saving to file
   
    scores = scores.reset_index()
    scores = scores.drop(['index'],1)
    
    total_scores = total_scores.append(scores)   
    total_scores = total_scores.reset_index()
    total_scores = total_scores.drop(['index'],1)
    
    
    
  
    f1_max = []
    f10_max = []
    f50_max = []
    f1_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F1']['Score'].astype('float64').idxmax()]).T
    f10_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F10']['Score'].astype('float64').idxmax()]).T
    f50_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F50']['Score'].astype('float64').idxmax()]).T
    
    summary = []
    summary = [f1_max,f10_max,f50_max]
    summary = pd.concat(summary,axis =0)
    summary.to_csv(file_name + "_summary.csv")
 
    