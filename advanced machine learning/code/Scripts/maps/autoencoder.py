#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../')

import pandas as pd
import datasplit.preprocessing as pp
import datasplit.datasplit as ds
import scoring.fb_score as fb
from sklearn.model_selection import train_test_split
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import datetime
import random

import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self,layers,input_size,layer_size):
        super(Net,self).__init__()
        self.layers = layers
        self.input_size = input_size-1
        
        vals = layer_size 
        self.fc1 = nn.Linear(self.input_size,vals[0])
        if self.layers ==2:
            self.fc4 = nn.Linear(vals[0],self.input_size)
        elif self.layers ==3:
            self.fc2 = nn.Linear(vals[0],vals[1])
            self.fc4 = nn.Linear(vals[1],self.input_size)
        elif self.layers == 4:
            self.fc2 = nn.Linear(vals[0],vals[1])
            self.fc3 = nn.Linear(vals[1],vals[2])
            self.fc4 = nn.Linear(vals[2],self.input_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        if self.layers == 3 or self.layers ==4:
            x = F.relu(self.fc2(x))
        if self.layers ==4:
            x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def layer_size(self):
        a = np.random.randint(self.input_size/3,self.input_size-1)
        b = np.random.randint(self.input_size/3,self.input_size-1)
        c = np.random.randint(self.input_size/3,self.input_size-1)
        return [a,b,c]
            
        
load_from_file = False
file_location = "./AE output/TOP1247-20180312T1405_epochs-20_Lin(31,27)-Lin(27,16)-Lin(16,31).pkl"

#parameters
no_epochs = 20
#Define threshold for negative/positive split
#threshold = np.logspace(np.log10(0.1), np.log10(100.0), num=200)
threshold = np.linspace(0,59,473)
#Number of iterations
no_iterations = 1

#set output folder
output_folder = "./AE output/"

#Reading and preprocessing data
df = pp.get_processed_data('../../Dataset/creditcard.csv')    

#Feature selection
#df = df[['V17','V14','V12','V7','V10','V3','V16','V11','V18','V5','V1','V4','Class']]

#error
my_error = []

#Reading splits
splits = pd.read_pickle("../cvsplits/cvsplits.pkl")

#File name
now = datetime.datetime.now()
time = now.strftime("%Y%m%dT%H%M")
file_name = output_folder + time + "_combinations_" + str(no_iterations)

#Total score
total_scores = pd.DataFrame(columns=['Name','Reference','F_score_name','Precision','Recall','TP','FP','TN','FN','Score','Threshold'])

#iterate 
for m in range(no_iterations):
    #Initialise net
    input_size = df.shape[1]
    
    #Number of units per layer
    a = np.random.randint(input_size/3,input_size)
    a = 25
    b = np.random.randint(input_size/3,input_size)
    c = np.random.randint(input_size/3,input_size)
    vals=[a,b,c]
    layer_no = 2#random.randint(3,4)
    train_size = random.choice([500,1000,10000,50000,100000,200000])
    train_size = random.choice([200000])
    
    
    
    #Statistics container
    total_statistics = []
    
    #Defining classifier name
    name = 'AE'
    reference= "epochs-" + str(no_epochs) + "_L(%i),vals(%i,%i,%i)(%i)" %(layer_no,a,b,c,train_size)
    
    
        
    #Architecture
    print ("\n********* Iteration %i/%i *********" % (m+1,no_iterations) )
    print("Classifier architecture:")
    print(name + "_" + reference)
    print("Input size: %i" % train_size)
    #Iterate through the CV sets
    for i,dataset in enumerate(splits[0:1]):
        print("------ Cross validation step %i ------" % (i+1))
        #indices for train and test set
        train_id = dataset[0]
        test_id = dataset[1]
        #generate the splits
        train = df.loc[train_id]
        train = ds.reduceSize(train,train_size)
        
        test = df.loc[test_id]
    
       
        #Divide set into training and test set
        #Train set. No need of y_train
        X_train = train[train.Class == 0]
        X_train = X_train.drop(['Class'], axis=1)
        
        #Test set
        X_test = test.drop(['Class'], axis=1)
        y_test = test['Class']
        
        #To array
        X_train = X_train.values
        X_test = X_test.values
        y_test = y_test.values
    
    
        #Generate batches to train the autoencoder
        trainloader = torch.utils.data.DataLoader(X_train, batch_size=32,shuffle=True, num_workers=2)
        
        #Turn X_test into a pyTorch Variable
        X_test = torch.from_numpy(X_test)
        X_test = Variable(X_test).float()
     
        #Initialise neural net
        net = Net(layer_no,input_size,vals)
        optimizer = torch.optim.Adam(net.parameters())    
        criterion = nn.MSELoss()
        net.zero_grad()
            
        loss_list = []
        
       
        if load_from_file == True:
            net = torch.load(file_location)
        else:
            #Model training
            for epoch in range(no_epochs):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    
                    #make inputs and labels the same
                    inp = Variable(data).float()
                    labels = Variable(data).float()
                    
                    net.zero_grad()
                    
                    #calculate outputs
                    outputs = net(inp)
                    
                    #update weights
                    optimizer.zero_grad()
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.data[0]
                print("Epoch: %s, loss: %.3f" %(epoch+1,running_loss))
                loss_list.append(running_loss)
        #prediction
        predictions = net(X_test)
        
        #calculate MSE of the reconstructed data
        mse = np.mean(np.power(X_test.data.numpy() - predictions.data.numpy(), 2), axis=1)
        my_error = mse
        error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})
    
        y_pred = []
      
        fb_list = []
        statistics = []
        for i,th in enumerate(threshold):       
            th_error = error_df.copy()
            y_pred = []
            error = th_error['reconstruction_error'].values
            for j in range(len(y_test)):
                if error[j] > th:
                    y_pred.append(1)
                else:
                    y_pred.append(0)        
    
            [precision,recall],[tp,fp,tn,fn] = fb.precisionRecallScore(y_test,y_pred)
            f1 = fb.fbScore(precision,recall,1)
            f10 = fb.fbScore(precision,recall,10)
            f50 = fb.fbScore(precision,recall,50)
            st = [th,precision,recall,tp,fp,tn,fn,f1,f10,f50]
            statistics.append(st)
        
        statistics = np.array(statistics)
        total_statistics.append(statistics)
        
    my_error.to_csv(file_name + "_error.csv")
        
    kfold_no = len(total_statistics)
    total = np.array(total_statistics)
    total_mean = total.mean(0)
    total_dev = total.std(0)
        
    th_df = pd.DataFrame({
            'Threshold': total_mean[:,0],
            'Precision':total_mean[:,1],
            'Recall':total_mean[:,2],
            'TP':total_mean[:,3],
            'FP':total_mean[:,4],
            'TN':total_mean[:,5],
            'FN':total_mean[:,6],
            'F1':total_mean[:,7],
            'F10':total_mean[:,8],
            'F50':total_mean[:,9],
            })
    th_df = th_df[['Threshold','Precision','Recall','TP','FP','TN','FN','F1','F10','F50']] 
    
    #Define reference name
    my_string=""
    for i, layer in enumerate(list(net.children())):
        if layer.__class__.__name__ != 'Dropout':
            my_string = "%s(%i,%i)" %(layer.__class__.__name__[:3],layer.in_features,layer.out_features)
        else:
            my_string = "%s(%.2f)" %(layer.__class__.__name__[:3],layer.p)
        reference = reference + my_string + "-"
    reference = reference[:-1]
    
    
    
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
    scores['Reference'] = reference
    scores = scores[['Name','Reference','F_score_name','Precision','Recall','TP','FP','TN','FN','Score','Threshold']] 
    #saving to file
    """
    now = datetime.datetime.now()
    time = now.strftime("%Y%m%dT%H%M")
    file_name = output_folder + time + "_epochs-" + str(no_epochs) + "_"
    
    for i, layer in enumerate(list(net.children())):
        if layer.__class__.__name__ != 'Dropout':
            my_string = "%s(%i,%i)" %(layer.__class__.__name__[:3],layer.in_features,layer.out_features)
        else:
            my_string = "%s(%.2f)" %(layer.__class__.__name__[:3],layer.p)
        file_name = file_name + my_string + "-"
        
    file_name = file_name[:-1] + ".csv"
    th_df.to_csv(file_name)
    """
    total_scores = total_scores.append(scores)    
    total_scores.to_csv(file_name+".csv")
    total_scores = total_scores.reset_index()
    total_scores = total_scores.drop(['index'],1)
    
    f1_max = []
    f10_max = []
    f50_max = []
    f1_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F1']['Score'].idxmax()]).T
    f10_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F10']['Score'].idxmax()]).T
    f50_max = pd.DataFrame(total_scores.loc[total_scores[total_scores['F_score_name']=='F50']['Score'].idxmax()]).T
    
    summary = []
    summary = [f1_max,f10_max,f50_max]
    summary = pd.concat(summary,axis =0)
    summary.to_csv(file_name + "_summary.csv")