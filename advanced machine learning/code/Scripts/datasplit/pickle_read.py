#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:35:18 2018

@author: diego
"""

import pandas as pd
import code.Scripts.datasplit.preprocessing as pp

splits = pd.read_pickle("../cvsplits/cvsplits.pkl")
df = pp.get_processed_data('../../Dataset/creditcard.csv')

for i, dataset in enumerate(splits):
    print ("hola")
    train_id = dataset[0]
    test_id = dataset[1]
    train = df.loc[train_id]
    test = df.loc[test_id]
