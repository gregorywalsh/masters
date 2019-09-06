import numpy as np
import pandas as pd
import pickle
import os.path as path
import random as rand

import code.Scripts.datasplit.preprocessing as pp
import code.Scripts.datasplit.clustering as cl
import code.Scripts.modeltesting.autovalidator.utilities as ut

from time import time
from multiprocessing import cpu_count
from warnings import warn

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, fbeta_score, precision_score, recall_score


# ============ SETTINGS ============ #

# Set estimator
classifier_map = SVC

# Apply subclassing
apply_subclassing = True

# Define targets of class IDs if required (dict or None) (subclass ID 2 is the most similar to the negatives)
class_targets = {0: -1, 1: 1, 2: -1, 3: 1}
class_targets = {0: 0, 1: 1, 2: 2, 3: 1}

# Set positive class ID
pos_class_id = 1

# Optionally define sample requirements (dict or None)
# sample_size_requirements = {0: 500, 1: 290, 2: 95, 3: 127}
sample_size_requirements = {0: 500, 1: 500, 2: 500}

# Number of machines to train
num_machines = 20

# Parameters and distributions to sample from
hyperparam_span = {'kernel': ['rbf'],
                  'nu': list(x for x in np.linspace(0.00001, 1, 1000)),
                  'gamma': list(1*10**exponent for exponent in np.linspace(-6, 2, 1000)),
                   #'coef0': list(x for x in np.linspace(-1, 1, 100)),
                  'cache_size': [6144]}

class_weight_span = [dict([(0, 1), (1, 1), (2, x)]) for x in np.linspace(1, 2, 100)]

hyperparam_span = {'C': list(1 * 10 ** exponent for exponent in np.linspace(-2, 1, 1000)),
                  'kernel': ['rbf'],
                  'gamma': list(1*10**exponent for exponent in np.linspace(-8, -3, 1000)),
                  'cache_size': [6144],
                  'class_weight': ['balanced', None]
                   }

# map_param_span = {'nu': [0.09090909090909091], 'kernel': ['rbf'], 'gamma': [8.697490026177834e-05], 'degree': range(0,1000), 'coef0': [0.2525252525252526], 'cache_size': [6144], 'shrinking': [True, False]}
hyperparam_span = {'kernel': ['rbf'], 'gamma': [0.00020621218039991424], 'degree': range(0, 100), 'class_weight': [{0: 1.0, 1: 1.0, 2: 0.6}], 'cache_size': [6144], 'C': [0.5764488282925876]}

# Scoring
fbeta = {'name': 'fbeta', 'function': fbeta_score, 'kwargs': {'beta': 1, 'pos_label': pos_class_id}}
precision = {'name': 'precision', 'function': precision_score, 'kwargs': {'pos_label': pos_class_id}}
recall = {'name': 'recall', 'function': recall_score, 'kwargs': {'pos_label': pos_class_id}}
scores = [fbeta, precision, recall]
ranking_score = fbeta

# CPU's to use
use_cpus = cpu_count() - 1

# Num splits to use
num_splits = 10

# ========== CHECK SETTINGS ========= #
if apply_subclassing:
    if len(sample_size_requirements) < 3:
        warn('Warning: You have selected subclass analysis, but only specified sample size requirements for two classes')


# ============ LOAD DATA ============ #

# Get data
print('======== LOADING DATA ========\n\n')
print()

df = pp.get_processed_data(path='../../Dataset/creditcard.csv')
df_x = df.loc[:, df.columns.difference(['Class'])]
sr_y = df.loc[:, 'Class'].copy()

# Subclass the positives if necessary
if apply_subclassing:
    if path.isfile('../cvsplits/df_subclasses.pkl'):
        sr_subclasses = pd.read_pickle('../cvsplits/df_subclasses.pkl')
    else:
        df_x_positives = df_x.loc[sr_y == 1]
        clusterer = cl.TSNEClusterer(df_x=df_x_positives)
        sr_subclasses = clusterer.get_subclasses()
        clusterer.plot_clusters()
        with open('../cvsplits/df_subclasses.pkl', 'wb') as f:
            pickle.dump(obj=sr_subclasses, file=f)
    sr_y.loc[sr_subclasses.index] = sr_subclasses

# Map class targets if specified
if class_targets is not None:
    sr_y = sr_y.map(class_targets)

# Load the splits
splits = pd.read_pickle("../cvsplits/cvsplits.pkl")
rand.shuffle(splits)
splits = splits[0:min(len(splits), num_splits)]


# ============ RUN VALIDATION ============ #

# Create pipeline
estimator = ut.MetaEstimator(estimator_class=classifier_map, sample_size_requirements=sample_size_requirements)
pipeline = make_pipeline(StandardScaler(), estimator)
pipeline_param_span = dict((type(estimator).__name__.lower() + '__' + key, value) for (key, value) in hyperparam_span.items())

# Create Scorers
scorers = {}
for score in scores:
    if apply_subclassing:
        score_function = ut.SubClassScore(score['function'])
    else:
        score_function = score['function']
    scorer = make_scorer(score_function, **score['kwargs'])
    scorers[score['name']] = scorer

# Create multi-threaded randomised search cross validator
random_search = RandomizedSearchCV(estimator=estimator,
                                   param_distributions=hyperparam_span,
                                   n_iter=num_machines,
                                   scoring=scorers,
                                   n_jobs=use_cpus,
                                   pre_dispatch=use_cpus,
                                   cv=splits,
                                   refit=False,
                                   verbose=True)

# Fit and validate the models
print('======== VALIDATING ========')
print()
print('Ranked by {0}'.format(ranking_score['name']))
print()

for score in scores:
    print(score['name'], 'parameters:')
    if len(score['kwargs']) == 0:
        print('\t None')
    else:
        for key, value in score['kwargs'].items():
            print('\t{0} = {1:.3f}'.format(key, value))
print()

start = time()
random_search.fit(X=df_x.values, y=sr_y.values)

# Report on performance
print()
print("%d machines - running time of %.2fs" % (num_machines, (time() - start)))
print()
ut.print_results(results=random_search.cv_results_,
                 score_names=scorers.keys(),
                 ranking_score_name='fbeta',
                 n_top=min(num_machines, 3))
