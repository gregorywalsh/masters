import numpy as np
import pandas as pd
import csv
import random as rand

import code.Scripts.datasplit.preprocessing as pp

from statistics import mean, stdev
from time import localtime, strftime
from sklearn.svm import LinearSVC, SVC, OneClassSVM

from code.Scripts.modeltesting import RandomisedClassifierValidator
from code.Scripts.modeltesting import ClassifierDefinition


def get_stats(results, model_index, num_splits, betas):

    def get_pos_neg_counts(results, model_index, num_splits):
        tps = []
        tns = []
        fps = []
        fns = []
        for split_num in range(num_splits):
            tps.append(results['split' + str(split_num) + '_test_True Pos'][model_index])
            tns.append(results['split' + str(split_num) + '_test_True Neg'][model_index])
            fps.append(results['split' + str(split_num) + '_test_False Pos'][model_index])
            fns.append(results['split' + str(split_num) + '_test_False Neg'][model_index])
        return tps, tns, fps, fns

    tps, tns, fps, fns = get_pos_neg_counts(
        results=results,
        model_index=model_index,
        num_splits=num_splits
    )

    pos_neg_counts = (tps, tns, fps, fns)
    tp_mean, tn_mean, fp_mean, fn_mean = [mean(counts) for counts in pos_neg_counts]
    # tp_std, tn_std, fp_std, fn_std = [stdev(counts) for counts in pos_neg_counts]

    precisions = [tp / (tp + fp) if (tp + fp) > 0 else 0 for tp, fp in zip(tps, fps)]
    recalls = [tp / (tp + fn)  if (tp + fn) > 0 else 0 for tp, fn in zip(tps, fns)]
    precision_mean, recall_mean = [mean(statistic) for statistic in [precisions, recalls]]
    # precision_std, recall_std = [stdev(statistic) for statistic in [precisions, recalls]]

    fbeta_means = []
    fbeta_stds = []
    for beta in betas:
        fbetas = []
        beta_sqd = beta ** 2
        for precision, recall in zip(precisions, recalls):
            fbetas.append(((1 + beta_sqd) * (precision * recall)) / ((beta_sqd * precision) + recall) if (precision + recall) > 0 else 0)
        fbeta_means.append(mean(fbetas))
        fbeta_stds.append(stdev(fbetas))

    return [pos_neg_counts, tp_mean, tn_mean, fp_mean, fn_mean, precision_mean, recall_mean] + fbeta_means + fbeta_stds


# ======== SCRIPT VARIABLES ======== #
df = pp.load_data(path='../../../Dataset/creditcard.csv')
df_x = df.loc[:, df.columns.difference(['Class'])]
sr_y = df.loc[:, 'Class']
splits = pd.read_pickle("../../cvsplits/cvsplits.pkl")
num_splits = 3
betas = [1, 10, 50]

# ======== BINARY CLASSIFIERS ========= #
binary_svm_hyperparam_span = {
    'C': list(1 * 10 ** exponent for exponent in np.linspace(-2, 1.5, 1000)),
    'kernel': ['rbf'],
    'gamma': list(1 * 10 ** exponent for exponent in np.linspace(-8, -3, 1000)),
    'shrinking': [False],
    'cache_size': [6144],
    'class_weight': ['balanced', None]
}

binary_svm_hyperparam_span_knn = {
    'C': [0.5002639399774802],
    'kernel': ['rbf'],
    'gamma': [0.0001618859690178199],
    'shrinking': [False],
    'cache_size': [6144],
    'class_weight': ['balanced', None]
}

binary_linearsvm_hyperparam_span = {
    # l1 hn t N
    # l1 hn f N
    # l1 sh t N
    # l1 sh f Y
    # l2 hn t Y
    # l2 hn f N
    # l2 sh t Y
    # l2 sh f Y
    'penalty': ['l1', 'l2',],
    'loss': ['squared_hinge',],
    'dual': [False],
    'tol': [1e-4],
    'C': list(1 * 10 ** exponent for exponent in np.linspace(-3, 3, 1000)),
    'multi_class': ['ovr', 'crammer_singer'],
    'intercept_scaling': [1],
    'class_weight': ['balanced', None]
}

# binary_definitions = [
#     ClassifierDefinition(
#         type='binary',
#         map=SVC,
#         map_hyperparams = binary_svm_hyperparam_span,
#         scaler=pp.CustomScaler,
#         scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
#         num_models=500,
#         class_id_mapping=None,
#         sample_size_reqs={0: 500, 1: 500},
#         positive_mapped_class_id=1
#     ),
#     ClassifierDefinition(
#         type='binary',
#         map=LinearSVC,
#         map_hyperparams = binary_linearsvm_hyperparam_span,
#         scaler=pp.CustomScaler,
#         scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
#         num_models=500,
#         class_id_mapping=None,
#         sample_size_reqs={0: 2000, 1: 2000},
#         positive_mapped_class_id=1
#     )
# ]

binary_definitions = [
    ClassifierDefinition(
        type='binary',
        map=SVC,
        map_hyperparams = binary_svm_hyperparam_span_knn,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
        num_models=2,
        class_id_mapping=None,
        sample_size_reqs={0: 500, 1: 500},
        positive_mapped_class_id=1
    )
]


# ======== EXPERIMENTAL BINARY WITH SUBCLASSES CLASSIFIERS ======== #
# sample_size_requirements = {0: 500, 1: 290, 2: 95, 3: 127}

binary_with_subclass_svm_hyperparam_span = {
    'C': list(1 * 10 ** exponent for exponent in np.linspace(-2, 1, 1000)),
    'kernel': ['rbf'],
    'gamma': list(1 * 10 ** exponent for exponent in np.linspace(-8, -3, 1000)),
    'shrinking': [False],
    'cache_size': [6144],
    'class_weight': ['balanced', None]
}

binary_with_subclass_definitions = [
    ClassifierDefinition(
        type='binary_with_subclasses',
        map=SVC,
        map_hyperparams=binary_with_subclass_svm_hyperparam_span,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
        num_models=100,
        class_id_mapping=None,
        sample_size_reqs={0: 500, 1: 290, 2: 95, 3: 127},
        positive_mapped_class_id=1
    ),
    ClassifierDefinition(
        type='binary_with_subclasses',
        map=SVC,
        map_hyperparams = binary_with_subclass_svm_hyperparam_span,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
        num_models=100,
        class_id_mapping={0: 0, 1: 1, 2:1, 3:2},
        sample_size_reqs={0: 500, 1: 500, 2:500},
        positive_mapped_class_id=1
    ),
    ClassifierDefinition(
        type='binary_with_subclasses',
        map=SVC,
        map_hyperparams = binary_with_subclass_svm_hyperparam_span,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
        num_models=100,
        class_id_mapping={0: 0, 1: 1, 2:2, 3:1},
        sample_size_reqs={0: 500, 1: 500, 2:500},
        positive_mapped_class_id=1
    ),
    ClassifierDefinition(
        type='binary_with_subclasses',
        map=SVC,
        map_hyperparams = binary_with_subclass_svm_hyperparam_span,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes':[[1, 2]]}, #TimeComponents
        num_models=100,
        class_id_mapping={0: 0, 1: 1, 2:2, 3:2},
        sample_size_reqs={0: 500, 1: 500, 2:500},
        positive_mapped_class_id=1
    )
]


# ======== OUTLIER CLASSIFIER ======== #

one_class_svm_hyperparam_span = {
    'kernel': ['rbf'],
    'nu': list(x for x in np.linspace(0.00001, 1, 1000)),
    'gamma': list(1 * 10 ** exponent for exponent in np.linspace(-6, 3, 1000)),
    'cache_size': [6144]
}

outlier_definitions = [
    ClassifierDefinition(
        type='outlier',
        map=OneClassSVM,
        map_hyperparams=one_class_svm_hyperparam_span,
        scaler=pp.CustomScaler,
        scaler_hyperparams={'ignored_column_indexes': [[1, 2]]}, #TimeComponents
        num_models=2,
        class_id_mapping = {0: -1, 1: 1},
        sample_size_reqs = {-1: 0, 1: 10},
        positive_mapped_class_id=1
    ),
    # ClassifierDefinition(
    #     type='outlier',
    #     map=OneClassSVM,
    #     map_hyperparams=one_class_svm_hyperparam_span,
    #     scaler=pp.CustomScaler,
    #     scaler_hyperparams={'ignored_column_indexes': [[1, 2]]}, #TimeComponents
    #     num_models=2,
    #     class_id_mapping={0: 1, 1: -1},
    #     sample_size_reqs={-1: 0, 1: 10},
    #     positive_mapped_class_id=-1
    # )
]


# ======== RUN VALIDATION ======== #
print('Started:{}\n'.format(strftime("%d %b %Y %H:%M", localtime())))
all_definitions = binary_definitions + binary_with_subclass_definitions + outlier_definitions
for classifier_definition in binary_definitions:

    print('MAP -', classifier_definition.map.__name__, '\n')
    for key, value in classifier_definition.__dict__.items():
        print('{0}:'.format(key))
        print('\t', value)
    print()

    sr_y_copy = sr_y.copy()
    if classifier_definition.type == 'binary_with_subclasses':
        sr_subclasses = pd.read_pickle('../cvsplits/df_subclasses.pkl')
        sr_y_copy.loc[sr_subclasses.index] = sr_subclasses
    if classifier_definition.class_id_mapping is not None:
        sr_y_copy = sr_y_copy.map(classifier_definition.class_id_mapping)

    rand.shuffle(splits)
    splits = splits[0:min(len(splits), num_splits)]

    rcv = RandomisedClassifierValidator(
        classifier_definition=classifier_definition,
        df_x=df_x,
        sr_y=sr_y_copy,
        splits=splits,
    )

    results = rcv.validate()
    with open(file='validator_results.csv', mode='a') as outfile:
        writer = csv.writer(outfile)
        models = results['params']
        for model_index, model in enumerate(models):
            row = [
                classifier_definition.type,
                classifier_definition.map.__name__,
                classifier_definition.scaler_hyperparams,
                classifier_definition.class_id_mapping,
                classifier_definition.sample_size_reqs,
                classifier_definition.positive_mapped_class_id,
                num_splits,
            ]
            row.append(model)

            stats = get_stats(results=results, model_index=model_index, num_splits=num_splits, betas=betas)
            row = row + stats
            writer.writerow(row)

    print('\n')
    print('='*60, '\n')
