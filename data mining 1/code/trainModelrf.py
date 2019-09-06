import numpy as np
import pandas as pd
import os
import multiprocessing

from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def find_best_hyperparams(model_FDM, feature_col_names, target_col_name):

    # Utility function to report best scores
    def report(results, n_top=3):

        # setting up the global path
        global_path = os.path.dirname(__file__)

        with open(file=global_path + '/model/parameters/rf_params.txt', mode='w+') as f:

            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:

                    f.write("Model with rank: {0}".format(i))
                    f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
                    f.write("Parameters: {0}".format(results['params'][candidate]))
                    f.write("")

    # get some data
    labels = model_FDM[target_col_name].as_matrix()
    train_feature = model_FDM[feature_col_names].as_matrix()

    # get a classifier
    clf = RandomForestClassifier(n_jobs=1)


    # specify parameters and distributions to sample from
    param_dist = {"n_estimators": [96],
                  "max_depth": [2, 3, 4, 5, None],
                  "max_features": [2, 3, 4, 5, 6, 7, 8],
                  "min_samples_split": sp_randint(2, 5),
                  "min_samples_leaf": sp_randint(2, 10),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 1
    random_search = RandomizedSearchCV(estimator=clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       n_jobs=multiprocessing.cpu_count(),
                                       cv=3)

    start = time()
    random_search.fit(train_feature, labels)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_, n_iter_search)


def train_RamdomForest(model_FDM, feature_col_names, target_col_name):
    """
           ARGS
           model_FDM            a dataframe containing the imputed FDM data
           feature_col_names    a set of features to be used for training the model
           target_col_name      name of the col containing the target

           RETURNS
           a LightGBModel
    """

    # need get labels
    labels = model_FDM[target_col_name].as_matrix()
    train_feature = model_FDM[feature_col_names].as_matrix()

    # check the Parameters setting on http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    random_model = RandomForestClassifier(n_estimators=3, n_jobs=multiprocessing.cpu_count(), criterion='gini', max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                          max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                          min_impurity_split=None, bootstrap=True, oob_score=False,
                                          random_state=None, verbose=0, warm_start=False, class_weight=None)

    # begin Training
    print("Train RandomForest :-)")
    random_model.fit(train_feature, labels)

    return random_model


if __name__ == '__main__':

    HYPERPARAM_SEARCH = True

    # setting up the global path
    global_path = os.path.dirname(__file__)

    # Load data_train_threshold
    FDM_model = pd.read_pickle(global_path + '/feature_data/data_train_model.pkl')

    # set which feature we want to use for training model
    feature_col_names = ['aisle_DIST_CNT', 'days_between_orders_LOG_AVG',
                         'days_between_orders_LOG_STD', 'days_since_last_purchased_CNT',
                         'delay_between_purchases_LOG_AVG', 'delay_between_reorders_LOG_AVG',
                         'departments_DIST_CNT', 'occurances_per_order_PROP',
                         'orders_CNT', 'orders_TOT_COUNT', 'position_in_cart_LOG_AVG',
                         'position_in_cart_LOG_STD', 'prev_ord_as_pred_F1_AVG', 'prev_ord_as_pred_F1_STD',
                         'prev_ord_as_pred_PRECISION_AVG', 'prev_ord_as_pred_PRECISION_STD',
                         'prev_ord_as_pred_RECALL_AVG',
                         'prev_ord_as_pred_RECALL_STD', 'prods_ordered_DIST_CNT', 'prods_ordered_TOT_CNT',
                         'prods_per_order_LOG_AVG', 'prods_per_order_LOG_STD',
                         'proportion_of_orders_containing_PROP',
                         'proportion_reordered_PROP', 'purchased_CNT', 'reorders_CNT']

    # set target column name
    target_col_name = 'target'

    if HYPERPARAM_SEARCH:
        find_best_hyperparams(model_FDM=FDM_model, feature_col_names=feature_col_names,
                                     target_col_name=target_col_name)
    else:
        trained_model = train_RamdomForest(model_FDM=FDM_model, feature_col_names=feature_col_names,
                                                  target_col_name=target_col_name)
        # save the random forest model
        joblib.dump(trained_model, global_path + '/model/randomforest.pkl', compress=3)