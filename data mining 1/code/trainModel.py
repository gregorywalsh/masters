from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import pandas as pd
import os
from time import time
import numpy as np
from sklearn.externals import joblib


def find_best_hyperparams(model_FDM, feature_col_names, target_col_name):
    # Utility function to report best scores

    # Utility function to report best scores
    def report(results, n_top=3):

        # setting up the global path
        global_path = os.path.dirname(__file__)

        with open(global_path + '/model/parameters/lgbm_params.txt','w') as f:

            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    f.write("Model with rank: {0}".format(i))
                    f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
                    f.write("Parameters: {0}".format(results['params'][candidate]))
                    f.write("")

    labels = model_FDM[target_col_name].as_matrix()
    train_feature = model_FDM[feature_col_names].as_matrix()
    del model_FDM

    lgb_classifier = LGBMClassifier(learning_rate=0.05,boosting_type='gbdt',objective='binary')


    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [8, 16, 24],
        'num_leaves': [6, 8, 12, 16],
        'colsample_bytree': [0.64, 0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
        'max_depth' : [5, 6, 7, 8, 9, 10]
    }

    random_search = RandomizedSearchCV(estimator=lgb_classifier,
                                       param_distributions=gridParams,
                                       n_iter=25,
                                       n_jobs=4)
    start = time()
    random_search.fit(train_feature, labels)
    report(random_search.cv_results_)
    quit()
# {'subsample': 0.7, 'reg_lambda': 1.4, 'reg_alpha': 1.2, 'num_leaves': 16, 'n_estimators': 24, 'max_depth': 7, 'learning_rate': 0.005, 'colsample_bytree': 0.65}
def train_LGBModel(model_FDM, feature_col_names, target_col_name):
    """
           ARGS
           model_FDM            a dataframe containing the imputed FDM data
           feature_col_names    a set of features to be used for training the model
           target_col_name      name of the col containing the target

           RETURNS
           a LightGBModel
    """

    labels = model_FDM[target_col_name].as_matrix()
    train_feature = model_FDM[feature_col_names].as_matrix()

    # delete the origin model_FDM for relese cache
    del model_FDM

    # begin Training process
    print('Light GMB training :-)')

    lgb_classifier = LGBMClassifier(
            subsample= 0.7,
             reg_lambda= 1.4,
             reg_alpha= 1.2,
             num_leaves= 16,
             n_estimators=24,
             max_depth = 7,
             learning_rate=0.005,
             colsample_bytree=0.65,
             boosting_type='gbdt',
             objective='binary')
    lgb_classifier.fit(train_feature,labels)
    # delete the training style lgb_train for relese cache
    # del lgb_train

    return lgb_classifier

if __name__ == '__main__':

    HYPERPARAM_SEARCH = False

    # setting up the global path
    global_path = os.path.dirname(__file__)

    # Load data_train_threshold
    # FDM_model = pd.read_pickle(global_path + '/feature_data/data_train_model.pkl')
    FDM_model = pd.read_pickle(global_path + '/feature_data/FDM_model.pkl')

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
        trained_model = train_LGBModel(model_FDM=FDM_model,
                                       feature_col_names=feature_col_names,
                                       target_col_name=target_col_name)

        # save the lightGBM model
        joblib.dump(trained_model, global_path + '/model/lightGBM.pkl')